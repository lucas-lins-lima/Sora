# sora_robot/audio_processing/speech_recognition.py

import threading
import time
import queue
import io
import numpy as np
from typing import Optional, List, Dict, Callable, Tuple, NamedTuple
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import wave
import json

# Google Speech-to-Text
try:
    from google.cloud import speech
    GOOGLE_STT_AVAILABLE = True
except ImportError:
    GOOGLE_STT_AVAILABLE = False
    speech = None

# Whisper (local)
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    whisper = None

# SpeechRecognition (fallback)
try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False
    sr = None

from utils.logger import get_logger
from utils.constants import SPEECH_RECOGNITION, AUDIO_SETTINGS
from audio_processing.microphone_handler import AudioChunk, VoiceActivity
import config

class RecognitionEngine(Enum):
    """Engines de reconhecimento de fala disponíveis."""
    GOOGLE_CLOUD = "google_cloud"
    WHISPER_LOCAL = "whisper_local"
    SPEECH_RECOGNITION = "speech_recognition"
    AUTO = "auto"

class RecognitionStatus(Enum):
    """Status do reconhecimento de fala."""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"

@dataclass
class SpeechSegment:
    """Representa um segmento de fala reconhecido."""
    text: str
    confidence: float
    start_time: float
    end_time: float
    language: str
    
    # Alternativas de reconhecimento
    alternatives: List[Tuple[str, float]] = field(default_factory=list)
    
    # Informações de palavras individuais (se disponível)
    words: List[Dict] = field(default_factory=list)  # [{"word": str, "confidence": float, "start_time": float, "end_time": float}]
    
    # Dados de áudio originais
    audio_data: Optional[np.ndarray] = None
    sample_rate: int = 44100
    
    # Metadados
    engine_used: str = ""
    processing_time: float = 0.0
    is_interim: bool = False  # Para resultados de streaming

@dataclass
class RecognitionResult:
    """Resultado completo de reconhecimento de fala."""
    segments: List[SpeechSegment] = field(default_factory=list)
    full_text: str = ""
    overall_confidence: float = 0.0
    total_duration: float = 0.0
    language_detected: str = ""
    
    # Estatísticas
    total_words: int = 0
    average_word_confidence: float = 0.0
    speech_rate: float = 0.0  # palavras por minuto
    
    # Metadados
    timestamp: float = 0.0
    engine_used: str = ""
    processing_time: float = 0.0

class SpeechRecognition:
    """
    Classe responsável pelo reconhecimento de fala em tempo real.
    Suporta múltiplos engines de reconhecimento e processamento streaming.
    """
    
    def __init__(self, engine: RecognitionEngine = RecognitionEngine.AUTO, language: str = "pt-BR"):
        """
        Inicializa o sistema de reconhecimento de fala.
        
        Args:
            engine: Engine de reconhecimento a ser usado
            language: Idioma principal para reconhecimento
        """
        self.logger = get_logger(__name__)
        
        # Configurações
        self.target_engine = engine
        self.language = language
        self.current_engine = None
        
        # Estado do sistema
        self.status = RecognitionStatus.IDLE
        self.is_processing = False
        
        # Threading
        self.processing_thread = None
        self.audio_queue = queue.Queue(maxsize=50)
        self.result_queue = queue.Queue(maxsize=20)
        self.stop_event = threading.Event()
        self.processing_lock = threading.Lock()
        
        # Buffers de áudio
        self.current_audio_buffer = deque()
        self.speech_segments = deque(maxlen=100)
        
        # Engines disponíveis
        self.available_engines = self._detect_available_engines()
        self.active_engine = None
        
        # Callbacks
        self.text_callbacks = []  # Callbacks para texto reconhecido
        self.interim_callbacks = []  # Callbacks para resultados intermediários
        self.error_callbacks = []  # Callbacks para erros
        
        # Cache de reconhecimento
        self.recognition_cache = {}
        self.cache_ttl = 60  # 1 minuto
        
        # Configurações de qualidade
        self.min_confidence = SPEECH_RECOGNITION.CONFIDENCE_THRESHOLD
        self.min_speech_length = SPEECH_RECOGNITION.MIN_SPEECH_LENGTH
        self.max_speech_length = SPEECH_RECOGNITION.MAX_SPEECH_LENGTH
        
        # Métricas de performance
        self.performance_metrics = {
            'total_recognitions': 0,
            'successful_recognitions': 0,
            'failed_recognitions': 0,
            'average_processing_time': 0.0,
            'processing_times': deque(maxlen=100),
            'confidence_scores': deque(maxlen=100),
            'engines_used': {},
            'languages_detected': {},
            'total_audio_processed': 0.0  # segundos
        }
        
        # Inicializa engine
        self._initialize_recognition_engine()
        
        self.logger.info(f"SpeechRecognition inicializado - Engine: {self.current_engine}, Idioma: {self.language}")
    
    def _detect_available_engines(self) -> List[RecognitionEngine]:
        """Detecta engines de reconhecimento disponíveis."""
        available = []
        
        if GOOGLE_STT_AVAILABLE and hasattr(config, 'API_KEYS') and config.API_KEYS.get('google_speech_to_text'):
            available.append(RecognitionEngine.GOOGLE_CLOUD)
            self.logger.info("Google Cloud Speech-to-Text disponível")
        
        if WHISPER_AVAILABLE:
            available.append(RecognitionEngine.WHISPER_LOCAL)
            self.logger.info("Whisper (local) disponível")
        
        if SPEECH_RECOGNITION_AVAILABLE:
            available.append(RecognitionEngine.SPEECH_RECOGNITION)
            self.logger.info("SpeechRecognition library disponível")
        
        if not available:
            self.logger.error("Nenhum engine de reconhecimento de fala disponível!")
        
        return available
    
    def _initialize_recognition_engine(self):
        """Inicializa o engine de reconhecimento selecionado."""
        if self.target_engine == RecognitionEngine.AUTO:
            # Seleciona melhor engine disponível
            if RecognitionEngine.GOOGLE_CLOUD in self.available_engines:
                self.current_engine = RecognitionEngine.GOOGLE_CLOUD
            elif RecognitionEngine.WHISPER_LOCAL in self.available_engines:
                self.current_engine = RecognitionEngine.WHISPER_LOCAL
            elif RecognitionEngine.SPEECH_RECOGNITION in self.available_engines:
                self.current_engine = RecognitionEngine.SPEECH_RECOGNITION
            else:
                raise Exception("Nenhum engine de reconhecimento disponível")
        else:
            if self.target_engine in self.available_engines:
                self.current_engine = self.target_engine
            else:
                raise Exception(f"Engine {self.target_engine} não disponível")
        
        # Inicializa engine específico
        try:
            if self.current_engine == RecognitionEngine.GOOGLE_CLOUD:
                self._initialize_google_stt()
            elif self.current_engine == RecognitionEngine.WHISPER_LOCAL:
                self._initialize_whisper()
            elif self.current_engine == RecognitionEngine.SPEECH_RECOGNITION:
                self._initialize_speech_recognition()
            
            self.logger.info(f"Engine {self.current_engine.value} inicializado com sucesso")
            
        except Exception as e:
            self.logger.error(f"Erro ao inicializar engine {self.current_engine}: {e}")
            raise
    
    def _initialize_google_stt(self):
        """Inicializa Google Cloud Speech-to-Text."""
        try:
            # Configura credenciais se necessário
            if hasattr(config, 'API_KEYS') and config.API_KEYS.get('google_speech_to_text'):
                # As credenciais são gerenciadas via environment variable ou service account
                pass
            
            self.google_client = speech.SpeechClient()
            
            # Configuração base
            self.google_config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=AUDIO_SETTINGS.SAMPLE_RATE,
                language_code=self.language,
                alternative_language_codes=['en-US'] if self.language != 'en-US' else [],
                max_alternatives=3,
                profanity_filter=True,
                enable_word_time_offsets=True,
                enable_word_confidence=True,
                enable_automatic_punctuation=True,
                model='latest_long'  # Modelo otimizado para áudio mais longo
            )
            
            # Configuração de streaming
            self.google_streaming_config = speech.StreamingRecognitionConfig(
                config=self.google_config,
                interim_results=True,
                single_utterance=False
            )
            
            self.active_engine = "google_cloud"
            
        except Exception as e:
            self.logger.error(f"Erro ao inicializar Google STT: {e}")
            raise
    
    def _initialize_whisper(self):
        """Inicializa Whisper (local)."""
        try:
            # Carrega modelo Whisper
            model_size = "base"  # small, base, large
            self.whisper_model = whisper.load_model(model_size)
            
            self.logger.info(f"Modelo Whisper '{model_size}' carregado")
            self.active_engine = "whisper_local"
            
        except Exception as e:
            self.logger.error(f"Erro ao inicializar Whisper: {e}")
            raise
    
    def _initialize_speech_recognition(self):
        """Inicializa SpeechRecognition library."""
        try:
            self.sr_recognizer = sr.Recognizer()
            
            # Ajusta parâmetros
            self.sr_recognizer.energy_threshold = 300
            self.sr_recognizer.dynamic_energy_threshold = True
            self.sr_recognizer.pause_threshold = 0.8
            self.sr_recognizer.phrase_threshold = 0.3
            
            self.active_engine = "speech_recognition"
            
        except Exception as e:
            self.logger.error(f"Erro ao inicializar SpeechRecognition: {e}")
            raise
    
    def start_recognition(self) -> bool:
        """
        Inicia processamento de reconhecimento de fala.
        
        Returns:
            bool: True se iniciado com sucesso
        """
        if self.is_processing:
            self.logger.warning("Reconhecimento de fala já está em execução")
            return True
        
        try:
            self.is_processing = True
            self.status = RecognitionStatus.LISTENING
            self.stop_event.clear()
            
            # Inicia thread de processamento
            self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
            self.processing_thread.start()
            
            self.logger.info("Reconhecimento de fala iniciado")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao iniciar reconhecimento: {e}")
            self.is_processing = False
            self.status = RecognitionStatus.ERROR
            return False
    
    def stop_recognition(self):
        """Para o processamento de reconhecimento."""
        if not self.is_processing:
            return
        
        self.logger.info("Parando reconhecimento de fala...")
        
        self.is_processing = False
        self.status = RecognitionStatus.IDLE
        self.stop_event.set()
        
        # Para thread
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5.0)
        
        # Limpa queues
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                break
        
        # Limpa buffers
        self.current_audio_buffer.clear()
        
        self.logger.info("Reconhecimento de fala parado")
    
    def process_audio_chunk(self, audio_chunk: AudioChunk):
        """
        Processa chunk de áudio para reconhecimento.
        
        Args:
            audio_chunk: Chunk de áudio do microfone
        """
        try:
            if not self.is_processing:
                return
            
            # Adiciona à queue para processamento
            try:
                self.audio_queue.put_nowait(audio_chunk)
            except queue.Full:
                # Remove chunk mais antigo se queue estiver cheia
                try:
                    self.audio_queue.get_nowait()
                    self.audio_queue.put_nowait(audio_chunk)
                except queue.Empty:
                    pass
            
        except Exception as e:
            self.logger.error(f"Erro ao processar chunk de áudio: {e}")
    
    def process_speech_segment(self, audio_data: np.ndarray, sample_rate: int = None) -> Optional[RecognitionResult]:
        """
        Processa um segmento completo de fala.
        
        Args:
            audio_data: Dados de áudio
            sample_rate: Taxa de amostragem (usa padrão se None)
            
        Returns:
            Optional[RecognitionResult]: Resultado do reconhecimento
        """
        if sample_rate is None:
            sample_rate = AUDIO_SETTINGS.SAMPLE_RATE
        
        start_time = time.time()
        
        try:
            # Verifica duração do áudio
            duration = len(audio_data) / sample_rate
            
            if duration < self.min_speech_length:
                self.logger.debug(f"Áudio muito curto para processamento: {duration:.2f}s")
                return None
            
            if duration > self.max_speech_length:
                self.logger.warning(f"Áudio muito longo, truncando: {duration:.2f}s")
                max_samples = int(self.max_speech_length * sample_rate)
                audio_data = audio_data[:max_samples]
                duration = self.max_speech_length
            
            # Verifica cache
            cache_key = self._generate_cache_key(audio_data)
            if cache_key in self.recognition_cache:
                cached_result, cache_time = self.recognition_cache[cache_key]
                if time.time() - cache_time < self.cache_ttl:
                    self.logger.debug("Usando resultado do cache")
                    return cached_result
            
            # Processa com engine atual
            result = None
            
            if self.current_engine == RecognitionEngine.GOOGLE_CLOUD:
                result = self._recognize_google_stt(audio_data, sample_rate)
            elif self.current_engine == RecognitionEngine.WHISPER_LOCAL:
                result = self._recognize_whisper(audio_data, sample_rate)
            elif self.current_engine == RecognitionEngine.SPEECH_RECOGNITION:
                result = self._recognize_speech_recognition(audio_data, sample_rate)
            
            if result:
                result.timestamp = start_time
                result.processing_time = time.time() - start_time
                result.engine_used = self.current_engine.value
                result.total_duration = duration
                
                # Atualiza cache
                self.recognition_cache[cache_key] = (result, time.time())
                
                # Limpa cache antigo
                self._cleanup_cache()
                
                # Atualiza métricas
                self._update_performance_metrics(result, True)
                
                self.logger.debug(f"Reconhecimento concluído: '{result.full_text[:50]}...' (conf: {result.overall_confidence:.2f})")
                
                return result
            else:
                self._update_performance_metrics(None, False)
                return None
                
        except Exception as e:
            self.logger.error(f"Erro no reconhecimento de fala: {e}")
            self._update_performance_metrics(None, False)
            return None
    
    def _processing_loop(self):
        """Loop principal de processamento de áudio."""
        self.logger.info("Iniciando loop de processamento de reconhecimento...")
        
        while self.is_processing and not self.stop_event.is_set():
            try:
                # Obtém chunk da queue
                try:
                    chunk = self.audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Adiciona ao buffer atual
                self.current_audio_buffer.extend(chunk.data)
                
                # Se chunk contém fala, processa
                if chunk.is_speech:
                    self.status = RecognitionStatus.PROCESSING
                    
                    # Converte buffer para array
                    if len(self.current_audio_buffer) > 0:
                        audio_array = np.array(list(self.current_audio_buffer))
                        
                        # Processa reconhecimento
                        result = self.process_speech_segment(audio_array, chunk.sample_rate)
                        
                        if result and result.full_text:
                            # Adiciona à queue de resultados
                            try:
                                self.result_queue.put_nowait(result)
                            except queue.Full:
                                # Remove resultado mais antigo
                                try:
                                    self.result_queue.get_nowait()
                                    self.result_queue.put_nowait(result)
                                except queue.Empty:
                                    pass
                            
                            # Executa callbacks
                            self._execute_text_callbacks(result)
                        
                        # Limpa buffer após processamento
                        self.current_audio_buffer.clear()
                    
                    self.status = RecognitionStatus.LISTENING
                
                # Marca task como completa
                self.audio_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Erro no loop de processamento: {e}")
                time.sleep(0.1)
        
        self.logger.info("Loop de processamento de reconhecimento finalizado")
    
    def _recognize_google_stt(self, audio_data: np.ndarray, sample_rate: int) -> Optional[RecognitionResult]:
        """Reconhecimento usando Google Cloud Speech-to-Text."""
        try:
            # Converte audio para bytes
            audio_bytes = audio_data.astype(np.int16).tobytes()
            
            # Cria objeto de áudio
            audio = speech.RecognitionAudio(content=audio_bytes)
            
            # Realiza reconhecimento
            response = self.google_client.recognize(config=self.google_config, audio=audio)
            
            if not response.results:
                return None
            
            # Processa resultados
            segments = []
            full_text_parts = []
            total_confidence = 0.0
            
            for result in response.results:
                if result.alternatives:
                    alternative = result.alternatives[0]  # Melhor alternativa
                    
                    # Cria segmento
                    segment = SpeechSegment(
                        text=alternative.transcript,
                        confidence=alternative.confidence,
                        start_time=0.0,  # Google não fornece tempo de início em non-streaming
                        end_time=len(audio_data) / sample_rate,
                        language=self.language,
                        engine_used="google_cloud"
                    )
                    
                    # Adiciona alternativas
                    for alt in result.alternatives[1:]:
                        segment.alternatives.append((alt.transcript, alt.confidence))
                    
                    # Adiciona palavras se disponível
                    if hasattr(alternative, 'words'):
                        for word_info in alternative.words:
                            word_data = {
                                'word': word_info.word,
                                'confidence': getattr(word_info, 'confidence', alternative.confidence),
                                'start_time': word_info.start_time.total_seconds() if hasattr(word_info, 'start_time') else 0.0,
                                'end_time': word_info.end_time.total_seconds() if hasattr(word_info, 'end_time') else 0.0
                            }
                            segment.words.append(word_data)
                    
                    segments.append(segment)
                    full_text_parts.append(alternative.transcript)
                    total_confidence += alternative.confidence
            
            if not segments:
                return None
            
            # Cria resultado final
            result = RecognitionResult(
                segments=segments,
                full_text=' '.join(full_text_parts),
                overall_confidence=total_confidence / len(segments),
                language_detected=self.language,
                total_words=len(' '.join(full_text_parts).split()),
                engine_used="google_cloud"
            )
            
            # Calcula estatísticas
            if result.total_words > 0 and segments:
                duration = segments[-1].end_time - segments[0].start_time
                if duration > 0:
                    result.speech_rate = (result.total_words / duration) * 60  # palavras por minuto
            
            return result
            
        except Exception as e:
            self.logger.error(f"Erro no reconhecimento Google STT: {e}")
            return None
    
    def _recognize_whisper(self, audio_data: np.ndarray, sample_rate: int) -> Optional[RecognitionResult]:
        """Reconhecimento usando Whisper (local)."""
        try:
            # Whisper espera float32 normalizado
            audio_float = audio_data.astype(np.float32) / 32768.0
            
            # Resample se necessário (Whisper espera 16kHz)
            if sample_rate != 16000:
                # Implementação simples de resampling
                target_length = int(len(audio_float) * 16000 / sample_rate)
                audio_float = np.interp(
                    np.linspace(0, len(audio_float), target_length),
                    np.arange(len(audio_float)),
                    audio_float
                )
            
            # Realiza reconhecimento
            result_whisper = self.whisper_model.transcribe(
                audio_float,
                language=self.language[:2] if self.language.startswith('pt') else 'en',
                task='transcribe',
                word_timestamps=True,
                condition_on_previous_text=False
            )
            
            if not result_whisper['text'].strip():
                return None
            
            # Processa resultado
            segments = []
            
            # Whisper pode retornar segmentos detalhados
            if 'segments' in result_whisper:
                for seg in result_whisper['segments']:
                    segment = SpeechSegment(
                        text=seg['text'].strip(),
                        confidence=seg.get('avg_logprob', 0.5) + 1.0,  # Converte logprob para confidence
                        start_time=seg.get('start', 0.0),
                        end_time=seg.get('end', len(audio_data) / sample_rate),
                        language=result_whisper.get('language', self.language),
                        engine_used="whisper_local"
                    )
                    
                    # Adiciona palavras se disponível
                    if 'words' in seg:
                        for word_info in seg['words']:
                            word_data = {
                                'word': word_info['word'],
                                'confidence': word_info.get('probability', segment.confidence),
                                'start_time': word_info.get('start', 0.0),
                                'end_time': word_info.get('end', 0.0)
                            }
                            segment.words.append(word_data)
                    
                    segments.append(segment)
            else:
                # Resultado simples
                segment = SpeechSegment(
                    text=result_whisper['text'].strip(),
                    confidence=0.8,  # Whisper não fornece confidence diretamente
                    start_time=0.0,
                    end_time=len(audio_data) / sample_rate,
                    language=result_whisper.get('language', self.language),
                    engine_used="whisper_local"
                )
                segments.append(segment)
            
            # Cria resultado final
            result = RecognitionResult(
                segments=segments,
                full_text=result_whisper['text'].strip(),
                overall_confidence=sum(seg.confidence for seg in segments) / len(segments),
                language_detected=result_whisper.get('language', self.language),
                total_words=len(result_whisper['text'].split()),
                engine_used="whisper_local"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Erro no reconhecimento Whisper: {e}")
            return None
    
    def _recognize_speech_recognition(self, audio_data: np.ndarray, sample_rate: int) -> Optional[RecognitionResult]:
        """Reconhecimento usando SpeechRecognition library."""
        try:
            # Converte para formato WAV em memória
            wav_buffer = io.BytesIO()
            
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_data.tobytes())
            
            wav_buffer.seek(0)
            
            # Cria objeto AudioData
            with sr.AudioFile(wav_buffer) as source:
                audio_sr = self.sr_recognizer.record(source)
            
            # Tenta reconhecimento com diferentes serviços
            try:
                # Google (gratuito, limitado)
                text = self.sr_recognizer.recognize_google(audio_sr, language=self.language)
                confidence = 0.8  # SpeechRecognition não fornece confidence
                engine_used = "speech_recognition_google"
                
            except sr.UnknownValueError:
                return None
            except sr.RequestError:
                try:
                    # Sphinx (offline)
                    text = self.sr_recognizer.recognize_sphinx(audio_sr)
                    confidence = 0.6
                    engine_used = "speech_recognition_sphinx"
                except:
                    return None
            
            if not text.strip():
                return None
            
            # Cria segmento
            segment = SpeechSegment(
                text=text.strip(),
                confidence=confidence,
                start_time=0.0,
                end_time=len(audio_data) / sample_rate,
                language=self.language,
                engine_used=engine_used
            )
            
            # Cria resultado
            result = RecognitionResult(
                segments=[segment],
                full_text=text.strip(),
                overall_confidence=confidence,
                language_detected=self.language,
                total_words=len(text.split()),
                engine_used=engine_used
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Erro no reconhecimento SpeechRecognition: {e}")
            return None
    
    def _generate_cache_key(self, audio_data: np.ndarray) -> str:
        """Gera chave de cache baseada no áudio."""
        # Hash simples baseado no conteúdo do áudio
        audio_hash = hash(audio_data.tobytes())
        return f"audio_{audio_hash}_{len(audio_data)}"
    
    def _cleanup_cache(self):
        """Remove entradas antigas do cache."""
        current_time = time.time()
        expired_keys = []
        
        for cache_key, (_, cache_time) in self.recognition_cache.items():
            if current_time - cache_time > self.cache_ttl:
                expired_keys.append(cache_key)
        
        for key in expired_keys:
            del self.recognition_cache[key]
    
    def _execute_text_callbacks(self, result: RecognitionResult):
        """Executa callbacks para texto reconhecido."""
        for callback in self.text_callbacks:
            try:
                callback(result)
            except Exception as e:
                self.logger.error(f"Erro ao executar callback de texto: {e}")
    
    def _execute_interim_callbacks(self, text: str, confidence: float):
        """Executa callbacks para resultados intermediários."""
        for callback in self.interim_callbacks:
            try:
                callback(text, confidence)
            except Exception as e:
                self.logger.error(f"Erro ao executar callback interim: {e}")
    
    def _execute_error_callbacks(self, error: Exception):
        """Executa callbacks para erros."""
        for callback in self.error_callbacks:
            try:
                callback(error)
            except Exception as e:
                self.logger.error(f"Erro ao executar callback de erro: {e}")
    
    def _update_performance_metrics(self, result: Optional[RecognitionResult], success: bool):
        """Atualiza métricas de performance."""
        self.performance_metrics['total_recognitions'] += 1
        
        if success and result:
            self.performance_metrics['successful_recognitions'] += 1
            
            # Atualiza tempos de processamento
            if result.processing_time > 0:
                self.performance_metrics['processing_times'].append(result.processing_time)
                if self.performance_metrics['processing_times']:
                    self.performance_metrics['average_processing_time'] = np.mean(
                        self.performance_metrics['processing_times']
                    )
            
            # Atualiza scores de confiança
            self.performance_metrics['confidence_scores'].append(result.overall_confidence)
            
            # Conta engines usados
            engine = result.engine_used
            if engine in self.performance_metrics['engines_used']:
                self.performance_metrics['engines_used'][engine] += 1
            else:
                self.performance_metrics['engines_used'][engine] = 1
            
            # Conta idiomas detectados
            language = result.language_detected
            if language in self.performance_metrics['languages_detected']:
                self.performance_metrics['languages_detected'][language] += 1
            else:
                self.performance_metrics['languages_detected'][language] = 1
            
            # Tempo total de áudio processado
            self.performance_metrics['total_audio_processed'] += result.total_duration
            
        else:
            self.performance_metrics['failed_recognitions'] += 1
    
    def get_latest_result(self) -> Optional[RecognitionResult]:
        """
        Retorna o resultado mais recente de reconhecimento.
        
        Returns:
            Optional[RecognitionResult]: Último resultado ou None
        """
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_all_pending_results(self) -> List[RecognitionResult]:
        """
        Retorna todos os resultados pendentes.
        
        Returns:
            List[RecognitionResult]: Lista de resultados pendentes
        """
        results = []
        while True:
            try:
                result = self.result_queue.get_nowait()
                results.append(result)
            except queue.Empty:
                break
        return results
    
    def add_text_callback(self, callback: Callable[[RecognitionResult], None]):
        """Adiciona callback para texto reconhecido."""
        if callback not in self.text_callbacks:
            self.text_callbacks.append(callback)
            self.logger.info(f"Callback de texto adicionado: {callback.__name__}")
    
    def add_interim_callback(self, callback: Callable[[str, float], None]):
        """Adiciona callback para resultados intermediários."""
        if callback not in self.interim_callbacks:
            self.interim_callbacks.append(callback)
            self.logger.info(f"Callback interim adicionado: {callback.__name__}")
    
    def add_error_callback(self, callback: Callable[[Exception], None]):
        """Adiciona callback para erros."""
        if callback not in self.error_callbacks:
            self.error_callbacks.append(callback)
            self.logger.info(f"Callback de erro adicionado: {callback.__name__}")
    
    def remove_callback(self, callback):
        """Remove callback de todas as listas."""
        removed_count = 0
        
        if callback in self.text_callbacks:
            self.text_callbacks.remove(callback)
            removed_count += 1
        
        if callback in self.interim_callbacks:
            self.interim_callbacks.remove(callback)
            removed_count += 1
        
        if callback in self.error_callbacks:
            self.error_callbacks.remove(callback)
            removed_count += 1
        
        if removed_count > 0:
            self.logger.info(f"Callback removido: {callback.__name__}")
    
    def get_performance_metrics(self) -> Dict:
        """Retorna métricas de performance."""
        metrics = self.performance_metrics.copy()
        
        # Adiciona métricas calculadas
        if metrics['total_recognitions'] > 0:
            metrics['success_rate'] = metrics['successful_recognitions'] / metrics['total_recognitions']
        else:
            metrics['success_rate'] = 0.0
        
        if metrics['confidence_scores']:
            metrics['average_confidence'] = np.mean(metrics['confidence_scores'])
        else:
            metrics['average_confidence'] = 0.0
        
        return metrics
    
    def get_recognition_info(self) -> Dict:
        """Retorna informações sobre configuração de reconhecimento."""
        return {
            'current_engine': self.current_engine.value if self.current_engine else None,
            'available_engines': [engine.value for engine in self.available_engines],
            'language': self.language,
            'is_processing': self.is_processing,
            'status': self.status.value,
            'min_confidence': self.min_confidence,
            'cache_size': len(self.recognition_cache),
            'audio_queue_size': self.audio_queue.qsize(),
            'result_queue_size': self.result_queue.qsize()
        }
    
    def change_language(self, new_language: str) -> bool:
        """
        Muda o idioma de reconhecimento.
        
        Args:
            new_language: Novo código de idioma (ex: 'pt-BR', 'en-US')
            
        Returns:
            bool: True se mudança bem-sucedida
        """
        try:
            old_language = self.language
            self.language = new_language
            
            # Reconfigura engine se necessário
            if self.current_engine == RecognitionEngine.GOOGLE_CLOUD:
                self.google_config.language_code = new_language
                
            self.logger.info(f"Idioma alterado de {old_language} para {new_language}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao alterar idioma: {e}")
            return False
    
    def change_engine(self, new_engine: RecognitionEngine) -> bool:
        """
        Muda o engine de reconhecimento.
        
        Args:
            new_engine: Novo engine
            
        Returns:
            bool: True se mudança bem-sucedida
        """
        if new_engine not in self.available_engines:
            self.logger.error(f"Engine {new_engine} não disponível")
            return False
        
        try:
            old_engine = self.current_engine
            was_processing = self.is_processing
            
            # Para processamento se estiver rodando
            if was_processing:
                self.stop_recognition()
            
            # Muda engine
            self.current_engine = new_engine
            self._initialize_recognition_engine()
            
            # Reinicia se estava processando
            if was_processing:
                self.start_recognition()
            
            self.logger.info(f"Engine alterado de {old_engine} para {new_engine}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao alterar engine: {e}")
            return False
    
    def clear_cache(self):
        """Limpa cache de reconhecimento."""
        self.recognition_cache.clear()
        self.logger.info("Cache de reconhecimento limpo")
    
    def __del__(self):
        """Cleanup automático."""
        self.stop_recognition()