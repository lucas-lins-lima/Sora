# sora_robot/audio_processing/microphone_handler.py

import pyaudio
import numpy as np
import threading
import time
import wave
import io
from typing import Optional, Callable, List, Tuple
from collections import deque
import queue
from dataclasses import dataclass
from enum import Enum

from utils.logger import get_logger
from utils.constants import AUDIO_SETTINGS, PERFORMANCE
import config

class AudioState(Enum):
    """Estados do sistema de áudio."""
    IDLE = "idle"
    LISTENING = "listening"
    RECORDING = "recording"
    PROCESSING = "processing"
    ERROR = "error"

@dataclass
class AudioChunk:
    """Estrutura para um chunk de áudio."""
    data: np.ndarray
    timestamp: float
    sample_rate: int
    channels: int
    duration: float
    volume_level: float
    is_speech: bool = False
    signal_quality: float = 0.0

@dataclass
class VoiceActivity:
    """Dados de atividade de voz detectada."""
    is_speaking: bool
    confidence: float
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration: float = 0.0
    volume_peak: float = 0.0
    volume_average: float = 0.0

class MicrophoneHandler:
    """
    Classe responsável pela captura e processamento de áudio em tempo real.
    Gerencia microfone, detecção de atividade de voz, e fornece chunks de áudio
    para processamento posterior.
    """
    
    def __init__(self, device_index: Optional[int] = None):
        """
        Inicializa o manipulador do microfone.
        
        Args:
            device_index: Índice do dispositivo de áudio (None = padrão)
        """
        self.logger = get_logger(__name__)
        
        # Configurações de áudio
        self.device_index = device_index if device_index is not None else config.MICROPHONE_DEVICE_INDEX
        self.sample_rate = AUDIO_SETTINGS.SAMPLE_RATE
        self.channels = AUDIO_SETTINGS.CHANNELS
        self.chunk_size = AUDIO_SETTINGS.CHUNK_SIZE
        self.format = pyaudio.paInt16  # 16-bit PCM
        
        # PyAudio
        self.audio = None
        self.stream = None
        
        # Estado do sistema
        self.current_state = AudioState.IDLE
        self.is_running = False
        self.is_recording = False
        
        # Threading e queue
        self.audio_thread = None
        self.processing_thread = None
        self.audio_queue = queue.Queue(maxsize=100)  # Buffer de chunks
        self.stop_event = threading.Event()
        self.audio_lock = threading.Lock()
        
        # Detecção de atividade de voz (VAD)
        self.voice_activity = VoiceActivity(is_speaking=False, confidence=0.0)
        self.silence_threshold = AUDIO_SETTINGS.SILENCE_THRESHOLD
        self.silence_duration = AUDIO_SETTINGS.SILENCE_DURATION
        self.speech_start_time = None
        self.last_speech_time = None
        
        # Buffers de áudio
        self.audio_buffer = deque(maxlen=int(self.sample_rate * 10))  # 10 segundos
        self.speech_buffer = deque(maxlen=int(self.sample_rate * 30))  # 30 segundos para gravação
        
        # Callbacks para diferentes eventos
        self.audio_callbacks = []  # Callbacks para cada chunk de áudio
        self.speech_start_callbacks = []  # Callbacks quando fala começar
        self.speech_end_callbacks = []  # Callbacks quando fala terminar
        self.speech_chunk_callbacks = []  # Callbacks para chunks com fala
        
        # Métricas de performance
        self.performance_metrics = {
            'total_chunks_processed': 0,
            'speech_chunks_detected': 0,
            'silence_chunks': 0,
            'audio_dropouts': 0,
            'average_processing_time': 0.0,
            'processing_times': deque(maxlen=100),
            'volume_levels': deque(maxlen=100),
            'speech_detection_accuracy': 0.0
        }
        
        # Configurações de processamento
        self.enable_noise_reduction = AUDIO_SETTINGS.NOISE_REDUCTION
        self.enable_auto_gain = AUDIO_SETTINGS.AUTO_GAIN
        self.volume_normalization = True
        
        # Parâmetros de qualidade de áudio
        self.min_volume_threshold = 100  # Mínimo para considerar áudio válido
        self.max_volume_threshold = 32000  # Máximo antes de considerar saturação
        
        self.logger.info(f"MicrophoneHandler inicializado - Device: {self.device_index}, SR: {self.sample_rate}Hz")
    
    def initialize_audio(self) -> bool:
        """
        Inicializa sistema de áudio e microfone.
        
        Returns:
            bool: True se inicialização bem-sucedida
        """
        try:
            self.logger.info("Inicializando sistema de áudio...")
            
            # Inicializa PyAudio
            self.audio = pyaudio.PyAudio()
            
            # Lista dispositivos disponíveis se device_index não especificado
            if self.device_index is None:
                self._list_audio_devices()
                self.device_index = self._find_best_input_device()
            
            # Verifica se dispositivo existe
            if not self._validate_audio_device():
                raise Exception(f"Dispositivo de áudio {self.device_index} não encontrado")
            
            # Configura stream de áudio
            self.stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.chunk_size,
                stream_callback=None  # Usaremos polling manual para maior controle
            )
            
            # Testa captura
            test_data = self.stream.read(self.chunk_size, exception_on_overflow=False)
            if not test_data:
                raise Exception("Falha ao capturar dados de teste do microfone")
            
            self.logger.info("Sistema de áudio inicializado com sucesso!")
            self._log_audio_info()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao inicializar áudio: {e}")
            if self.stream:
                self.stream.close()
                self.stream = None
            if self.audio:
                self.audio.terminate()
                self.audio = None
            return False
    
    def _list_audio_devices(self):
        """Lista dispositivos de áudio disponíveis."""
        try:
            device_count = self.audio.get_device_count()
            self.logger.info(f"Dispositivos de áudio disponíveis ({device_count}):")
            
            for i in range(device_count):
                device_info = self.audio.get_device_info_by_index(i)
                if device_info['maxInputChannels'] > 0:  # Dispositivo de entrada
                    self.logger.info(f"  [{i}] {device_info['name']} - {device_info['maxInputChannels']} canais")
                    
        except Exception as e:
            self.logger.error(f"Erro ao listar dispositivos de áudio: {e}")
    
    def _find_best_input_device(self) -> int:
        """Encontra o melhor dispositivo de entrada disponível."""
        try:
            # Tenta usar dispositivo padrão primeiro
            default_device = self.audio.get_default_input_device_info()
            return default_device['index']
            
        except Exception as e:
            self.logger.warning(f"Não foi possível obter dispositivo padrão: {e}")
            
            # Procura por dispositivo de entrada válido
            try:
                device_count = self.audio.get_device_count()
                for i in range(device_count):
                    device_info = self.audio.get_device_info_by_index(i)
                    if device_info['maxInputChannels'] > 0:
                        return i
            except Exception as e2:
                self.logger.error(f"Erro ao procurar dispositivos: {e2}")
            
            return 0  # Fallback para índice 0
    
    def _validate_audio_device(self) -> bool:
        """Valida se o dispositivo de áudio é válido."""
        try:
            if self.device_index is None:
                return False
            
            device_info = self.audio.get_device_info_by_index(self.device_index)
            
            if device_info['maxInputChannels'] < self.channels:
                self.logger.error(f"Dispositivo não suporta {self.channels} canais de entrada")
                return False
            
            # Testa se sample rate é suportado
            supported = self.audio.is_format_supported(
                rate=self.sample_rate,
                input_device=self.device_index,
                input_channels=self.channels,
                input_format=self.format
            )
            
            if not supported:
                self.logger.warning(f"Sample rate {self.sample_rate}Hz pode não ser suportado")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erro na validação do dispositivo: {e}")
            return False
    
    def _log_audio_info(self):
        """Registra informações do sistema de áudio configurado."""
        try:
            device_info = self.audio.get_device_info_by_index(self.device_index)
            
            self.logger.info("Configuração de áudio:")
            self.logger.info(f"  - Dispositivo: {device_info['name']}")
            self.logger.info(f"  - Sample Rate: {self.sample_rate}Hz")
            self.logger.info(f"  - Canais: {self.channels}")
            self.logger.info(f"  - Chunk Size: {self.chunk_size}")
            self.logger.info(f"  - Formato: 16-bit PCM")
            self.logger.info(f"  - Latência: ~{(self.chunk_size / self.sample_rate) * 1000:.1f}ms")
            
        except Exception as e:
            self.logger.error(f"Erro ao registrar informações de áudio: {e}")
    
    def start_listening(self) -> bool:
        """
        Inicia captura contínua de áudio.
        
        Returns:
            bool: True se iniciado com sucesso
        """
        if self.is_running:
            self.logger.warning("Captura de áudio já está em execução")
            return True
        
        if not self.stream:
            if not self.initialize_audio():
                return False
        
        try:
            self.is_running = True
            self.current_state = AudioState.LISTENING
            self.stop_event.clear()
            
            # Inicia thread de captura de áudio
            self.audio_thread = threading.Thread(target=self._audio_capture_loop, daemon=True)
            self.audio_thread.start()
            
            # Inicia thread de processamento
            self.processing_thread = threading.Thread(target=self._audio_processing_loop, daemon=True)
            self.processing_thread.start()
            
            self.logger.info("Captura de áudio iniciada")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao iniciar captura de áudio: {e}")
            self.is_running = False
            self.current_state = AudioState.ERROR
            return False
    
    def stop_listening(self):
        """Para captura de áudio."""
        if not self.is_running:
            return
        
        self.logger.info("Parando captura de áudio...")
        
        self.is_running = False
        self.current_state = AudioState.IDLE
        self.stop_event.set()
        
        # Para threads
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=3.0)
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=3.0)
        
        # Fecha stream
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        if self.audio:
            self.audio.terminate()
            self.audio = None
        
        # Limpa buffers
        with self.audio_lock:
            self.audio_buffer.clear()
            self.speech_buffer.clear()
            
            # Limpa queue
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break
        
        self.logger.info("Captura de áudio parada")
    
    def _audio_capture_loop(self):
        """Loop principal de captura de áudio."""
        self.logger.info("Iniciando loop de captura de áudio...")
        
        while self.is_running and not self.stop_event.is_set():
            try:
                if not self.stream or not self.stream.is_active():
                    self.logger.warning("Stream de áudio não está ativo")
                    time.sleep(0.1)
                    continue
                
                # Captura chunk de áudio
                raw_data = self.stream.read(
                    self.chunk_size,
                    exception_on_overflow=False
                )
                
                if not raw_data:
                    self.performance_metrics['audio_dropouts'] += 1
                    continue
                
                # Converte para numpy array
                audio_data = np.frombuffer(raw_data, dtype=np.int16)
                
                # Cria estrutura de chunk
                chunk = self._create_audio_chunk(audio_data)
                
                # Adiciona à queue para processamento
                try:
                    self.audio_queue.put_nowait(chunk)
                except queue.Full:
                    # Remove chunk mais antigo se queue estiver cheia
                    try:
                        self.audio_queue.get_nowait()
                        self.audio_queue.put_nowait(chunk)
                        self.performance_metrics['audio_dropouts'] += 1
                    except queue.Empty:
                        pass
                
                # Atualiza buffer
                with self.audio_lock:
                    self.audio_buffer.extend(audio_data)
                
            except Exception as e:
                self.logger.error(f"Erro no loop de captura: {e}")
                self.performance_metrics['audio_dropouts'] += 1
                time.sleep(0.01)
        
        self.logger.info("Loop de captura de áudio finalizado")
    
    def _audio_processing_loop(self):
        """Loop principal de processamento de áudio."""
        self.logger.info("Iniciando loop de processamento de áudio...")
        
        while self.is_running and not self.stop_event.is_set():
            try:
                # Obtém chunk da queue
                try:
                    chunk = self.audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                start_time = time.time()
                
                # Processa chunk
                processed_chunk = self._process_audio_chunk(chunk)
                
                # Detecção de atividade de voz
                self._detect_voice_activity(processed_chunk)
                
                # Executa callbacks
                self._execute_audio_callbacks(processed_chunk)
                
                # Atualiza métricas
                processing_time = time.time() - start_time
                self._update_performance_metrics(processed_chunk, processing_time)
                
                # Marca task como completa
                self.audio_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Erro no loop de processamento: {e}")
                time.sleep(0.01)
        
        self.logger.info("Loop de processamento de áudio finalizado")
    
    def _create_audio_chunk(self, audio_data: np.ndarray) -> AudioChunk:
        """Cria estrutura AudioChunk a partir dos dados brutos."""
        timestamp = time.time()
        duration = len(audio_data) / self.sample_rate
        
        # Calcula nível de volume
        volume_level = self._calculate_volume_level(audio_data)
        
        # Calcula qualidade do sinal
        signal_quality = self._calculate_signal_quality(audio_data)
        
        return AudioChunk(
            data=audio_data,
            timestamp=timestamp,
            sample_rate=self.sample_rate,
            channels=self.channels,
            duration=duration,
            volume_level=volume_level,
            signal_quality=signal_quality
        )
    
    def _process_audio_chunk(self, chunk: AudioChunk) -> AudioChunk:
        """Processa chunk de áudio (filtros, normalizações, etc.)."""
        processed_data = chunk.data.copy()
        
        try:
            # Redução de ruído se habilitada
            if self.enable_noise_reduction:
                processed_data = self._apply_noise_reduction(processed_data)
            
            # Controle automático de ganho se habilitado
            if self.enable_auto_gain:
                processed_data = self._apply_auto_gain(processed_data)
            
            # Normalização de volume se habilitada
            if self.volume_normalization:
                processed_data = self._normalize_volume(processed_data)
            
            # Atualiza chunk com dados processados
            chunk.data = processed_data
            chunk.volume_level = self._calculate_volume_level(processed_data)
            chunk.signal_quality = self._calculate_signal_quality(processed_data)
            
        except Exception as e:
            self.logger.error(f"Erro no processamento de áudio: {e}")
        
        return chunk
    
    def _calculate_volume_level(self, audio_data: np.ndarray) -> float:
        """Calcula nível de volume RMS do áudio."""
        try:
            if len(audio_data) == 0:
                return 0.0
            
            # RMS (Root Mean Square)
            rms = np.sqrt(np.mean(audio_data.astype(np.float64) ** 2))
            
            # Converte para escala logarítmica (aproximação de dB)
            if rms > 0:
                db_level = 20 * np.log10(rms / 32768.0)  # 32768 = max value for 16-bit
                # Normaliza para 0-1 (assumindo range de -60dB a 0dB)
                normalized_level = max(0.0, min(1.0, (db_level + 60) / 60))
            else:
                normalized_level = 0.0
            
            return normalized_level
            
        except Exception as e:
            self.logger.error(f"Erro ao calcular volume: {e}")
            return 0.0
    
    def _calculate_signal_quality(self, audio_data: np.ndarray) -> float:
        """Calcula qualidade do sinal de áudio."""
        try:
            if len(audio_data) == 0:
                return 0.0
            
            # Fatores de qualidade
            
            # 1. SNR aproximado (relação sinal/ruído)
            signal_power = np.var(audio_data.astype(np.float64))
            if signal_power > 0:
                snr_estimate = min(1.0, signal_power / 1000000)  # Normalização aproximada
            else:
                snr_estimate = 0.0
            
            # 2. Ausência de saturação
            max_value = np.max(np.abs(audio_data))
            saturation_factor = 1.0 - max(0, (max_value - 30000) / 2768)  # Penaliza se próximo de saturação
            
            # 3. Consistência do sinal
            consistency = 1.0 - min(1.0, np.std(audio_data) / (np.mean(np.abs(audio_data)) + 1))
            
            # Combina fatores
            quality = (snr_estimate * 0.4 + saturation_factor * 0.4 + consistency * 0.2)
            
            return max(0.0, min(1.0, quality))
            
        except Exception as e:
            self.logger.error(f"Erro ao calcular qualidade do sinal: {e}")
            return 0.0
    
    def _apply_noise_reduction(self, audio_data: np.ndarray) -> np.ndarray:
        """Aplica redução básica de ruído."""
        try:
            # Implementação simplificada: filtro passa-alta
            # Remove frequências muito baixas que geralmente são ruído
            
            # Para implementação completa, usaria bibliotecas como scipy
            # Por enquanto, apenas aplica threshold de amplitude
            
            noise_threshold = np.max(np.abs(audio_data)) * 0.05  # 5% do pico
            filtered_data = np.where(np.abs(audio_data) < noise_threshold, 0, audio_data)
            
            return filtered_data
            
        except Exception as e:
            self.logger.error(f"Erro na redução de ruído: {e}")
            return audio_data
    
    def _apply_auto_gain(self, audio_data: np.ndarray) -> np.ndarray:
        """Aplica controle automático de ganho."""
        try:
            if len(audio_data) == 0:
                return audio_data
            
            # Calcula nível atual
            current_level = np.max(np.abs(audio_data))
            
            if current_level == 0:
                return audio_data
            
            # Nível alvo (cerca de 70% do máximo)
            target_level = 22000  # Para 16-bit PCM
            
            # Calcula fator de ganho
            gain_factor = target_level / current_level
            
            # Limita ganho para evitar distorção
            gain_factor = max(0.1, min(3.0, gain_factor))
            
            # Aplica ganho
            gained_data = (audio_data * gain_factor).astype(np.int16)
            
            return gained_data
            
        except Exception as e:
            self.logger.error(f"Erro no controle de ganho: {e}")
            return audio_data
    
    def _normalize_volume(self, audio_data: np.ndarray) -> np.ndarray:
        """Normaliza volume do áudio."""
        try:
            if len(audio_data) == 0:
                return audio_data
            
            # Normalização simples para utilizar melhor a faixa dinâmica
            max_value = np.max(np.abs(audio_data))
            
            if max_value == 0:
                return audio_data
            
            # Normaliza para 80% do máximo para evitar saturação
            target_max = 26214  # 80% de 32767 (max 16-bit)
            normalization_factor = target_max / max_value
            
            # Só aplica se for necessário amplificar significativamente
            if normalization_factor > 1.2:
                normalized_data = (audio_data * normalization_factor).astype(np.int16)
                return normalized_data
            
            return audio_data
            
        except Exception as e:
            self.logger.error(f"Erro na normalização: {e}")
            return audio_data
    
    def _detect_voice_activity(self, chunk: AudioChunk):
        """Detecta atividade de voz no chunk de áudio."""
        try:
            current_time = chunk.timestamp
            
            # Critério simples: volume acima do threshold
            is_speech_volume = chunk.volume_level > (self.silence_threshold / 32768.0)
            
            # Critério de qualidade: sinal deve ter qualidade mínima
            is_speech_quality = chunk.signal_quality > 0.3
            
            # Critério de energia: deve haver energia suficiente em frequências de fala
            is_speech_energy = self._has_speech_energy(chunk.data)
            
            # Combina critérios
            is_speech = is_speech_volume and is_speech_quality and is_speech_energy
            confidence = (chunk.volume_level + chunk.signal_quality + (1.0 if is_speech_energy else 0.0)) / 3.0
            
            chunk.is_speech = is_speech
            
            # Atualiza estado de atividade de voz
            if is_speech:
                if not self.voice_activity.is_speaking:
                    # Início da fala
                    self.voice_activity.is_speaking = True
                    self.voice_activity.start_time = current_time
                    self.speech_start_time = current_time
                    
                    self.logger.debug("Início de fala detectado")
                    self._execute_speech_start_callbacks()
                
                # Atualiza métricas de fala
                self.voice_activity.confidence = confidence
                self.voice_activity.volume_peak = max(self.voice_activity.volume_peak, chunk.volume_level)
                self.last_speech_time = current_time
                
                # Adiciona ao buffer de fala
                with self.audio_lock:
                    self.speech_buffer.extend(chunk.data)
                
                # Executa callbacks de chunk de fala
                self._execute_speech_chunk_callbacks(chunk)
                
            else:
                # Verifica se devemos finalizar detecção de fala
                if self.voice_activity.is_speaking and self.last_speech_time:
                    silence_duration = current_time - self.last_speech_time
                    
                    if silence_duration >= self.silence_duration:
                        # Fim da fala
                        self.voice_activity.is_speaking = False
                        self.voice_activity.end_time = current_time
                        
                        if self.speech_start_time:
                            self.voice_activity.duration = current_time - self.speech_start_time
                            
                            # Calcula volume médio durante a fala
                            if self.voice_activity.duration > 0:
                                # Simplificação: usa último valor de pico
                                self.voice_activity.volume_average = self.voice_activity.volume_peak * 0.7
                        
                        self.logger.debug(f"Fim de fala detectado - Duração: {self.voice_activity.duration:.2f}s")
                        self._execute_speech_end_callbacks()
                        
                        # Reset para próxima detecção
                        self.voice_activity.volume_peak = 0.0
                        self.speech_start_time = None
            
        except Exception as e:
            self.logger.error(f"Erro na detecção de atividade de voz: {e}")
    
    def _has_speech_energy(self, audio_data: np.ndarray) -> bool:
        """Verifica se áudio tem energia em frequências típicas de fala."""
        try:
            # Implementação simplificada
            # Análise de energia em diferentes bandas de frequência seria mais precisa
            
            # Por enquanto, usa variação do sinal como indicador
            if len(audio_data) < 10:
                return False
            
            # Calcula variação do sinal
            signal_variation = np.std(audio_data)
            
            # Threshold empírico para energia de fala
            speech_energy_threshold = 500
            
            return signal_variation > speech_energy_threshold
            
        except Exception as e:
            self.logger.error(f"Erro ao analisar energia de fala: {e}")
            return False
    
    def _execute_audio_callbacks(self, chunk: AudioChunk):
        """Executa callbacks para chunks de áudio."""
        for callback in self.audio_callbacks:
            try:
                callback(chunk)
            except Exception as e:
                self.logger.error(f"Erro ao executar callback de áudio: {e}")
    
    def _execute_speech_start_callbacks(self):
        """Executa callbacks de início de fala."""
        for callback in self.speech_start_callbacks:
            try:
                callback(self.voice_activity)
            except Exception as e:
                self.logger.error(f"Erro ao executar callback de início de fala: {e}")
    
    def _execute_speech_end_callbacks(self):
        """Executa callbacks de fim de fala."""
        for callback in self.speech_end_callbacks:
            try:
                callback(self.voice_activity)
            except Exception as e:
                self.logger.error(f"Erro ao executar callback de fim de fala: {e}")
    
    def _execute_speech_chunk_callbacks(self, chunk: AudioChunk):
        """Executa callbacks para chunks com fala."""
        for callback in self.speech_chunk_callbacks:
            try:
                callback(chunk)
            except Exception as e:
                self.logger.error(f"Erro ao executar callback de chunk de fala: {e}")
    
    def _update_performance_metrics(self, chunk: AudioChunk, processing_time: float):
        """Atualiza métricas de performance."""
        self.performance_metrics['total_chunks_processed'] += 1
        
        if chunk.is_speech:
            self.performance_metrics['speech_chunks_detected'] += 1
        else:
            self.performance_metrics['silence_chunks'] += 1
        
        # Atualiza tempos de processamento
        self.performance_metrics['processing_times'].append(processing_time)
        if self.performance_metrics['processing_times']:
            self.performance_metrics['average_processing_time'] = np.mean(
                self.performance_metrics['processing_times']
            )
        
        # Atualiza níveis de volume
        self.performance_metrics['volume_levels'].append(chunk.volume_level)
    
    def get_current_audio_data(self, duration_seconds: float = 1.0) -> Optional[np.ndarray]:
        """
        Retorna dados de áudio dos últimos N segundos.
        
        Args:
            duration_seconds: Duração em segundos
            
        Returns:
            Optional[np.ndarray]: Dados de áudio ou None se não disponível
        """
        try:
            with self.audio_lock:
                if not self.audio_buffer:
                    return None
                
                # Calcula número de samples necessários
                samples_needed = int(duration_seconds * self.sample_rate)
                
                # Pega os últimos N samples
                if len(self.audio_buffer) >= samples_needed:
                    audio_data = np.array(list(self.audio_buffer)[-samples_needed:])
                    return audio_data
                else:
                    # Retorna todo o buffer se não há samples suficientes
                    return np.array(list(self.audio_buffer))
                    
        except Exception as e:
            self.logger.error(f"Erro ao obter dados de áudio atuais: {e}")
            return None
    
    def get_speech_audio(self) -> Optional[np.ndarray]:
        """
        Retorna áudio da última fala detectada.
        
        Returns:
            Optional[np.ndarray]: Dados de áudio da fala ou None
        """
        try:
            with self.audio_lock:
                if not self.speech_buffer:
                    return None
                
                return np.array(list(self.speech_buffer))
                
        except Exception as e:
            self.logger.error(f"Erro ao obter áudio de fala: {e}")
            return None
    
    def save_audio_to_file(self, audio_data: np.ndarray, filename: str) -> bool:
        """
        Salva dados de áudio em arquivo WAV.
        
        Args:
            audio_data: Dados de áudio
            filename: Nome do arquivo
            
        Returns:
            bool: True se salvo com sucesso
        """
        try:
            with wave.open(filename, 'wb') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(2)  # 16-bit = 2 bytes
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_data.tobytes())
            
            self.logger.info(f"Áudio salvo em: {filename}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao salvar áudio: {e}")
            return False
    
    def add_audio_callback(self, callback: Callable[[AudioChunk], None]):
        """Adiciona callback para processamento de chunks de áudio."""
        if callback not in self.audio_callbacks:
            self.audio_callbacks.append(callback)
            self.logger.info(f"Callback de áudio adicionado: {callback.__name__}")
    
    def add_speech_start_callback(self, callback: Callable[[VoiceActivity], None]):
        """Adiciona callback para início de fala."""
        if callback not in self.speech_start_callbacks:
            self.speech_start_callbacks.append(callback)
            self.logger.info(f"Callback de início de fala adicionado: {callback.__name__}")
    
    def add_speech_end_callback(self, callback: Callable[[VoiceActivity], None]):
        """Adiciona callback para fim de fala."""
        if callback not in self.speech_end_callbacks:
            self.speech_end_callbacks.append(callback)
            self.logger.info(f"Callback de fim de fala adicionado: {callback.__name__}")
    
    def add_speech_chunk_callback(self, callback: Callable[[AudioChunk], None]):
        """Adiciona callback para chunks com fala."""
        if callback not in self.speech_chunk_callbacks:
            self.speech_chunk_callbacks.append(callback)
            self.logger.info(f"Callback de chunk de fala adicionado: {callback.__name__}")
    
    def remove_callback(self, callback):
        """Remove callback de todas as listas."""
        removed_count = 0
        
        if callback in self.audio_callbacks:
            self.audio_callbacks.remove(callback)
            removed_count += 1
        
        if callback in self.speech_start_callbacks:
            self.speech_start_callbacks.remove(callback)
            removed_count += 1
        
        if callback in self.speech_end_callbacks:
            self.speech_end_callbacks.remove(callback)
            removed_count += 1
        
        if callback in self.speech_chunk_callbacks:
            self.speech_chunk_callbacks.remove(callback)
            removed_count += 1
        
        if removed_count > 0:
            self.logger.info(f"Callback removido: {callback.__name__}")
    
    def get_voice_activity_status(self) -> VoiceActivity:
        """Retorna status atual de atividade de voz."""
        return self.voice_activity
    
    def get_performance_metrics(self) -> dict:
        """Retorna métricas de performance."""
        metrics = self.performance_metrics.copy()
        
        # Adiciona métricas calculadas
        if metrics['total_chunks_processed'] > 0:
            metrics['speech_detection_ratio'] = (
                metrics['speech_chunks_detected'] / metrics['total_chunks_processed']
            )
        else:
            metrics['speech_detection_ratio'] = 0.0
        
        return metrics
    
    def get_audio_info(self) -> dict:
        """Retorna informações sobre configuração de áudio."""
        info = {
            'device_index': self.device_index,
            'sample_rate': self.sample_rate,
            'channels': self.channels,
            'chunk_size': self.chunk_size,
            'is_running': self.is_running,
            'current_state': self.current_state.value,
            'buffer_size': len(self.audio_buffer) if self.audio_buffer else 0,
            'speech_buffer_size': len(self.speech_buffer) if self.speech_buffer else 0
        }
        
        # Adiciona informações do dispositivo se disponível
        if self.audio and self.device_index is not None:
            try:
                device_info = self.audio.get_device_info_by_index(self.device_index)
                info['device_name'] = device_info['name']
                info['device_channels'] = device_info['maxInputChannels']
            except:
                pass
        
        return info
    
    def is_audio_available(self) -> bool:
        """Verifica se áudio está disponível e funcionando."""
        return (self.is_running and 
                self.stream is not None and 
                self.stream.is_active() and
                self.current_state != AudioState.ERROR)
    
    def __enter__(self):
        """Context manager entry."""
        if self.initialize_audio():
            self.start_listening()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_listening()
    
    def __del__(self):
        """Destructor para limpeza automática."""
        self.stop_listening()