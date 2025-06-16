# sora_robot/response_generation/speech_synthesis.py

import time
import threading
import asyncio
import io
import wave
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import tempfile
import os

# Google Text-to-Speech
try:
    from google.cloud import texttospeech
    GOOGLE_TTS_AVAILABLE = True
except ImportError:
    GOOGLE_TTS_AVAILABLE = False
    texttospeech = None

# Azure Cognitive Services
try:
    import azure.cognitiveservices.speech as speechsdk
    AZURE_TTS_AVAILABLE = True
except ImportError:
    AZURE_TTS_AVAILABLE = False
    speechsdk = None

# AWS Polly
try:
    import boto3
    AWS_POLLY_AVAILABLE = True
except ImportError:
    AWS_POLLY_AVAILABLE = False
    boto3 = None

# pyttsx3 (offline TTS)
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False
    pyttsx3 = None

# gTTS (Google Text-to-Speech, simpler API)
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False
    gTTS = None

# PyAudio para reprodução
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    pyaudio = None

from utils.logger import get_logger
from utils.constants import AUDIO_SETTINGS
from response_generation.llm_integration import LLMResponse
import config

class TTSProvider(Enum):
    """Provedores de Text-to-Speech disponíveis."""
    GOOGLE_CLOUD = "google_cloud"
    AZURE_COGNITIVE = "azure_cognitive"
    AWS_POLLY = "aws_polly"
    GOOGLE_GTTS = "google_gtts"
    PYTTSX3_OFFLINE = "pyttsx3_offline"
    AUTO = "auto"

class VoiceCharacteristics(Enum):
    """Características de voz disponíveis."""
    FRIENDLY = "friendly"
    PROFESSIONAL = "professional"
    EMPATHETIC = "empathetic"
    ENERGETIC = "energetic"
    CALM = "calm"
    AUTHORITATIVE = "authoritative"

class SpeechQuality(Enum):
    """Qualidade do áudio de síntese."""
    LOW = "low"        # 16kHz, menor qualidade
    MEDIUM = "medium"  # 24kHz, qualidade padrão
    HIGH = "high"      # 48kHz, alta qualidade

@dataclass
class VoiceConfig:
    """Configuração de voz para síntese."""
    provider: TTSProvider
    voice_name: str
    language: str = "pt-BR"
    gender: str = "female"  # "male", "female", "neutral"
    age: str = "adult"      # "child", "adult", "elderly"
    
    # Parâmetros prosódicos
    speaking_rate: float = 1.0    # 0.25 a 4.0 (velocidade)
    pitch: float = 0.0            # -20.0 a 20.0 (tom)
    volume_gain: float = 0.0      # -96.0 a 16.0 (volume)
    
    # Características emocionais
    emotion: str = "neutral"      # "neutral", "happy", "sad", "angry", etc.
    emotion_intensity: float = 0.5  # 0.0 a 1.0
    
    # Qualidade e formato
    sample_rate: int = 24000
    audio_format: str = "wav"
    
    # Configurações específicas do provedor
    provider_config: Dict = field(default_factory=dict)

@dataclass
class SpeechSegment:
    """Segmento de fala processado."""
    text: str
    audio_data: Optional[bytes] = None
    duration: float = 0.0
    
    # Metadados do segmento
    start_time: float = 0.0
    end_time: float = 0.0
    word_timings: List[Dict] = field(default_factory=list)  # [{"word": str, "start": float, "end": float}]
    
    # Características aplicadas
    voice_config: Optional[VoiceConfig] = None
    emotion_applied: str = "neutral"
    prosody_adjustments: Dict = field(default_factory=dict)

@dataclass
class SynthesisResult:
    """Resultado completo da síntese de fala."""
    
    # Áudio principal
    audio_data: bytes
    sample_rate: int
    duration: float
    
    # Metadados
    provider_used: str = ""
    voice_used: str = ""
    text_processed: str = ""
    
    # Segmentação
    segments: List[SpeechSegment] = field(default_factory=list)
    
    # Qualidade e timing
    synthesis_time: float = 0.0
    audio_quality: float = 0.0
    prosody_score: float = 0.0
    
    # Informações de controle
    playback_ready: bool = False
    file_path: Optional[str] = None
    
    # Marcadores para sincronização
    word_boundaries: List[Dict] = field(default_factory=list)
    sentence_boundaries: List[Dict] = field(default_factory=list)
    
    # Erro se houver
    error: Optional[str] = None
    fallback_used: bool = False

class SpeechSynthesis:
    """
    Classe responsável pela síntese de fala (Text-to-Speech).
    Suporta múltiplos provedores e oferece controle avançado de prosódia e emoção.
    """
    
    def __init__(self, default_voice_config: Optional[VoiceConfig] = None):
        """
        Inicializa o sistema de síntese de fala.
        
        Args:
            default_voice_config: Configuração padrão de voz
        """
        self.logger = get_logger(__name__)
        
        # Configuração de voz padrão
        self.default_voice_config = default_voice_config or self._create_default_voice_config()
        
        # Provedores configurados
        self.providers_config = {}
        self.active_provider = None
        self.fallback_providers = []
        
        # Estado do sistema
        self.is_initialized = False
        
        # Threading e controle de reprodução
        self.synthesis_lock = threading.Lock()
        self.playback_lock = threading.Lock()
        self.current_playback = None
        
        # Cache de síntese
        self.synthesis_cache = {}
        self.cache_ttl = 600  # 10 minutos
        
        # Histórico de sínteses
        self.synthesis_history = deque(maxlen=50)
        
        # Modelos de emoção para prosódia
        self.emotion_models = self._load_emotion_models()
        
        # Perfis de voz pré-configurados
        self.voice_profiles = self._load_voice_profiles()
        
        # Sistema de reprodução de áudio
        self.audio_player = None
        self.playback_callbacks = []
        
        # Métricas de performance
        self.performance_metrics = {
            'total_syntheses': 0,
            'successful_syntheses': 0,
            'failed_syntheses': 0,
            'average_synthesis_time': 0.0,
            'synthesis_times': deque(maxlen=100),
            'providers_used': {},
            'cache_hits': 0,
            'total_audio_duration': 0.0,
            'quality_scores': deque(maxlen=100)
        }
        
        # Inicializa componentes
        self._initialize_providers()
        self._initialize_audio_player()
        
        self.logger.info("SpeechSynthesis inicializado")
    
    def _create_default_voice_config(self) -> VoiceConfig:
        """Cria configuração padrão de voz."""
        return VoiceConfig(
            provider=TTSProvider.AUTO,
            voice_name="pt-BR-Wavenet-A",  # Voz feminina Google
            language="pt-BR",
            gender="female",
            speaking_rate=1.1,
            pitch=2.0,
            emotion="friendly",
            sample_rate=24000
        )
    
    def _initialize_providers(self):
        """Inicializa provedores de TTS disponíveis."""
        try:
            # Google Cloud TTS
            if GOOGLE_TTS_AVAILABLE and hasattr(config, 'API_KEYS') and config.API_KEYS.get('google_cloud'):
                self.providers_config[TTSProvider.GOOGLE_CLOUD] = {
                    'client': texttospeech.TextToSpeechClient(),
                    'voices': self._get_google_voices(),
                    'enabled': True
                }
                self.logger.info("Google Cloud TTS configurado")
            
            # Azure Cognitive Services
            if AZURE_TTS_AVAILABLE and hasattr(config, 'API_KEYS') and config.API_KEYS.get('azure_speech'):
                speech_config = speechsdk.SpeechConfig(
                    subscription=config.API_KEYS['azure_speech'],
                    region=config.API_KEYS.get('azure_region', 'eastus')
                )
                self.providers_config[TTSProvider.AZURE_COGNITIVE] = {
                    'config': speech_config,
                    'voices': self._get_azure_voices(),
                    'enabled': True
                }
                self.logger.info("Azure Cognitive Services TTS configurado")
            
            # AWS Polly
            if AWS_POLLY_AVAILABLE and hasattr(config, 'API_KEYS') and config.API_KEYS.get('aws_access_key'):
                polly_client = boto3.client(
                    'polly',
                    aws_access_key_id=config.API_KEYS['aws_access_key'],
                    aws_secret_access_key=config.API_KEYS['aws_secret_key'],
                    region_name=config.API_KEYS.get('aws_region', 'us-east-1')
                )
                self.providers_config[TTSProvider.AWS_POLLY] = {
                    'client': polly_client,
                    'voices': self._get_polly_voices(),
                    'enabled': True
                }
                self.logger.info("AWS Polly configurado")
            
            # Google gTTS (simples, gratuito)
            if GTTS_AVAILABLE:
                self.providers_config[TTSProvider.GOOGLE_GTTS] = {
                    'enabled': True,
                    'languages': ['pt', 'en', 'es', 'fr']
                }
                self.logger.info("Google gTTS configurado")
            
            # pyttsx3 (offline)
            if PYTTSX3_AVAILABLE:
                try:
                    engine = pyttsx3.init()
                    voices = engine.getProperty('voices')
                    self.providers_config[TTSProvider.PYTTSX3_OFFLINE] = {
                        'engine': engine,
                        'voices': voices,
                        'enabled': True
                    }
                    self.logger.info("pyttsx3 (offline) configurado")
                except Exception as e:
                    self.logger.warning(f"Erro ao configurar pyttsx3: {e}")
            
            # Seleciona provedor principal
            provider_priority = [
                TTSProvider.GOOGLE_CLOUD,
                TTSProvider.AZURE_COGNITIVE,
                TTSProvider.AWS_POLLY,
                TTSProvider.GOOGLE_GTTS,
                TTSProvider.PYTTSX3_OFFLINE
            ]
            
            for provider in provider_priority:
                if provider in self.providers_config and self.providers_config[provider]['enabled']:
                    self.active_provider = provider
                    break
            
            # Define fallbacks
            self.fallback_providers = [
                p for p in self.providers_config.keys() 
                if p != self.active_provider and self.providers_config[p]['enabled']
            ]
            
            if self.active_provider:
                self.is_initialized = True
                self.logger.info(f"Provedor principal TTS: {self.active_provider.value}")
            else:
                self.logger.error("Nenhum provedor TTS disponível")
                
        except Exception as e:
            self.logger.error(f"Erro ao inicializar provedores TTS: {e}")
    
    def _get_google_voices(self) -> List[Dict]:
        """Obtém vozes disponíveis no Google Cloud TTS."""
        try:
            client = self.providers_config[TTSProvider.GOOGLE_CLOUD]['client']
            response = client.list_voices()
            
            voices = []
            for voice in response.voices:
                if 'pt-BR' in voice.language_codes:
                    voices.append({
                        'name': voice.name,
                        'language': voice.language_codes[0],
                        'gender': voice.ssml_gender.name.lower()
                    })
            
            return voices
            
        except Exception as e:
            self.logger.error(f"Erro ao obter vozes Google: {e}")
            return [{'name': 'pt-BR-Wavenet-A', 'language': 'pt-BR', 'gender': 'female'}]
    
    def _get_azure_voices(self) -> List[Dict]:
        """Obtém vozes disponíveis no Azure."""
        # Lista simplificada das principais vozes Azure para pt-BR
        return [
            {'name': 'pt-BR-FranciscaNeural', 'language': 'pt-BR', 'gender': 'female'},
            {'name': 'pt-BR-AntonioNeural', 'language': 'pt-BR', 'gender': 'male'},
            {'name': 'pt-BR-BrendaNeural', 'language': 'pt-BR', 'gender': 'female'},
            {'name': 'pt-BR-DonatoNeural', 'language': 'pt-BR', 'gender': 'male'}
        ]
    
    def _get_polly_voices(self) -> List[Dict]:
        """Obtém vozes disponíveis no AWS Polly."""
        return [
            {'name': 'Camila', 'language': 'pt-BR', 'gender': 'female'},
            {'name': 'Vitoria', 'language': 'pt-BR', 'gender': 'female'},
            {'name': 'Ricardo', 'language': 'pt-BR', 'gender': 'male'}
        ]
    
    def _load_emotion_models(self) -> Dict:
        """Carrega modelos de emoção para ajustes prosódicos."""
        return {
            'happy': {
                'speaking_rate': 1.15,
                'pitch': 3.0,
                'volume_gain': 2.0,
                'energy': 'high'
            },
            'sad': {
                'speaking_rate': 0.85,
                'pitch': -3.0,
                'volume_gain': -2.0,
                'energy': 'low'
            },
            'excited': {
                'speaking_rate': 1.25,
                'pitch': 5.0,
                'volume_gain': 4.0,
                'energy': 'very_high'
            },
            'calm': {
                'speaking_rate': 0.95,
                'pitch': -1.0,
                'volume_gain': -1.0,
                'energy': 'low'
            },
            'empathetic': {
                'speaking_rate': 0.9,
                'pitch': -0.5,
                'volume_gain': 0.0,
                'energy': 'medium'
            },
            'confident': {
                'speaking_rate': 1.0,
                'pitch': 1.0,
                'volume_gain': 1.0,
                'energy': 'medium_high'
            },
            'neutral': {
                'speaking_rate': 1.0,
                'pitch': 0.0,
                'volume_gain': 0.0,
                'energy': 'medium'
            }
        }
    
    def _load_voice_profiles(self) -> Dict:
        """Carrega perfis de voz pré-configurados."""
        return {
            'sora_default': VoiceConfig(
                provider=TTSProvider.AUTO,
                voice_name="pt-BR-Wavenet-A",
                language="pt-BR",
                gender="female",
                speaking_rate=1.1,
                pitch=2.0,
                emotion="friendly",
                sample_rate=24000
            ),
            'sora_empathetic': VoiceConfig(
                provider=TTSProvider.AUTO,
                voice_name="pt-BR-Wavenet-A",
                language="pt-BR",
                gender="female",
                speaking_rate=0.9,
                pitch=-0.5,
                emotion="empathetic",
                sample_rate=24000
            ),
            'sora_energetic': VoiceConfig(
                provider=TTSProvider.AUTO,
                voice_name="pt-BR-Wavenet-A",
                language="pt-BR",
                gender="female",
                speaking_rate=1.2,
                pitch=4.0,
                emotion="excited",
                sample_rate=24000
            )
        }
    
    def _initialize_audio_player(self):
        """Inicializa sistema de reprodução de áudio."""
        try:
            if PYAUDIO_AVAILABLE:
                self.audio_player = pyaudio.PyAudio()
                self.logger.info("PyAudio inicializado para reprodução")
            else:
                self.logger.warning("PyAudio não disponível - reprodução limitada")
                
        except Exception as e:
            self.logger.error(f"Erro ao inicializar reprodução de áudio: {e}")
    
    async def synthesize_speech(self, text: str, voice_config: Optional[VoiceConfig] = None, 
                               emotion_context: Dict = None) -> SynthesisResult:
        """
        Sintetiza fala a partir de texto.
        
        Args:
            text: Texto para sintetizar
            voice_config: Configuração de voz (usa padrão se None)
            emotion_context: Contexto emocional para ajustar prosódia
            
        Returns:
            SynthesisResult: Resultado da síntese
        """
        start_time = time.time()
        
        try:
            # Usa configuração padrão se não fornecida
            if voice_config is None:
                voice_config = self.default_voice_config.copy() if hasattr(self.default_voice_config, 'copy') else self.default_voice_config
            
            # Aplica contexto emocional
            if emotion_context:
                voice_config = self._apply_emotion_context(voice_config, emotion_context)
            
            # Verifica cache
            cache_key = self._generate_cache_key(text, voice_config)
            cached_result = self._get_cached_synthesis(cache_key)
            if cached_result:
                self.performance_metrics['cache_hits'] += 1
                return cached_result
            
            # Pré-processa texto
            processed_text = self._preprocess_text(text)
            
            # Tenta síntese com provedor principal
            result = None
            if self.active_provider and self.is_initialized:
                result = await self._synthesize_with_provider(processed_text, voice_config, self.active_provider)
            
            # Fallback para outros provedores se necessário
            if not result or result.error:
                for fallback_provider in self.fallback_providers:
                    try:
                        result = await self._synthesize_with_provider(processed_text, voice_config, fallback_provider)
                        if result and not result.error:
                            break
                    except Exception as e:
                        self.logger.warning(f"Fallback {fallback_provider.value} falhou: {e}")
                        continue
            
            # Fallback final para síntese offline
            if not result or result.error:
                result = await self._synthesize_offline_fallback(processed_text, voice_config)
                result.fallback_used = True
            
            # Pós-processamento
            if result and not result.error:
                result.synthesis_time = time.time() - start_time
                result.text_processed = processed_text
                result = self._post_process_synthesis(result, voice_config)
                
                # Atualiza cache
                self._cache_synthesis(cache_key, result)
                
                # Atualiza histórico
                self.synthesis_history.append(result)
                
                # Atualiza métricas
                self._update_performance_metrics(result, True)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Erro na síntese de fala: {e}")
            error_result = SynthesisResult(
                audio_data=b'',
                sample_rate=24000,
                duration=0.0,
                error=str(e),
                synthesis_time=time.time() - start_time
            )
            self._update_performance_metrics(error_result, False)
            return error_result
    
    def synthesize_speech_sync(self, text: str, voice_config: Optional[VoiceConfig] = None,
                              emotion_context: Dict = None) -> SynthesisResult:
        """Versão síncrona da síntese de fala."""
        try:
            return asyncio.run(self.synthesize_speech(text, voice_config, emotion_context))
        except Exception as e:
            self.logger.error(f"Erro na síntese síncrona: {e}")
            return self._create_error_result(str(e))
    
    def synthesize_from_llm_response(self, llm_response: LLMResponse, 
                                   emotion_override: str = None) -> SynthesisResult:
        """
        Sintetiza fala a partir de uma resposta LLM, aplicando contexto emocional.
        
        Args:
            llm_response: Resposta do LLM
            emotion_override: Emoção específica para sobrescrever
            
        Returns:
            SynthesisResult: Resultado da síntese
        """
        try:
            # Determina emoção a ser aplicada
            emotion = emotion_override or llm_response.emotional_tone or "neutral"
            
            # Seleciona perfil de voz baseado na emoção
            voice_profile = self._select_voice_profile(emotion, llm_response)
            
            # Contexto emocional
            emotion_context = {
                'emotion': emotion,
                'intensity': 0.7,  # Intensidade padrão
                'confidence': llm_response.confidence,
                'strategy': getattr(llm_response, 'strategy', 'neutral')
            }
            
            # Sintetiza
            return self.synthesize_speech_sync(
                llm_response.text,
                voice_profile,
                emotion_context
            )
            
        except Exception as e:
            self.logger.error(f"Erro ao sintetizar resposta LLM: {e}")
            return self._create_error_result(str(e))
    
    def _apply_emotion_context(self, voice_config: VoiceConfig, emotion_context: Dict) -> VoiceConfig:
        """Aplica contexto emocional à configuração de voz."""
        try:
            # Cria cópia da configuração
            import copy
            modified_config = copy.deepcopy(voice_config)
            
            emotion = emotion_context.get('emotion', 'neutral')
            intensity = emotion_context.get('intensity', 0.5)
            
            if emotion in self.emotion_models:
                emotion_model = self.emotion_models[emotion]
                
                # Aplica ajustes prosódicos com intensidade
                base_rate = modified_config.speaking_rate
                rate_adjustment = (emotion_model['speaking_rate'] - 1.0) * intensity
                modified_config.speaking_rate = max(0.25, min(4.0, base_rate + rate_adjustment))
                
                base_pitch = modified_config.pitch
                pitch_adjustment = emotion_model['pitch'] * intensity
                modified_config.pitch = max(-20.0, min(20.0, base_pitch + pitch_adjustment))
                
                base_volume = modified_config.volume_gain
                volume_adjustment = emotion_model['volume_gain'] * intensity
                modified_config.volume_gain = max(-96.0, min(16.0, base_volume + volume_adjustment))
                
                # Atualiza emoção
                modified_config.emotion = emotion
                modified_config.emotion_intensity = intensity
            
            return modified_config
            
        except Exception as e:
            self.logger.error(f"Erro ao aplicar contexto emocional: {e}")
            return voice_config
    
    def _select_voice_profile(self, emotion: str, llm_response: LLMResponse) -> VoiceConfig:
        """Seleciona perfil de voz mais apropriado baseado na emoção."""
        try:
            # Mapeamento de emoção para perfil
            emotion_profile_map = {
                'empathetic': 'sora_empathetic',
                'caring': 'sora_empathetic',
                'excited': 'sora_energetic',
                'energetic': 'sora_energetic',
                'happy': 'sora_energetic',
                'calm': 'sora_default',
                'neutral': 'sora_default',
                'confident': 'sora_default'
            }
            
            profile_name = emotion_profile_map.get(emotion, 'sora_default')
            return self.voice_profiles.get(profile_name, self.default_voice_config)
            
        except Exception as e:
            self.logger.error(f"Erro ao selecionar perfil de voz: {e}")
            return self.default_voice_config
    
    def _preprocess_text(self, text: str) -> str:
        """Pré-processa texto para síntese."""
        try:
            # Remove quebras de linha desnecessárias
            text = text.replace('\n', ' ').replace('\r', ' ')
            
            # Normaliza espaços
            import re
            text = re.sub(r'\s+', ' ', text)
            
            # Remove ou substitui caracteres problemáticos
            text = text.replace('&', 'e')
            text = text.replace('@', 'arroba')
            text = text.replace('#', 'hashtag')
            
            # Normaliza pontuação para melhor prosódia
            text = re.sub(r'\.{2,}', '...', text)  # Múltiplos pontos
            text = re.sub(r'\!{2,}', '!', text)   # Múltiplas exclamações
            text = re.sub(r'\?{2,}', '?', text)   # Múltiplas interrogações
            
            # Adiciona pausas em listas
            text = re.sub(r';', ',', text)
            
            return text.strip()
            
        except Exception as e:
            self.logger.error(f"Erro no pré-processamento: {e}")
            return text
    
    async def _synthesize_with_provider(self, text: str, voice_config: VoiceConfig, 
                                       provider: TTSProvider) -> Optional[SynthesisResult]:
        """Sintetiza com provedor específico."""
        try:
            if provider == TTSProvider.GOOGLE_CLOUD:
                return await self._synthesize_google_cloud(text, voice_config)
            elif provider == TTSProvider.AZURE_COGNITIVE:
                return await self._synthesize_azure(text, voice_config)
            elif provider == TTSProvider.AWS_POLLY:
                return await self._synthesize_polly(text, voice_config)
            elif provider == TTSProvider.GOOGLE_GTTS:
                return await self._synthesize_gtts(text, voice_config)
            elif provider == TTSProvider.PYTTSX3_OFFLINE:
                return await self._synthesize_pyttsx3(text, voice_config)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Erro com provedor {provider.value}: {e}")
            return SynthesisResult(
                audio_data=b'',
                sample_rate=voice_config.sample_rate,
                duration=0.0,
                error=str(e),
                provider_used=provider.value
            )
    
    async def _synthesize_google_cloud(self, text: str, voice_config: VoiceConfig) -> SynthesisResult:
        """Sintetiza usando Google Cloud TTS."""
        try:
            client = self.providers_config[TTSProvider.GOOGLE_CLOUD]['client']
            
            # Configura síntese
            synthesis_input = texttospeech.SynthesisInput(text=text)
            
            voice = texttospeech.VoiceSelectionParams(
                language_code=voice_config.language,
                name=voice_config.voice_name,
                ssml_gender=texttospeech.SsmlVoiceGender.FEMALE if voice_config.gender == 'female' else texttospeech.SsmlVoiceGender.MALE
            )
            
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.LINEAR16,
                sample_rate_hertz=voice_config.sample_rate,
                speaking_rate=voice_config.speaking_rate,
                pitch=voice_config.pitch,
                volume_gain_db=voice_config.volume_gain
            )
            
            # Realiza síntese
            response = await asyncio.get_event_loop().run_in_executor(
                None, client.synthesize_speech,
                {"input": synthesis_input, "voice": voice, "audio_config": audio_config}
            )
            
            # Calcula duração estimada
            duration = self._estimate_duration(text, voice_config.speaking_rate)
            
            return SynthesisResult(
                audio_data=response.audio_content,
                sample_rate=voice_config.sample_rate,
                duration=duration,
                provider_used="google_cloud",
                voice_used=voice_config.voice_name,
                playback_ready=True
            )
            
        except Exception as e:
            self.logger.error(f"Erro Google Cloud TTS: {e}")
            return SynthesisResult(
                audio_data=b'',
                sample_rate=voice_config.sample_rate,
                duration=0.0,
                error=str(e),
                provider_used="google_cloud"
            )
    
    async def _synthesize_azure(self, text: str, voice_config: VoiceConfig) -> SynthesisResult:
        """Sintetiza usando Azure Cognitive Services."""
        try:
            speech_config = self.providers_config[TTSProvider.AZURE_COGNITIVE]['config']
            speech_config.speech_synthesis_voice_name = voice_config.voice_name
            
            # Cria sintetizador
            synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)
            
            # SSML para controle prosódico
            ssml = f"""
            <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="{voice_config.language}">
                <voice name="{voice_config.voice_name}">
                    <prosody rate="{voice_config.speaking_rate}" pitch="{voice_config.pitch:+.1f}Hz" volume="{voice_config.volume_gain:+.1f}dB">
                        {text}
                    </prosody>
                </voice>
            </speak>
            """
            
            # Realiza síntese
            result = await asyncio.get_event_loop().run_in_executor(
                None, synthesizer.speak_ssml, ssml
            )
            
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                duration = self._estimate_duration(text, voice_config.speaking_rate)
                
                return SynthesisResult(
                    audio_data=result.audio_data,
                    sample_rate=voice_config.sample_rate,
                    duration=duration,
                    provider_used="azure_cognitive",
                    voice_used=voice_config.voice_name,
                    playback_ready=True
                )
            else:
                raise Exception(f"Azure synthesis failed: {result.reason}")
                
        except Exception as e:
            self.logger.error(f"Erro Azure TTS: {e}")
            return SynthesisResult(
                audio_data=b'',
                sample_rate=voice_config.sample_rate,
                duration=0.0,
                error=str(e),
                provider_used="azure_cognitive"
            )
    
    async def _synthesize_polly(self, text: str, voice_config: VoiceConfig) -> SynthesisResult:
        """Sintetiza usando AWS Polly."""
        try:
            polly_client = self.providers_config[TTSProvider.AWS_POLLY]['client']
            
            # Mapeia configurações para Polly
            voice_id = voice_config.voice_name
            engine = 'neural' if 'Neural' in voice_config.voice_name else 'standard'
            
            # SSML para controle prosódico
            ssml_text = f"""
            <speak>
                <prosody rate="{voice_config.speaking_rate * 100}%" pitch="{voice_config.pitch:+.1f}Hz" volume="{voice_config.volume_gain:+.1f}dB">
                    {text}
                </prosody>
            </speak>
            """
            
            # Realiza síntese
            response = await asyncio.get_event_loop().run_in_executor(
                None, polly_client.synthesize_speech,
                {
                    'Text': ssml_text,
                    'TextType': 'ssml',
                    'VoiceId': voice_id,
                    'OutputFormat': 'pcm',
                    'SampleRate': str(voice_config.sample_rate),
                    'Engine': engine
                }
            )
            
            # Lê dados de áudio
            audio_data = response['AudioStream'].read()
            duration = self._estimate_duration(text, voice_config.speaking_rate)
            
            return SynthesisResult(
                audio_data=audio_data,
                sample_rate=voice_config.sample_rate,
                duration=duration,
                provider_used="aws_polly",
                voice_used=voice_id,
                playback_ready=True
            )
            
        except Exception as e:
            self.logger.error(f"Erro AWS Polly: {e}")
            return SynthesisResult(
                audio_data=b'',
                sample_rate=voice_config.sample_rate,
                duration=0.0,
                error=str(e),
                provider_used="aws_polly"
            )
    
    async def _synthesize_gtts(self, text: str, voice_config: VoiceConfig) -> SynthesisResult:
        """Sintetiza usando Google gTTS."""
        try:
            # gTTS é mais simples, sem controle prosódico avançado
            lang = 'pt' if voice_config.language.startswith('pt') else 'en'
            
            tts = gTTS(text=text, lang=lang, slow=voice_config.speaking_rate < 0.9)
            
            # Salva em buffer temporário
            audio_buffer = io.BytesIO()
            await asyncio.get_event_loop().run_in_executor(
                None, tts.write_to_fp, audio_buffer
            )
            
            audio_data = audio_buffer.getvalue()
            duration = self._estimate_duration(text, voice_config.speaking_rate)
            
            return SynthesisResult(
                audio_data=audio_data,
                sample_rate=22050,  # gTTS padrão
                duration=duration,
                provider_used="google_gtts",
                voice_used="default",
                playback_ready=True
            )
            
        except Exception as e:
            self.logger.error(f"Erro gTTS: {e}")
            return SynthesisResult(
                audio_data=b'',
                sample_rate=22050,
                duration=0.0,
                error=str(e),
                provider_used="google_gtts"
            )
    
    async def _synthesize_pyttsx3(self, text: str, voice_config: VoiceConfig) -> SynthesisResult:
        """Sintetiza usando pyttsx3 (offline)."""
        try:
            engine = self.providers_config[TTSProvider.PYTTSX3_OFFLINE]['engine']
            
            # Configura parâmetros
            engine.setProperty('rate', int(200 * voice_config.speaking_rate))
            engine.setProperty('volume', min(1.0, max(0.0, 0.7 + voice_config.volume_gain / 16.0)))
            
            # Seleciona voz
            voices = engine.getProperty('voices')
            if voices:
                # Tenta encontrar voz feminina para português
                for voice in voices:
                    if 'portuguese' in voice.name.lower() or 'brasil' in voice.name.lower():
                        engine.setProperty('voice', voice.id)
                        break
            
            # Salva em arquivo temporário
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_path = temp_file.name
            temp_file.close()
            
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: (engine.save_to_file(text, temp_path), engine.runAndWait())
            )
            
            # Lê arquivo
            with open(temp_path, 'rb') as f:
                audio_data = f.read()
            
            # Remove arquivo temporário
            os.unlink(temp_path)
            
            duration = self._estimate_duration(text, voice_config.speaking_rate)
            
            return SynthesisResult(
                audio_data=audio_data,
                sample_rate=22050,
                duration=duration,
                provider_used="pyttsx3_offline",
                voice_used="system_default",
                playback_ready=True
            )
            
        except Exception as e:
            self.logger.error(f"Erro pyttsx3: {e}")
            return SynthesisResult(
                audio_data=b'',
                sample_rate=22050,
                duration=0.0,
                error=str(e),
                provider_used="pyttsx3_offline"
            )
    
    async def _synthesize_offline_fallback(self, text: str, voice_config: VoiceConfig) -> SynthesisResult:
        """Fallback offline quando todos os provedores falham."""
        try:
            # Usa pyttsx3 se disponível
            if TTSProvider.PYTTSX3_OFFLINE in self.providers_config:
                return await self._synthesize_pyttsx3(text, voice_config)
            
            # Fallback final: áudio silencioso com duração estimada
            duration = self._estimate_duration(text, voice_config.speaking_rate)
            silence_samples = int(duration * voice_config.sample_rate)
            
            # Gera silêncio
            silence_audio = np.zeros(silence_samples, dtype=np.int16)
            audio_data = silence_audio.tobytes()
            
            return SynthesisResult(
                audio_data=audio_data,
                sample_rate=voice_config.sample_rate,
                duration=duration,
                provider_used="silence_fallback",
                voice_used="none",
                error="No TTS provider available - silence generated",
                fallback_used=True
            )
            
        except Exception as e:
            self.logger.error(f"Erro no fallback offline: {e}")
            return self._create_error_result(str(e))
    
    def _estimate_duration(self, text: str, speaking_rate: float) -> float:
        """Estima duração do áudio baseado no texto."""
        try:
            # Estimativa: ~150 palavras por minuto em velocidade normal
            word_count = len(text.split())
            base_wpm = 150
            adjusted_wpm = base_wpm * speaking_rate
            
            duration = (word_count / adjusted_wpm) * 60
            
            # Adiciona tempo para pontuação
            punctuation_pause = text.count('.') * 0.5 + text.count(',') * 0.2 + text.count('!') * 0.3 + text.count('?') * 0.4
            
            return max(0.5, duration + punctuation_pause)
            
        except Exception as e:
            self.logger.error(f"Erro ao estimar duração: {e}")
            return max(1.0, len(text) * 0.1)  # Fallback simples
    
    def _post_process_synthesis(self, result: SynthesisResult, voice_config: VoiceConfig) -> SynthesisResult:
        """Pós-processa resultado da síntese."""
        try:
            # Calcula métricas de qualidade
            result.audio_quality = self._calculate_audio_quality(result)
            result.prosody_score = self._calculate_prosody_score(result, voice_config)
            
            # Gera word boundaries (simplificado)
            result.word_boundaries = self._generate_word_boundaries(result.text_processed, result.duration)
            
            # Marca como pronto para reprodução
            result.playback_ready = len(result.audio_data) > 0
            
            return result
            
        except Exception as e:
            self.logger.error(f"Erro no pós-processamento: {e}")
            return result
    
    def _calculate_audio_quality(self, result: SynthesisResult) -> float:
        """Calcula score de qualidade do áudio."""
        try:
            quality = 0.5  # Base
            
            # Qualidade baseada no provedor
            provider_quality = {
                'google_cloud': 0.9,
                'azure_cognitive': 0.9,
                'aws_polly': 0.85,
                'google_gtts': 0.7,
                'pyttsx3_offline': 0.6,
                'silence_fallback': 0.0
            }
            
            quality = provider_quality.get(result.provider_used, 0.5)
            
            # Penaliza se houve erro
            if result.error:
                quality *= 0.3
            
            # Bonus por duração razoável
            if 1.0 <= result.duration <= 30.0:
                quality += 0.1
            
            return min(1.0, max(0.0, quality))
            
        except Exception as e:
            self.logger.error(f"Erro ao calcular qualidade: {e}")
            return 0.5
    
    def _calculate_prosody_score(self, result: SynthesisResult, voice_config: VoiceConfig) -> float:
        """Calcula score de prosódia aplicada."""
        try:
            # Score baseado na aplicação bem-sucedida dos parâmetros prosódicos
            prosody = 0.5
            
            # Provedor suporta prosódia avançada
            advanced_prosody_providers = ['google_cloud', 'azure_cognitive', 'aws_polly']
            if result.provider_used in advanced_prosody_providers:
                prosody += 0.3
            
            # Emoção aplicada
            if voice_config.emotion != 'neutral':
                prosody += 0.2
            
            return min(1.0, max(0.0, prosody))
            
        except Exception as e:
            self.logger.error(f"Erro ao calcular prosódia: {e}")
            return 0.5
    
    def _generate_word_boundaries(self, text: str, duration: float) -> List[Dict]:
        """Gera marcadores de palavras simplificados."""
        try:
            words = text.split()
            if not words:
                return []
            
            boundaries = []
            time_per_word = duration / len(words)
            
            current_time = 0.0
            for word in words:
                boundaries.append({
                    'word': word,
                    'start_time': current_time,
                    'end_time': current_time + time_per_word
                })
                current_time += time_per_word
            
            return boundaries
            
        except Exception as e:
            self.logger.error(f"Erro ao gerar word boundaries: {e}")
            return []
    
    def play_synthesis_result(self, result: SynthesisResult, blocking: bool = False) -> bool:
        """
        Reproduz resultado de síntese.
        
        Args:
            result: Resultado da síntese
            blocking: Se deve bloquear até terminar reprodução
            
        Returns:
            bool: True se reprodução iniciada com sucesso
        """
        try:
            if not result.playback_ready or not result.audio_data:
                self.logger.warning("Áudio não está pronto para reprodução")
                return False
            
            if not self.audio_player:
                self.logger.warning("Sistema de áudio não disponível")
                return False
            
            # Para reprodução atual se houver
            self.stop_current_playback()
            
            # Inicia nova reprodução
            if blocking:
                return self._play_audio_blocking(result)
            else:
                return self._play_audio_async(result)
                
        except Exception as e:
            self.logger.error(f"Erro na reprodução: {e}")
            return False
    
    def _play_audio_blocking(self, result: SynthesisResult) -> bool:
        """Reproduz áudio de forma bloqueante."""
        try:
            with self.playback_lock:
                # Configura stream de saída
                stream = self.audio_player.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=result.sample_rate,
                    output=True
                )
                
                # Reproduz em chunks
                chunk_size = 1024
                audio_data = result.audio_data
                
                for i in range(0, len(audio_data), chunk_size):
                    chunk = audio_data[i:i + chunk_size]
                    stream.write(chunk)
                
                stream.stop_stream()
                stream.close()
                
                # Executa callbacks
                self._execute_playback_callbacks('completed', result)
                
                return True
                
        except Exception as e:
            self.logger.error(f"Erro na reprodução bloqueante: {e}")
            return False
    
    def _play_audio_async(self, result: SynthesisResult) -> bool:
        """Reproduz áudio de forma assíncrona."""
        try:
            def playback_thread():
                self._play_audio_blocking(result)
            
            thread = threading.Thread(target=playback_thread, daemon=True)
            thread.start()
            
            self.current_playback = thread
            self._execute_playback_callbacks('started', result)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erro na reprodução assíncrona: {e}")
            return False
    
    def stop_current_playback(self):
        """Para reprodução atual."""
        try:
            if self.current_playback and self.current_playback.is_alive():
                # Note: PyAudio não tem stop direto, thread terminará naturalmente
                self.current_playback = None
                self._execute_playback_callbacks('stopped', None)
                
        except Exception as e:
            self.logger.error(f"Erro ao parar reprodução: {e}")
    
    def add_playback_callback(self, callback: Callable):
        """Adiciona callback para eventos de reprodução."""
        if callback not in self.playback_callbacks:
            self.playback_callbacks.append(callback)
    
    def remove_playback_callback(self, callback: Callable):
        """Remove callback de reprodução."""
        if callback in self.playback_callbacks:
            self.playback_callbacks.remove(callback)
    
    def _execute_playback_callbacks(self, event: str, result: Optional[SynthesisResult]):
        """Executa callbacks de reprodução."""
        for callback in self.playback_callbacks:
            try:
                callback(event, result)
            except Exception as e:
                self.logger.error(f"Erro ao executar callback: {e}")
    
    def _generate_cache_key(self, text: str, voice_config: VoiceConfig) -> str:
        """Gera chave de cache."""
        key_components = [
            text,
            voice_config.provider.value,
            voice_config.voice_name,
            str(voice_config.speaking_rate),
            str(voice_config.pitch),
            voice_config.emotion
        ]
        combined = "|".join(key_components)
        return str(hash(combined))
    
    def _get_cached_synthesis(self, cache_key: str) -> Optional[SynthesisResult]:
        """Recupera síntese do cache."""
        if cache_key in self.synthesis_cache:
            cached_result, cache_time = self.synthesis_cache[cache_key]
            if time.time() - cache_time < self.cache_ttl:
                return cached_result
        return None
    
    def _cache_synthesis(self, cache_key: str, result: SynthesisResult):
        """Armazena síntese no cache."""
        self.synthesis_cache[cache_key] = (result, time.time())
        
        # Limpa cache antigo
        current_time = time.time()
        expired_keys = [
            key for key, (_, cache_time) in self.synthesis_cache.items()
            if current_time - cache_time > self.cache_ttl
        ]
        for key in expired_keys:
            del self.synthesis_cache[key]
    
    def _create_error_result(self, error_message: str) -> SynthesisResult:
        """Cria resultado de erro."""
        return SynthesisResult(
            audio_data=b'',
            sample_rate=24000,
            duration=0.0,
            error=error_message,
            provider_used="error",
            fallback_used=True
        )
    
    def _update_performance_metrics(self, result: SynthesisResult, success: bool):
        """Atualiza métricas de performance."""
        self.performance_metrics['total_syntheses'] += 1
        
        if success and not result.error:
            self.performance_metrics['successful_syntheses'] += 1
        else:
            self.performance_metrics['failed_syntheses'] += 1
        
        # Tempo de síntese
        if result.synthesis_time > 0:
            self.performance_metrics['synthesis_times'].append(result.synthesis_time)
            if self.performance_metrics['synthesis_times']:
                self.performance_metrics['average_synthesis_time'] = np.mean(
                    self.performance_metrics['synthesis_times']
                )
        
        # Provedor usado
        provider = result.provider_used
        if provider in self.performance_metrics['providers_used']:
            self.performance_metrics['providers_used'][provider] += 1
        else:
            self.performance_metrics['providers_used'][provider] = 1
        
        # Duração total de áudio
        self.performance_metrics['total_audio_duration'] += result.duration
        
        # Score de qualidade
        if result.audio_quality > 0:
            self.performance_metrics['quality_scores'].append(result.audio_quality)
    
    def get_performance_metrics(self) -> Dict:
        """Retorna métricas de performance."""
        metrics = self.performance_metrics.copy()
        
        # Adiciona métricas calculadas
        if metrics['total_syntheses'] > 0:
            metrics['success_rate'] = metrics['successful_syntheses'] / metrics['total_syntheses']
        else:
            metrics['success_rate'] = 0.0
        
        if metrics['quality_scores']:
            metrics['average_quality'] = np.mean(metrics['quality_scores'])
        else:
            metrics['average_quality'] = 0.0
        
        if metrics['total_syntheses'] > 0:
            metrics['cache_hit_rate'] = metrics['cache_hits'] / metrics['total_syntheses']
        else:
            metrics['cache_hit_rate'] = 0.0
        
        return metrics
    
    def get_available_voices(self) -> Dict:
        """Retorna vozes disponíveis por provedor."""
        voices = {}
        
        for provider, config in self.providers_config.items():
            if config.get('enabled', False):
                voices[provider.value] = config.get('voices', [])
        
        return voices
    
    def clear_cache(self):
        """Limpa cache de síntese."""
        self.synthesis_cache.clear()
        self.logger.info("Cache de síntese de fala limpo")
    
    def clear_history(self):
        """Limpa histórico de sínteses."""
        self.synthesis_history.clear()
        self.logger.info("Histórico de síntese de fala limpo")
    
    def __del__(self):
        """Cleanup automático."""
        self.stop_current_playback()
        if self.audio_player:
            self.audio_player.terminate()