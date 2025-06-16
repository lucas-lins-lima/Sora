# sora_robot/audio_processing/audio_analysis.py

import numpy as np
import librosa
import scipy.signal
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import threading
import time
import math

from utils.logger import get_logger
from utils.constants import AUDIO_SETTINGS, EMOTIONS
from audio_processing.microphone_handler import AudioChunk, VoiceActivity
import config

class VocalEmotion(Enum):
    """Emoções vocais detectáveis através de características do áudio."""
    NEUTRAL = "neutral"
    EXCITED = "excited"
    CALM = "calm"
    STRESSED = "stressed"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    SURPRISED = "surprised"
    CONFIDENT = "confident"
    UNCERTAIN = "uncertain"

class SpeechPattern(Enum):
    """Padrões de fala detectáveis."""
    NORMAL = "normal"
    FAST = "fast"
    SLOW = "slow"
    HESITANT = "hesitant"
    FLUENT = "fluent"
    MONOTONE = "monotone"
    EXPRESSIVE = "expressive"

@dataclass
class ProsodyFeatures:
    """Características prosódicas extraídas do áudio."""
    
    # Pitch (frequência fundamental)
    pitch_mean: float = 0.0
    pitch_std: float = 0.0
    pitch_range: float = 0.0
    pitch_median: float = 0.0
    
    # Energia e intensidade
    energy_mean: float = 0.0
    energy_std: float = 0.0
    energy_max: float = 0.0
    intensity_contour: List[float] = field(default_factory=list)
    
    # Ritmo e timing
    speech_rate: float = 0.0  # palavras por minuto (estimado)
    pause_frequency: float = 0.0  # pausas por minuto
    pause_duration_mean: float = 0.0  # duração média das pausas
    silence_ratio: float = 0.0  # proporção de silêncio
    
    # Articulação
    articulation_rate: float = 0.0  # taxa de articulação
    voice_breaks: int = 0  # quebras na voz
    voice_stability: float = 0.0  # estabilidade vocal
    
    # Características espectrais
    spectral_centroid: float = 0.0  # "brilho" do som
    spectral_rolloff: float = 0.0  # ponto de roll-off espectral
    spectral_contrast: List[float] = field(default_factory=list)  # contraste espectral
    mfcc_coefficients: List[float] = field(default_factory=list)  # coeficientes MFCC
    
    # Qualidade vocal  
    jitter: float = 0.0  # variação do pitch
    shimmer: float = 0.0  # variação da amplitude
    harmonicity: float = 0.0  # relação harmônico-ruído
    voice_quality_score: float = 0.0

@dataclass
class AudioAnalysisResult:
    """Resultado completo da análise de áudio."""
    
    # Características prosódicas
    prosody: ProsodyFeatures = field(default_factory=ProsodyFeatures)
    
    # Emoção vocal detectada
    vocal_emotion: VocalEmotion = VocalEmotion.NEUTRAL
    emotion_confidence: float = 0.0
    emotion_scores: Dict[str, float] = field(default_factory=dict)
    
    # Padrão de fala
    speech_pattern: SpeechPattern = SpeechPattern.NORMAL
    pattern_confidence: float = 0.0
    
    # Níveis de características
    arousal_level: float = 0.0  # 0.0 = calmo, 1.0 = muito excitado
    valence_level: float = 0.0  # 0.0 = negativo, 1.0 = positivo
    dominance_level: float = 0.0  # 0.0 = submisso, 1.0 = dominante
    stress_level: float = 0.0  # 0.0 = relaxado, 1.0 = muito estressado
    confidence_level: float = 0.0  # 0.0 = inseguro, 1.0 = muito confiante
    
    # Qualidade da análise
    analysis_quality: float = 0.0
    audio_duration: float = 0.0
    sample_rate: int = 44100
    
    # Metadados
    timestamp: float = 0.0
    processing_time: float = 0.0

class AudioAnalysis:
    """
    Classe responsável pela análise de características prosódicas e emocionais do áudio.
    Extrai informações sobre como o usuário está falando, complementando o reconhecimento de fala.
    """
    
    def __init__(self):
        """Inicializa o sistema de análise de áudio."""
        self.logger = get_logger(__name__)
        
        # Configurações de análise
        self.sample_rate = AUDIO_SETTINGS.SAMPLE_RATE
        self.hop_length = 512  # Para análise espectral
        self.frame_length = 2048
        
        # Parâmetros de análise prosódica
        self.pitch_range = (50, 400)  # Hz - range típico de voz humana
        self.min_analysis_duration = 0.5  # segundos
        self.max_analysis_duration = 30.0  # segundos
        
        # Threading
        self.analysis_lock = threading.Lock()
        
        # Cache para análises
        self.analysis_cache = {}
        self.cache_ttl = 30  # 30 segundos
        
        # Histórico para análise temporal
        self.analysis_history = deque(maxlen=50)
        
        # Modelos de referência para classificação emocional
        self._initialize_emotion_models()
        
        # Métricas de performance
        self.performance_metrics = {
            'total_analyses': 0,
            'successful_analyses': 0,
            'processing_times': deque(maxlen=100),
            'average_processing_time': 0.0,
            'emotions_detected': {emotion.value: 0 for emotion in VocalEmotion},
            'patterns_detected': {pattern.value: 0 for pattern in SpeechPattern}
        }
        
        self.logger.info("AudioAnalysis inicializado com sucesso")
    
    def _initialize_emotion_models(self):
        """Inicializa modelos de referência para classificação emocional."""
        
        # Características prosódicas típicas para cada emoção
        self.emotion_prosody_models = {
            VocalEmotion.HAPPY: {
                'pitch_mean_range': (150, 250),
                'pitch_std_range': (20, 50),
                'energy_mean_range': (0.6, 1.0),
                'speech_rate_range': (140, 200),  # palavras por minuto
                'spectral_centroid_range': (1000, 3000)
            },
            VocalEmotion.SAD: {
                'pitch_mean_range': (80, 150),
                'pitch_std_range': (5, 20),
                'energy_mean_range': (0.2, 0.6),
                'speech_rate_range': (80, 130),
                'spectral_centroid_range': (500, 1500)
            },
            VocalEmotion.ANGRY: {
                'pitch_mean_range': (120, 200),
                'pitch_std_range': (30, 70),
                'energy_mean_range': (0.7, 1.0),
                'speech_rate_range': (160, 220),
                'spectral_centroid_range': (1500, 4000)
            },
            VocalEmotion.CALM: {
                'pitch_mean_range': (100, 180),
                'pitch_std_range': (5, 25),
                'energy_mean_range': (0.3, 0.7),
                'speech_rate_range': (120, 160),
                'spectral_centroid_range': (800, 2000)
            },
            VocalEmotion.EXCITED: {
                'pitch_mean_range': (140, 280),
                'pitch_std_range': (25, 60),
                'energy_mean_range': (0.6, 1.0),
                'speech_rate_range': (160, 240),
                'spectral_centroid_range': (1200, 3500)
            },
            VocalEmotion.STRESSED: {
                'pitch_mean_range': (130, 220),
                'pitch_std_range': (35, 80),
                'energy_mean_range': (0.5, 0.9),
                'speech_rate_range': (180, 260),
                'spectral_centroid_range': (1000, 3000)
            }
        }
        
        # Padrões de fala característicos
        self.speech_pattern_models = {
            SpeechPattern.FAST: {
                'speech_rate_min': 180,
                'pause_frequency_max': 3,
                'articulation_rate_min': 5.0
            },
            SpeechPattern.SLOW: {
                'speech_rate_max': 100,
                'pause_frequency_min': 8,
                'pause_duration_mean_min': 0.8
            },
            SpeechPattern.HESITANT: {
                'voice_breaks_min': 2,
                'pause_frequency_min': 10,
                'voice_stability_max': 0.6
            },
            SpeechPattern.MONOTONE: {
                'pitch_std_max': 15,
                'energy_std_max': 0.2,
                'spectral_contrast_mean_max': 10
            }
        }
    
    def analyze_audio(self, audio_data: np.ndarray, sample_rate: int = None) -> Optional[AudioAnalysisResult]:
        """
        Analisa características prosódicas e emocionais do áudio.
        
        Args:
            audio_data: Dados de áudio para análise
            sample_rate: Taxa de amostragem (usa padrão se None)
            
        Returns:
            Optional[AudioAnalysisResult]: Resultado da análise ou None se falhou
        """
        if sample_rate is None:
            sample_rate = self.sample_rate
        
        start_time = time.time()
        
        try:
            with self.analysis_lock:
                # Validações básicas
                if len(audio_data) == 0:
                    return None
                
                duration = len(audio_data) / sample_rate
                
                if duration < self.min_analysis_duration:
                    self.logger.debug(f"Áudio muito curto para análise: {duration:.2f}s")
                    return None
                
                if duration > self.max_analysis_duration:
                    self.logger.debug(f"Áudio muito longo, truncando: {duration:.2f}s")
                    max_samples = int(self.max_analysis_duration * sample_rate)
                    audio_data = audio_data[:max_samples]
                    duration = self.max_analysis_duration
                
                # Cache check
                cache_key = self._generate_cache_key(audio_data)
                if cache_key in self.analysis_cache:
                    cached_result, cache_time = self.analysis_cache[cache_key]
                    if time.time() - cache_time < self.cache_ttl:
                        return cached_result
                
                # Normaliza áudio para análise
                normalized_audio = self._normalize_audio(audio_data)
                
                # Extrai características prosódicas
                prosody = self._extract_prosody_features(normalized_audio, sample_rate)
                
                # Classifica emoção vocal
                vocal_emotion, emotion_confidence, emotion_scores = self._classify_vocal_emotion(prosody)
                
                # Detecta padrão de fala
                speech_pattern, pattern_confidence = self._detect_speech_pattern(prosody)
                
                # Calcula níveis dimensionais (arousal, valence, dominance)
                arousal, valence, dominance = self._calculate_dimensional_levels(prosody, vocal_emotion)
                
                # Calcula níveis de stress e confiança
                stress_level = self._calculate_stress_level(prosody)
                confidence_level = self._calculate_confidence_level(prosody)
                
                # Calcula qualidade da análise
                analysis_quality = self._calculate_analysis_quality(prosody, audio_data)
                
                # Cria resultado
                result = AudioAnalysisResult(
                    prosody=prosody,
                    vocal_emotion=vocal_emotion,
                    emotion_confidence=emotion_confidence,
                    emotion_scores=emotion_scores,
                    speech_pattern=speech_pattern,
                    pattern_confidence=pattern_confidence,
                    arousal_level=arousal,
                    valence_level=valence,
                    dominance_level=dominance,
                    stress_level=stress_level,
                    confidence_level=confidence_level,
                    analysis_quality=analysis_quality,
                    audio_duration=duration,
                    sample_rate=sample_rate,
                    timestamp=start_time,
                    processing_time=time.time() - start_time
                )
                
                # Atualiza cache
                self.analysis_cache[cache_key] = (result, time.time())
                
                # Atualiza histórico
                self.analysis_history.append(result)
                
                # Limpa cache antigo
                self._cleanup_cache()
                
                # Atualiza métricas
                self._update_performance_metrics(result, True)
                
                self.logger.debug(f"Análise de áudio concluída: {vocal_emotion.value} (confiança: {emotion_confidence:.2f})")
                
                return result
                
        except Exception as e:
            self.logger.error(f"Erro na análise de áudio: {e}")
            self._update_performance_metrics(None, False)
            return None
    
    def analyze_audio_chunk(self, audio_chunk: AudioChunk) -> Optional[AudioAnalysisResult]:
        """
        Analisa um chunk de áudio específico.
        
        Args:
            audio_chunk: Chunk de áudio do microfone
            
        Returns:
            Optional[AudioAnalysisResult]: Resultado da análise
        """
        return self.analyze_audio(audio_chunk.data, audio_chunk.sample_rate)
    
    def _normalize_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Normaliza áudio para análise consistente."""
        try:
            # Converte para float
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Normaliza amplitude
            if audio_data.dtype == np.int16:
                audio_data = audio_data / 32768.0
            elif audio_data.dtype == np.int32:
                audio_data = audio_data / 2147483648.0
            
            # Remove DC offset
            audio_data = audio_data - np.mean(audio_data)
            
            # Normaliza RMS
            rms = np.sqrt(np.mean(audio_data ** 2))
            if rms > 0:
                target_rms = 0.1  # RMS alvo
                audio_data = audio_data * (target_rms / rms)
            
            return audio_data
            
        except Exception as e:
            self.logger.error(f"Erro na normalização de áudio: {e}")
            return audio_data
    
    def _extract_prosody_features(self, audio_data: np.ndarray, sample_rate: int) -> ProsodyFeatures:
        """Extrai características prosódicas do áudio."""
        prosody = ProsodyFeatures()
        
        try:
            # Extração de pitch
            prosody = self._extract_pitch_features(audio_data, sample_rate, prosody)
            
            # Extração de energia
            prosody = self._extract_energy_features(audio_data, sample_rate, prosody)
            
            # Extração de características temporais
            prosody = self._extract_temporal_features(audio_data, sample_rate, prosody)
            
            # Extração de características espectrais
            prosody = self._extract_spectral_features(audio_data, sample_rate, prosody)
            
            # Extração de qualidade vocal
            prosody = self._extract_voice_quality_features(audio_data, sample_rate, prosody)
            
        except Exception as e:
            self.logger.error(f"Erro na extração de características prosódicas: {e}")
        
        return prosody
    
    def _extract_pitch_features(self, audio_data: np.ndarray, sample_rate: int, prosody: ProsodyFeatures) -> ProsodyFeatures:
        """Extrai características relacionadas ao pitch."""
        try:
            # Extrai pitch usando librosa
            pitches, magnitudes = librosa.piptrack(
                y=audio_data,
                sr=sample_rate,
                hop_length=self.hop_length,
                fmin=self.pitch_range[0],
                fmax=self.pitch_range[1]
            )
            
            # Extrai valores de pitch válidos
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if pitch_values:
                prosody.pitch_mean = np.mean(pitch_values)
                prosody.pitch_std = np.std(pitch_values)
                prosody.pitch_range = np.max(pitch_values) - np.min(pitch_values)
                prosody.pitch_median = np.median(pitch_values)
            else:
                # Fallback: usa método alternativo
                pitch_alt = self._extract_pitch_autocorr(audio_data, sample_rate)
                if pitch_alt > 0:
                    prosody.pitch_mean = pitch_alt
                    prosody.pitch_std = 10.0  # Valor padrão
                    prosody.pitch_median = pitch_alt
            
        except Exception as e:
            self.logger.error(f"Erro na extração de pitch: {e}")
        
        return prosody
    
    def _extract_pitch_autocorr(self, audio_data: np.ndarray, sample_rate: int) -> float:
        """Extrai pitch usando autocorrelação (método alternativo)."""
        try:
            # Calcula autocorrelação
            autocorr = np.correlate(audio_data, audio_data, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Encontra picos
            min_period = int(sample_rate / self.pitch_range[1])
            max_period = int(sample_rate / self.pitch_range[0])
            
            if max_period < len(autocorr):
                peak_region = autocorr[min_period:max_period]
                if len(peak_region) > 0:
                    peak_index = np.argmax(peak_region) + min_period
                    pitch = sample_rate / peak_index
                    return pitch
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Erro na autocorrelação de pitch: {e}")
            return 0.0
    
    def _extract_energy_features(self, audio_data: np.ndarray, sample_rate: int, prosody: ProsodyFeatures) -> ProsodyFeatures:
        """Extrai características relacionadas à energia e intensidade."""
        try:
            # RMS Energy
            frame_length = int(0.025 * sample_rate)  # 25ms frames
            hop_length = int(0.01 * sample_rate)     # 10ms hop
            
            # Calcula RMS por frame
            rms_frames = []
            for i in range(0, len(audio_data) - frame_length, hop_length):
                frame = audio_data[i:i + frame_length]
                rms = np.sqrt(np.mean(frame ** 2))
                rms_frames.append(rms)
            
            rms_frames = np.array(rms_frames)
            
            if len(rms_frames) > 0:
                prosody.energy_mean = np.mean(rms_frames)
                prosody.energy_std = np.std(rms_frames)
                prosody.energy_max = np.max(rms_frames)
                prosody.intensity_contour = rms_frames.tolist()
            
        except Exception as e:
            self.logger.error(f"Erro na extração de energia: {e}")
        
        return prosody
    
    def _extract_temporal_features(self, audio_data: np.ndarray, sample_rate: int, prosody: ProsodyFeatures) -> ProsodyFeatures:
        """Extrai características temporais (ritmo, pausas, etc.)."""
        try:
            duration = len(audio_data) / sample_rate
            
            # Detecção de pausas baseada em energia
            energy_threshold = np.mean(prosody.intensity_contour) * 0.1 if prosody.intensity_contour else 0.01
            
            # Identifica segmentos de fala e pausas
            speech_segments = []
            pause_segments = []
            
            if prosody.intensity_contour:
                in_speech = False
                segment_start = 0
                frame_duration = 0.01  # 10ms por frame
                
                for i, energy in enumerate(prosody.intensity_contour):
                    if energy > energy_threshold and not in_speech:
                        # Início de fala
                        if segment_start < i:
                            pause_duration = (i - segment_start) * frame_duration
                            if pause_duration > 0.1:  # Pausas > 100ms
                                pause_segments.append(pause_duration)
                        in_speech = True
                        segment_start = i
                    elif energy <= energy_threshold and in_speech:
                        # Fim de fala
                        speech_duration = (i - segment_start) * frame_duration
                        if speech_duration > 0.1:
                            speech_segments.append(speech_duration)
                        in_speech = False
                        segment_start = i
                
                # Segmento final
                if in_speech:
                    speech_duration = (len(prosody.intensity_contour) - segment_start) * frame_duration
                    speech_segments.append(speech_duration)
            
            # Calcula métricas temporais
            if speech_segments:
                total_speech_time = sum(speech_segments)
                total_pause_time = sum(pause_segments) if pause_segments else 0
                
                # Taxa de fala estimada (baseada em heurística)
                estimated_syllables = total_speech_time * 4  # ~4 sílabas por segundo
                estimated_words = estimated_syllables / 2    # ~2 sílabas por palavra
                prosody.speech_rate = (estimated_words / duration) * 60  # palavras por minuto
                
                # Frequência de pausas
                prosody.pause_frequency = (len(pause_segments) / duration) * 60  # pausas por minuto
                
                # Duração média das pausas
                if pause_segments:
                    prosody.pause_duration_mean = np.mean(pause_segments)
                
                # Proporção de silêncio
                prosody.silence_ratio = total_pause_time / duration
                
                # Taxa de articulação (fala ativa)
                if total_speech_time > 0:
                    prosody.articulation_rate = estimated_words / (total_speech_time / 60)
            
        except Exception as e:
            self.logger.error(f"Erro na extração de características temporais: {e}")
        
        return prosody
    
    def _extract_spectral_features(self, audio_data: np.ndarray, sample_rate: int, prosody: ProsodyFeatures) -> ProsodyFeatures:
        """Extrai características espectrais."""
        try:
            # Calcula STFT
            stft = librosa.stft(audio_data, hop_length=self.hop_length, n_fft=self.frame_length)
            magnitude = np.abs(stft)
            
            # Spectral Centroid
            spectral_centroids = librosa.feature.spectral_centroid(S=magnitude, sr=sample_rate)
            prosody.spectral_centroid = np.mean(spectral_centroids)
            
            # Spectral Rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(S=magnitude, sr=sample_rate, roll_percent=0.85)
            prosody.spectral_rolloff = np.mean(spectral_rolloff)
            
            # Spectral Contrast
            spectral_contrast = librosa.feature.spectral_contrast(S=magnitude, sr=sample_rate)
            prosody.spectral_contrast = np.mean(spectral_contrast, axis=1).tolist()
            
            # MFCC Coefficients
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
            prosody.mfcc_coefficients = np.mean(mfccs, axis=1).tolist()
            
        except Exception as e:
            self.logger.error(f"Erro na extração de características espectrais: {e}")
        
        return prosody
    
    def _extract_voice_quality_features(self, audio_data: np.ndarray, sample_rate: int, prosody: ProsodyFeatures) -> ProsodyFeatures:
        """Extrai características de qualidade vocal."""
        try:
            # Jitter (variação do período do pitch)
            if prosody.pitch_mean > 0:
                # Implementação simplificada
                prosody.jitter = min(0.1, prosody.pitch_std / prosody.pitch_mean) if prosody.pitch_mean > 0 else 0.0
            
            # Shimmer (variação da amplitude)
            if prosody.intensity_contour:
                mean_intensity = np.mean(prosody.intensity_contour)
                if mean_intensity > 0:
                    prosody.shimmer = min(0.3, prosody.energy_std / prosody.energy_mean) if prosody.energy_mean > 0 else 0.0
            
            # Harmonicity (relação harmônico-ruído) - implementação simplificada
            # Baseado na regularidade do sinal
            if len(audio_data) > sample_rate:  # Pelo menos 1 segundo
                # Calcula autocorrelação para medir periodicidade
                autocorr = np.correlate(audio_data, audio_data, mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                
                # Pico de autocorrelação indica harmonicidade
                if len(autocorr) > 100:
                    peak_value = np.max(autocorr[50:500])  # Evita pico em zero
                    prosody.harmonicity = min(1.0, peak_value / autocorr[0]) if autocorr[0] > 0 else 0.0
            
            # Voice breaks (quedas abruptas na energia)
            if prosody.intensity_contour:
                energy_diff = np.diff(prosody.intensity_contour)
                threshold = np.std(energy_diff) * 2
                voice_breaks = np.sum(energy_diff < -threshold)
                prosody.voice_breaks = int(voice_breaks)
            
            # Voice stability
            prosody.voice_stability = 1.0 - min(1.0, prosody.jitter + prosody.shimmer)
            
            # Overall voice quality score
            quality_factors = [
                1.0 - prosody.jitter,
                1.0 - prosody.shimmer,
                prosody.harmonicity,
                prosody.voice_stability
            ]
            prosody.voice_quality_score = np.mean([f for f in quality_factors if f >= 0])
            
        except Exception as e:
            self.logger.error(f"Erro na extração de qualidade vocal: {e}")
        
        return prosody
    
    def _classify_vocal_emotion(self, prosody: ProsodyFeatures) -> Tuple[VocalEmotion, float, Dict[str, float]]:
        """Classifica emoção vocal baseada nas características prosódicas."""
        emotion_scores = {}
        
        try:
            for emotion, model in self.emotion_prosody_models.items():
                score = 0.0
                factors = 0
                
                # Score baseado em pitch
                if 'pitch_mean_range' in model:
                    pitch_score = self._score_in_range(prosody.pitch_mean, model['pitch_mean_range'])
                    score += pitch_score
                    factors += 1
                
                # Score baseado em variação de pitch
                if 'pitch_std_range' in model:
                    pitch_std_score = self._score_in_range(prosody.pitch_std, model['pitch_std_range'])
                    score += pitch_std_score
                    factors += 1
                
                # Score baseado em energia
                if 'energy_mean_range' in model:
                    energy_score = self._score_in_range(prosody.energy_mean, model['energy_mean_range'])
                    score += energy_score
                    factors += 1
                
                # Score baseado em taxa de fala
                if 'speech_rate_range' in model and prosody.speech_rate > 0:
                    speech_rate_score = self._score_in_range(prosody.speech_rate, model['speech_rate_range'])
                    score += speech_rate_score
                    factors += 1
                
                # Score baseado em centroide espectral
                if 'spectral_centroid_range' in model and prosody.spectral_centroid > 0:
                    spectral_score = self._score_in_range(prosody.spectral_centroid, model['spectral_centroid_range'])
                    score += spectral_score
                    factors += 1
                
                # Normaliza score
                if factors > 0:
                    emotion_scores[emotion.value] = score / factors
                else:
                    emotion_scores[emotion.value] = 0.0
            
            # Encontra emoção com maior score
            if emotion_scores:
                best_emotion_name = max(emotion_scores, key=emotion_scores.get)
                best_emotion = VocalEmotion(best_emotion_name)
                confidence = emotion_scores[best_emotion_name]
                
                # Ajusta confiança baseada na diferença entre primeira e segunda opções
                sorted_scores = sorted(emotion_scores.values(), reverse=True)
                if len(sorted_scores) > 1:
                    confidence_adjustment = (sorted_scores[0] - sorted_scores[1]) * 0.5
                    confidence = min(1.0, confidence + confidence_adjustment)
                
                return best_emotion, confidence, emotion_scores
            else:
                return VocalEmotion.NEUTRAL, 0.5, {VocalEmotion.NEUTRAL.value: 0.5}
                
        except Exception as e:
            self.logger.error(f"Erro na classificação de emoção vocal: {e}")
            return VocalEmotion.NEUTRAL, 0.0, {}
    
    def _detect_speech_pattern(self, prosody: ProsodyFeatures) -> Tuple[SpeechPattern, float]:
        """Detecta padrão de fala baseado nas características prosódicas."""
        try:
            pattern_scores = {}
            
            for pattern, model in self.speech_pattern_models.items():
                score = 0.0
                factors = 0
                
                # Verifica cada critério do modelo
                for criterion, value in model.items():
                    prosody_value = getattr(prosody, criterion.replace('_min', '').replace('_max', ''), 0)
                    
                    if criterion.endswith('_min'):
                        if prosody_value >= value:
                            score += 1.0
                        else:
                            score += max(0.0, prosody_value / value)
                        factors += 1
                    elif criterion.endswith('_max'):
                        if prosody_value <= value:
                            score += 1.0
                        else:
                            score += max(0.0, value / prosody_value)
                        factors += 1
                
                if factors > 0:
                    pattern_scores[pattern] = score / factors
                else:
                    pattern_scores[pattern] = 0.0
            
            # Encontra padrão com maior score
            if pattern_scores:
                best_pattern = max(pattern_scores, key=pattern_scores.get)
                confidence = pattern_scores[best_pattern]
                
                # Se nenhum padrão tem score alto, retorna NORMAL
                if confidence < 0.6:
                    return SpeechPattern.NORMAL, 0.7
                
                return best_pattern, confidence
            else:
                return SpeechPattern.NORMAL, 0.5
                
        except Exception as e:
            self.logger.error(f"Erro na detecção de padrão de fala: {e}")
            return SpeechPattern.NORMAL, 0.0
    
    def _score_in_range(self, value: float, range_tuple: Tuple[float, float]) -> float:
        """Calcula score de um valor dentro de um range (0.0 a 1.0)."""
        if value == 0:
            return 0.0
        
        min_val, max_val = range_tuple
        
        if min_val <= value <= max_val:
            # Valor dentro do range ideal
            center = (min_val + max_val) / 2
            distance_from_center = abs(value - center)
            range_width = (max_val - min_val) / 2
            
            if range_width > 0:
                return 1.0 - (distance_from_center / range_width)
            else:
                return 1.0
        else:
            # Valor fora do range - score baseado na distância
            if value < min_val:
                distance = min_val - value
                penalty_range = min_val * 0.5  # 50% do valor mínimo
            else:
                distance = value - max_val
                penalty_range = max_val * 0.5  # 50% do valor máximo
            
            if penalty_range > 0:
                return max(0.0, 1.0 - (distance / penalty_range))
            else:
                return 0.0
    
    def _calculate_dimensional_levels(self, prosody: ProsodyFeatures, emotion: VocalEmotion) -> Tuple[float, float, float]:
        """Calcula níveis dimensionais (arousal, valence, dominance)."""
        try:
            # Arousal (ativação) - baseado em energia, pitch range, taxa de fala
            arousal_factors = []
            
            if prosody.energy_mean > 0:
                arousal_factors.append(min(1.0, prosody.energy_mean / 0.5))  # Normaliza energia
            
            if prosody.pitch_range > 0:
                arousal_factors.append(min(1.0, prosody.pitch_range / 100))  # Normaliza range de pitch
            
            if prosody.speech_rate > 0:
                arousal_factors.append(min(1.0, prosody.speech_rate / 200))  # Normaliza taxa de fala
            
            arousal = np.mean(arousal_factors) if arousal_factors else 0.5
            
            # Valence (valência) - baseado na emoção detectada
            emotion_valences = {
                VocalEmotion.HAPPY: 0.8,
                VocalEmotion.EXCITED: 0.7,
                VocalEmotion.CALM: 0.6,
                VocalEmotion.CONFIDENT: 0.7,
                VocalEmotion.NEUTRAL: 0.5,
                VocalEmotion.SAD: 0.2,
                VocalEmotion.ANGRY: 0.3,
                VocalEmotion.STRESSED: 0.3,
                VocalEmotion.UNCERTAIN: 0.4,
                VocalEmotion.SURPRISED: 0.6
            }
            
            valence = emotion_valences.get(emotion, 0.5)
            
            # Dominance (dominância) - baseado em energia, pitch médio, qualidade vocal
            dominance_factors = []
            
            if prosody.energy_mean > 0:
                dominance_factors.append(min(1.0, prosody.energy_mean / 0.6))
            
            if prosody.pitch_mean > 0:
                # Pitch mais baixo pode indicar mais dominância
                normalized_pitch = 1.0 - min(1.0, (prosody.pitch_mean - 80) / 200)
                dominance_factors.append(normalized_pitch)
            
            if prosody.voice_quality_score > 0:
                dominance_factors.append(prosody.voice_quality_score)
            
            dominance = np.mean(dominance_factors) if dominance_factors else 0.5
            
            return arousal, valence, dominance
            
        except Exception as e:
            self.logger.error(f"Erro no cálculo de níveis dimensionais: {e}")
            return 0.5, 0.5, 0.5
    
    def _calculate_stress_level(self, prosody: ProsodyFeatures) -> float:
        """Calcula nível de stress baseado em características prosódicas."""
        try:
            stress_indicators = []
            
            # Variabilidade alta do pitch indica stress
            if prosody.pitch_std > 0:
                stress_indicators.append(min(1.0, prosody.pitch_std / 50))
            
            # Taxa de fala muito alta ou baixa pode indicar stress
            if prosody.speech_rate > 0:
                if prosody.speech_rate > 180 or prosody.speech_rate < 100:
                    stress_indicators.append(0.7)
                else:
                    stress_indicators.append(0.3)
            
            # Qualidade vocal baixa indica stress
            if prosody.voice_quality_score > 0:
                stress_indicators.append(1.0 - prosody.voice_quality_score)
            
            # Jitter e shimmer altos indicam stress
            stress_indicators.append(min(1.0, prosody.jitter * 10))
            stress_indicators.append(min(1.0, prosody.shimmer * 3))
            
            # Voice breaks indicam stress
            if prosody.voice_breaks > 0:
                stress_indicators.append(min(1.0, prosody.voice_breaks * 0.2))
            
            return np.mean(stress_indicators) if stress_indicators else 0.0
            
        except Exception as e:
            self.logger.error(f"Erro no cálculo de nível de stress: {e}")
            return 0.0
    
    def _calculate_confidence_level(self, prosody: ProsodyFeatures) -> float:
        """Calcula nível de confiança baseado em características prosódicas."""
        try:
            confidence_indicators = []
            
            # Qualidade vocal alta indica confiança
            confidence_indicators.append(prosody.voice_quality_score)
            
            # Estabilidade vocal indica confiança
            confidence_indicators.append(prosody.voice_stability)
            
            # Taxa de fala moderada indica confiança
            if prosody.speech_rate > 0:
                if 120 <= prosody.speech_rate <= 180:
                    confidence_indicators.append(0.8)
                else:
                    confidence_indicators.append(0.4)
            
            # Pausas controladas indicam confiança
            if prosody.pause_frequency > 0:
                if 2 <= prosody.pause_frequency <= 8:  # pausas por minuto
                    confidence_indicators.append(0.7)
                else:
                    confidence_indicators.append(0.3)
            
            # Energia consistente indica confiança
            if prosody.energy_std > 0 and prosody.energy_mean > 0:
                energy_consistency = 1.0 - min(1.0, prosody.energy_std / prosody.energy_mean)
                confidence_indicators.append(energy_consistency)
            
            return np.mean(confidence_indicators) if confidence_indicators else 0.5
            
        except Exception as e:
            self.logger.error(f"Erro no cálculo de nível de confiança: {e}")
            return 0.5
    
    def _calculate_analysis_quality(self, prosody: ProsodyFeatures, audio_data: np.ndarray) -> float:
        """Calcula qualidade da análise realizada."""
        try:
            quality_factors = []
            
            # Qualidade baseada na duração do áudio
            duration = len(audio_data) / self.sample_rate
            if duration >= 2.0:
                quality_factors.append(1.0)
            elif duration >= 1.0:
                quality_factors.append(0.8)
            else:
                quality_factors.append(0.5)
            
            # Qualidade baseada na energia do sinal
            if prosody.energy_mean > 0.1:
                quality_factors.append(min(1.0, prosody.energy_mean / 0.3))
            else:
                quality_factors.append(0.3)
            
            # Qualidade baseada na presença de pitch
            if prosody.pitch_mean > 0:
                quality_factors.append(0.9)
            else:
                quality_factors.append(0.4)
            
            # Qualidade baseada na qualidade vocal
            quality_factors.append(prosody.voice_quality_score)
            
            return np.mean(quality_factors)
            
        except Exception as e:
            self.logger.error(f"Erro no cálculo de qualidade da análise: {e}")
            return 0.5
    
    def _generate_cache_key(self, audio_data: np.ndarray) -> str:
        """Gera chave de cache baseada no áudio."""
        audio_hash = hash(audio_data.tobytes())
        return f"audio_analysis_{audio_hash}_{len(audio_data)}"
    
    def _cleanup_cache(self):
        """Remove entradas antigas do cache."""
        current_time = time.time()
        expired_keys = []
        
        for cache_key, (_, cache_time) in self.analysis_cache.items():
            if current_time - cache_time > self.cache_ttl:
                expired_keys.append(cache_key)
        
        for key in expired_keys:
            del self.analysis_cache[key]
    
    def _update_performance_metrics(self, result: Optional[AudioAnalysisResult], success: bool):
        """Atualiza métricas de performance."""
        self.performance_metrics['total_analyses'] += 1
        
        if success and result:
            self.performance_metrics['successful_analyses'] += 1
            
            # Atualiza tempos de processamento
            self.performance_metrics['processing_times'].append(result.processing_time)
            if self.performance_metrics['processing_times']:
                self.performance_metrics['average_processing_time'] = np.mean(
                    self.performance_metrics['processing_times']
                )
            
            # Conta emoções detectadas
            emotion_name = result.vocal_emotion.value
            if emotion_name in self.performance_metrics['emotions_detected']:
                self.performance_metrics['emotions_detected'][emotion_name] += 1
            
            # Conta padrões detectados
            pattern_name = result.speech_pattern.value
            if pattern_name in self.performance_metrics['patterns_detected']:
                self.performance_metrics['patterns_detected'][pattern_name] += 1
    
    def get_analysis_summary(self, duration_seconds: float = 60.0) -> Dict:
        """
        Retorna resumo das análises dos últimos N segundos.
        
        Args:
            duration_seconds: Período para análise
            
        Returns:
            Dict: Resumo das análises
        """
        try:
            current_time = time.time()
            cutoff_time = current_time - duration_seconds
            
            # Filtra análises recentes
            recent_analyses = [
                analysis for analysis in self.analysis_history
                if analysis.timestamp >= cutoff_time
            ]
            
            if not recent_analyses:
                return {'message': 'Nenhuma análise recente encontrada'}
            
            # Calcula estatísticas
            emotions = [analysis.vocal_emotion.value for analysis in recent_analyses]
            patterns = [analysis.speech_pattern.value for analysis in recent_analyses]
            
            emotion_counts = {}
            for emotion in emotions:
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            pattern_counts = {}
            for pattern in patterns:
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
            
            # Médias de níveis
            avg_arousal = np.mean([a.arousal_level for a in recent_analyses])
            avg_valence = np.mean([a.valence_level for a in recent_analyses])
            avg_stress = np.mean([a.stress_level for a in recent_analyses])
            avg_confidence = np.mean([a.confidence_level for a in recent_analyses])
            
            return {
                'period_seconds': duration_seconds,
                'total_analyses': len(recent_analyses),
                'dominant_emotion': max(emotion_counts, key=emotion_counts.get),
                'dominant_pattern': max(pattern_counts, key=pattern_counts.get),
                'emotion_distribution': emotion_counts,
                'pattern_distribution': pattern_counts,
                'average_levels': {
                    'arousal': avg_arousal,
                    'valence': avg_valence,
                    'stress': avg_stress,
                    'confidence': avg_confidence
                },
                'latest_analysis': {
                    'emotion': recent_analyses[-1].vocal_emotion.value,
                    'pattern': recent_analyses[-1].speech_pattern.value,
                    'confidence': recent_analyses[-1].emotion_confidence
                }
            }
            
        except Exception as e:
            self.logger.error(f"Erro ao gerar resumo de análises: {e}")
            return {'error': str(e)}
    
    def get_performance_metrics(self) -> Dict:
        """Retorna métricas de performance."""
        metrics = self.performance_metrics.copy()
        
        # Adiciona métricas calculadas
        if metrics['total_analyses'] > 0:
            metrics['success_rate'] = metrics['successful_analyses'] / metrics['total_analyses']
        else:
            metrics['success_rate'] = 0.0
        
        return metrics
    
    def clear_cache(self):
        """Limpa cache de análises."""
        self.analysis_cache.clear()
        self.logger.info("Cache de análise de áudio limpo")
    
    def clear_history(self):
        """Limpa histórico de análises."""
        self.analysis_history.clear()
        self.logger.info("Histórico de análise de áudio limpo")