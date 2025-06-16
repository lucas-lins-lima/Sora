# sora_robot/response_generation/avatar_animation.py

import time
import threading
import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import json

# Bibliotecas para animação 3D (opcional)
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    pygame = None

# OpenCV para processamento de imagem
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    cv2 = None

from utils.logger import get_logger
from response_generation.speech_synthesis import SynthesisResult, VoiceConfig
from response_generation.llm_integration import LLMResponse
from nlp.dialogue_manager import ResponseStrategy, UserMood
import config

class AnimationType(Enum):
    """Tipos de animação disponíveis."""
    FACIAL_EXPRESSION = "facial_expression"
    LIP_SYNC = "lip_sync"
    EYE_MOVEMENT = "eye_movement"
    HEAD_MOVEMENT = "head_movement"
    HAND_GESTURE = "hand_gesture"
    BODY_POSTURE = "body_posture"
    BREATHING = "breathing"
    IDLE_ANIMATION = "idle_animation"

class ExpressionType(Enum):
    """Tipos de expressão facial."""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    SURPRISED = "surprised"
    ANGRY = "angry"
    CONFUSED = "confused"
    THINKING = "thinking"
    EMPATHETIC = "empathetic"
    EXCITED = "excited"
    CALM = "calm"
    CONFIDENT = "confident"
    FRIENDLY = "friendly"

class GestureType(Enum):
    """Tipos de gestos."""
    NONE = "none"
    POINTING = "pointing"
    EXPLAINING = "explaining"
    WELCOMING = "welcoming"
    THINKING_POSE = "thinking_pose"
    THUMBS_UP = "thumbs_up"
    OPEN_ARMS = "open_arms"
    HAND_TO_CHEST = "hand_to_chest"
    COUNTING = "counting"
    DISMISSIVE = "dismissive"
    ENCOURAGING = "encouraging"

class AnimationIntensity(Enum):
    """Intensidade da animação."""
    SUBTLE = "subtle"      # 0.2-0.4
    MODERATE = "moderate"  # 0.4-0.6  
    STRONG = "strong"      # 0.6-0.8
    INTENSE = "intense"    # 0.8-1.0

@dataclass
class AnimationKeyframe:
    """Keyframe de animação."""
    timestamp: float
    parameters: Dict[str, float]  # {"mouth_open": 0.5, "eyebrow_raise": 0.3, etc.}
    easing: str = "linear"  # "linear", "ease_in", "ease_out", "ease_in_out"

@dataclass
class FacialAnimation:
    """Animação facial completa."""
    expression: ExpressionType
    intensity: float
    duration: float
    
    # Parâmetros específicos
    eyebrow_position: float = 0.0  # -1.0 (baixo) a 1.0 (alto)
    eye_openness: float = 1.0      # 0.0 (fechado) a 1.0 (aberto)
    eye_direction: Tuple[float, float] = (0.0, 0.0)  # (x, y) -1.0 a 1.0
    mouth_shape: str = "neutral"   # "neutral", "smile", "frown", "open", "speaking"
    mouth_openness: float = 0.0    # 0.0 (fechado) a 1.0 (aberto)
    
    # Keyframes para animação suave
    keyframes: List[AnimationKeyframe] = field(default_factory=list)

@dataclass
class LipSyncData:
    """Dados de sincronização labial."""
    phonemes: List[Dict] = field(default_factory=list)  # [{"phoneme": "A", "start": 0.1, "end": 0.3}]
    visemes: List[Dict] = field(default_factory=list)   # [{"viseme": "A", "intensity": 0.8, "time": 0.2}]
    word_emphasis: List[Dict] = field(default_factory=list)  # Palavras com ênfase especial

@dataclass
class GestureAnimation:
    """Animação de gesto."""
    gesture_type: GestureType
    start_time: float
    duration: float
    intensity: float
    
    # Parâmetros do gesto
    hand_positions: List[Tuple[float, float, float]] = field(default_factory=list)  # (x, y, z)
    arm_positions: List[Tuple[float, float]] = field(default_factory=list)  # (shoulder, elbow angles)
    synchronize_with_speech: bool = True

@dataclass
class BodyAnimation:
    """Animação corporal."""
    posture: str = "neutral"  # "neutral", "leaning_forward", "confident", "relaxed"
    head_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # (pitch, yaw, roll)
    breathing_pattern: str = "normal"  # "normal", "calm", "excited", "nervous"
    
    # Micro-movimentos para naturalidade
    idle_sway: float = 0.3  # Balanceio sutil
    blink_frequency: float = 1.0  # Piscadas por segundo

@dataclass
class AnimationSequence:
    """Sequência completa de animação."""
    total_duration: float
    
    # Componentes da animação
    facial_animation: FacialAnimation
    lip_sync_data: LipSyncData
    gesture_animation: Optional[GestureAnimation] = None
    body_animation: BodyAnimation = field(default_factory=BodyAnimation)
    
    # Metadados
    emotion_conveyed: str = "neutral"
    response_strategy: str = "informative"
    sync_with_audio: bool = True
    
    # Controle de timing
    start_time: float = 0.0
    playback_speed: float = 1.0

class AvatarAnimation:
    """
    Classe responsável pela geração e controle de animações do avatar.
    Sincroniza expressões faciais, gestos e movimentos corporais com a fala.
    """
    
    def __init__(self, avatar_config: Dict = None):
        """
        Inicializa o sistema de animação do avatar.
        
        Args:
            avatar_config: Configurações específicas do avatar
        """
        self.logger = get_logger(__name__)
        
        # Configurações do avatar
        self.avatar_config = avatar_config or self._default_avatar_config()
        
        # Estado atual
        self.current_animation: Optional[AnimationSequence] = None
        self.is_animating = False
        self.animation_start_time = 0.0
        
        # Threading
        self.animation_lock = threading.Lock()
        self.animation_thread: Optional[threading.Thread] = None
        self.stop_animation_event = threading.Event()
        
        # Histórico de animações
        self.animation_history = deque(maxlen=20)
        
        # Modelos de expressão
        self.expression_models = self._load_expression_models()
        
        # Modelos de gesto
        self.gesture_models = self._load_gesture_models()
        
        # Mapeamento de fonemas para visemas
        self.phoneme_to_viseme = self._load_phoneme_mapping()
        
        # Sistema de idle (animações quando não está falando)
        self.idle_animations = self._load_idle_animations()
        self.idle_animation_active = False
        
        # Callbacks para eventos de animação
        self.animation_callbacks = []
        self.frame_callbacks = []  # Callbacks por frame
        
        # Cache de animações
        self.animation_cache = {}
        self.cache_ttl = 300  # 5 minutos
        
        # Métricas de performance
        self.performance_metrics = {
            'total_animations': 0,
            'successful_animations': 0,
            'average_generation_time': 0.0,
            'generation_times': deque(maxlen=100),
            'expressions_used': {},
            'gestures_used': {},
            'sync_accuracy': deque(maxlen=50)
        }
        
        # Sistema de renderização (se disponível)
        self.renderer = None
        if PYGAME_AVAILABLE or OPENCV_AVAILABLE:
            self._initialize_renderer()
        
        self.logger.info("AvatarAnimation inicializado")
    
    def _default_avatar_config(self) -> Dict:
        """Configuração padrão do avatar."""
        return {
            'name': 'Sora',
            'appearance': {
                'gender': 'female',
                'age_appearance': 'young_adult',
                'style': 'friendly_professional'
            },
            'animation_style': {
                'expressiveness': 0.7,  # 0.0 a 1.0
                'gesture_frequency': 0.6,
                'micro_expressions': True,
                'idle_animation_enabled': True
            },
            'synchronization': {
                'lip_sync_precision': 'high',
                'gesture_timing': 'natural',
                'expression_transitions': 'smooth'
            },
            'personality_traits': {
                'enthusiasm': 0.7,
                'empathy': 0.8,
                'formality': 0.6,
                'energy_level': 0.7
            }
        }
    
    def _load_expression_models(self) -> Dict:
        """Carrega modelos de expressão facial."""
        return {
            ExpressionType.NEUTRAL: {
                'eyebrow_position': 0.0,
                'eye_openness': 1.0,
                'mouth_shape': 'neutral',
                'mouth_openness': 0.0,
                'duration_multiplier': 1.0
            },
            ExpressionType.HAPPY: {
                'eyebrow_position': 0.2,
                'eye_openness': 0.8,  # Ligeiramente fechados pelo sorriso
                'mouth_shape': 'smile',
                'mouth_openness': 0.0,
                'cheek_elevation': 0.6,
                'duration_multiplier': 1.2
            },
            ExpressionType.SAD: {
                'eyebrow_position': -0.3,
                'eye_openness': 0.6,
                'mouth_shape': 'frown',
                'mouth_openness': 0.0,
                'duration_multiplier': 1.5
            },
            ExpressionType.SURPRISED: {
                'eyebrow_position': 0.8,
                'eye_openness': 1.0,
                'mouth_shape': 'open',
                'mouth_openness': 0.6,
                'duration_multiplier': 0.8
            },
            ExpressionType.THINKING: {
                'eyebrow_position': 0.1,
                'eye_openness': 0.7,
                'mouth_shape': 'slight_frown',
                'mouth_openness': 0.0,
                'head_tilt': -0.1,
                'duration_multiplier': 2.0
            },
            ExpressionType.EMPATHETIC: {
                'eyebrow_position': -0.1,
                'eye_openness': 0.9,
                'mouth_shape': 'gentle_smile',
                'mouth_openness': 0.0,
                'head_nod': True,
                'duration_multiplier': 1.3
            },
            ExpressionType.EXCITED: {
                'eyebrow_position': 0.4,
                'eye_openness': 1.0,
                'mouth_shape': 'big_smile',
                'mouth_openness': 0.2,
                'energy_level': 1.0,
                'duration_multiplier': 1.1
            },
            ExpressionType.CONFIDENT: {
                'eyebrow_position': 0.1,
                'eye_openness': 1.0,
                'mouth_shape': 'slight_smile',
                'mouth_openness': 0.0,
                'posture_adjustment': 'straight',
                'duration_multiplier': 1.0
            }
        }
    
    def _load_gesture_models(self) -> Dict:
        """Carrega modelos de gestos."""
        return {
            GestureType.EXPLAINING: {
                'description': 'Gestos de explicação com as mãos',
                'hand_movements': [
                    {'time': 0.0, 'position': (0.0, 0.0, 0.0), 'intensity': 0.0},
                    {'time': 0.3, 'position': (0.2, 0.1, 0.0), 'intensity': 0.6},
                    {'time': 0.7, 'position': (-0.2, 0.1, 0.0), 'intensity': 0.6},
                    {'time': 1.0, 'position': (0.0, 0.0, 0.0), 'intensity': 0.0}
                ],
                'triggers': ['informative', 'explaining', 'teaching']
            },
            GestureType.WELCOMING: {
                'description': 'Gestos de boas-vindas',
                'hand_movements': [
                    {'time': 0.0, 'position': (0.0, 0.0, 0.0), 'intensity': 0.0},
                    {'time': 0.5, 'position': (0.3, 0.3, 0.1), 'intensity': 0.8},
                    {'time': 1.0, 'position': (0.0, 0.0, 0.0), 'intensity': 0.0}
                ],
                'triggers': ['greeting', 'welcoming', 'casual']
            },
            GestureType.THINKING_POSE: {
                'description': 'Pose pensativa',
                'hand_movements': [
                    {'time': 0.0, 'position': (0.0, 0.0, 0.0), 'intensity': 0.0},
                    {'time': 0.8, 'position': (0.1, 0.4, 0.3), 'intensity': 0.7},
                    {'time': 2.0, 'position': (0.1, 0.4, 0.3), 'intensity': 0.7}
                ],
                'triggers': ['thinking', 'considering', 'clarifying']
            },
            GestureType.ENCOURAGING: {
                'description': 'Gestos encorajadores',
                'hand_movements': [
                    {'time': 0.0, 'position': (0.0, 0.0, 0.0), 'intensity': 0.0},
                    {'time': 0.4, 'position': (0.0, 0.2, 0.1), 'intensity': 0.9},
                    {'time': 0.8, 'position': (0.0, 0.0, 0.0), 'intensity': 0.0}
                ],
                'triggers': ['encouraging', 'supportive', 'motivational']
            },
            GestureType.OPEN_ARMS: {
                'description': 'Braços abertos receptivos',
                'hand_movements': [
                    {'time': 0.0, 'position': (0.0, 0.0, 0.0), 'intensity': 0.0},
                    {'time': 0.6, 'position': (0.4, 0.2, 0.0), 'intensity': 0.8},
                    {'time': 1.5, 'position': (0.4, 0.2, 0.0), 'intensity': 0.8},
                    {'time': 2.0, 'position': (0.0, 0.0, 0.0), 'intensity': 0.0}
                ],
                'triggers': ['open', 'receptive', 'inclusive']
            }
        }
    
    def _load_phoneme_mapping(self) -> Dict:
        """Carrega mapeamento de fonemas para visemas."""
        return {
            # Vogais
            'A': {'mouth_openness': 0.8, 'mouth_width': 0.6, 'lip_rounding': 0.0},
            'E': {'mouth_openness': 0.4, 'mouth_width': 0.8, 'lip_rounding': 0.0},
            'I': {'mouth_openness': 0.2, 'mouth_width': 1.0, 'lip_rounding': 0.0},
            'O': {'mouth_openness': 0.6, 'mouth_width': 0.3, 'lip_rounding': 0.8},
            'U': {'mouth_openness': 0.3, 'mouth_width': 0.2, 'lip_rounding': 1.0},
            
            # Consoantes
            'P': {'mouth_openness': 0.0, 'lip_compression': 1.0},
            'B': {'mouth_openness': 0.0, 'lip_compression': 1.0},
            'M': {'mouth_openness': 0.0, 'lip_compression': 1.0},
            'F': {'mouth_openness': 0.1, 'lip_bite': 0.7},
            'V': {'mouth_openness': 0.1, 'lip_bite': 0.7},
            'T': {'mouth_openness': 0.2, 'tongue_tip': 1.0},
            'D': {'mouth_openness': 0.2, 'tongue_tip': 1.0},
            'S': {'mouth_openness': 0.1, 'teeth_show': 0.8},
            'Z': {'mouth_openness': 0.1, 'teeth_show': 0.8},
            'L': {'mouth_openness': 0.3, 'tongue_lateral': 1.0},
            'R': {'mouth_openness': 0.2, 'tongue_curl': 0.8},
            'N': {'mouth_openness': 0.2, 'tongue_tip': 0.8},
            
            # Silêncio/pausa
            'SIL': {'mouth_openness': 0.0, 'mouth_width': 0.0, 'lip_rounding': 0.0}
        }
    
    def _load_idle_animations(self) -> Dict:
        """Carrega animações idle."""
        return {
            'breathing': {
                'type': 'breathing',
                'duration': 4.0,  # segundos por ciclo
                'intensity': 0.2,
                'chest_movement': 0.1,
                'shoulder_movement': 0.05
            },
            'eye_movement': {
                'type': 'eye_movement',
                'duration': 3.0,
                'patterns': ['look_left', 'look_right', 'look_up', 'center'],
                'intensity': 0.3
            },
            'micro_expressions': {
                'type': 'micro_expressions',
                'duration': 2.0,
                'expressions': ['slight_smile', 'eyebrow_raise', 'blink'],
                'frequency': 0.2  # por segundo
            },
            'posture_adjustment': {
                'type': 'posture',
                'duration': 8.0,
                'adjustments': ['slight_lean', 'head_tilt', 'shoulder_adjust'],
                'intensity': 0.1
            }
        }
    
    def _initialize_renderer(self):
        """Inicializa sistema de renderização visual."""
        try:
            if PYGAME_AVAILABLE:
                pygame.init()
                self.renderer = 'pygame'
                self.logger.info("Renderer pygame inicializado")
            elif OPENCV_AVAILABLE:
                self.renderer = 'opencv'
                self.logger.info("Renderer OpenCV inicializado")
        except Exception as e:
            self.logger.warning(f"Falha ao inicializar renderer: {e}")
    
    def generate_animation_from_speech(self, synthesis_result: SynthesisResult, 
                                     llm_response: LLMResponse = None,
                                     emotion_override: str = None) -> AnimationSequence:
        """
        Gera sequência de animação baseada no resultado de síntese de fala.
        
        Args:
            synthesis_result: Resultado da síntese de fala
            llm_response: Resposta do LLM (para contexto)
            emotion_override: Emoção específica para sobrescrever
            
        Returns:
            AnimationSequence: Sequência completa de animação
        """
        start_time = time.time()
        
        try:
            # Determina emoção e estratégia
            emotion = emotion_override
            strategy = "informative"
            
            if llm_response:
                emotion = emotion or llm_response.emotional_tone or "neutral"
                strategy = getattr(llm_response, 'strategy', 'informative')
            
            # Mapeia emoção para expressão
            expression = self._map_emotion_to_expression(emotion)
            
            # Gera animação facial
            facial_animation = self._generate_facial_animation(
                expression, synthesis_result.duration, emotion
            )
            
            # Gera lip sync
            lip_sync_data = self._generate_lip_sync(synthesis_result)
            
            # Gera gestos se apropriado
            gesture_animation = self._generate_gesture_animation(
                strategy, synthesis_result.duration, synthesis_result.text_processed
            )
            
            # Gera animação corporal
            body_animation = self._generate_body_animation(emotion, strategy)
            
            # Cria sequência completa
            sequence = AnimationSequence(
                total_duration=synthesis_result.duration,
                facial_animation=facial_animation,
                lip_sync_data=lip_sync_data,
                gesture_animation=gesture_animation,
                body_animation=body_animation,
                emotion_conveyed=emotion,
                response_strategy=strategy,
                sync_with_audio=True
            )
            
            # Atualiza métricas
            generation_time = time.time() - start_time
            self._update_performance_metrics(sequence, generation_time, True)
            
            self.logger.debug(f"Animação gerada: {expression.value}, {gesture_animation.gesture_type.value if gesture_animation else 'sem gesto'}")
            
            return sequence
            
        except Exception as e:
            self.logger.error(f"Erro ao gerar animação: {e}")
            return self._create_fallback_animation(synthesis_result.duration)
    
    def _map_emotion_to_expression(self, emotion: str) -> ExpressionType:
        """Mapeia emoção para tipo de expressão."""
        emotion_mapping = {
            'happy': ExpressionType.HAPPY,
            'excited': ExpressionType.EXCITED,
            'sad': ExpressionType.SAD,
            'empathetic': ExpressionType.EMPATHETIC,
            'caring': ExpressionType.EMPATHETIC,
            'surprised': ExpressionType.SURPRISED,
            'confident': ExpressionType.CONFIDENT,
            'thinking': ExpressionType.THINKING,
            'confused': ExpressionType.CONFUSED,
            'calm': ExpressionType.CALM,
            'friendly': ExpressionType.FRIENDLY,
            'neutral': ExpressionType.NEUTRAL
        }
        
        return emotion_mapping.get(emotion, ExpressionType.NEUTRAL)
    
    def _generate_facial_animation(self, expression: ExpressionType, 
                                  duration: float, emotion: str) -> FacialAnimation:
        """Gera animação facial baseada na expressão."""
        try:
            model = self.expression_models[expression]
            expressiveness = self.avatar_config['animation_style']['expressiveness']
            
            # Ajusta intensidade baseada na personalidade
            intensity = expressiveness * 0.8
            
            # Cria keyframes para transição suave
            keyframes = []
            
            # Keyframe inicial (neutro)
            keyframes.append(AnimationKeyframe(
                timestamp=0.0,
                parameters={
                    'eyebrow_position': 0.0,
                    'eye_openness': 1.0,
                    'mouth_openness': 0.0
                },
                easing="ease_out"
            ))
            
            # Keyframe de expressão principal
            keyframes.append(AnimationKeyframe(
                timestamp=duration * 0.2,
                parameters={
                    'eyebrow_position': model['eyebrow_position'] * intensity,
                    'eye_openness': model['eye_openness'],
                    'mouth_openness': model['mouth_openness'] * intensity
                },
                easing="ease_in_out"
            ))
            
            # Keyframe de sustentação
            if duration > 2.0:
                keyframes.append(AnimationKeyframe(
                    timestamp=duration * 0.8,
                    parameters={
                        'eyebrow_position': model['eyebrow_position'] * intensity * 0.7,
                        'eye_openness': model['eye_openness'],
                        'mouth_openness': model['mouth_openness'] * intensity * 0.5
                    },
                    easing="linear"
                ))
            
            # Keyframe final (retorno suave)
            keyframes.append(AnimationKeyframe(
                timestamp=duration,
                parameters={
                    'eyebrow_position': 0.0,
                    'eye_openness': 1.0,
                    'mouth_openness': 0.0
                },
                easing="ease_in"
            ))
            
            return FacialAnimation(
                expression=expression,
                intensity=intensity,
                duration=duration,
                eyebrow_position=model['eyebrow_position'] * intensity,
                eye_openness=model['eye_openness'],
                mouth_shape=model['mouth_shape'],
                keyframes=keyframes
            )
            
        except Exception as e:
            self.logger.error(f"Erro ao gerar animação facial: {e}")
            return self._create_default_facial_animation(duration)
    
    def _generate_lip_sync(self, synthesis_result: SynthesisResult) -> LipSyncData:
        """Gera dados de sincronização labial."""
        try:
            lip_sync = LipSyncData()
            
            # Usa word boundaries se disponível
            if synthesis_result.word_boundaries:
                for word_data in synthesis_result.word_boundaries:
                    word = word_data['word']
                    start_time = word_data['start_time']
                    end_time = word_data['end_time']
                    
                    # Converte palavra em fonemas (simplificado)
                    phonemes = self._word_to_phonemes(word)
                    
                    # Distribui fonemas no tempo
                    word_duration = end_time - start_time
                    phoneme_duration = word_duration / len(phonemes) if phonemes else 0.1
                    
                    current_time = start_time
                    for phoneme in phonemes:
                        # Adiciona fonema
                        lip_sync.phonemes.append({
                            'phoneme': phoneme,
                            'start': current_time,
                            'end': current_time + phoneme_duration
                        })
                        
                        # Converte para visema
                        if phoneme in self.phoneme_to_viseme:
                            viseme_data = self.phoneme_to_viseme[phoneme].copy()
                            viseme_data['time'] = current_time + phoneme_duration / 2
                            viseme_data['duration'] = phoneme_duration
                            lip_sync.visemes.append(viseme_data)
                        
                        current_time += phoneme_duration
            
            else:
                # Fallback: estimativa baseada no texto
                lip_sync = self._estimate_lip_sync_from_text(
                    synthesis_result.text_processed, synthesis_result.duration
                )
            
            return lip_sync
            
        except Exception as e:
            self.logger.error(f"Erro ao gerar lip sync: {e}")
            return LipSyncData()
    
    def _word_to_phonemes(self, word: str) -> List[str]:
        """Converte palavra em lista de fonemas (simplificado)."""
        # Implementação muito simplificada
        # Em produção, usaria um dicionário fonético ou biblioteca especializada
        
        phoneme_map = {
            'a': ['A'], 'e': ['E'], 'i': ['I'], 'o': ['O'], 'u': ['U'],
            'p': ['P'], 'b': ['B'], 'm': ['M'], 'f': ['F'], 'v': ['V'],
            't': ['T'], 'd': ['D'], 's': ['S'], 'z': ['Z'], 'l': ['L'],
            'r': ['R'], 'n': ['N']
        }
        
        phonemes = []
        word = word.lower().strip('.,!?;:')
        
        for char in word:
            if char in phoneme_map:
                phonemes.extend(phoneme_map[char])
            elif char == ' ':
                phonemes.append('SIL')
        
        return phonemes if phonemes else ['A']  # Fallback
    
    def _estimate_lip_sync_from_text(self, text: str, duration: float) -> LipSyncData:
        """Estima lip sync baseado apenas no texto."""
        try:
            words = text.split()
            word_duration = duration / len(words) if words else duration
            
            lip_sync = LipSyncData()
            current_time = 0.0
            
            for word in words:
                phonemes = self._word_to_phonemes(word)
                phoneme_duration = word_duration / len(phonemes) if phonemes else 0.1
                
                for phoneme in phonemes:
                    lip_sync.phonemes.append({
                        'phoneme': phoneme,
                        'start': current_time,
                        'end': current_time + phoneme_duration
                    })
                    
                    if phoneme in self.phoneme_to_viseme:
                        viseme_data = self.phoneme_to_viseme[phoneme].copy()
                        viseme_data['time'] = current_time + phoneme_duration / 2
                        viseme_data['duration'] = phoneme_duration
                        lip_sync.visemes.append(viseme_data)
                    
                    current_time += phoneme_duration
                
                # Pausa entre palavras
                current_time += word_duration * 0.1
            
            return lip_sync
            
        except Exception as e:
            self.logger.error(f"Erro ao estimar lip sync: {e}")
            return LipSyncData()
    
    def _generate_gesture_animation(self, strategy: str, duration: float, 
                                   text: str) -> Optional[GestureAnimation]:
        """Gera animação de gesto baseada na estratégia e conteúdo."""
        try:
            gesture_frequency = self.avatar_config['animation_style']['gesture_frequency']
            
            # Decide se deve usar gesto
            should_gesture = (
                gesture_frequency > 0.4 and 
                duration > 2.0 and  # Fala longa o suficiente
                self._text_benefits_from_gesture(text)
            )
            
            if not should_gesture:
                return None
            
            # Seleciona tipo de gesto baseado na estratégia
            gesture_type = self._select_gesture_for_strategy(strategy, text)
            
            if gesture_type == GestureType.NONE:
                return None
            
            # Obtém modelo do gesto
            gesture_model = self.gesture_models[gesture_type]
            
            # Ajusta timing do gesto
            gesture_start = duration * 0.1  # Começa após 10% da fala
            gesture_duration = min(duration * 0.8, 3.0)  # Máximo 3 segundos
            
            return GestureAnimation(
                gesture_type=gesture_type,
                start_time=gesture_start,
                duration=gesture_duration,
                intensity=gesture_frequency,
                hand_positions=[(move['position']) for move in gesture_model['hand_movements']],
                synchronize_with_speech=True
            )
            
        except Exception as e:
            self.logger.error(f"Erro ao gerar gesto: {e}")
            return None
    
    def _text_benefits_from_gesture(self, text: str) -> bool:
        """Determina se o texto se beneficiaria de gestos."""
        gesture_keywords = [
            'explicar', 'mostrar', 'ver', 'aqui', 'ali', 'grande', 'pequeno',
            'primeiro', 'segundo', 'próximo', 'anterior', 'isso', 'aquilo',
            'direita', 'esquerda', 'acima', 'abaixo', 'junto', 'separado'
        ]
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in gesture_keywords)
    
    def _select_gesture_for_strategy(self, strategy: str, text: str) -> GestureType:
        """Seleciona gesto apropriado para a estratégia."""
        strategy_gesture_map = {
            'informative': [GestureType.EXPLAINING, GestureType.POINTING],
            'empathetic': [GestureType.OPEN_ARMS, GestureType.HAND_TO_CHEST],
            'encouraging': [GestureType.ENCOURAGING, GestureType.THUMBS_UP],
            'clarifying': [GestureType.THINKING_POSE, GestureType.EXPLAINING],
            'casual': [GestureType.WELCOMING, GestureType.EXPLAINING],
            'directive': [GestureType.POINTING, GestureType.EXPLAINING]
        }
        
        possible_gestures = strategy_gesture_map.get(strategy, [GestureType.EXPLAINING])
        
        # Seleciona baseado no conteúdo
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['bem-vindo', 'olá', 'oi']):
            return GestureType.WELCOMING
        elif any(word in text_lower for word in ['pensar', 'considerar', 'analisar']):
            return GestureType.THINKING_POSE
        elif any(word in text_lower for word in ['ótimo', 'excelente', 'parabéns']):
            return GestureType.ENCOURAGING
        elif any(word in text_lower for word in ['aqui', 'ali', 'este', 'esse']):
            return GestureType.POINTING
        
        # Fallback para primeiro da lista
        return possible_gestures[0] if possible_gestures else GestureType.NONE
    
    def _generate_body_animation(self, emotion: str, strategy: str) -> BodyAnimation:
        """Gera animação corporal baseada na emoção e estratégia."""
        try:
            # Postura baseada na estratégia
            posture_map = {
                'informative': 'confident',
                'empathetic': 'leaning_forward',
                'encouraging': 'confident',
                'clarifying': 'attentive',
                'casual': 'relaxed',
                'directive': 'authoritative'
            }
            
            posture = posture_map.get(strategy, 'neutral')
            
            # Ajusta cabeça baseado na emoção
            head_positions = {
                'empathetic': (0.0, 0.0, -0.1),  # Leve inclinação
                'thinking': (-0.1, 0.1, 0.0),   # Cabeça ligeiramente baixa
                'confident': (0.1, 0.0, 0.0),   # Cabeça alta
                'excited': (0.0, 0.0, 0.0),     # Neutra
                'sad': (-0.2, 0.0, 0.0)         # Cabeça baixa
            }
            
            head_position = head_positions.get(emotion, (0.0, 0.0, 0.0))
            
            # Padrão de respiração baseado na emoção
            breathing_patterns = {
                'excited': 'excited',
                'calm': 'calm',
                'nervous': 'nervous',
                'confident': 'normal'
            }
            
            breathing = breathing_patterns.get(emotion, 'normal')
            
            return BodyAnimation(
                posture=posture,
                head_position=head_position,
                breathing_pattern=breathing,
                idle_sway=0.3,
                blink_frequency=1.0
            )
            
        except Exception as e:
            self.logger.error(f"Erro ao gerar animação corporal: {e}")
            return BodyAnimation()
    
    def play_animation(self, animation_sequence: AnimationSequence, 
                      start_immediately: bool = True) -> bool:
        """
        Reproduz sequência de animação.
        
        Args:
            animation_sequence: Sequência a ser reproduzida
            start_immediately: Se deve iniciar imediatamente
            
        Returns:
            bool: True se reprodução iniciada com sucesso
        """
        try:
            with self.animation_lock:
                # Para animação atual se houver
                self.stop_current_animation()
                
                # Define nova animação
                self.current_animation = animation_sequence
                self.is_animating = True
                
                if start_immediately:
                    self.animation_start_time = time.time()
                    animation_sequence.start_time = self.animation_start_time
                    
                    # Inicia thread de animação
                    self.animation_thread = threading.Thread(
                        target=self._animation_playback_loop,
                        args=(animation_sequence,),
                        daemon=True
                    )
                    self.animation_thread.start()
                    
                    # Executa callbacks
                    self._execute_animation_callbacks('started', animation_sequence)
                    
                    self.logger.debug(f"Animação iniciada: {animation_sequence.emotion_conveyed}")
                    return True
                
                return True
                
        except Exception as e:
            self.logger.error(f"Erro ao reproduzir animação: {e}")
            return False
    
    def _animation_playback_loop(self, animation_sequence: AnimationSequence):
        """Loop principal de reprodução de animação."""
        try:
            start_time = animation_sequence.start_time
            duration = animation_sequence.total_duration
            
            while (self.is_animating and 
                   not self.stop_animation_event.is_set() and
                   time.time() - start_time < duration):
                
                current_time = time.time() - start_time
                
                # Calcula estado atual da animação
                animation_state = self._calculate_animation_state(animation_sequence, current_time)
                
                # Executa callbacks de frame
                self._execute_frame_callbacks(animation_state, current_time)
                
                # Renderiza se disponível
                if self.renderer:
                    self._render_frame(animation_state)
                
                # Controla framerate (30 FPS)
                time.sleep(1.0 / 30.0)
            
            # Animação concluída
            self.is_animating = False
            self._execute_animation_callbacks('completed', animation_sequence)
            
        except Exception as e:
            self.logger.error(f"Erro no loop de animação: {e}")
            self.is_animating = False
            self._execute_animation_callbacks('error', animation_sequence)
    
    def _calculate_animation_state(self, sequence: AnimationSequence, current_time: float) -> Dict:
        """Calcula estado atual de todas as animações."""
        try:
            state = {
                'time': current_time,
                'facial': {},
                'lip_sync': {},
                'gesture': {},
                'body': {}
            }
            
            # Estado facial
            if sequence.facial_animation:
                state['facial'] = self._calculate_facial_state(
                    sequence.facial_animation, current_time
                )
            
            # Estado de lip sync
            if sequence.lip_sync_data:
                state['lip_sync'] = self._calculate_lip_sync_state(
                    sequence.lip_sync_data, current_time
                )
            
            # Estado de gesto
            if sequence.gesture_animation:
                state['gesture'] = self._calculate_gesture_state(
                    sequence.gesture_animation, current_time
                )
            
            # Estado corporal
            if sequence.body_animation:
                state['body'] = self._calculate_body_state(
                    sequence.body_animation, current_time
                )
            
            return state
            
        except Exception as e:
            self.logger.error(f"Erro ao calcular estado da animação: {e}")
            return {'time': current_time}
    
    def _calculate_facial_state(self, facial_anim: FacialAnimation, current_time: float) -> Dict:
        """Calcula estado atual da animação facial."""
        try:
            # Interpola entre keyframes
            state = {
                'eyebrow_position': 0.0,
                'eye_openness': 1.0,
                'mouth_openness': 0.0,
                'expression': facial_anim.expression.value
            }
            
            if not facial_anim.keyframes:
                return state
            
            # Encontra keyframes adjacentes
            prev_frame = None
            next_frame = None
            
            for i, keyframe in enumerate(facial_anim.keyframes):
                if keyframe.timestamp <= current_time:
                    prev_frame = keyframe
                else:
                    next_frame = keyframe
                    break
            
            if prev_frame and next_frame:
                # Interpola entre keyframes
                duration = next_frame.timestamp - prev_frame.timestamp
                if duration > 0:
                    progress = (current_time - prev_frame.timestamp) / duration
                    
                    # Aplica easing
                    progress = self._apply_easing(progress, next_frame.easing)
                    
                    # Interpola parâmetros
                    for param in prev_frame.parameters:
                        if param in next_frame.parameters:
                            prev_val = prev_frame.parameters[param]
                            next_val = next_frame.parameters[param]
                            state[param] = prev_val + (next_val - prev_val) * progress
            
            elif prev_frame:
                # Usa último keyframe
                state.update(prev_frame.parameters)
            
            return state
            
        except Exception as e:
            self.logger.error(f"Erro ao calcular estado facial: {e}")
            return {'expression': 'neutral'}
    
    def _calculate_lip_sync_state(self, lip_sync: LipSyncData, current_time: float) -> Dict:
        """Calcula estado atual do lip sync."""
        try:
            state = {
                'mouth_openness': 0.0,
                'mouth_width': 0.0,
                'lip_rounding': 0.0,
                'current_viseme': 'SIL'
            }
            
            # Encontra visema atual
            current_viseme = None
            for viseme in lip_sync.visemes:
                viseme_time = viseme.get('time', 0.0)
                viseme_duration = viseme.get('duration', 0.1)
                
                if viseme_time <= current_time <= viseme_time + viseme_duration:
                    current_viseme = viseme
                    break
            
            if current_viseme:
                state.update(current_viseme)
                state['current_viseme'] = current_viseme.get('viseme', 'SIL')
            
            return state
            
        except Exception as e:
            self.logger.error(f"Erro ao calcular lip sync: {e}")
            return {'current_viseme': 'SIL'}
    
    def _calculate_gesture_state(self, gesture_anim: GestureAnimation, current_time: float) -> Dict:
        """Calcula estado atual do gesto."""
        try:
            state = {
                'gesture_type': gesture_anim.gesture_type.value,
                'active': False,
                'hand_position': (0.0, 0.0, 0.0),
                'intensity': 0.0
            }
            
            gesture_time = current_time - gesture_anim.start_time
            
            if 0 <= gesture_time <= gesture_anim.duration:
                state['active'] = True
                
                # Interpola posição das mãos
                if gesture_anim.hand_positions:
                    progress = gesture_time / gesture_anim.duration
                    num_positions = len(gesture_anim.hand_positions)
                    
                    if num_positions > 1:
                        segment_duration = gesture_anim.duration / (num_positions - 1)
                        segment_index = min(int(gesture_time / segment_duration), num_positions - 2)
                        
                        segment_progress = (gesture_time - segment_index * segment_duration) / segment_duration
                        
                        pos1 = gesture_anim.hand_positions[segment_index]
                        pos2 = gesture_anim.hand_positions[segment_index + 1]
                        
                        # Interpola posição
                        state['hand_position'] = (
                            pos1[0] + (pos2[0] - pos1[0]) * segment_progress,
                            pos1[1] + (pos2[1] - pos1[1]) * segment_progress,
                            pos1[2] + (pos2[2] - pos1[2]) * segment_progress
                        )
                    else:
                        state['hand_position'] = gesture_anim.hand_positions[0]
                
                # Intensidade baseada no progresso
                state['intensity'] = gesture_anim.intensity * min(1.0, 2.0 * min(progress, 1.0 - progress))
            
            return state
            
        except Exception as e:
            self.logger.error(f"Erro ao calcular gesto: {e}")
            return {'gesture_type': 'none', 'active': False}
    
    def _calculate_body_state(self, body_anim: BodyAnimation, current_time: float) -> Dict:
        """Calcula estado atual da animação corporal."""
        try:
            state = {
                'posture': body_anim.posture,
                'head_position': body_anim.head_position,
                'breathing_phase': 0.0,
                'idle_sway': 0.0
            }
            
            # Calcula fase da respiração
            if body_anim.breathing_pattern == 'excited':
                breathing_rate = 1.5  # Mais rápida
            elif body_anim.breathing_pattern == 'calm':
                breathing_rate = 0.7  # Mais lenta
            else:
                breathing_rate = 1.0  # Normal
            
            breathing_cycle = 4.0 / breathing_rate  # Duração do ciclo
            breathing_progress = (current_time % breathing_cycle) / breathing_cycle
            state['breathing_phase'] = math.sin(breathing_progress * 2 * math.pi) * 0.1
            
            # Calcula movimento idle
            sway_cycle = 6.0  # Ciclo de 6 segundos
            sway_progress = (current_time % sway_cycle) / sway_cycle
            state['idle_sway'] = math.sin(sway_progress * 2 * math.pi) * body_anim.idle_sway
            
            return state
            
        except Exception as e:
            self.logger.error(f"Erro ao calcular estado corporal: {e}")
            return {'posture': 'neutral'}
    
    def _apply_easing(self, progress: float, easing_type: str) -> float:
        """Aplica função de easing à progressão."""
        try:
            progress = max(0.0, min(1.0, progress))
            
            if easing_type == "ease_in":
                return progress * progress
            elif easing_type == "ease_out":
                return 1.0 - (1.0 - progress) * (1.0 - progress)
            elif easing_type == "ease_in_out":
                if progress < 0.5:
                    return 2.0 * progress * progress
                else:
                    return 1.0 - 2.0 * (1.0 - progress) * (1.0 - progress)
            else:  # linear
                return progress
                
        except Exception as e:
            self.logger.error(f"Erro no easing: {e}")
            return progress
    
    def _render_frame(self, animation_state: Dict):
        """Renderiza frame atual (implementação básica)."""
        try:
            if self.renderer == 'pygame':
                self._render_pygame_frame(animation_state)
            elif self.renderer == 'opencv':
                self._render_opencv_frame(animation_state)
                
        except Exception as e:
            self.logger.error(f"Erro na renderização: {e}")
    
    def _render_pygame_frame(self, animation_state: Dict):
        """Renderização usando pygame (placeholder)."""
        # Implementação básica - em produção seria muito mais complexa
        pass
    
    def _render_opencv_frame(self, animation_state: Dict):
        """Renderização usando OpenCV (placeholder)."""
        # Implementação básica - em produção seria muito mais complexa
        pass
    
    def stop_current_animation(self):
        """Para animação atual."""
        try:
            self.is_animating = False
            self.stop_animation_event.set()
            
            if self.animation_thread and self.animation_thread.is_alive():
                self.animation_thread.join(timeout=1.0)
            
            self.current_animation = None
            self._execute_animation_callbacks('stopped', None)
            
        except Exception as e:
            self.logger.error(f"Erro ao parar animação: {e}")
    
    def start_idle_animation(self):
        """Inicia animações idle quando não está falando."""
        try:
            if not self.idle_animation_active and not self.is_animating:
                self.idle_animation_active = True
                # Implementaria ciclo de animações idle aqui
                
        except Exception as e:
            self.logger.error(f"Erro ao iniciar idle: {e}")
    
    def stop_idle_animation(self):
        """Para animações idle."""
        self.idle_animation_active = False
    
    def add_animation_callback(self, callback: Callable):
        """Adiciona callback para eventos de animação."""
        if callback not in self.animation_callbacks:
            self.animation_callbacks.append(callback)
    
    def add_frame_callback(self, callback: Callable):
        """Adiciona callback para frames de animação."""
        if callback not in self.frame_callbacks:
            self.frame_callbacks.append(callback)
    
    def remove_animation_callback(self, callback: Callable):
        """Remove callback de animação."""
        if callback in self.animation_callbacks:
            self.animation_callbacks.remove(callback)
    
    def remove_frame_callback(self, callback: Callable):
        """Remove callback de frame."""
        if callback in self.frame_callbacks:
            self.frame_callbacks.remove(callback)
    
    def _execute_animation_callbacks(self, event: str, sequence: Optional[AnimationSequence]):
        """Executa callbacks de animação."""
        for callback in self.animation_callbacks:
            try:
                callback(event, sequence)
            except Exception as e:
                self.logger.error(f"Erro ao executar callback: {e}")
    
    def _execute_frame_callbacks(self, animation_state: Dict, current_time: float):
        """Executa callbacks de frame."""
        for callback in self.frame_callbacks:
            try:
                callback(animation_state, current_time)
            except Exception as e:
                self.logger.error(f"Erro ao executar callback de frame: {e}")
    
    def _create_fallback_animation(self, duration: float) -> AnimationSequence:
        """Cria animação fallback básica."""
        return AnimationSequence(
            total_duration=duration,
            facial_animation=self._create_default_facial_animation(duration),
            lip_sync_data=LipSyncData(),
            body_animation=BodyAnimation(),
            emotion_conveyed="neutral",
            response_strategy="informative"
        )
    
    def _create_default_facial_animation(self, duration: float) -> FacialAnimation:
        """Cria animação facial padrão."""
        return FacialAnimation(
            expression=ExpressionType.NEUTRAL,
            intensity=0.5,
            duration=duration,
            keyframes=[
                AnimationKeyframe(
                    timestamp=0.0,
                    parameters={'eyebrow_position': 0.0, 'eye_openness': 1.0, 'mouth_openness': 0.0}
                ),
                AnimationKeyframe(
                    timestamp=duration,
                    parameters={'eyebrow_position': 0.0, 'eye_openness': 1.0, 'mouth_openness': 0.0}
                )
            ]
        )
    
    def _update_performance_metrics(self, sequence: AnimationSequence, 
                                   generation_time: float, success: bool):
        """Atualiza métricas de performance."""
        self.performance_metrics['total_animations'] += 1
        
        if success:
            self.performance_metrics['successful_animations'] += 1
        
        # Tempo de geração
        self.performance_metrics['generation_times'].append(generation_time)
        if self.performance_metrics['generation_times']:
            self.performance_metrics['average_generation_time'] = np.mean(
                self.performance_metrics['generation_times']
            )
        
        # Expressões usadas
        expression = sequence.facial_animation.expression.value
        if expression in self.performance_metrics['expressions_used']:
            self.performance_metrics['expressions_used'][expression] += 1
        else:
            self.performance_metrics['expressions_used'][expression] = 1
        
        # Gestos usados
        if sequence.gesture_animation:
            gesture = sequence.gesture_animation.gesture_type.value
            if gesture in self.performance_metrics['gestures_used']:
                self.performance_metrics['gestures_used'][gesture] += 1
            else:
                self.performance_metrics['gestures_used'][gesture] = 1
    
    def get_performance_metrics(self) -> Dict:
        """Retorna métricas de performance."""
        metrics = self.performance_metrics.copy()
        
        # Adiciona métricas calculadas
        if metrics['total_animations'] > 0:
            metrics['success_rate'] = metrics['successful_animations'] / metrics['total_animations']
        else:
            metrics['success_rate'] = 0.0
        
        return metrics
    
    def get_animation_status(self) -> Dict:
        """Retorna status atual da animação."""
        return {
            'is_animating': self.is_animating,
            'idle_active': self.idle_animation_active,
            'current_animation': {
                'emotion': self.current_animation.emotion_conveyed if self.current_animation else None,
                'duration': self.current_animation.total_duration if self.current_animation else 0.0,
                'elapsed': time.time() - self.animation_start_time if self.is_animating else 0.0
            }
        }
    
    def clear_cache(self):
        """Limpa cache de animações."""
        self.animation_cache.clear()
        self.logger.info("Cache de animações limpo")
    
    def clear_history(self):
        """Limpa histórico de animações."""
        self.animation_history.clear()
        self.logger.info("Histórico de animações limpo")
    
    def __del__(self):
        """Cleanup automático."""
        self.stop_current_animation()