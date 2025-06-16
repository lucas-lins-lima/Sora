# sora_robot/vision_processing/body_pose_estimation.py

import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, List, Optional, Tuple, NamedTuple
import threading
import time
import math
from dataclasses import dataclass, field
from collections import deque
from enum import Enum

from utils.logger import get_logger
from utils.constants import BODY_POSE, EMOTIONS, PERFORMANCE
import config

class GestureType(Enum):
    """Tipos de gestos detectáveis."""
    WAVE = "wave"
    POINTING = "pointing"
    THUMBS_UP = "thumbs_up"
    OPEN_ARMS = "open_arms"
    CROSSED_ARMS = "crossed_arms"
    HAND_ON_HIP = "hand_on_hip"
    CLAPPING = "clapping"
    RAISED_HAND = "raised_hand"
    PEACE_SIGN = "peace_sign"
    THINKING_POSE = "thinking_pose"

class PostureType(Enum):
    """Tipos de postura corporal."""
    UPRIGHT = "upright"
    LEANING_FORWARD = "leaning_forward"
    LEANING_BACK = "leaning_back"
    SLOUCHING = "slouching"
    DEFENSIVE = "defensive"
    OPEN = "open"
    CONFIDENT = "confident"
    NERVOUS = "nervous"

@dataclass
class BodyKeypoint:
    """Representa um ponto-chave do corpo."""
    x: float
    y: float
    z: float
    confidence: float
    visible: bool

@dataclass
class BodyPoseData:
    """Estrutura completa de dados de pose corporal."""
    
    # Keypoints detectados
    keypoints: Dict[str, BodyKeypoint] = field(default_factory=dict)
    
    # Gestos detectados
    detected_gestures: List[Tuple[GestureType, float]] = field(default_factory=list)  # (gesture, confidence)
    primary_gesture: Optional[GestureType] = None
    gesture_confidence: float = 0.0
    
    # Postura corporal
    posture_type: PostureType = PostureType.UPRIGHT
    posture_confidence: float = 0.0
    
    # Análises derivadas
    body_orientation: float = 0.0  # Ângulo em graus (-180 a 180)
    engagement_level: float = 0.0  # 0.0 a 1.0 (quão engajada a pessoa parece)
    energy_level: float = 0.0      # 0.0 a 1.0 (quão energética/ativa)
    openness_score: float = 0.0    # 0.0 a 1.0 (quão aberta/receptiva)
    
    # Métricas de movimento
    movement_intensity: float = 0.0  # Quantidade de movimento detectado
    stability_score: float = 0.0     # Quão estável é a pose
    
    # Informações temporais
    timestamp: float = 0.0
    tracking_id: Optional[int] = None
    
    # Dados de qualidade
    detection_quality: float = 0.0   # Qualidade geral da detecção
    visible_keypoints: int = 0       # Número de keypoints visíveis
    
    # Histórico de poses (para análise temporal)
    pose_history: List['BodyPoseData'] = field(default_factory=list)

class BodyPoseEstimation:
    """
    Classe responsável pela estimativa de pose corporal e análise de gestos.
    Utiliza MediaPipe Pose para detecção robusta e análise de linguagem corporal.
    """
    
    def __init__(self):
        """Inicializa o sistema de estimativa de pose corporal."""
        self.logger = get_logger(__name__)
        
        # Inicialização do MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Modelo MediaPipe Pose
        self.pose_detector = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # 0=lite, 1=full, 2=heavy
            smooth_landmarks=True,
            enable_segmentation=False,  # Desabilitado para performance
            smooth_segmentation=False,
            min_detection_confidence=BODY_POSE.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=BODY_POSE.MIN_TRACKING_CONFIDENCE
        )
        
        # Sistema de tracking de pessoas
        self.person_tracker = {}  # tracking_id -> BodyPoseData
        self.next_tracking_id = 1
        self.tracking_threshold = 100  # pixels para considerar mesma pessoa
        
        # Cache para performance
        self.pose_cache = {}
        self.cache_ttl = 0.05  # 50ms
        
        # Histórico para análise temporal
        self.pose_history = {}  # tracking_id -> deque of BodyPoseData
        self.history_length = 15  # Número de poses anteriores para manter
        
        # Threading
        self.processing_lock = threading.Lock()
        
        # Configurações de análise
        self.gesture_confidence_threshold = 0.6
        self.movement_sensitivity = 0.1
        self.stability_window = 5  # frames para calcular estabilidade
        
        # Métricas de performance
        self.performance_metrics = {
            'total_detections': 0,
            'successful_detections': 0,
            'processing_times': deque(maxlen=100),
            'average_processing_time': 0.0,
            'gestures_detected': {gesture.value: 0 for gesture in GestureType},
            'postures_detected': {posture.value: 0 for posture in PostureType}
        }
        
        # Inicializa mapeamentos de gestos
        self._initialize_gesture_rules()
        self._initialize_posture_rules()
        
        self.logger.info("BodyPoseEstimation inicializado com sucesso")
    
    def _initialize_gesture_rules(self):
        """Inicializa regras para detecção de gestos."""
        
        self.gesture_rules = {
            GestureType.WAVE: {
                'description': 'Mão levantada com movimento lateral',
                'requirements': [
                    lambda kp: kp.get('RIGHT_WRIST', BodyKeypoint(0,0,0,0,False)).y < kp.get('RIGHT_SHOULDER', BodyKeypoint(0,0,0,0,False)).y,
                    lambda kp: kp.get('RIGHT_WRIST', BodyKeypoint(0,0,0,0,False)).confidence > 0.5
                ],
                'movement_pattern': 'lateral_hand_movement'
            },
            
            GestureType.POINTING: {
                'description': 'Braço estendido apontando',
                'requirements': [
                    lambda kp: abs(kp.get('RIGHT_WRIST', BodyKeypoint(0,0,0,0,False)).y - 
                                  kp.get('RIGHT_SHOULDER', BodyKeypoint(0,0,0,0,False)).y) < 50,
                    lambda kp: kp.get('RIGHT_WRIST', BodyKeypoint(0,0,0,0,False)).x > 
                              kp.get('RIGHT_ELBOW', BodyKeypoint(0,0,0,0,False)).x
                ]
            },
            
            GestureType.THUMBS_UP: {
                'description': 'Polegar para cima',
                'requirements': [
                    lambda kp: kp.get('RIGHT_THUMB', BodyKeypoint(0,0,0,0,False)).y < 
                              kp.get('RIGHT_WRIST', BodyKeypoint(0,0,0,0,False)).y,
                    lambda kp: kp.get('RIGHT_THUMB', BodyKeypoint(0,0,0,0,False)).confidence > 0.7
                ]
            },
            
            GestureType.OPEN_ARMS: {
                'description': 'Braços abertos lateralmente',
                'requirements': [
                    lambda kp: kp.get('RIGHT_WRIST', BodyKeypoint(0,0,0,0,False)).x > 
                              kp.get('RIGHT_SHOULDER', BodyKeypoint(0,0,0,0,False)).x + 100,
                    lambda kp: kp.get('LEFT_WRIST', BodyKeypoint(0,0,0,0,False)).x < 
                              kp.get('LEFT_SHOULDER', BodyKeypoint(0,0,0,0,False)).x - 100
                ]
            },
            
            GestureType.CROSSED_ARMS: {
                'description': 'Braços cruzados no peito',
                'requirements': [
                    lambda kp: abs(kp.get('RIGHT_WRIST', BodyKeypoint(0,0,0,0,False)).x - 
                                  kp.get('LEFT_SHOULDER', BodyKeypoint(0,0,0,0,False)).x) < 100,
                    lambda kp: abs(kp.get('LEFT_WRIST', BodyKeypoint(0,0,0,0,False)).x - 
                                  kp.get('RIGHT_SHOULDER', BodyKeypoint(0,0,0,0,False)).x) < 100
                ]
            },
            
            GestureType.RAISED_HAND: {
                'description': 'Mão levantada (pergunta/cumprimento)',
                'requirements': [
                    lambda kp: kp.get('RIGHT_WRIST', BodyKeypoint(0,0,0,0,False)).y < 
                              kp.get('RIGHT_SHOULDER', BodyKeypoint(0,0,0,0,False)).y - 50,
                    lambda kp: abs(kp.get('RIGHT_WRIST', BodyKeypoint(0,0,0,0,False)).x - 
                                  kp.get('RIGHT_SHOULDER', BodyKeypoint(0,0,0,0,False)).x) < 100
                ]
            },
            
            GestureType.THINKING_POSE: {
                'description': 'Mão no queixo (pensativo)',
                'requirements': [
                    lambda kp: abs(kp.get('RIGHT_WRIST', BodyKeypoint(0,0,0,0,False)).y - 
                                  kp.get('NOSE', BodyKeypoint(0,0,0,0,False)).y) < 100,
                    lambda kp: kp.get('RIGHT_WRIST', BodyKeypoint(0,0,0,0,False)).x > 
                              kp.get('NOSE', BodyKeypoint(0,0,0,0,False)).x - 50
                ]
            }
        }
    
    def _initialize_posture_rules(self):
        """Inicializa regras para análise de postura."""
        
        self.posture_rules = {
            PostureType.UPRIGHT: {
                'description': 'Postura ereta e alinhada',
                'spine_alignment': 'straight',
                'shoulder_level': 'even',
                'confidence_indicator': True
            },
            
            PostureType.LEANING_FORWARD: {
                'description': 'Inclinado para frente (interesse/atenção)',
                'torso_angle': 'forward',
                'engagement_level': 'high'
            },
            
            PostureType.LEANING_BACK: {
                'description': 'Inclinado para trás (relaxado/distante)',
                'torso_angle': 'backward',
                'engagement_level': 'low'
            },
            
            PostureType.SLOUCHING: {
                'description': 'Postura curvada (cansaço/desinteresse)',
                'spine_alignment': 'curved',
                'energy_level': 'low'
            },
            
            PostureType.DEFENSIVE: {
                'description': 'Postura defensiva (braços cruzados, fechada)',
                'arm_position': 'closed',
                'openness_level': 'low'
            },
            
            PostureType.OPEN: {
                'description': 'Postura aberta e receptiva',
                'arm_position': 'open',
                'openness_level': 'high'
            },
            
            PostureType.CONFIDENT: {
                'description': 'Postura confiante (ombros para trás, peito aberto)',
                'chest_position': 'open',
                'shoulder_position': 'back'
            }
        }
    
    def estimate_pose(self, frame: np.ndarray) -> Optional[BodyPoseData]:
        """
        Estima pose corporal em um frame.
        
        Args:
            frame: Frame de entrada (BGR)
            
        Returns:
            Optional[BodyPoseData]: Dados de pose estimados ou None se falhou
        """
        start_time = time.time()
        
        try:
            with self.processing_lock:
                # Verifica cache
                frame_hash = hash(frame.tobytes())
                current_time = time.time()
                
                if frame_hash in self.pose_cache:
                    cached_data, cache_time = self.pose_cache[frame_hash]
                    if current_time - cache_time < self.cache_ttl:
                        return cached_data
                
                # Converte para RGB para MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Detecção de pose
                results = self.pose_detector.process(rgb_frame)
                
                if not results.pose_landmarks:
                    return None
                
                # Extrai keypoints
                keypoints = self._extract_keypoints(results.pose_landmarks, frame.shape)
                
                # Cria estrutura de dados da pose
                pose_data = BodyPoseData(
                    keypoints=keypoints,
                    timestamp=current_time,
                    visible_keypoints=len([kp for kp in keypoints.values() if kp.visible])
                )
                
                # Calcula qualidade da detecção
                pose_data.detection_quality = self._calculate_detection_quality(keypoints)
                
                # Análise de gestos
                pose_data = self._analyze_gestures(pose_data)
                
                # Análise de postura
                pose_data = self._analyze_posture(pose_data)
                
                # Análise de orientação corporal
                pose_data.body_orientation = self._calculate_body_orientation(keypoints)
                
                # Análise de níveis (engajamento, energia, abertura)
                pose_data.engagement_level = self._calculate_engagement_level(pose_data)
                pose_data.energy_level = self._calculate_energy_level(pose_data)
                pose_data.openness_score = self._calculate_openness_score(pose_data)
                
                # Sistema de tracking
                pose_data = self._update_pose_tracking(pose_data, current_time)
                
                # Análise temporal se há histórico
                if pose_data.tracking_id and pose_data.tracking_id in self.pose_history:
                    pose_data = self._analyze_temporal_features(pose_data)
                
                # Atualiza histórico
                self._update_pose_history(pose_data)
                
                # Atualiza cache
                self.pose_cache[frame_hash] = (pose_data, current_time)
                
                # Limpa cache antigo
                self._cleanup_cache(current_time)
                
                # Atualiza métricas
                processing_time = time.time() - start_time
                self._update_metrics(pose_data, processing_time)
                
                return pose_data
                
        except Exception as e:
            self.logger.error(f"Erro na estimativa de pose: {e}")
            return None
    
    def _extract_keypoints(self, pose_landmarks, frame_shape: Tuple[int, int, int]) -> Dict[str, BodyKeypoint]:
        """Extrai keypoints do resultado do MediaPipe."""
        keypoints = {}
        
        try:
            h, w = frame_shape[:2]
            
            # Mapeia landmarks do MediaPipe para nossos keypoints
            landmark_mapping = {
                0: 'NOSE',
                2: 'LEFT_EYE',
                5: 'RIGHT_EYE',
                7: 'LEFT_EAR',
                8: 'RIGHT_EAR',
                9: 'MOUTH_LEFT',
                10: 'MOUTH_RIGHT',
                11: 'LEFT_SHOULDER',
                12: 'RIGHT_SHOULDER',
                13: 'LEFT_ELBOW',
                14: 'RIGHT_ELBOW',
                15: 'LEFT_WRIST',
                16: 'RIGHT_WRIST',
                17: 'LEFT_PINKY',
                18: 'RIGHT_PINKY',
                19: 'LEFT_INDEX',
                20: 'RIGHT_INDEX',
                21: 'LEFT_THUMB',
                22: 'RIGHT_THUMB',
                23: 'LEFT_HIP',
                24: 'RIGHT_HIP',
                25: 'LEFT_KNEE',
                26: 'RIGHT_KNEE',
                27: 'LEFT_ANKLE',
                28: 'RIGHT_ANKLE'
            }
            
            for i, landmark in enumerate(pose_landmarks.landmark):
                if i in landmark_mapping:
                    keypoint_name = landmark_mapping[i]
                    
                    # Converte coordenadas normalizadas para pixels
                    x = landmark.x * w
                    y = landmark.y * h
                    z = landmark.z  # Profundidade relativa
                    
                    # Determina visibilidade (MediaPipe usa visibility)
                    confidence = landmark.visibility if hasattr(landmark, 'visibility') else 0.5
                    visible = confidence > 0.3
                    
                    keypoints[keypoint_name] = BodyKeypoint(
                        x=x, y=y, z=z,
                        confidence=confidence,
                        visible=visible
                    )
            
        except Exception as e:
            self.logger.error(f"Erro ao extrair keypoints: {e}")
        
        return keypoints
    
    def _calculate_detection_quality(self, keypoints: Dict[str, BodyKeypoint]) -> float:
        """Calcula qualidade geral da detecção de pose."""
        if not keypoints:
            return 0.0
        
        try:
            # Keypoints importantes para qualidade
            important_keypoints = [
                'NOSE', 'LEFT_SHOULDER', 'RIGHT_SHOULDER',
                'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_HIP', 'RIGHT_HIP'
            ]
            
            total_confidence = 0.0
            visible_count = 0
            
            for keypoint_name in important_keypoints:
                if keypoint_name in keypoints:
                    kp = keypoints[keypoint_name]
                    if kp.visible:
                        total_confidence += kp.confidence
                        visible_count += 1
            
            if visible_count == 0:
                return 0.0
            
            # Score baseado na confiança média e número de keypoints visíveis
            avg_confidence = total_confidence / visible_count
            visibility_ratio = visible_count / len(important_keypoints)
            
            quality_score = (avg_confidence * 0.7) + (visibility_ratio * 0.3)
            
            return min(1.0, max(0.0, quality_score))
            
        except Exception as e:
            self.logger.error(f"Erro ao calcular qualidade da detecção: {e}")
            return 0.0
    
    def _analyze_gestures(self, pose_data: BodyPoseData) -> BodyPoseData:
        """Analisa gestos baseado nos keypoints."""
        try:
            detected_gestures = []
            
            for gesture_type, gesture_config in self.gesture_rules.items():
                confidence = self._calculate_gesture_confidence(
                    gesture_type, pose_data.keypoints, gesture_config
                )
                
                if confidence > self.gesture_confidence_threshold:
                    detected_gestures.append((gesture_type, confidence))
            
            # Ordena por confiança
            detected_gestures.sort(key=lambda x: x[1], reverse=True)
            
            pose_data.detected_gestures = detected_gestures
            
            # Define gesto primário
            if detected_gestures:
                pose_data.primary_gesture = detected_gestures[0][0]
                pose_data.gesture_confidence = detected_gestures[0][1]
            
        except Exception as e:
            self.logger.error(f"Erro na análise de gestos: {e}")
        
        return pose_data
    
    def _calculate_gesture_confidence(self, gesture_type: GestureType, 
                                    keypoints: Dict[str, BodyKeypoint], 
                                    gesture_config: Dict) -> float:
        """Calcula confiança de um gesto específico."""
        try:
            requirements = gesture_config.get('requirements', [])
            if not requirements:
                return 0.0
            
            satisfied_requirements = 0
            
            for requirement in requirements:
                try:
                    if requirement(keypoints):
                        satisfied_requirements += 1
                except Exception:
                    # Requirement failed (missing keypoints, etc.)
                    continue
            
            # Confiança baseada na proporção de requisitos satisfeitos
            confidence = satisfied_requirements / len(requirements)
            
            # Ajusta confiança baseado na qualidade dos keypoints envolvidos
            quality_bonus = self._calculate_keypoint_quality_bonus(keypoints, gesture_type)
            
            final_confidence = min(1.0, confidence + quality_bonus)
            
            return final_confidence
            
        except Exception as e:
            self.logger.error(f"Erro ao calcular confiança do gesto {gesture_type}: {e}")
            return 0.0
    
    def _calculate_keypoint_quality_bonus(self, keypoints: Dict[str, BodyKeypoint], 
                                        gesture_type: GestureType) -> float:
        """Calcula bônus de qualidade baseado nos keypoints relevantes."""
        try:
            # Keypoints relevantes para cada gesto
            relevant_keypoints = {
                GestureType.WAVE: ['RIGHT_WRIST', 'RIGHT_ELBOW', 'RIGHT_SHOULDER'],
                GestureType.POINTING: ['RIGHT_WRIST', 'RIGHT_ELBOW', 'RIGHT_SHOULDER'],
                GestureType.THUMBS_UP: ['RIGHT_THUMB', 'RIGHT_WRIST'],
                GestureType.OPEN_ARMS: ['LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_SHOULDER', 'RIGHT_SHOULDER'],
                GestureType.CROSSED_ARMS: ['LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_SHOULDER', 'RIGHT_SHOULDER'],
                GestureType.RAISED_HAND: ['RIGHT_WRIST', 'RIGHT_SHOULDER'],
                GestureType.THINKING_POSE: ['RIGHT_WRIST', 'NOSE']
            }
            
            relevant_kps = relevant_keypoints.get(gesture_type, [])
            if not relevant_kps:
                return 0.0
            
            total_confidence = 0.0
            count = 0
            
            for kp_name in relevant_kps:
                if kp_name in keypoints and keypoints[kp_name].visible:
                    total_confidence += keypoints[kp_name].confidence
                    count += 1
            
            if count == 0:
                return 0.0
            
            avg_confidence = total_confidence / count
            # Bônus máximo de 0.2
            return min(0.2, (avg_confidence - 0.5) * 0.4)
            
        except Exception as e:
            self.logger.error(f"Erro ao calcular bônus de qualidade: {e}")
            return 0.0
    
    def _analyze_posture(self, pose_data: BodyPoseData) -> BodyPoseData:
        """Analisa postura corporal."""
        try:
            keypoints = pose_data.keypoints
            
            # Calcula ângulos e proporções corporais
            posture_features = self._extract_posture_features(keypoints)
            
            # Avalia cada tipo de postura
            posture_scores = {}
            
            for posture_type in PostureType:
                score = self._calculate_posture_score(posture_type, posture_features)
                posture_scores[posture_type] = score
            
            # Encontra postura predominante
            best_posture = max(posture_scores, key=posture_scores.get)
            best_score = posture_scores[best_posture]
            
            pose_data.posture_type = best_posture
            pose_data.posture_confidence = best_score
            
        except Exception as e:
            self.logger.error(f"Erro na análise de postura: {e}")
        
        return pose_data
    
    def _extract_posture_features(self, keypoints: Dict[str, BodyKeypoint]) -> Dict[str, float]:
        """Extrai características relevantes para análise de postura."""
        features = {}
        
        try:
            # Verifica se keypoints essenciais estão disponíveis
            essential_kps = ['LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_HIP', 'RIGHT_HIP']
            if not all(kp in keypoints and keypoints[kp].visible for kp in essential_kps):
                return features
            
            left_shoulder = keypoints['LEFT_SHOULDER']
            right_shoulder = keypoints['RIGHT_SHOULDER']
            left_hip = keypoints['LEFT_HIP']
            right_hip = keypoints['RIGHT_HIP']
            
            # Ângulo dos ombros (indica inclinação lateral)
            shoulder_angle = math.atan2(
                right_shoulder.y - left_shoulder.y,
                right_shoulder.x - left_shoulder.x
            ) * 180 / math.pi
            features['shoulder_angle'] = abs(shoulder_angle)
            
            # Ângulo do torso (indica inclinação para frente/trás)
            shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
            hip_center_y = (left_hip.y + right_hip.y) / 2
            shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
            hip_center_x = (left_hip.x + right_hip.x) / 2
            
            if hip_center_y != shoulder_center_y:
                torso_angle = math.atan2(
                    hip_center_x - shoulder_center_x,
                    hip_center_y - shoulder_center_y
                ) * 180 / math.pi
                features['torso_angle'] = torso_angle
            
            # Largura dos ombros vs quadris
            shoulder_width = abs(right_shoulder.x - left_shoulder.x)
            hip_width = abs(right_hip.x - left_hip.x)
            if hip_width > 0:
                features['shoulder_hip_ratio'] = shoulder_width / hip_width
            
            # Altura relativa dos ombros (simetria)
            features['shoulder_symmetry'] = 1.0 - abs(left_shoulder.y - right_shoulder.y) / max(1, shoulder_width)
            
            # Posição dos braços se disponível
            if 'LEFT_WRIST' in keypoints and 'RIGHT_WRIST' in keypoints:
                left_wrist = keypoints['LEFT_WRIST']
                right_wrist = keypoints['RIGHT_WRIST']
                
                # Altura relativa dos pulsos em relação aos ombros
                left_arm_height = (left_shoulder.y - left_wrist.y) / max(1, shoulder_width)
                right_arm_height = (right_shoulder.y - right_wrist.y) / max(1, shoulder_width)
                features['arms_height'] = (left_arm_height + right_arm_height) / 2
                
                # Distância dos braços do corpo
                left_arm_distance = abs(left_wrist.x - left_shoulder.x) / max(1, shoulder_width)
                right_arm_distance = abs(right_wrist.x - right_shoulder.x) / max(1, shoulder_width)
                features['arms_openness'] = (left_arm_distance + right_arm_distance) / 2
            
        except Exception as e:
            self.logger.error(f"Erro ao extrair características de postura: {e}")
        
        return features
    
    def _calculate_posture_score(self, posture_type: PostureType, features: Dict[str, float]) -> float:
        """Calcula score para um tipo específico de postura."""
        try:
            score = 0.0
            
            if posture_type == PostureType.UPRIGHT:
                # Postura ereta: torso reto, ombros nivelados
                if 'torso_angle' in features:
                    score += max(0, 1.0 - abs(features['torso_angle']) / 20.0) * 0.4
                if 'shoulder_symmetry' in features:
                    score += features['shoulder_symmetry'] * 0.3
                
            elif posture_type == PostureType.LEANING_FORWARD:
                # Inclinado para frente: torso_angle positivo
                if 'torso_angle' in features and features['torso_angle'] > 5:
                    score += min(1.0, features['torso_angle'] / 30.0) * 0.6
                    
            elif posture_type == PostureType.LEANING_BACK:
                # Inclinado para trás: torso_angle negativo
                if 'torso_angle' in features and features['torso_angle'] < -5:
                    score += min(1.0, abs(features['torso_angle']) / 30.0) * 0.6
                    
            elif posture_type == PostureType.OPEN:
                # Postura aberta: braços afastados do corpo
                if 'arms_openness' in features:
                    score += min(1.0, features['arms_openness']) * 0.5
                    
            elif posture_type == PostureType.DEFENSIVE:
                # Postura defensiva: braços próximos ao corpo
                if 'arms_openness' in features:
                    score += max(0, 1.0 - features['arms_openness']) * 0.5
                    
            elif posture_type == PostureType.CONFIDENT:
                # Postura confiante: peito aberto, ombros para trás
                if 'shoulder_hip_ratio' in features and features['shoulder_hip_ratio'] > 1.0:
                    score += 0.4
                if 'torso_angle' in features and features['torso_angle'] < 0:
                    score += 0.3
            
            return min(1.0, max(0.0, score))
            
        except Exception as e:
            self.logger.error(f"Erro ao calcular score de postura {posture_type}: {e}")
            return 0.0
    
    def _calculate_body_orientation(self, keypoints: Dict[str, BodyKeypoint]) -> float:
        """Calcula orientação corporal em graus."""
        try:
            if 'LEFT_SHOULDER' not in keypoints or 'RIGHT_SHOULDER' not in keypoints:
                return 0.0
            
            left_shoulder = keypoints['LEFT_SHOULDER']
            right_shoulder = keypoints['RIGHT_SHOULDER']
            
            # Calcula ângulo da linha dos ombros
            angle = math.atan2(
                right_shoulder.y - left_shoulder.y,
                right_shoulder.x - left_shoulder.x
            ) * 180 / math.pi
            
            # Normaliza para -180 a 180
            while angle > 180:
                angle -= 360
            while angle < -180:
                angle += 360
            
            return angle
            
        except Exception as e:
            self.logger.error(f"Erro ao calcular orientação corporal: {e}")
            return 0.0
    
    def _calculate_engagement_level(self, pose_data: BodyPoseData) -> float:
        """Calcula nível de engajamento baseado na postura e gestos."""
        try:
            engagement = 0.5  # baseline
            
            # Bonus por postura
            if pose_data.posture_type == PostureType.LEANING_FORWARD:
                engagement += 0.3
            elif pose_data.posture_type == PostureType.UPRIGHT:
                engagement += 0.1
            elif pose_data.posture_type == PostureType.LEANING_BACK:
                engagement -= 0.2
            elif pose_data.posture_type == PostureType.SLOUCHING:
                engagement -= 0.3
            
            # Bonus por gestos
            if pose_data.primary_gesture in [GestureType.WAVE, GestureType.POINTING, GestureType.RAISED_HAND]:
                engagement += 0.2
            elif pose_data.primary_gesture == GestureType.CROSSED_ARMS:
                engagement -= 0.2
            
            # Bonus por orientação (frontal = mais engajado)
            if abs(pose_data.body_orientation) < 15:
                engagement += 0.1
            elif abs(pose_data.body_orientation) > 45:
                engagement -= 0.1
            
            return min(1.0, max(0.0, engagement))
            
        except Exception as e:
            self.logger.error(f"Erro ao calcular nível de engajamento: {e}")
            return 0.5
    
    def _calculate_energy_level(self, pose_data: BodyPoseData) -> float:
        """Calcula nível de energia baseado na postura e movimento."""
        try:
            energy = 0.5  # baseline
            
            # Bonus por postura
            if pose_data.posture_type == PostureType.UPRIGHT:
                energy += 0.2
            elif pose_data.posture_type == PostureType.CONFIDENT:
                energy += 0.3
            elif pose_data.posture_type == PostureType.SLOUCHING:
                energy -= 0.3
            
            # Bonus por gestos ativos
            active_gestures = [GestureType.WAVE, GestureType.POINTING, GestureType.OPEN_ARMS]
            if pose_data.primary_gesture in active_gestures:
                energy += 0.2
            
            # Bonus por intensidade de movimento (se disponível)
            if pose_data.movement_intensity > 0:
                energy += pose_data.movement_intensity * 0.3
            
            return min(1.0, max(0.0, energy))
            
        except Exception as e:
            self.logger.error(f"Erro ao calcular nível de energia: {e}")
            return 0.5
    
    def _calculate_openness_score(self, pose_data: BodyPoseData) -> float:
        """Calcula score de abertura/receptividade."""
        try:
            openness = 0.5  # baseline
            
            # Bonus por postura aberta
            if pose_data.posture_type == PostureType.OPEN:
                openness += 0.3
            elif pose_data.posture_type == PostureType.CONFIDENT:
                openness += 0.2
            elif pose_data.posture_type == PostureType.DEFENSIVE:
                openness -= 0.3
            
            # Bonus por gestos abertos
            open_gestures = [GestureType.OPEN_ARMS, GestureType.WAVE, GestureType.THUMBS_UP]
            if pose_data.primary_gesture in open_gestures:
                openness += 0.2
            elif pose_data.primary_gesture == GestureType.CROSSED_ARMS:
                openness -= 0.3
            
            return min(1.0, max(0.0, openness))
            
        except Exception as e:
            self.logger.error(f"Erro ao calcular score de abertura: {e}")
            return 0.5
    
    def _update_pose_tracking(self, current_pose: BodyPoseData, current_time: float) -> BodyPoseData:
        """Atualiza sistema de tracking de pessoas."""
        try:
            # Remove tracks antigos
            expired_ids = []
            for track_id, track_data in self.person_tracker.items():
                if current_time - track_data.timestamp > 3.0:  # 3 segundos
                    expired_ids.append(track_id)
            
            for track_id in expired_ids:
                del self.person_tracker[track_id]
                if track_id in self.pose_history:
                    del self.pose_history[track_id]
            
            # Associa pose atual com track existente
            best_match_id = None
            best_distance = float('inf')
            
            # Calcula centro do torso da pose atual
            current_center = self._calculate_torso_center(current_pose.keypoints)
            
            for track_id, track_data in self.person_tracker.items():
                track_center = self._calculate_torso_center(track_data.keypoints)
                
                if current_center and track_center:
                    distance = np.sqrt(
                        (current_center[0] - track_center[0]) ** 2 +
                        (current_center[1] - track_center[1]) ** 2
                    )
                    
                    if distance < self.tracking_threshold and distance < best_distance:
                        best_distance = distance
                        best_match_id = track_id
            
            # Atualiza track existente ou cria novo
            if best_match_id is not None:
                current_pose.tracking_id = best_match_id
            else:
                current_pose.tracking_id = self.next_tracking_id
                self.next_tracking_id += 1
            
            # Atualiza tracker
            self.person_tracker[current_pose.tracking_id] = current_pose
            
            return current_pose
            
        except Exception as e:
            self.logger.error(f"Erro no tracking de poses: {e}")
            return current_pose
    
    def _calculate_torso_center(self, keypoints: Dict[str, BodyKeypoint]) -> Optional[Tuple[float, float]]:
        """Calcula centro do torso baseado nos keypoints."""
        try:
            torso_keypoints = ['LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_HIP', 'RIGHT_HIP']
            available_points = []
            
            for kp_name in torso_keypoints:
                if kp_name in keypoints and keypoints[kp_name].visible:
                    kp = keypoints[kp_name]
                    available_points.append((kp.x, kp.y))
            
            if len(available_points) < 2:
                return None
            
            # Calcula centro
            center_x = sum(point[0] for point in available_points) / len(available_points)
            center_y = sum(point[1] for point in available_points) / len(available_points)
            
            return (center_x, center_y)
            
        except Exception as e:
            self.logger.error(f"Erro ao calcular centro do torso: {e}")
            return None
    
    def _analyze_temporal_features(self, current_pose: BodyPoseData) -> BodyPoseData:
        """Analisa características temporais baseadas no histórico."""
        try:
            if not current_pose.tracking_id or current_pose.tracking_id not in self.pose_history:
                return current_pose
            
            history = list(self.pose_history[current_pose.tracking_id])
            if len(history) < 2:
                return current_pose
            
            # Calcula intensidade de movimento
            movement_intensity = self._calculate_movement_intensity(current_pose, history)
            current_pose.movement_intensity = movement_intensity
            
            # Calcula estabilidade
            stability = self._calculate_pose_stability(current_pose, history)
            current_pose.stability_score = stability
            
            return current_pose
            
        except Exception as e:
            self.logger.error(f"Erro na análise temporal: {e}")
            return current_pose
    
    def _calculate_movement_intensity(self, current_pose: BodyPoseData, 
                                    history: List[BodyPoseData]) -> float:
        """Calcula intensidade de movimento comparando com poses anteriores."""
        try:
            if not history:
                return 0.0
            
            last_pose = history[-1]
            
            # Calcula distância entre keypoints correspondentes
            total_movement = 0.0
            compared_keypoints = 0
            
            for kp_name in current_pose.keypoints:
                if (kp_name in last_pose.keypoints and 
                    current_pose.keypoints[kp_name].visible and 
                    last_pose.keypoints[kp_name].visible):
                    
                    current_kp = current_pose.keypoints[kp_name]
                    last_kp = last_pose.keypoints[kp_name]
                    
                    distance = np.sqrt(
                        (current_kp.x - last_kp.x) ** 2 +
                        (current_kp.y - last_kp.y) ** 2
                    )
                    
                    total_movement += distance
                    compared_keypoints += 1
            
            if compared_keypoints == 0:
                return 0.0
            
            # Normaliza movimento (valores típicos: 0-100 pixels)
            avg_movement = total_movement / compared_keypoints
            normalized_movement = min(1.0, avg_movement / 50.0)
            
            return normalized_movement
            
        except Exception as e:
            self.logger.error(f"Erro ao calcular intensidade de movimento: {e}")
            return 0.0
    
    def _calculate_pose_stability(self, current_pose: BodyPoseData, 
                                 history: List[BodyPoseData]) -> float:
        """Calcula estabilidade da pose ao longo do tempo."""
        try:
            if len(history) < self.stability_window:
                return 0.5
            
            # Analisa últimas N poses
            recent_poses = history[-self.stability_window:]
            recent_poses.append(current_pose)
            
            # Calcula variabilidade da postura
            posture_changes = 0
            for i in range(1, len(recent_poses)):
                if recent_poses[i].posture_type != recent_poses[i-1].posture_type:
                    posture_changes += 1
            
            posture_stability = 1.0 - (posture_changes / len(recent_poses))
            
            # Calcula variabilidade de movimento
            movements = []
            for pose in recent_poses:
                movements.append(pose.movement_intensity)
            
            if movements:
                movement_variance = np.var(movements)
                movement_stability = 1.0 - min(1.0, movement_variance)
            else:
                movement_stability = 0.5
            
            # Combina estabilidades
            overall_stability = (posture_stability * 0.6) + (movement_stability * 0.4)
            
            return min(1.0, max(0.0, overall_stability))
            
        except Exception as e:
            self.logger.error(f"Erro ao calcular estabilidade: {e}")
            return 0.5
    
    def _update_pose_history(self, pose_data: BodyPoseData):
        """Atualiza histórico de poses."""
        if pose_data.tracking_id is None:
            return
        
        tracking_id = pose_data.tracking_id
        
        if tracking_id not in self.pose_history:
            self.pose_history[tracking_id] = deque(maxlen=self.history_length)
        
        self.pose_history[tracking_id].append(pose_data)
        
        # Atualiza histórico na própria estrutura
        pose_data.pose_history = list(self.pose_history[tracking_id])
    
    def _cleanup_cache(self, current_time: float):
        """Remove entradas antigas do cache."""
        expired_keys = []
        for cache_key, (_, cache_time) in self.pose_cache.items():
            if current_time - cache_time > self.cache_ttl * 20:
                expired_keys.append(cache_key)
        
        for key in expired_keys:
            del self.pose_cache[key]
    
    def _update_metrics(self, pose_data: Optional[BodyPoseData], processing_time: float):
        """Atualiza métricas de performance."""
        self.performance_metrics['total_detections'] += 1
        
        if pose_data is not None:
            self.performance_metrics['successful_detections'] += 1
            
            # Conta gesto detectado
            if pose_data.primary_gesture:
                gesture_name = pose_data.primary_gesture.value
                if gesture_name in self.performance_metrics['gestures_detected']:
                    self.performance_metrics['gestures_detected'][gesture_name] += 1
            
            # Conta postura detectada
            posture_name = pose_data.posture_type.value
            if posture_name in self.performance_metrics['postures_detected']:
                self.performance_metrics['postures_detected'][posture_name] += 1
        
        # Atualiza tempo de processamento
        self.performance_metrics['processing_times'].append(processing_time)
        
        if self.performance_metrics['processing_times']:
            self.performance_metrics['average_processing_time'] = np.mean(
                self.performance_metrics['processing_times']
            )
    
    def draw_pose(self, frame: np.ndarray, pose_data: BodyPoseData, 
                  draw_keypoints: bool = True, draw_connections: bool = True,
                  draw_info: bool = True) -> np.ndarray:
        """
        Desenha pose detectada no frame.
        
        Args:
            frame: Frame original
            pose_data: Dados de pose detectados
            draw_keypoints: Se deve desenhar keypoints
            draw_connections: Se deve desenhar conexões entre keypoints
            draw_info: Se deve desenhar informações da pose
            
        Returns:
            np.ndarray: Frame com pose desenhada
        """
        result_frame = frame.copy()
        
        try:
            if draw_keypoints:
                # Desenha keypoints
                for kp_name, keypoint in pose_data.keypoints.items():
                    if keypoint.visible:
                        # Cor baseada na confiança
                        if keypoint.confidence > 0.8:
                            color = (0, 255, 0)  # Verde
                        elif keypoint.confidence > 0.5:
                            color = (0, 255, 255)  # Amarelo
                        else:
                            color = (0, 0, 255)  # Vermelho
                        
                        cv2.circle(result_frame, 
                                 (int(keypoint.x), int(keypoint.y)), 
                                 5, color, -1)
            
            if draw_connections:
                # Desenha conexões básicas (esqueleto)
                connections = [
                    ('LEFT_SHOULDER', 'RIGHT_SHOULDER'),
                    ('LEFT_SHOULDER', 'LEFT_ELBOW'),
                    ('LEFT_ELBOW', 'LEFT_WRIST'),
                    ('RIGHT_SHOULDER', 'RIGHT_ELBOW'),
                    ('RIGHT_ELBOW', 'RIGHT_WRIST'),
                    ('LEFT_SHOULDER', 'LEFT_HIP'),
                    ('RIGHT_SHOULDER', 'RIGHT_HIP'),
                    ('LEFT_HIP', 'RIGHT_HIP'),
                    ('LEFT_HIP', 'LEFT_KNEE'),
                    ('LEFT_KNEE', 'LEFT_ANKLE'),
                    ('RIGHT_HIP', 'RIGHT_KNEE'),
                    ('RIGHT_KNEE', 'RIGHT_ANKLE')
                ]
                
                for start_kp, end_kp in connections:
                    if (start_kp in pose_data.keypoints and end_kp in pose_data.keypoints and
                        pose_data.keypoints[start_kp].visible and pose_data.keypoints[end_kp].visible):
                        
                        start_point = pose_data.keypoints[start_kp]
                        end_point = pose_data.keypoints[end_kp]
                        
                        cv2.line(result_frame,
                               (int(start_point.x), int(start_point.y)),
                               (int(end_point.x), int(end_point.y)),
                               (255, 255, 255), 2)
            
            if draw_info:
                # Desenha informações da pose
                info_y = 30
                
                # ID de tracking
                if pose_data.tracking_id:
                    cv2.putText(result_frame, f"ID: {pose_data.tracking_id}",
                              (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    info_y += 25
                
                # Gesto principal
                if pose_data.primary_gesture:
                    gesture_text = f"Gesto: {pose_data.primary_gesture.value} ({pose_data.gesture_confidence:.2f})"
                    cv2.putText(result_frame, gesture_text,
                              (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    info_y += 20
                
                # Postura
                posture_text = f"Postura: {pose_data.posture_type.value} ({pose_data.posture_confidence:.2f})"
                cv2.putText(result_frame, posture_text,
                          (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                info_y += 20
                
                # Níveis
                cv2.putText(result_frame, f"Engajamento: {pose_data.engagement_level:.2f}",
                          (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                info_y += 15
                
                cv2.putText(result_frame, f"Energia: {pose_data.energy_level:.2f}",
                          (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                info_y += 15
                
                cv2.putText(result_frame, f"Abertura: {pose_data.openness_score:.2f}",
                          (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
        except Exception as e:
            self.logger.error(f"Erro ao desenhar pose: {e}")
        
        return result_frame
    
    def get_performance_metrics(self) -> Dict:
        """Retorna métricas de performance."""
        return self.performance_metrics.copy()
    
    def clear_history(self, tracking_id: Optional[int] = None):
        """Limpa histórico de poses."""
        if tracking_id is not None:
            if tracking_id in self.pose_history:
                del self.pose_history[tracking_id]
            if tracking_id in self.person_tracker:
                del self.person_tracker[tracking_id]
        else:
            self.pose_history.clear()
            self.person_tracker.clear()
        
        self.logger.info(f"Histórico de poses limpo {'para tracking_id ' + str(tracking_id) if tracking_id else 'completamente'}")
    
    def __del__(self):
        """Cleanup automático."""
        if hasattr(self, 'pose_detector'):
            self.pose_detector.close()