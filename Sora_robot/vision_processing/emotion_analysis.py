# sora_robot/vision_processing/emotion_analysis.py

import cv2
import numpy as np
import tensorflow as tf
from typing import Dict, List, Optional, Tuple, NamedTuple
import threading
import time
from dataclasses import dataclass, field
from collections import deque
import pickle
import os

from utils.logger import get_logger
from utils.constants import EMOTIONS, PERFORMANCE
from vision_processing.facial_recognition import FaceData
import config

@dataclass
class EmotionData:
    """Estrutura para dados de emoção analisados."""
    
    # Emoção detectada
    primary_emotion: str
    emotion_confidence: float
    
    # Distribuição de todas as emoções
    emotion_scores: Dict[str, float] = field(default_factory=dict)
    
    # Métricas adicionais
    emotional_intensity: float = 0.0  # 0.0 a 1.0
    emotional_stability: float = 0.0  # Quão estável é a emoção ao longo do tempo
    
    # Características faciais que contribuem para a emoção
    facial_features: Dict[str, float] = field(default_factory=dict)
    
    # Contexto temporal
    timestamp: float = 0.0
    tracking_id: Optional[int] = None
    
    # Histórico de emoções para esta face
    emotion_history: List[Tuple[str, float, float]] = field(default_factory=list)  # (emotion, confidence, timestamp)

class EmotionAnalysis:
    """
    Classe responsável pela análise de emoções em faces detectadas.
    Utiliza modelos de deep learning e análise de características faciais.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Inicializa o analisador de emoções.
        
        Args:
            model_path: Caminho para modelo personalizado (opcional)
        """
        self.logger = get_logger(__name__)
        
        # Configurações
        self.model_path = model_path
        self.emotion_model = None
        self.model_input_size = (48, 48)  # Tamanho padrão para modelos de emoção
        
        # Cache e histórico
        self.emotion_cache = {}
        self.emotion_history = {}  # tracking_id -> deque of EmotionData
        self.cache_ttl = 0.1  # 100ms
        
        # Threading
        self.analysis_lock = threading.Lock()
        
        # Configurações de análise
        self.history_length = 10  # Número de análises anteriores para manter
        self.smoothing_factor = 0.3  # Para suavização temporal das emoções
        self.confidence_threshold = 0.4  # Confiança mínima para aceitar uma emoção
        
        # Métricas de performance
        self.performance_metrics = {
            'total_analyses': 0,
            'successful_analyses': 0,
            'processing_times': deque(maxlen=100),
            'average_processing_time': 0.0,
            'emotions_detected': {emotion: 0 for emotion in EMOTIONS.ALL_EMOTIONS}
        }
        
        # Mapeamentos para análise heurística
        self._init_heuristic_mappings()
        
        # Inicializa modelo
        self._initialize_model()
        
        self.logger.info("EmotionAnalysis inicializado com sucesso")
    
    def _initialize_model(self):
        """Inicializa o modelo de análise de emoções."""
        try:
            if self.model_path and os.path.exists(self.model_path):
                # Carrega modelo personalizado
                self.logger.info(f"Carregando modelo personalizado: {self.model_path}")
                self.emotion_model = tf.keras.models.load_model(self.model_path)
                
                # Obtém tamanho de entrada do modelo
                input_shape = self.emotion_model.input_shape
                if len(input_shape) >= 3:
                    self.model_input_size = (input_shape[1], input_shape[2])
                
                self.logger.info(f"Modelo carregado - Input size: {self.model_input_size}")
                
            else:
                # Usa análise heurística baseada em características faciais
                self.logger.info("Usando análise heurística de emoções (sem modelo de ML)")
                self.emotion_model = None
                
        except Exception as e:
            self.logger.error(f"Erro ao carregar modelo de emoções: {e}")
            self.logger.info("Fallback para análise heurística")
            self.emotion_model = None
    
    def _init_heuristic_mappings(self):
        """Inicializa mapeamentos para análise heurística baseada em características faciais."""
        
        # Características faciais que indicam emoções específicas
        self.emotion_features = {
            EMOTIONS.HAPPY: {
                'mouth_curvature': (0.6, 1.0),  # Boca curvada para cima
                'eye_crinkles': (0.4, 1.0),     # Rugas nos olhos (sorriso genuíno)
                'cheek_elevation': (0.5, 1.0),  # Bochechas elevadas
                'mouth_width': (0.7, 1.0)       # Boca mais larga
            },
            EMOTIONS.SAD: {
                'mouth_curvature': (0.0, 0.3),  # Boca curvada para baixo
                'eyebrow_angle': (0.0, 0.4),    # Sobrancelhas caídas
                'eye_openness': (0.0, 0.6),     # Olhos menos abertos
                'mouth_width': (0.0, 0.4)       # Boca mais estreita
            },
            EMOTIONS.ANGRY: {
                'eyebrow_furrow': (0.6, 1.0),   # Sobrancelhas franzidas
                'eye_narrowing': (0.5, 1.0),    # Olhos estreitados
                'nostril_flare': (0.4, 1.0),    # Narinas dilatadas
                'lip_tightness': (0.6, 1.0)     # Lábios contraídos
            },
            EMOTIONS.SURPRISED: {
                'eyebrow_raise': (0.7, 1.0),    # Sobrancelhas levantadas
                'eye_widening': (0.6, 1.0),     # Olhos arregalados
                'mouth_opening': (0.5, 1.0),    # Boca aberta
                'forehead_wrinkles': (0.4, 1.0) # Rugas na testa
            },
            EMOTIONS.FEAR: {
                'eyebrow_raise': (0.5, 0.8),    # Sobrancelhas levantadas (menos que surpresa)
                'eye_widening': (0.6, 1.0),     # Olhos arregalados
                'mouth_tension': (0.5, 1.0),    # Tensão na boca
                'facial_pallor': (0.3, 1.0)     # Palidez facial
            },
            EMOTIONS.DISGUST: {
                'nose_wrinkle': (0.6, 1.0),     # Nariz franzido
                'upper_lip_raise': (0.5, 1.0),  # Lábio superior levantado
                'eye_narrowing': (0.4, 0.8),    # Leve estreitamento dos olhos
                'mouth_curvature': (0.0, 0.4)   # Boca ligeiramente para baixo
            }
        }
    
    def analyze_emotions(self, face_data: FaceData, frame: np.ndarray) -> Optional[EmotionData]:
        """
        Analisa emoções de uma face detectada.
        
        Args:
            face_data: Dados da face detectada
            frame: Frame original onde a face foi detectada
            
        Returns:
            Optional[EmotionData]: Dados de emoção analisados ou None se falhou
        """
        start_time = time.time()
        
        try:
            with self.analysis_lock:
                # Verifica cache
                cache_key = self._generate_cache_key(face_data, frame)
                current_time = time.time()
                
                if cache_key in self.emotion_cache:
                    cached_data, cache_time = self.emotion_cache[cache_key]
                    if current_time - cache_time < self.cache_ttl:
                        return cached_data
                
                # Extrai região da face
                face_roi = self._extract_face_roi(face_data, frame)
                if face_roi is None:
                    return None
                
                # Análise de emoção
                emotion_data = None
                
                if self.emotion_model is not None:
                    # Usa modelo de ML
                    emotion_data = self._analyze_with_model(face_roi, face_data)
                else:
                    # Usa análise heurística
                    emotion_data = self._analyze_heuristic(face_data, face_roi)
                
                if emotion_data is not None:
                    emotion_data.timestamp = current_time
                    emotion_data.tracking_id = face_data.tracking_id
                    
                    # Aplica suavização temporal se há histórico
                    emotion_data = self._apply_temporal_smoothing(emotion_data)
                    
                    # Atualiza histórico
                    self._update_emotion_history(emotion_data)
                    
                    # Atualiza cache
                    self.emotion_cache[cache_key] = (emotion_data, current_time)
                
                # Limpa cache antigo
                self._cleanup_cache(current_time)
                
                # Atualiza métricas
                processing_time = time.time() - start_time
                self._update_metrics(emotion_data, processing_time)
                
                return emotion_data
                
        except Exception as e:
            self.logger.error(f"Erro na análise de emoções: {e}")
            return None
    
    def _extract_face_roi(self, face_data: FaceData, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extrai região de interesse da face do frame."""
        try:
            x, y, w, h = face_data.bbox
            
            # Adiciona padding para capturar contexto facial
            padding = 0.1
            pad_x = int(w * padding)
            pad_y = int(h * padding)
            
            # Coordenadas expandidas com verificação de limites
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(frame.shape[1], x + w + pad_x)
            y2 = min(frame.shape[0], y + h + pad_y)
            
            face_roi = frame[y1:y2, x1:x2]
            
            if face_roi.size == 0:
                return None
            
            return face_roi
            
        except Exception as e:
            self.logger.error(f"Erro ao extrair ROI da face: {e}")
            return None
    
    def _analyze_with_model(self, face_roi: np.ndarray, face_data: FaceData) -> Optional[EmotionData]:
        """Análise usando modelo de machine learning."""
        try:
            # Preprocessa imagem para o modelo
            processed_face = self._preprocess_for_model(face_roi)
            if processed_face is None:
                return None
            
            # Predição
            predictions = self.emotion_model.predict(processed_face[np.newaxis, ...], verbose=0)
            emotion_scores = predictions[0]
            
            # Mapeia scores para emoções
            emotion_dict = {}
            for i, emotion in enumerate(EMOTIONS.ALL_EMOTIONS):
                if i < len(emotion_scores):
                    emotion_dict[emotion] = float(emotion_scores[i])
            
            # Encontra emoção primária
            primary_emotion = max(emotion_dict, key=emotion_dict.get)
            primary_confidence = emotion_dict[primary_emotion]
            
            # Verifica confiança mínima
            if primary_confidence < self.confidence_threshold:
                primary_emotion = EMOTIONS.NEUTRAL
                primary_confidence = emotion_dict.get(EMOTIONS.NEUTRAL, 0.5)
            
            # Calcula intensidade emocional
            emotional_intensity = self._calculate_emotional_intensity(emotion_dict)
            
            return EmotionData(
                primary_emotion=primary_emotion,
                emotion_confidence=primary_confidence,
                emotion_scores=emotion_dict,
                emotional_intensity=emotional_intensity,
                facial_features={},  # Modelo não fornece características específicas
                timestamp=time.time()
            )
            
        except Exception as e:
            self.logger.error(f"Erro na análise com modelo: {e}")
            return None
    
    def _analyze_heuristic(self, face_data: FaceData, face_roi: np.ndarray) -> Optional[EmotionData]:
        """Análise heurística baseada em características faciais."""
        try:
            # Extrai características faciais
            facial_features = self._extract_facial_features(face_data, face_roi)
            
            # Calcula scores para cada emoção
            emotion_scores = {}
            
            for emotion in EMOTIONS.ALL_EMOTIONS:
                if emotion == EMOTIONS.NEUTRAL:
                    # Score neutro é inversamente proporcional a outras emoções
                    emotion_scores[emotion] = 0.5
                else:
                    score = self._calculate_emotion_score_heuristic(emotion, facial_features)
                    emotion_scores[emotion] = score
            
            # Normaliza scores para somar 1
            total_score = sum(emotion_scores.values())
            if total_score > 0:
                emotion_scores = {k: v/total_score for k, v in emotion_scores.items()}
            else:
                emotion_scores = {emotion: 1.0/len(EMOTIONS.ALL_EMOTIONS) for emotion in EMOTIONS.ALL_EMOTIONS}
            
            # Ajusta score neutro
            max_non_neutral = max(score for emotion, score in emotion_scores.items() if emotion != EMOTIONS.NEUTRAL)
            emotion_scores[EMOTIONS.NEUTRAL] = max(0.1, 1.0 - max_non_neutral * 1.5)
            
            # Re-normaliza
            total_score = sum(emotion_scores.values())
            emotion_scores = {k: v/total_score for k, v in emotion_scores.items()}
            
            # Encontra emoção primária
            primary_emotion = max(emotion_scores, key=emotion_scores.get)
            primary_confidence = emotion_scores[primary_emotion]
            
            # Calcula intensidade emocional
            emotional_intensity = self._calculate_emotional_intensity(emotion_scores)
            
            return EmotionData(
                primary_emotion=primary_emotion,
                emotion_confidence=primary_confidence,
                emotion_scores=emotion_scores,
                emotional_intensity=emotional_intensity,
                facial_features=facial_features,
                timestamp=time.time()
            )
            
        except Exception as e:
            self.logger.error(f"Erro na análise heurística: {e}")
            return None
    
    def _preprocess_for_model(self, face_roi: np.ndarray) -> Optional[np.ndarray]:
        """Preprocessa imagem da face para entrada no modelo."""
        try:
            # Converte para escala de cinza se necessário
            if len(face_roi.shape) == 3:
                gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            else:
                gray_face = face_roi
            
            # Redimensiona para tamanho do modelo
            resized_face = cv2.resize(gray_face, self.model_input_size)
            
            # Normaliza pixel values
            normalized_face = resized_face.astype(np.float32) / 255.0
            
            # Adiciona dimensão de canal se necessário
            if len(normalized_face.shape) == 2:
                normalized_face = np.expand_dims(normalized_face, axis=-1)
            
            return normalized_face
            
        except Exception as e:
            self.logger.error(f"Erro no preprocessamento: {e}")
            return None
    
    def _extract_facial_features(self, face_data: FaceData, face_roi: np.ndarray) -> Dict[str, float]:
        """Extrai características faciais para análise heurística."""
        features = {}
        
        try:
            # Análise básica da imagem
            gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) if len(face_roi.shape) == 3 else face_roi
            
            # Características baseadas em intensidade de pixels
            features['brightness'] = np.mean(gray_roi) / 255.0
            features['contrast'] = np.std(gray_roi) / 255.0
            
            # Análise de gradientes (indicador de rugas/tensão)
            grad_x = cv2.Sobel(gray_roi, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray_roi, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            features['texture_complexity'] = np.mean(gradient_magnitude) / 255.0
            
            # Análise de simetria horizontal
            h, w = gray_roi.shape
            left_half = gray_roi[:, :w//2]
            right_half = cv2.flip(gray_roi[:, w//2:], 1)
            
            # Redimensiona para mesmo tamanho se necessário
            min_width = min(left_half.shape[1], right_half.shape[1])
            left_half = left_half[:, :min_width]
            right_half = right_half[:, :min_width]
            
            if left_half.shape == right_half.shape:
                symmetry = 1.0 - np.mean(np.abs(left_half.astype(float) - right_half.astype(float))) / 255.0
                features['facial_symmetry'] = max(0.0, symmetry)
            else:
                features['facial_symmetry'] = 0.5
            
            # Características específicas baseadas em regiões
            if face_data.landmarks is not None:
                features.update(self._analyze_landmark_features(face_data.landmarks, face_roi))
            else:
                # Estimativas baseadas em regiões da face
                features.update(self._analyze_region_features(gray_roi))
            
        except Exception as e:
            self.logger.error(f"Erro ao extrair características faciais: {e}")
            # Retorna características padrão em caso de erro
            features = {
                'brightness': 0.5,
                'contrast': 0.5,
                'texture_complexity': 0.5,
                'facial_symmetry': 0.5
            }
        
        return features
    
    def _analyze_landmark_features(self, landmarks: np.ndarray, face_roi: np.ndarray) -> Dict[str, float]:
        """Analisa características específicas usando landmarks faciais."""
        features = {}
        
        try:
            # Esta é uma implementação simplificada
            # Em uma implementação completa, seria necessário mapear landmarks específicos
            
            # Por enquanto, retorna estimativas baseadas na distribuição dos landmarks
            if len(landmarks) > 10:
                # Análise da distribuição vertical dos landmarks (indica abertura da boca, olhos)
                y_coords = landmarks[:, 1]
                y_range = np.max(y_coords) - np.min(y_coords)
                features['facial_openness'] = min(1.0, y_range / face_roi.shape[0])
                
                # Análise da distribuição horizontal (indica largura da boca/olhos)
                x_coords = landmarks[:, 0]
                x_range = np.max(x_coords) - np.min(x_coords)
                features['facial_width'] = min(1.0, x_range / face_roi.shape[1])
            
        except Exception as e:
            self.logger.error(f"Erro na análise de landmarks: {e}")
        
        return features
    
    def _analyze_region_features(self, gray_roi: np.ndarray) -> Dict[str, float]:
        """Analisa características baseadas em regiões da face."""
        features = {}
        
        try:
            h, w = gray_roi.shape
            
            # Divide face em regiões: testa, olhos, nariz, boca
            forehead = gray_roi[:h//3, :]
            eye_region = gray_roi[h//3:2*h//3, :]
            mouth_region = gray_roi[2*h//3:, :]
            
            # Análise de intensidade por região
            features['forehead_intensity'] = np.mean(forehead) / 255.0
            features['eye_region_intensity'] = np.mean(eye_region) / 255.0
            features['mouth_region_intensity'] = np.mean(mouth_region) / 255.0
            
            # Análise de variação (indicador de rugas/expressões)
            features['forehead_variation'] = np.std(forehead) / 255.0
            features['eye_region_variation'] = np.std(eye_region) / 255.0
            features['mouth_region_variation'] = np.std(mouth_region) / 255.0
            
        except Exception as e:
            self.logger.error(f"Erro na análise de regiões: {e}")
        
        return features
    
    def _calculate_emotion_score_heuristic(self, emotion: str, features: Dict[str, float]) -> float:
        """Calcula score de uma emoção específica baseado em características."""
        if emotion not in self.emotion_features:
            return 0.1
        
        emotion_rules = self.emotion_features[emotion]
        score = 0.0
        matched_features = 0
        
        for feature_name, (min_val, max_val) in emotion_rules.items():
            if feature_name in features:
                feature_value = features[feature_name]
                
                # Verifica se a característica está no range esperado para esta emoção
                if min_val <= feature_value <= max_val:
                    # Score baseado na proximidade ao centro do range
                    center = (min_val + max_val) / 2
                    range_size = max_val - min_val
                    if range_size > 0:
                        proximity = 1.0 - abs(feature_value - center) / (range_size / 2)
                        score += proximity
                    else:
                        score += 1.0
                    matched_features += 1
        
        # Normaliza pelo número de características verificadas
        if matched_features > 0:
            score = score / matched_features
        else:
            score = 0.1  # Score mínimo se nenhuma característica foi encontrada
        
        return min(1.0, max(0.0, score))
    
    def _calculate_emotional_intensity(self, emotion_scores: Dict[str, float]) -> float:
        """Calcula intensidade emocional geral."""
        # Remove o score neutro para calcular intensidade
        non_neutral_scores = [score for emotion, score in emotion_scores.items() 
                             if emotion != EMOTIONS.NEUTRAL]
        
        if not non_neutral_scores:
            return 0.0
        
        # Intensidade baseada no score máximo e na distribuição
        max_score = max(non_neutral_scores)
        score_variance = np.var(non_neutral_scores)
        
        # Maior intensidade = score alto + baixa dispersão (emoção bem definida)
        intensity = max_score * (1.0 - min(0.5, score_variance))
        
        return min(1.0, max(0.0, intensity))
    
    def _apply_temporal_smoothing(self, current_emotion: EmotionData) -> EmotionData:
        """Aplica suavização temporal usando histórico de emoções."""
        if current_emotion.tracking_id is None:
            return current_emotion
        
        tracking_id = current_emotion.tracking_id
        
        # Verifica se há histórico para esta face
        if tracking_id not in self.emotion_history:
            return current_emotion
        
        history = self.emotion_history[tracking_id]
        if len(history) == 0:
            return current_emission
        
        try:
            # Pega a última emoção do histórico
            last_emotion = history[-1]
            
            # Suavização simples: média ponderada
            smoothed_scores = {}
            for emotion in EMOTIONS.ALL_EMOTIONS:
                current_score = current_emotion.emotion_scores.get(emotion, 0.0)
                last_score = last_emotion.emotion_scores.get(emotion, 0.0)
                
                smoothed_score = (
                    self.smoothing_factor * current_score + 
                    (1.0 - self.smoothing_factor) * last_score
                )
                smoothed_scores[emotion] = smoothed_score
            
            # Atualiza emoção primária baseada nos scores suavizados
            primary_emotion = max(smoothed_scores, key=smoothed_scores.get)
            primary_confidence = smoothed_scores[primary_emotion]
            
            # Calcula estabilidade emocional
            emotional_stability = self._calculate_emotional_stability(history, current_emotion)
            
            # Cria nova instância com dados suavizados
            smoothed_emotion = EmotionData(
                primary_emotion=primary_emotion,
                emotion_confidence=primary_confidence,
                emotion_scores=smoothed_scores,
                emotional_intensity=current_emotion.emotional_intensity,
                emotional_stability=emotional_stability,
                facial_features=current_emotion.facial_features,
                timestamp=current_emotion.timestamp,
                tracking_id=current_emotion.tracking_id
            )
            
            return smoothed_emotion
            
        except Exception as e:
            self.logger.error(f"Erro na suavização temporal: {e}")
            return current_emotion
    
    def _calculate_emotional_stability(self, history: deque, current_emotion: EmotionData) -> float:
        """Calcula quão estável tem sido a emoção ao longo do tempo."""
        if len(history) < 2:
            return 0.5
        
        try:
            # Analisa consistência da emoção primária nos últimos frames
            recent_emotions = [data.primary_emotion for data in list(history)[-5:]]
            recent_emotions.append(current_emotion.primary_emotion)
            
            # Calcula frequência da emoção mais comum
            emotion_counts = {}
            for emotion in recent_emotions:
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            most_common_count = max(emotion_counts.values())
            stability = most_common_count / len(recent_emotions)
            
            return stability
            
        except Exception as e:
            self.logger.error(f"Erro ao calcular estabilidade emocional: {e}")
            return 0.5
    
    def _update_emotion_history(self, emotion_data: EmotionData):
        """Atualiza histórico de emoções para uma face."""
        if emotion_data.tracking_id is None:
            return
        
        tracking_id = emotion_data.tracking_id
        
        # Inicializa histórico se necessário
        if tracking_id not in self.emotion_history:
            self.emotion_history[tracking_id] = deque(maxlen=self.history_length)
        
        # Adiciona emoção atual ao histórico
        self.emotion_history[tracking_id].append(emotion_data)
        
        # Atualiza histórico de emoções na própria estrutura
        emotion_data.emotion_history = list(self.emotion_history[tracking_id])
    
    def _generate_cache_key(self, face_data: FaceData, frame: np.ndarray) -> str:
        """Gera chave de cache baseada nos dados da face."""
        # Usa hash da região da face para cache
        x, y, w, h = face_data.bbox
        face_roi = frame[y:y+h, x:x+w]
        
        # Hash simples baseado em características da face
        cache_key = f"{face_data.tracking_id}_{hash(face_roi.tobytes())}"
        return cache_key
    
    def _cleanup_cache(self, current_time: float):
        """Remove entradas antigas do cache."""
        expired_keys = []
        for cache_key, (_, cache_time) in self.emotion_cache.items():
            if current_time - cache_time > self.cache_ttl * 10:
                expired_keys.append(cache_key)
        
        for key in expired_keys:
            del self.emotion_cache[key]
    
    def _update_metrics(self, emotion_data: Optional[EmotionData], processing_time: float):
        """Atualiza métricas de performance."""
        self.performance_metrics['total_analyses'] += 1
        
        if emotion_data is not None:
            self.performance_metrics['successful_analyses'] += 1
            
            # Conta emoção detectada
            if emotion_data.primary_emotion in self.performance_metrics['emotions_detected']:
                self.performance_metrics['emotions_detected'][emotion_data.primary_emotion] += 1
        
        # Atualiza tempo de processamento
        self.performance_metrics['processing_times'].append(processing_time)
        
        if self.performance_metrics['processing_times']:
            self.performance_metrics['average_processing_time'] = np.mean(
                self.performance_metrics['processing_times']
            )
    
    def get_emotion_summary(self, tracking_id: int) -> Optional[Dict]:
        """
        Retorna resumo das emoções para uma face específica.
        
        Args:
            tracking_id: ID de tracking da face
            
        Returns:
            Optional[Dict]: Resumo das emoções ou None se não encontrado
        """
        if tracking_id not in self.emotion_history:
            return None
        
        history = list(self.emotion_history[tracking_id])
        if not history:
            return None
        
        try:
            # Calcula estatísticas do histórico
            emotions_count = {}
            total_intensity = 0.0
            total_stability = 0.0
            
            for emotion_data in history:
                primary = emotion_data.primary_emotion
                emotions_count[primary] = emotions_count.get(primary, 0) + 1
                total_intensity += emotion_data.emotional_intensity
                total_stability += emotion_data.emotional_stability
            
            # Emoção predominante
            dominant_emotion = max(emotions_count, key=emotions_count.get)
            
            # Médias
            avg_intensity = total_intensity / len(history)
            avg_stability = total_stability / len(history)
            
            # Última emoção
            last_emotion = history[-1]
            
            return {
                'tracking_id': tracking_id,
                'dominant_emotion': dominant_emotion,
                'current_emotion': last_emotion.primary_emotion,
                'current_confidence': last_emotion.emotion_confidence,
                'average_intensity': avg_intensity,
                'average_stability': avg_stability,
                'emotions_distribution': emotions_count,
                'total_analyses': len(history),
                'time_span': history[-1].timestamp - history[0].timestamp
            }
            
        except Exception as e:
            self.logger.error(f"Erro ao gerar resumo de emoções: {e}")
            return None
    
    def get_performance_metrics(self) -> Dict:
        """Retorna métricas de performance da análise de emoções."""
        return self.performance_metrics.copy()
    
    def clear_history(self, tracking_id: Optional[int] = None):
        """
        Limpa histórico de emoções.
        
        Args:
            tracking_id: ID específico para limpar (None = limpa tudo)
        """
        if tracking_id is not None:
            if tracking_id in self.emotion_history:
                del self.emotion_history[tracking_id]
        else:
            self.emotion_history.clear()
        
        self.logger.info(f"Histórico de emoções limpo {'para tracking_id ' + str(tracking_id) if tracking_id else 'completamente'}")
    
    def save_emotion_model(self, model_path: str):
        """
        Salva dados de emoções coletados para treinamento futuro.
        
        Args:
            model_path: Caminho onde salvar os dados
        """
        try:
            # Coleta dados de todas as análises realizadas
            training_data = {
                'emotion_history': dict(self.emotion_history),
                'performance_metrics': self.performance_metrics,
                'emotion_features': self.emotion_features,
                'timestamp': time.time()
            }
            
            with open(model_path, 'wb') as f:
                pickle.dump(training_data, f)
            
            self.logger.info(f"Dados de emoção salvos em: {model_path}")
            
        except Exception as e:
            self.logger.error(f"Erro ao salvar dados de emoção: {e}")
    
    def __del__(self):
        """Cleanup automático."""
        if hasattr(self, 'emotion_model') and self.emotion_model is not None:
            try:
                del self.emotion_model
            except:
                pass