# sora_robot/vision_processing/facial_recognition.py

import cv2
import numpy as np
import mediapipe as mp
from typing import List, Dict, Optional, Tuple, NamedTuple
import threading
import time
from dataclasses import dataclass
from utils.logger import get_logger
from utils.constants import FACE_DETECTION, EMOTIONS
import config

@dataclass
class FaceData:
    """Estrutura de dados para informações de uma face detectada."""
    
    # Informações básicas da detecção
    face_id: int
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    confidence: float
    
    # Landmarks faciais
    landmarks: Optional[np.ndarray] = None
    
    # Características extraídas
    age_estimate: Optional[int] = None
    gender_estimate: Optional[str] = None
    
    # Dados para reconhecimento
    face_encoding: Optional[np.ndarray] = None
    
    # Métricas de qualidade
    quality_score: float = 0.0
    blur_score: float = 0.0
    brightness_score: float = 0.0
    
    # Tracking
    tracking_id: Optional[int] = None
    first_seen: float = 0.0
    last_seen: float = 0.0
    
    # Status da face
    is_frontal: bool = True
    eyes_open: bool = True
    mouth_open: bool = False

class FaceRecognition:
    """
    Classe responsável pela detecção e reconhecimento facial.
    Utiliza MediaPipe para detecção eficiente e cv2 para processamento adicional.
    """
    
    def __init__(self):
        """Inicializa o sistema de reconhecimento facial."""
        self.logger = get_logger(__name__)
        
        # Inicialização do MediaPipe
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Modelos MediaPipe
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=FACE_DETECTION.MIN_DETECTION_CONFIDENCE,
            model_selection=1  # Modelo melhor para faces próximas
        )
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=FACE_DETECTION.MAX_FACES,
            refine_landmarks=True,
            min_detection_confidence=FACE_DETECTION.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=FACE_DETECTION.MIN_TRACKING_CONFIDENCE
        )
        
        # Cascata Haar como backup
        try:
            self.haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        except Exception as e:
            self.logger.warning(f"Não foi possível carregar Haar Cascade: {e}")
            self.haar_cascade = None
        
        # Sistema de tracking
        self.face_tracker = {}  # tracking_id -> FaceData
        self.next_tracking_id = 1
        self.tracking_threshold = 50  # pixels para considerar mesma face
        
        # Cache para performance
        self.detection_cache = {}
        self.cache_ttl = 0.1  # 100ms
        
        # Threading
        self.processing_lock = threading.Lock()
        
        # Métricas de performance
        self.performance_metrics = {
            'total_detections': 0,
            'successful_detections': 0,
            'processing_times': [],
            'average_processing_time': 0.0,
            'faces_tracked': 0
        }
        
        self.logger.info("FaceRecognition inicializado com sucesso")
    
    def detect_faces(self, frame: np.ndarray, return_landmarks: bool = True) -> List[FaceData]:
        """
        Detecta faces em um frame.
        
        Args:
            frame: Frame de entrada (BGR)
            return_landmarks: Se deve retornar landmarks faciais
            
        Returns:
            List[FaceData]: Lista de faces detectadas
        """
        start_time = time.time()
        
        try:
            with self.processing_lock:
                # Converte para RGB para MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Cache check
                frame_hash = hash(rgb_frame.tobytes())
                current_time = time.time()
                
                if frame_hash in self.detection_cache:
                    cache_data, cache_time = self.detection_cache[frame_hash]
                    if current_time - cache_time < self.cache_ttl:
                        return cache_data
                
                # Detecção principal com MediaPipe
                faces = self._detect_with_mediapipe(rgb_frame, frame, return_landmarks)
                
                # Fallback para Haar Cascade se necessário
                if not faces and self.haar_cascade is not None:
                    faces = self._detect_with_haar(frame)
                
                # Sistema de tracking
                faces = self._update_tracking(faces, current_time)
                
                # Atualizar cache
                self.detection_cache[frame_hash] = (faces, current_time)
                
                # Limpar cache antigo
                self._cleanup_cache(current_time)
                
                # Atualizar métricas
                processing_time = time.time() - start_time
                self._update_metrics(len(faces), processing_time)
                
                return faces
                
        except Exception as e:
            self.logger.error(f"Erro na detecção de faces: {e}")
            return []
    
    def _detect_with_mediapipe(self, rgb_frame: np.ndarray, bgr_frame: np.ndarray, 
                              return_landmarks: bool) -> List[FaceData]:
        """Detecção usando MediaPipe."""
        faces = []
        
        try:
            # Detecção básica
            detection_results = self.face_detection.process(rgb_frame)
            
            if detection_results.detections:
                h, w = rgb_frame.shape[:2]
                
                for i, detection in enumerate(detection_results.detections):
                    # Extrai bounding box
                    bbox = detection.location_data.relative_bounding_box
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)
                    
                    # Verifica se a face tem tamanho mínimo
                    if width < FACE_DETECTION.MIN_FACE_SIZE[0] or height < FACE_DETECTION.MIN_FACE_SIZE[1]:
                        continue
                    
                    face_data = FaceData(
                        face_id=i,
                        bbox=(x, y, width, height),
                        confidence=detection.score[0] if detection.score else 0.0,
                        first_seen=time.time(),
                        last_seen=time.time()
                    )
                    
                    # Análise de qualidade da face
                    face_roi = bgr_frame[y:y+height, x:x+width]
                    if face_roi.size > 0:
                        face_data.quality_score = self._calculate_face_quality(face_roi)
                        face_data.blur_score = self._calculate_blur_score(face_roi)
                        face_data.brightness_score = self._calculate_brightness_score(face_roi)
                    
                    # Landmarks se solicitado
                    if return_landmarks:
                        landmarks = self._extract_landmarks(rgb_frame, (x, y, width, height))
                        if landmarks is not None:
                            face_data.landmarks = landmarks
                            face_data.is_frontal = self._is_frontal_face(landmarks)
                            face_data.eyes_open = self._are_eyes_open(landmarks)
                            face_data.mouth_open = self._is_mouth_open(landmarks)
                    
                    faces.append(face_data)
                    
        except Exception as e:
            self.logger.error(f"Erro na detecção MediaPipe: {e}")
        
        return faces
    
    def _detect_with_haar(self, frame: np.ndarray) -> List[FaceData]:
        """Detecção de fallback usando Haar Cascade."""
        faces = []
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detections = self.haar_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=FACE_DETECTION.MIN_FACE_SIZE
            )
            
            for i, (x, y, w, h) in enumerate(detections):
                face_data = FaceData(
                    face_id=i,
                    bbox=(x, y, w, h),
                    confidence=0.5,  # Haar não fornece score de confiança
                    first_seen=time.time(),
                    last_seen=time.time()
                )
                
                # Análise básica de qualidade
                face_roi = frame[y:y+h, x:x+w]
                if face_roi.size > 0:
                    face_data.quality_score = self._calculate_face_quality(face_roi)
                
                faces.append(face_data)
                
        except Exception as e:
            self.logger.error(f"Erro na detecção Haar: {e}")
        
        return faces
    
    def _extract_landmarks(self, rgb_frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Extrai landmarks faciais de uma região específica."""
        try:
            mesh_results = self.face_mesh.process(rgb_frame)
            
            if mesh_results.multi_face_landmarks:
                h, w = rgb_frame.shape[:2]
                
                # Pega o primeiro conjunto de landmarks (face mais próxima da bbox)
                face_landmarks = mesh_results.multi_face_landmarks[0]
                
                # Converte landmarks normalizados para coordenadas de pixel
                landmarks = []
                for landmark in face_landmarks.landmark:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    z = landmark.z if hasattr(landmark, 'z') else 0
                    landmarks.append([x, y, z])
                
                return np.array(landmarks)
                
        except Exception as e:
            self.logger.error(f"Erro ao extrair landmarks: {e}")
        
        return None
    
    def _calculate_face_quality(self, face_roi: np.ndarray) -> float:
        """
        Calcula score de qualidade da face baseado em vários fatores.
        
        Returns:
            float: Score de 0.0 a 1.0 (maior = melhor qualidade)
        """
        if face_roi.size == 0:
            return 0.0
        
        try:
            # Fatores de qualidade
            blur_score = self._calculate_blur_score(face_roi)
            brightness_score = self._calculate_brightness_score(face_roi)
            contrast_score = self._calculate_contrast_score(face_roi)
            size_score = self._calculate_size_score(face_roi)
            
            # Média ponderada
            quality_score = (
                blur_score * 0.3 +
                brightness_score * 0.2 +
                contrast_score * 0.2 +
                size_score * 0.3
            )
            
            return min(1.0, max(0.0, quality_score))
            
        except Exception as e:
            self.logger.error(f"Erro ao calcular qualidade da face: {e}")
            return 0.0
    
    def _calculate_blur_score(self, face_roi: np.ndarray) -> float:
        """Calcula score de nitidez (menor blur = maior score)."""
        try:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Normaliza (valores típicos: 0-500+)
            return min(1.0, laplacian_var / 100.0)
            
        except Exception:
            return 0.0
    
    def _calculate_brightness_score(self, face_roi: np.ndarray) -> float:
        """Calcula score de brilho (nem muito escuro nem muito claro)."""
        try:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray)
            
            # Ideal entre 80-180 (0-255 range)
            if 80 <= mean_brightness <= 180:
                return 1.0
            elif mean_brightness < 50 or mean_brightness > 220:
                return 0.2
            else:
                return 0.6
                
        except Exception:
            return 0.0
    
    def _calculate_contrast_score(self, face_roi: np.ndarray) -> float:
        """Calcula score de contraste."""
        try:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            contrast = np.std(gray)
            
            # Normaliza (valores típicos: 0-60+)
            return min(1.0, contrast / 40.0)
            
        except Exception:
            return 0.0
    
    def _calculate_size_score(self, face_roi: np.ndarray) -> float:
        """Calcula score baseado no tamanho da face."""
        try:
            h, w = face_roi.shape[:2]
            size = min(w, h)
            
            # Faces maiores = melhor score
            if size >= 200:
                return 1.0
            elif size >= 100:
                return 0.8
            elif size >= 64:
                return 0.6
            else:
                return 0.3
                
        except Exception:
            return 0.0
    
    def _is_frontal_face(self, landmarks: np.ndarray) -> bool:
        """Determina se a face está frontal baseado nos landmarks."""
        try:
            if landmarks is None or len(landmarks) < 10:
                return False
            
            # Usa landmarks do nariz e olhos para determinar orientação
            nose_tip = landmarks[1]  # Aproximação
            left_eye = landmarks[33] if len(landmarks) > 33 else landmarks[2]
            right_eye = landmarks[362] if len(landmarks) > 362 else landmarks[5]
            
            # Calcula simetria horizontal
            eye_center_x = (left_eye[0] + right_eye[0]) / 2
            nose_x = nose_tip[0]
            
            # Se o nariz está aproximadamente no centro entre os olhos
            asymmetry = abs(nose_x - eye_center_x)
            eye_distance = abs(left_eye[0] - right_eye[0])
            
            if eye_distance > 0:
                asymmetry_ratio = asymmetry / eye_distance
                return asymmetry_ratio < 0.15  # 15% de tolerância
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao calcular orientação facial: {e}")
            return True
    
    def _are_eyes_open(self, landmarks: np.ndarray) -> bool:
        """Determina se os olhos estão abertos."""
        try:
            if landmarks is None or len(landmarks) < 468:
                return True  # Assume olhos abertos se não conseguir detectar
            
            # Landmarks simplificados para detecção de olhos
            # Esta é uma aproximação, seria necessário landmarks específicos
            return True  # Por enquanto sempre retorna True
            
        except Exception:
            return True
    
    def _is_mouth_open(self, landmarks: np.ndarray) -> bool:
        """Determina se a boca está aberta."""
        try:
            if landmarks is None:
                return False
            
            # Implementação simplificada
            # Seria necessário landmarks específicos da boca
            return False  # Por enquanto sempre retorna False
            
        except Exception:
            return False
    
    def _update_tracking(self, current_faces: List[FaceData], current_time: float) -> List[FaceData]:
        """Atualiza sistema de tracking das faces."""
        try:
            # Remove tracks antigos (mais de 2 segundos sem atualização)
            expired_ids = []
            for track_id, track_data in self.face_tracker.items():
                if current_time - track_data.last_seen > 2.0:
                    expired_ids.append(track_id)
            
            for track_id in expired_ids:
                del self.face_tracker[track_id]
            
            # Associa faces atuais com tracks existentes
            for face in current_faces:
                best_match_id = None
                best_distance = float('inf')
                
                face_center = (
                    face.bbox[0] + face.bbox[2] // 2,
                    face.bbox[1] + face.bbox[3] // 2
                )
                
                # Procura por track existente próximo
                for track_id, track_data in self.face_tracker.items():
                    track_center = (
                        track_data.bbox[0] + track_data.bbox[2] // 2,
                        track_data.bbox[1] + track_data.bbox[3] // 2
                    )
                    
                    distance = np.sqrt(
                        (face_center[0] - track_center[0]) ** 2 +
                        (face_center[1] - track_center[1]) ** 2
                    )
                    
                    if distance < self.tracking_threshold and distance < best_distance:
                        best_distance = distance
                        best_match_id = track_id
                
                # Atualiza track existente ou cria novo
                if best_match_id is not None:
                    face.tracking_id = best_match_id
                    face.first_seen = self.face_tracker[best_match_id].first_seen
                    face.last_seen = current_time
                    self.face_tracker[best_match_id] = face
                else:
                    # Novo track
                    face.tracking_id = self.next_tracking_id
                    face.first_seen = current_time
                    face.last_seen = current_time
                    self.face_tracker[self.next_tracking_id] = face
                    self.next_tracking_id += 1
            
            return current_faces
            
        except Exception as e:
            self.logger.error(f"Erro no sistema de tracking: {e}")
            return current_faces
    
    def _cleanup_cache(self, current_time: float):
        """Remove entradas antigas do cache."""
        expired_keys = []
        for cache_key, (_, cache_time) in self.detection_cache.items():
            if current_time - cache_time > self.cache_ttl * 10:  # 10x TTL
                expired_keys.append(cache_key)
        
        for key in expired_keys:
            del self.detection_cache[key]
    
    def _update_metrics(self, num_faces: int, processing_time: float):
        """Atualiza métricas de performance."""
        self.performance_metrics['total_detections'] += 1
        if num_faces > 0:
            self.performance_metrics['successful_detections'] += 1
        
        self.performance_metrics['processing_times'].append(processing_time)
        
        # Mantém apenas os últimos 100 tempos para cálculo da média
        if len(self.performance_metrics['processing_times']) > 100:
            self.performance_metrics['processing_times'].pop(0)
        
        # Atualiza média
        if self.performance_metrics['processing_times']:
            self.performance_metrics['average_processing_time'] = np.mean(
                self.performance_metrics['processing_times']
            )
        
        self.performance_metrics['faces_tracked'] = len(self.face_tracker)
    
    def get_performance_metrics(self) -> Dict:
        """Retorna métricas de performance."""
        return self.performance_metrics.copy()
    
    def draw_faces(self, frame: np.ndarray, faces: List[FaceData], 
                   draw_landmarks: bool = False, draw_info: bool = True) -> np.ndarray:
        """
        Desenha faces detectadas no frame.
        
        Args:
            frame: Frame original
            faces: Lista de faces detectadas
            draw_landmarks: Se deve desenhar landmarks
            draw_info: Se deve desenhar informações da face
            
        Returns:
            np.ndarray: Frame com faces desenhadas
        """
        result_frame = frame.copy()
        
        try:
            for face in faces:
                x, y, w, h = face.bbox
                
                # Cor baseada na qualidade
                if face.quality_score > 0.7:
                    color = (0, 255, 0)  # Verde - boa qualidade
                elif face.quality_score > 0.4:
                    color = (0, 255, 255)  # Amarelo - qualidade média
                else:
                    color = (0, 0, 255)  # Vermelho - baixa qualidade
                
                # Desenha bounding box
                cv2.rectangle(result_frame, (x, y), (x + w, y + h), color, 2)
                
                # Desenha informações
                if draw_info:
                    info_text = f"ID:{face.tracking_id} Q:{face.quality_score:.2f}"
                    cv2.putText(result_frame, info_text, (x, y - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Desenha landmarks se disponível
                if draw_landmarks and face.landmarks is not None:
                    for point in face.landmarks:
                        cv2.circle(result_frame, (int(point[0]), int(point[1])), 1, (255, 0, 0), -1)
                        
        except Exception as e:
            self.logger.error(f"Erro ao desenhar faces: {e}")
        
        return result_frame
    
    def get_best_face(self, faces: List[FaceData]) -> Optional[FaceData]:
        """
        Retorna a melhor face baseada em qualidade e outros critérios.
        
        Args:
            faces: Lista de faces detectadas
            
        Returns:
            Optional[FaceData]: Melhor face ou None se lista vazia
        """
        if not faces:
            return None
        
        # Ordena por score combinado de qualidade, tamanho e centralidade
        def score_face(face: FaceData) -> float:
            quality_score = face.quality_score
            
            # Bonus para faces maiores
            size_bonus = min(face.bbox[2] * face.bbox[3] / (200 * 200), 1.0) * 0.2
            
            # Bonus para faces frontais
            frontal_bonus = 0.1 if face.is_frontal else 0.0
            
            return quality_score + size_bonus + frontal_bonus
        
        best_face = max(faces, key=score_face)
        return best_face
    
    def __del__(self):
        """Cleanup automático."""
        if hasattr(self, 'face_detection'):
            self.face_detection.close()
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()