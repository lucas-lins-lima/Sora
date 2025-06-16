# sora_robot/vision_processing/camera_handler.py

import cv2
import threading
import time
import numpy as np
from typing import Optional, Callable, Tuple
import logging
from utils.logger import get_logger
from utils.constants import CAMERA_SETTINGS
import config

class CameraHandler:
    """
    Gerencia a captura de frames da câmera de forma thread-safe e eficiente.
    Suporta múltiplas resoluções, taxa de frames configurável e callbacks para processamento.
    """
    
    def __init__(self, camera_index: int = None, resolution: Tuple[int, int] = None):
        """
        Inicializa o manipulador da câmera.
        
        Args:
            camera_index: Índice da câmera (padrão: config.CAMERA_INDEX)
            resolution: Tupla (largura, altura) para resolução (padrão: CAMERA_SETTINGS.DEFAULT_RESOLUTION)
        """
        self.logger = get_logger(__name__)
        
        # Configurações da câmera
        self.camera_index = camera_index if camera_index is not None else config.CAMERA_INDEX
        self.resolution = resolution if resolution is not None else CAMERA_SETTINGS.DEFAULT_RESOLUTION
        
        # Estado da câmera
        self.cap = None
        self.is_running = False
        self.current_frame = None
        self.frame_count = 0
        self.fps_counter = 0
        self.last_fps_time = time.time()
        
        # Threading
        self.capture_thread = None
        self.frame_lock = threading.Lock()
        self.stop_event = threading.Event()
        
        # Callbacks para processamento de frames
        self.frame_callbacks = []
        
        # Métricas de performance
        self.performance_metrics = {
            'frames_captured': 0,
            'frames_dropped': 0,
            'average_fps': 0.0,
            'last_update': time.time()
        }
        
        self.logger.info(f"CameraHandler inicializado - Câmera: {self.camera_index}, Resolução: {self.resolution}")
    
    def initialize_camera(self) -> bool:
        """
        Inicializa a conexão com a câmera e configura parâmetros.
        
        Returns:
            bool: True se a inicialização foi bem-sucedida, False caso contrário
        """
        try:
            self.logger.info(f"Inicializando câmera {self.camera_index}...")
            
            # Tenta diferentes backends caso o padrão falhe
            backends = [cv2.CAP_DSHOW, cv2.CAP_V4L2, cv2.CAP_ANY]
            
            for backend in backends:
                try:
                    self.cap = cv2.VideoCapture(self.camera_index, backend)
                    if self.cap.isOpened():
                        self.logger.info(f"Câmera inicializada com backend: {backend}")
                        break
                except Exception as e:
                    self.logger.warning(f"Falha ao inicializar com backend {backend}: {e}")
                    continue
            
            if not self.cap or not self.cap.isOpened():
                raise Exception(f"Não foi possível abrir a câmera {self.camera_index}")
            
            # Configura resolução
            width, height = self.resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            # Configura FPS
            self.cap.set(cv2.CAP_PROP_FPS, CAMERA_SETTINGS.TARGET_FPS)
            
            # Configura buffer size para reduzir latência
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Verifica configurações aplicadas
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            self.logger.info(f"Configurações da câmera aplicadas:")
            self.logger.info(f"  - Resolução: {actual_width}x{actual_height} (solicitado: {width}x{height})")
            self.logger.info(f"  - FPS: {actual_fps} (solicitado: {CAMERA_SETTINGS.TARGET_FPS})")
            
            # Testa captura de um frame
            ret, test_frame = self.cap.read()
            if not ret or test_frame is None:
                raise Exception("Falha ao capturar frame de teste")
            
            self.logger.info("Câmera inicializada com sucesso!")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao inicializar câmera: {e}")
            if self.cap:
                self.cap.release()
                self.cap = None
            return False
    
    def start_capture(self) -> bool:
        """
        Inicia a captura de frames em uma thread separada.
        
        Returns:
            bool: True se a captura foi iniciada com sucesso
        """
        if self.is_running:
            self.logger.warning("Captura já está em execução")
            return True
        
        if not self.cap or not self.cap.isOpened():
            if not self.initialize_camera():
                return False
        
        try:
            self.is_running = True
            self.stop_event.clear()
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
            
            self.logger.info("Captura de câmera iniciada")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao iniciar captura: {e}")
            self.is_running = False
            return False
    
    def stop_capture(self):
        """Para a captura de frames e limpa recursos."""
        if not self.is_running:
            return
        
        self.logger.info("Parando captura de câmera...")
        
        self.is_running = False
        self.stop_event.set()
        
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=5.0)
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        with self.frame_lock:
            self.current_frame = None
        
        self.logger.info("Captura de câmera parada")
    
    def _capture_loop(self):
        """Loop principal de captura de frames executado em thread separada."""
        self.logger.info("Iniciando loop de captura...")
        
        while self.is_running and not self.stop_event.is_set():
            try:
                ret, frame = self.cap.read()
                
                if not ret or frame is None:
                    self.performance_metrics['frames_dropped'] += 1
                    self.logger.warning("Frame inválido capturado")
                    time.sleep(0.01)  # Pequena pausa para evitar loop tight
                    continue
                
                # Processa frame se necessário
                processed_frame = self._process_frame(frame)
                
                # Atualiza frame atual de forma thread-safe
                with self.frame_lock:
                    self.current_frame = processed_frame.copy()
                    self.frame_count += 1
                
                # Atualiza métricas
                self.performance_metrics['frames_captured'] += 1
                self._update_fps_metrics()
                
                # Executa callbacks registrados
                self._execute_callbacks(processed_frame)
                
                # Controla taxa de frames se necessário
                if CAMERA_SETTINGS.ENABLE_FPS_LIMIT:
                    time.sleep(1.0 / CAMERA_SETTINGS.TARGET_FPS)
                
            except Exception as e:
                self.logger.error(f"Erro no loop de captura: {e}")
                self.performance_metrics['frames_dropped'] += 1
                time.sleep(0.1)
        
        self.logger.info("Loop de captura finalizado")
    
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Processa o frame capturado (filtros, correções, etc.).
        
        Args:
            frame: Frame original da câmera
            
        Returns:
            np.ndarray: Frame processado
        """
        processed_frame = frame.copy()
        
        # Aplicar correções se necessário
        if CAMERA_SETTINGS.AUTO_BRIGHTNESS:
            processed_frame = self._adjust_brightness(processed_frame)
        
        if CAMERA_SETTINGS.AUTO_CONTRAST:
            processed_frame = self._adjust_contrast(processed_frame)
        
        # Flipar horizontalmente se necessário (efeito espelho)
        if CAMERA_SETTINGS.MIRROR_MODE:
            processed_frame = cv2.flip(processed_frame, 1)
        
        return processed_frame
    
    def _adjust_brightness(self, frame: np.ndarray) -> np.ndarray:
        """Ajusta brilho automaticamente baseado na luminosidade média."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        
        target_brightness = 128  # Valor alvo
        adjustment = target_brightness - mean_brightness
        
        if abs(adjustment) > 10:  # Só ajusta se a diferença for significativa
            adjustment = np.clip(adjustment * 0.5, -50, 50)  # Suaviza o ajuste
            frame = cv2.convertScaleAbs(frame, alpha=1, beta=adjustment)
        
        return frame
    
    def _adjust_contrast(self, frame: np.ndarray) -> np.ndarray:
        """Ajusta contraste automaticamente."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        contrast = np.std(gray)
        
        if contrast < 30:  # Baixo contraste
            alpha = 1.2  # Aumenta contraste
            frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=0)
        
        return frame
    
    def _update_fps_metrics(self):
        """Atualiza métricas de FPS."""
        current_time = time.time()
        self.fps_counter += 1
        
        if current_time - self.last_fps_time >= 1.0:  # Atualiza a cada segundo
            self.performance_metrics['average_fps'] = self.fps_counter / (current_time - self.last_fps_time)
            self.fps_counter = 0
            self.last_fps_time = current_time
            self.performance_metrics['last_update'] = current_time
    
    def _execute_callbacks(self, frame: np.ndarray):
        """Executa todos os callbacks registrados com o frame atual."""
        for callback in self.frame_callbacks:
            try:
                callback(frame.copy())
            except Exception as e:
                self.logger.error(f"Erro ao executar callback: {e}")
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """
        Retorna o frame atual capturado.
        
        Returns:
            Optional[np.ndarray]: Frame atual ou None se não disponível
        """
        with self.frame_lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
            return None
    
    def add_frame_callback(self, callback: Callable[[np.ndarray], None]):
        """
        Adiciona um callback para processar frames em tempo real.
        
        Args:
            callback: Função que recebe um frame (np.ndarray) como parâmetro
        """
        if callback not in self.frame_callbacks:
            self.frame_callbacks.append(callback)
            self.logger.info(f"Callback adicionado: {callback.__name__}")
    
    def remove_frame_callback(self, callback: Callable[[np.ndarray], None]):
        """Remove um callback da lista."""
        if callback in self.frame_callbacks:
            self.frame_callbacks.remove(callback)
            self.logger.info(f"Callback removido: {callback.__name__}")
    
    def get_performance_metrics(self) -> dict:
        """
        Retorna métricas de performance da câmera.
        
        Returns:
            dict: Métricas incluindo FPS, frames capturados/perdidos, etc.
        """
        return self.performance_metrics.copy()
    
    def is_camera_available(self) -> bool:
        """
        Verifica se a câmera está disponível e funcionando.
        
        Returns:
            bool: True se a câmera está funcionando
        """
        return self.is_running and self.cap is not None and self.cap.isOpened()
    
    def get_camera_info(self) -> dict:
        """
        Retorna informações sobre a câmera atual.
        
        Returns:
            dict: Informações da câmera (resolução, FPS, etc.)
        """
        if not self.cap or not self.cap.isOpened():
            return {}
        
        return {
            'camera_index': self.camera_index,
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': self.cap.get(cv2.CAP_PROP_FPS),
            'backend': self.cap.getBackendName(),
            'frame_count': self.frame_count,
            'is_running': self.is_running
        }
    
    def capture_single_frame(self) -> Optional[np.ndarray]:
        """
        Captura um único frame sem iniciar o loop contínuo.
        Útil para testes ou capturas pontuais.
        
        Returns:
            Optional[np.ndarray]: Frame capturado ou None se falhou
        """
        if not self.cap or not self.cap.isOpened():
            if not self.initialize_camera():
                return None
        
        try:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                return self._process_frame(frame)
            return None
        except Exception as e:
            self.logger.error(f"Erro ao capturar frame único: {e}")
            return None
    
    def __enter__(self):
        """Context manager entry."""
        if self.initialize_camera():
            self.start_capture()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_capture()
    
    def __del__(self):
        """Destructor para limpeza automática."""
        self.stop_capture()