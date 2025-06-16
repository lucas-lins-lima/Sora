# =========================================================================
# SORA ROBOT - VIDEO ANIMATION PLAYER
# Sistema avançado de reprodução de animações de vídeo para o avatar
# =========================================================================

import cv2
import asyncio
import threading
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import time
import queue
from concurrent.futures import ThreadPoolExecutor
import logging

from utils.logger import get_logger
from utils.helpers import validate_input, ensure_directory_exists
from utils.constants import SYSTEM_CONFIG

# =========================================================================
# CONFIGURAÇÕES E ENUMS
# =========================================================================

class AnimationType(Enum):
    """Tipos de animação de vídeo"""
    FACIAL_EXPRESSION = "facial_expression"
    BODY_GESTURE = "body_gesture"
    LIP_SYNC = "lip_sync"
    EYE_MOVEMENT = "eye_movement"
    HEAD_MOVEMENT = "head_movement"
    FULL_BODY = "full_body"
    BACKGROUND_EFFECT = "background_effect"
    PARTICLE_EFFECT = "particle_effect"
    TRANSITION = "transition"
    IDLE_ANIMATION = "idle_animation"

class PlaybackState(Enum):
    """Estados de reprodução"""
    STOPPED = "stopped"
    PLAYING = "playing"
    PAUSED = "paused"
    BUFFERING = "buffering"
    ERROR = "error"
    LOADING = "loading"

class BlendMode(Enum):
    """Modos de mistura de animações"""
    REPLACE = "replace"           # Substitui completamente
    OVERLAY = "overlay"           # Sobrepõe
    MULTIPLY = "multiply"         # Multiplica valores
    ADD = "add"                   # Adiciona valores
    ALPHA_BLEND = "alpha_blend"   # Mistura com alpha

class SyncMode(Enum):
    """Modos de sincronização"""
    AUDIO_SYNC = "audio_sync"     # Sincronizar com áudio
    TIME_SYNC = "time_sync"       # Sincronizar com tempo
    MANUAL_SYNC = "manual_sync"   # Controle manual
    AUTO_SYNC = "auto_sync"       # Sincronização automática

# =========================================================================
# ESTRUTURAS DE DADOS
# =========================================================================

@dataclass
class AnimationClip:
    """Clipe de animação individual"""
    clip_id: str
    name: str
    animation_type: AnimationType
    file_path: Path
    
    # Propriedades de reprodução
    duration_ms: float
    fps: float
    resolution: Tuple[int, int]
    frame_count: int
    
    # Propriedades de blending
    blend_mode: BlendMode = BlendMode.REPLACE
    alpha: float = 1.0
    priority: int = 0
    
    # Metadados
    loop: bool = False
    start_frame: int = 0
    end_frame: Optional[int] = None
    speed_multiplier: float = 1.0
    
    # Triggers e condições
    trigger_conditions: List[str] = field(default_factory=list)
    emotion_tags: List[str] = field(default_factory=list)
    context_tags: List[str] = field(default_factory=list)
    
    # Cache
    frames_cache: Optional[List[np.ndarray]] = None
    is_loaded: bool = False
    
    def __post_init__(self):
        if self.end_frame is None:
            self.end_frame = self.frame_count - 1

@dataclass
class PlaybackSettings:
    """Configurações de reprodução"""
    sync_mode: SyncMode = SyncMode.AUTO_SYNC
    audio_file: Optional[Path] = None
    start_time_ms: float = 0.0
    end_time_ms: Optional[float] = None
    speed_multiplier: float = 1.0
    volume: float = 1.0
    fade_in_ms: float = 0.0
    fade_out_ms: float = 0.0
    auto_loop: bool = False
    max_loops: int = -1  # -1 = infinito

@dataclass
class AnimationLayer:
    """Camada de animação para composição"""
    layer_id: str
    clip: AnimationClip
    settings: PlaybackSettings
    blend_mode: BlendMode
    alpha: float
    z_order: int
    is_active: bool = True
    start_time: float = 0.0
    current_frame: int = 0

@dataclass
class PlaybackState:
    """Estado atual de reprodução"""
    is_playing: bool = False
    is_paused: bool = False
    current_time_ms: float = 0.0
    total_duration_ms: float = 0.0
    current_frame: int = 0
    total_frames: int = 0
    playback_speed: float = 1.0
    volume: float = 1.0
    active_layers: List[str] = field(default_factory=list)

# =========================================================================
# PLAYER DE ANIMAÇÕES DE VÍDEO
# =========================================================================

class VideoAnimationPlayer:
    """Sistema avançado de reprodução de animações de vídeo"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.logger = get_logger(__name__)
        self.config = config or SYSTEM_CONFIG.get("video_animation", {})
        
        # Configurações
        self.animations_dir = Path(self.config.get("animations_directory", "data/animations"))
        self.cache_dir = Path(self.config.get("cache_directory", "cache/animations"))
        self.max_cache_size_mb = self.config.get("max_cache_size_mb", 500)
        self.preload_animations = self.config.get("preload_animations", True)
        self.enable_hardware_acceleration = self.config.get("enable_hardware_acceleration", True)
        
        # Configurações de performance
        self.max_concurrent_loads = self.config.get("max_concurrent_loads", 3)
        self.frame_buffer_size = self.config.get("frame_buffer_size", 30)
        self.target_fps = self.config.get("target_fps", 30)
        
        # Estado do player
        self.playback_state = PlaybackState()
        self.animation_clips: Dict[str, AnimationClip] = {}
        self.active_layers: Dict[str, AnimationLayer] = {}
        self.animation_queue: queue.Queue = queue.Queue()
        
        # Threading
        self._playback_thread: Optional[threading.Thread] = None
        self._loading_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._executor = ThreadPoolExecutor(max_workers=self.max_concurrent_loads)
        
        # Callbacks
        self._frame_callbacks: List[Callable[[np.ndarray, float], None]] = []
        self._state_callbacks: List[Callable[[PlaybackState], None]] = []
        self._animation_callbacks: List[Callable[str, str], None] = []  # (event, clip_id)
        
        # Cache de frames
        self._frame_cache: Dict[str, List[np.ndarray]] = {}
        self._cache_size_mb = 0.0
        
        # Output canvas
        self._canvas_size = (512, 512)  # Tamanho padrão
        self._output_frame: Optional[np.ndarray] = None
        
        # Inicialização
        self._initialize_directories()
        self._load_animation_library()
        self.logger.info("VideoAnimationPlayer inicializado")
    
    def _initialize_directories(self):
        """Inicializa diretórios necessários"""
        try:
            ensure_directory_exists(self.animations_dir)
            ensure_directory_exists(self.cache_dir)
            
            # Criar subdiretórios por tipo
            for anim_type in AnimationType:
                type_dir = self.animations_dir / anim_type.value
                ensure_directory_exists(type_dir)
                
        except Exception as e:
            self.logger.error(f"Erro ao inicializar diretórios: {e}")
            raise
    
    def _load_animation_library(self):
        """Carrega biblioteca de animações"""
        try:
            # Procurar por arquivos de configuração JSON
            config_files = list(self.animations_dir.glob("**/*.json"))
            
            for config_file in config_files:
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        animation_config = json.load(f)
                    
                    self._load_animation_from_config(animation_config, config_file.parent)
                    
                except Exception as e:
                    self.logger.error(f"Erro ao carregar config {config_file}: {e}")
            
            # Procurar por vídeos sem configuração
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
            for ext in video_extensions:
                video_files = list(self.animations_dir.glob(f"**/*{ext}"))
                
                for video_file in video_files:
                    if video_file.stem not in self.animation_clips:
                        self._load_animation_from_video(video_file)
            
            self.logger.info(f"Carregadas {len(self.animation_clips)} animações")
            
            # Pré-carregar animações se configurado
            if self.preload_animations:
                self._preload_critical_animations()
                
        except Exception as e:
            self.logger.error(f"Erro ao carregar biblioteca de animações: {e}")
    
    def _load_animation_from_config(self, config: Dict[str, Any], base_path: Path):
        """Carrega animação a partir de configuração JSON"""
        try:
            clip_id = config['clip_id']
            video_path = base_path / config['file_path']
            
            if not video_path.exists():
                self.logger.warning(f"Arquivo de vídeo não encontrado: {video_path}")
                return
            
            # Obter informações do vídeo
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                self.logger.error(f"Não foi possível abrir vídeo: {video_path}")
                return
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration_ms = (frame_count / fps) * 1000 if fps > 0 else 0
            
            cap.release()
            
            # Criar clip de animação
            clip = AnimationClip(
                clip_id=clip_id,
                name=config.get('name', clip_id),
                animation_type=AnimationType(config.get('animation_type', 'idle_animation')),
                file_path=video_path,
                duration_ms=duration_ms,
                fps=fps,
                resolution=(width, height),
                frame_count=frame_count,
                blend_mode=BlendMode(config.get('blend_mode', 'replace')),
                alpha=config.get('alpha', 1.0),
                priority=config.get('priority', 0),
                loop=config.get('loop', False),
                speed_multiplier=config.get('speed_multiplier', 1.0),
                trigger_conditions=config.get('trigger_conditions', []),
                emotion_tags=config.get('emotion_tags', []),
                context_tags=config.get('context_tags', [])
            )
            
            self.animation_clips[clip_id] = clip
            self.logger.debug(f"Animação carregada: {clip_id}")
            
        except Exception as e:
            self.logger.error(f"Erro ao carregar animação da config: {e}")
    
    def _load_animation_from_video(self, video_path: Path):
        """Carrega animação diretamente de arquivo de vídeo"""
        try:
            clip_id = video_path.stem
            
            # Obter informações do vídeo
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration_ms = (frame_count / fps) * 1000 if fps > 0 else 0
            
            cap.release()
            
            # Inferir tipo de animação a partir do diretório
            animation_type = AnimationType.IDLE_ANIMATION
            for anim_type in AnimationType:
                if anim_type.value in str(video_path.parent):
                    animation_type = anim_type
                    break
            
            # Criar clip básico
            clip = AnimationClip(
                clip_id=clip_id,
                name=clip_id.replace('_', ' ').title(),
                animation_type=animation_type,
                file_path=video_path,
                duration_ms=duration_ms,
                fps=fps,
                resolution=(width, height),
                frame_count=frame_count
            )
            
            self.animation_clips[clip_id] = clip
            self.logger.debug(f"Animação carregada automaticamente: {clip_id}")
            
        except Exception as e:
            self.logger.error(f"Erro ao carregar vídeo {video_path}: {e}")
    
    def _preload_critical_animations(self):
        """Pré-carrega animações críticas"""
        try:
            # Priorizar animações idle e de transição
            critical_types = [
                AnimationType.IDLE_ANIMATION,
                AnimationType.TRANSITION,
                AnimationType.FACIAL_EXPRESSION
            ]
            
            critical_clips = [
                clip for clip in self.animation_clips.values()
                if clip.animation_type in critical_types
            ]
            
            # Carregar em paralelo
            futures = []
            for clip in critical_clips[:5]:  # Limitar a 5 animações críticas
                future = self._executor.submit(self._load_clip_frames, clip)
                futures.append(future)
            
            # Aguardar carregamento
            for future in futures:
                try:
                    future.result(timeout=10)  # 10 segundos de timeout
                except Exception as e:
                    self.logger.error(f"Erro no pré-carregamento: {e}")
                    
        except Exception as e:
            self.logger.error(f"Erro no pré-carregamento crítico: {e}")
    
    def _load_clip_frames(self, clip: AnimationClip) -> bool:
        """Carrega frames de um clipe para cache"""
        try:
            if clip.is_loaded or clip.clip_id in self._frame_cache:
                return True
            
            cap = cv2.VideoCapture(str(clip.file_path))
            if not cap.isOpened():
                return False
            
            frames = []
            frame_idx = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Redimensionar para tamanho do canvas se necessário
                if frame.shape[:2] != self._canvas_size:
                    frame = cv2.resize(frame, self._canvas_size)
                
                frames.append(frame)
                frame_idx += 1
                
                # Limitar número de frames para controlar memória
                if frame_idx >= 1000:  # Máximo 1000 frames por clipe
                    break
            
            cap.release()
            
            if frames:
                self._frame_cache[clip.clip_id] = frames
                clip.frames_cache = frames
                clip.is_loaded = True
                
                # Calcular tamanho em cache
                frame_size_mb = (frames[0].nbytes * len(frames)) / (1024 * 1024)
                self._cache_size_mb += frame_size_mb
                
                # Verificar limite de cache
                if self._cache_size_mb > self.max_cache_size_mb:
                    self._cleanup_cache()
                
                self.logger.debug(f"Frames carregados para {clip.clip_id}: {len(frames)} frames")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Erro ao carregar frames do clipe {clip.clip_id}: {e}")
            return False
    
    def _cleanup_cache(self):
        """Limpa cache baseado em LRU"""
        try:
            # Ordenar clipes por último uso (simplificado)
            clips_by_usage = sorted(
                self.animation_clips.values(),
                key=lambda c: c.priority,
                reverse=True
            )
            
            # Remover clips de baixa prioridade
            for clip in clips_by_usage:
                if self._cache_size_mb <= self.max_cache_size_mb * 0.8:
                    break
                
                if clip.clip_id in self._frame_cache:
                    frames = self._frame_cache[clip.clip_id]
                    frame_size_mb = (frames[0].nbytes * len(frames)) / (1024 * 1024)
                    
                    del self._frame_cache[clip.clip_id]
                    clip.frames_cache = None
                    clip.is_loaded = False
                    self._cache_size_mb -= frame_size_mb
                    
                    self.logger.debug(f"Removido do cache: {clip.clip_id}")
                    
        except Exception as e:
            self.logger.error(f"Erro na limpeza de cache: {e}")
    
    def play_animation(self, 
                      clip_id: str,
                      settings: Optional[PlaybackSettings] = None,
                      layer_id: str = "main") -> bool:
        """
        Reproduz uma animação
        
        Args:
            clip_id: ID do clipe de animação
            settings: Configurações de reprodução
            layer_id: ID da camada
            
        Returns:
            True se iniciado com sucesso
        """
        try:
            if clip_id not in self.animation_clips:
                self.logger.error(f"Clipe de animação não encontrado: {clip_id}")
                return False
            
            clip = self.animation_clips[clip_id]
            
            # Configurações padrão
            if settings is None:
                settings = PlaybackSettings()
            
            # Carregar frames se necessário
            if not clip.is_loaded:
                if not self._load_clip_frames(clip):
                    self.logger.error(f"Falha ao carregar frames: {clip_id}")
                    return False
            
            # Criar camada de animação
            layer = AnimationLayer(
                layer_id=layer_id,
                clip=clip,
                settings=settings,
                blend_mode=clip.blend_mode,
                alpha=clip.alpha,
                z_order=clip.priority,
                start_time=time.time()
            )
            
            # Adicionar à lista de camadas ativas
            self.active_layers[layer_id] = layer
            
            # Iniciar reprodução se não estiver rodando
            if not self.playback_state.is_playing:
                self._start_playback()
            
            # Callback de início de animação
            self._trigger_animation_callback("animation_started", clip_id)
            
            self.logger.info(f"Iniciando reprodução: {clip_id} na camada {layer_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao reproduzir animação {clip_id}: {e}")
            return False
    
    def stop_animation(self, layer_id: str = None) -> bool:
        """
        Para uma animação específica ou todas
        
        Args:
            layer_id: ID da camada (None para todas)
            
        Returns:
            True se parado com sucesso
        """
        try:
            if layer_id is None:
                # Parar todas as animações
                for lid in list(self.active_layers.keys()):
                    layer = self.active_layers[lid]
                    self._trigger_animation_callback("animation_stopped", layer.clip.clip_id)
                    del self.active_layers[lid]
                
                self.playback_state.is_playing = False
                self.playback_state.active_layers.clear()
                
            else:
                # Parar animação específica
                if layer_id in self.active_layers:
                    layer = self.active_layers[layer_id]
                    self._trigger_animation_callback("animation_stopped", layer.clip.clip_id)
                    del self.active_layers[layer_id]
                    
                    if layer_id in self.playback_state.active_layers:
                        self.playback_state.active_layers.remove(layer_id)
                
                # Parar playback se não há mais camadas
                if not self.active_layers:
                    self.playback_state.is_playing = False
            
            self.logger.info(f"Animação parada: {layer_id or 'todas'}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao parar animação: {e}")
            return False
    
    def _start_playback(self):
        """Inicia thread de reprodução"""
        try:
            if self._playback_thread and self._playback_thread.is_alive():
                return
            
            self._stop_event.clear()
            self.playback_state.is_playing = True
            
            self._playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
            self._playback_thread.start()
            
        except Exception as e:
            self.logger.error(f"Erro ao iniciar playback: {e}")
    
    def _playback_loop(self):
        """Loop principal de reprodução"""
        try:
            last_frame_time = time.time()
            frame_interval = 1.0 / self.target_fps
            
            while not self._stop_event.is_set() and self.active_layers:
                current_time = time.time()
                
                # Controlar FPS
                time_since_last_frame = current_time - last_frame_time
                if time_since_last_frame < frame_interval:
                    time.sleep(frame_interval - time_since_last_frame)
                    continue
                
                # Processar frame atual
                self._process_frame()
                
                # Atualizar estado
                self._update_playback_state()
                
                # Callbacks de estado
                self._trigger_state_callbacks()
                
                last_frame_time = time.time()
            
            # Finalizar reprodução
            self.playback_state.is_playing = False
            self.logger.debug("Loop de reprodução finalizado")
            
        except Exception as e:
            self.logger.error(f"Erro no loop de reprodução: {e}")
            self.playback_state.is_playing = False
    
    def _process_frame(self):
        """Processa frame atual combinando todas as camadas"""
        try:
            # Criar canvas vazio
            canvas = np.zeros((*self._canvas_size, 3), dtype=np.uint8)
            
            # Ordenar camadas por z-order
            sorted_layers = sorted(
                self.active_layers.values(),
                key=lambda l: l.z_order
            )
            
            layers_to_remove = []
            
            for layer in sorted_layers:
                if not layer.is_active:
                    continue
                
                # Calcular frame atual
                elapsed_time = time.time() - layer.start_time
                elapsed_ms = elapsed_time * 1000 * layer.settings.speed_multiplier
                
                # Verificar se animação terminou
                if (not layer.clip.loop and 
                    elapsed_ms >= layer.clip.duration_ms):
                    layers_to_remove.append(layer.layer_id)
                    self._trigger_animation_callback("animation_completed", layer.clip.clip_id)
                    continue
                
                # Loop de animação
                if layer.clip.loop and elapsed_ms >= layer.clip.duration_ms:
                    elapsed_ms = elapsed_ms % layer.clip.duration_ms
                
                # Calcular frame
                frame_index = int((elapsed_ms / layer.clip.duration_ms) * layer.clip.frame_count)
                frame_index = max(0, min(frame_index, layer.clip.frame_count - 1))
                
                # Obter frame
                frame = self._get_frame(layer.clip, frame_index)
                if frame is not None:
                    # Aplicar blending
                    canvas = self._blend_frame(canvas, frame, layer.blend_mode, layer.alpha)
                
                layer.current_frame = frame_index
            
            # Remover camadas finalizadas
            for layer_id in layers_to_remove:
                if layer_id in self.active_layers:
                    del self.active_layers[layer_id]
            
            # Atualizar output frame
            self._output_frame = canvas
            
            # Callbacks de frame
            self._trigger_frame_callbacks(canvas, time.time())
            
        except Exception as e:
            self.logger.error(f"Erro ao processar frame: {e}")
    
    def _get_frame(self, clip: AnimationClip, frame_index: int) -> Optional[np.ndarray]:
        """Obtém frame específico de um clipe"""
        try:
            if clip.frames_cache and frame_index < len(clip.frames_cache):
                return clip.frames_cache[frame_index].copy()
            
            # Carregar frame diretamente do arquivo se não estiver em cache
            cap = cv2.VideoCapture(str(clip.file_path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                # Redimensionar se necessário
                if frame.shape[:2] != self._canvas_size:
                    frame = cv2.resize(frame, self._canvas_size)
                return frame
            
            return None
            
        except Exception as e:
            self.logger.error(f"Erro ao obter frame {frame_index} do clipe {clip.clip_id}: {e}")
            return None
    
    def _blend_frame(self, canvas: np.ndarray, frame: np.ndarray, 
                    blend_mode: BlendMode, alpha: float) -> np.ndarray:
        """Aplica blending entre canvas e frame"""
        try:
            if blend_mode == BlendMode.REPLACE:
                return cv2.addWeighted(canvas, 1.0 - alpha, frame, alpha, 0)
            
            elif blend_mode == BlendMode.OVERLAY:
                return cv2.addWeighted(canvas, 1.0, frame, alpha, 0)
            
            elif blend_mode == BlendMode.MULTIPLY:
                normalized_canvas = canvas.astype(np.float32) / 255.0
                normalized_frame = frame.astype(np.float32) / 255.0
                result = normalized_canvas * normalized_frame * alpha + normalized_canvas * (1.0 - alpha)
                return (result * 255).astype(np.uint8)
            
            elif blend_mode == BlendMode.ADD:
                return cv2.add(canvas, (frame * alpha).astype(np.uint8))
            
            elif blend_mode == BlendMode.ALPHA_BLEND:
                return cv2.addWeighted(canvas, 1.0 - alpha, frame, alpha, 0)
            
            else:
                return cv2.addWeighted(canvas, 1.0 - alpha, frame, alpha, 0)
                
        except Exception as e:
            self.logger.error(f"Erro no blending: {e}")
            return canvas
    
    def _update_playback_state(self):
        """Atualiza estado de reprodução"""
        try:
            if self.active_layers:
                # Calcular tempo total e atual
                max_duration = 0
                current_times = []
                
                for layer in self.active_layers.values():
                    elapsed_time = time.time() - layer.start_time
                    elapsed_ms = elapsed_time * 1000 * layer.settings.speed_multiplier
                    
                    current_times.append(elapsed_ms)
                    
                    duration = layer.clip.duration_ms
                    if layer.clip.loop:
                        duration = float('inf')  # Duração infinita para loops
                    
                    max_duration = max(max_duration, duration)
                
                self.playback_state.current_time_ms = max(current_times) if current_times else 0
                self.playback_state.total_duration_ms = max_duration if max_duration != float('inf') else 0
                self.playback_state.active_layers = list(self.active_layers.keys())
                
        except Exception as e:
            self.logger.error(f"Erro ao atualizar estado: {e}")
    
    def _trigger_frame_callbacks(self, frame: np.ndarray, timestamp: float):
        """Dispara callbacks de frame"""
        for callback in self._frame_callbacks:
            try:
                callback(frame, timestamp)
            except Exception as e:
                self.logger.error(f"Erro em callback de frame: {e}")
    
    def _trigger_state_callbacks(self):
        """Dispara callbacks de estado"""
        for callback in self._state_callbacks:
            try:
                callback(self.playback_state)
            except Exception as e:
                self.logger.error(f"Erro em callback de estado: {e}")
    
    def _trigger_animation_callback(self, event: str, clip_id: str):
        """Dispara callbacks de animação"""
        for callback in self._animation_callbacks:
            try:
                callback(event, clip_id)
            except Exception as e:
                self.logger.error(f"Erro em callback de animação: {e}")
    
    def add_frame_callback(self, callback: Callable[[np.ndarray, float], None]):
        """Adiciona callback para frames"""
        self._frame_callbacks.append(callback)
    
    def add_state_callback(self, callback: Callable[[PlaybackState], None]):
        """Adiciona callback para mudanças de estado"""
        self._state_callbacks.append(callback)
    
    def add_animation_callback(self, callback: Callable[[str, str], None]):
        """Adiciona callback para eventos de animação"""
        self._animation_callbacks.append(callback)
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """Retorna frame atual"""
        return self._output_frame.copy() if self._output_frame is not None else None
    
    def get_playback_state(self) -> PlaybackState:
        """Retorna estado atual de reprodução"""
        return self.playback_state
    
    def list_animations(self, animation_type: Optional[AnimationType] = None) -> List[Dict[str, Any]]:
        """Lista animações disponíveis"""
        animations = []
        
        for clip in self.animation_clips.values():
            if animation_type is None or clip.animation_type == animation_type:
                animations.append({
                    'clip_id': clip.clip_id,
                    'name': clip.name,
                    'type': clip.animation_type.value,
                    'duration_ms': clip.duration_ms,
                    'fps': clip.fps,
                    'resolution': clip.resolution,
                    'is_loaded': clip.is_loaded,
                    'emotion_tags': clip.emotion_tags,
                    'context_tags': clip.context_tags
                })
        
        return animations
    
    def get_animations_by_emotion(self, emotion: str) -> List[str]:
        """Retorna animações compatíveis com emoção"""
        compatible_clips = []
        
        for clip in self.animation_clips.values():
            if emotion.lower() in [tag.lower() for tag in clip.emotion_tags]:
                compatible_clips.append(clip.clip_id)
        
        return compatible_clips
    
    def get_animations_by_context(self, context: str) -> List[str]:
        """Retorna animações compatíveis com contexto"""
        compatible_clips = []
        
        for clip in self.animation_clips.values():
            if context.lower() in [tag.lower() for tag in clip.context_tags]:
                compatible_clips.append(clip.clip_id)
        
        return compatible_clips
    
    def set_canvas_size(self, width: int, height: int):
        """Define tamanho do canvas de output"""
        self._canvas_size = (width, height)
        self.logger.info(f"Canvas size alterado para: {width}x{height}")
    
    def shutdown(self):
        """Finaliza o player"""
        try:
            self._stop_event.set()
            self.stop_animation()
            
            if self._playback_thread:
                self._playback_thread.join(timeout=2)
            
            self._executor.shutdown(wait=True)
            
            self.logger.info("VideoAnimationPlayer finalizado")
            
        except Exception as e:
            self.logger.error(f"Erro ao finalizar player: {e}")

# =========================================================================
# FUNÇÕES DE CONVENIÊNCIA
# =========================================================================

# Instância global do player
_video_animation_player = None

def get_video_animation_player() -> VideoAnimationPlayer:
    """Retorna instância global do player de animações"""
    global _video_animation_player
    if _video_animation_player is None:
        _video_animation_player = VideoAnimationPlayer()
    return _video_animation_player

def play_emotion_animation(emotion: str, intensity: float = 1.0) -> bool:
    """Reproduz animação baseada em emoção"""
    player = get_video_animation_player()
    
    # Buscar animações compatíveis
    compatible_clips = player.get_animations_by_emotion(emotion)
    
    if not compatible_clips:
        return False
    
    # Selecionar primeira animação disponível
    clip_id = compatible_clips[0]
    
    # Configurar reprodução
    settings = PlaybackSettings(
        speed_multiplier=intensity,
        auto_loop=False
    )
    
    return player.play_animation(clip_id, settings, f"emotion_{emotion}")

def play_context_animation(context: str, loop: bool = False) -> bool:
    """Reproduz animação baseada em contexto"""
    player = get_video_animation_player()
    
    # Buscar animações compatíveis
    compatible_clips = player.get_animations_by_context(context)
    
    if not compatible_clips:
        return False
    
    # Selecionar animação
    clip_id = compatible_clips[0]
    
    # Configurar reprodução
    settings = PlaybackSettings(
        auto_loop=loop
    )
    
    return player.play_animation(clip_id, settings, f"context_{context}")

def stop_all_animations():
    """Para todas as animações"""
    player = get_video_animation_player()
    return player.stop_animation()

# =========================================================================
# EXEMPLO DE USO
# =========================================================================

if __name__ == "__main__":
    # Exemplo de uso do player de animações
    
    # Criar instância do player
    player = get_video_animation_player()
    
    # Callback para frames
    def on_frame(frame: np.ndarray, timestamp: float):
        # Aqui você pode processar cada frame
        # Por exemplo, salvar, exibir ou transmitir
        cv2.imshow("Sora Animation", frame)
        cv2.waitKey(1)
    
    # Callback para eventos de animação
    def on_animation_event(event: str, clip_id: str):
        print(f"Evento de animação: {event} - {clip_id}")
    
    # Registrar callbacks
    player.add_frame_callback(on_frame)
    player.add_animation_callback(on_animation_event)
    
    # Listar animações disponíveis
    animations = player.list_animations()
    print(f"Animações disponíveis: {len(animations)}")
    
    for anim in animations:
        print(f"- {anim['name']} ({anim['type']}) - {anim['duration_ms']:.0f}ms")
    
    # Reproduzir animação de emoção feliz
    if play_emotion_animation("happy", intensity=1.2):
        print("Animação de felicidade iniciada")
    
    # Reproduzir animação idle em loop
    idle_animations = player.get_animations_by_context("idle")
    if idle_animations:
        settings = PlaybackSettings(auto_loop=True)
        player.play_animation(idle_animations[0], settings, "idle_layer")
        print("Animação idle iniciada em loop")
    
    # Aguardar reprodução por alguns segundos
    try:
        time.sleep(10)
    except KeyboardInterrupt:
        print("Interrompido pelo usuário")
    
    # Parar todas as animações
    stop_all_animations()
    print("Todas as animações paradas")
    
    # Finalizar player
    player.shutdown()
    cv2.destroyAllWindows()
    
    print("Player finalizado")