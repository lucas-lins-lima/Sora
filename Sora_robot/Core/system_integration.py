# sora_robot/core/system_integration.py

import asyncio
import threading
import time
import queue
import numpy as np
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum
import json
import uuid

from utils.logger import get_logger
from utils.constants import SYSTEM_STATES, PERFORMANCE

# Importações dos módulos principais
from vision_processing.camera_handler import CameraHandler
from vision_processing.facial_recognition import FacialRecognition
from vision_processing.emotion_analysis import EmotionAnalysis
from vision_processing.body_pose_estimation import BodyPoseEstimation

from audio_processing.microphone_handler import MicrophoneHandler
from audio_processing.speech_recognition import SpeechRecognition
from audio_processing.audio_analysis import AudioAnalysis

from nlp.sentiment_analysis import SentimentAnalysis
from nlp.intent_recognition import IntentRecognition
from nlp.dialogue_manager import DialogueManager

from response_generation.llm_integration import LLMIntegration, PromptContext
from response_generation.speech_synthesis import SpeechSynthesis
from response_generation.avatar_animation import AvatarAnimation

import config

class SystemState(Enum):
    """Estados do sistema."""
    INITIALIZING = "initializing"
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    RESPONDING = "responding"
    ERROR = "error"
    SHUTDOWN = "shutdown"

class ProcessingMode(Enum):
    """Modos de processamento."""
    REAL_TIME = "real_time"
    BATCH = "batch"
    ON_DEMAND = "on_demand"

class PerformanceLevel(Enum):
    """Níveis de performance."""
    LOW = "low"          # Componentes básicos
    MEDIUM = "medium"    # Componentes essenciais
    HIGH = "high"        # Todos os componentes
    MAXIMUM = "maximum"  # Máxima qualidade

@dataclass
class SystemConfig:
    """Configuração do sistema integrado."""
    
    # Componentes habilitados
    enable_vision: bool = True
    enable_audio: bool = True
    enable_nlp: bool = True
    enable_response: bool = True
    
    # Configurações de performance
    performance_level: PerformanceLevel = PerformanceLevel.HIGH
    processing_mode: ProcessingMode = ProcessingMode.REAL_TIME
    max_concurrent_processes: int = 4
    
    # Configurações de qualidade
    vision_quality: str = "medium"
    audio_quality: str = "high"
    response_quality: str = "high"
    
    # Configurações de timing
    response_timeout: float = 30.0
    processing_timeout: float = 15.0
    idle_timeout: float = 300.0
    
    # Configurações de cache
    enable_caching: bool = True
    cache_size_mb: int = 100
    
    # Configurações de debug
    debug_mode: bool = False
    log_performance: bool = True
    save_interactions: bool = False

@dataclass
class InteractionData:
    """Dados de uma interação completa."""
    
    interaction_id: str
    timestamp: float
    duration: float
    
    # Dados de entrada
    visual_data: Optional[Any] = None
    audio_data: Optional[Any] = None
    text_input: Optional[str] = None
    
    # Resultados de processamento
    face_data: Optional[Any] = None
    emotion_data: Optional[Any] = None
    pose_data: Optional[Any] = None
    speech_data: Optional[Any] = None
    audio_analysis_data: Optional[Any] = None
    sentiment_data: Optional[Any] = None
    intent_data: Optional[Any] = None
    
    # Resposta gerada
    dialogue_response: Optional[Dict] = None
    llm_response: Optional[Any] = None
    speech_synthesis: Optional[Any] = None
    animation_sequence: Optional[Any] = None
    
    # Metadados
    processing_times: Dict[str, float] = field(default_factory=dict)
    quality_scores: Dict[str, float] = field(default_factory=dict)
    user_satisfaction: Optional[float] = None

class SystemIntegration:
    """
    Classe principal de integração do sistema Sora.
    Orquestra todos os componentes e gerencia o fluxo completo de processamento.
    """
    
    def __init__(self, system_config: Optional[SystemConfig] = None):
        """
        Inicializa o sistema integrado.
        
        Args:
            system_config: Configuração do sistema
        """
        self.logger = get_logger(__name__)
        
        # Configuração do sistema
        self.config = system_config or SystemConfig()
        
        # Estado do sistema
        self.current_state = SystemState.INITIALIZING
        self.is_running = False
        self.is_processing = False
        
        # ID da sessão atual
        self.session_id: Optional[str] = None
        
        # Threading e sincronização
        self.main_lock = threading.RLock()
        self.processing_lock = threading.Lock()
        self.shutdown_event = threading.Event()
        
        # Queues para comunicação entre componentes
        self.vision_queue = queue.Queue(maxsize=10)
        self.audio_queue = queue.Queue(maxsize=20)
        self.processing_queue = queue.Queue(maxsize=15)
        self.response_queue = queue.Queue(maxsize=10)
        
        # Componentes do sistema
        self.components = {}
        self.component_threads = {}
        
        # Histórico de interações
        self.interaction_history = deque(maxlen=100)
        self.current_interaction: Optional[InteractionData] = None
        
        # Sistema de callbacks
        self.state_change_callbacks = []
        self.interaction_callbacks = []
        self.error_callbacks = []
        
        # Métricas de sistema
        self.system_metrics = {
            'uptime': 0.0,
            'total_interactions': 0,
            'successful_interactions': 0,
            'average_response_time': 0.0,
            'response_times': deque(maxlen=200),
            'component_health': {},
            'resource_usage': {},
            'error_count': 0,
            'last_error': None
        }
        
        # Monitoramento de recursos
        self.resource_monitor = None
        
        # Inicialização
        self.startup_time = time.time()
        self._initialize_system()
        
        self.logger.info("SystemIntegration inicializado")
    
    def _initialize_system(self):
        """Inicializa todos os componentes do sistema."""
        try:
            self.logger.info("Inicializando componentes do sistema...")
            
            # Inicializa componentes baseado na configuração
            if self.config.enable_vision:
                self._initialize_vision_components()
            
            if self.config.enable_audio:
                self._initialize_audio_components()
            
            if self.config.enable_nlp:
                self._initialize_nlp_components()
            
            if self.config.enable_response:
                self._initialize_response_components()
            
            # Inicializa monitoramento
            self._initialize_monitoring()
            
            # Verifica saúde dos componentes
            self._health_check()
            
            self.logger.info("Sistema inicializado com sucesso")
            self.current_state = SystemState.IDLE
            
        except Exception as e:
            self.logger.error(f"Erro na inicialização do sistema: {e}")
            self.current_state = SystemState.ERROR
            raise
    
    def _initialize_vision_components(self):
        """Inicializa componentes de visão."""
        try:
            self.logger.info("Inicializando componentes de visão...")
            
            # Camera Handler
            self.components['camera'] = CameraHandler()
            if not self.components['camera'].initialize_camera():
                raise Exception("Falha ao inicializar câmera")
            
            # Facial Recognition
            self.components['facial_recognition'] = FacialRecognition()
            
            # Emotion Analysis
            self.components['emotion_analysis'] = EmotionAnalysis()
            
            # Body Pose Estimation
            if self.config.performance_level in [PerformanceLevel.HIGH, PerformanceLevel.MAXIMUM]:
                self.components['pose_estimation'] = BodyPoseEstimation()
            
            self.logger.info("Componentes de visão inicializados")
            
        except Exception as e:
            self.logger.error(f"Erro ao inicializar visão: {e}")
            raise
    
    def _initialize_audio_components(self):
        """Inicializa componentes de áudio."""
        try:
            self.logger.info("Inicializando componentes de áudio...")
            
            # Microphone Handler
            self.components['microphone'] = MicrophoneHandler()
            if not self.components['microphone'].initialize_audio():
                raise Exception("Falha ao inicializar microfone")
            
            # Speech Recognition
            self.components['speech_recognition'] = SpeechRecognition()
            
            # Audio Analysis
            if self.config.performance_level in [PerformanceLevel.MEDIUM, PerformanceLevel.HIGH, PerformanceLevel.MAXIMUM]:
                self.components['audio_analysis'] = AudioAnalysis()
            
            self.logger.info("Componentes de áudio inicializados")
            
        except Exception as e:
            self.logger.error(f"Erro ao inicializar áudio: {e}")
            raise
    
    def _initialize_nlp_components(self):
        """Inicializa componentes de NLP."""
        try:
            self.logger.info("Inicializando componentes de NLP...")
            
            # Sentiment Analysis
            self.components['sentiment_analysis'] = SentimentAnalysis()
            
            # Intent Recognition
            self.components['intent_recognition'] = IntentRecognition()
            
            # Dialogue Manager
            self.components['dialogue_manager'] = DialogueManager()
            
            self.logger.info("Componentes de NLP inicializados")
            
        except Exception as e:
            self.logger.error(f"Erro ao inicializar NLP: {e}")
            raise
    
    def _initialize_response_components(self):
        """Inicializa componentes de geração de resposta."""
        try:
            self.logger.info("Inicializando componentes de resposta...")
            
            # LLM Integration
            self.components['llm_integration'] = LLMIntegration()
            
            # Speech Synthesis
            self.components['speech_synthesis'] = SpeechSynthesis()
            
            # Avatar Animation
            if self.config.performance_level in [PerformanceLevel.HIGH, PerformanceLevel.MAXIMUM]:
                self.components['avatar_animation'] = AvatarAnimation()
            
            self.logger.info("Componentes de resposta inicializados")
            
        except Exception as e:
            self.logger.error(f"Erro ao inicializar resposta: {e}")
            raise
    
    def _initialize_monitoring(self):
        """Inicializa sistema de monitoramento."""
        try:
            # Inicia monitoramento de recursos
            if self.config.log_performance:
                self.resource_monitor = threading.Thread(
                    target=self._resource_monitoring_loop,
                    daemon=True
                )
                self.resource_monitor.start()
            
        except Exception as e:
            self.logger.error(f"Erro ao inicializar monitoramento: {e}")
    
    def start_system(self) -> bool:
        """
        Inicia o sistema completo.
        
        Returns:
            bool: True se iniciado com sucesso
        """
        try:
            with self.main_lock:
                if self.is_running:
                    self.logger.warning("Sistema já está em execução")
                    return True
                
                self.logger.info("Iniciando sistema Sora...")
                
                # Inicia nova sessão
                self.session_id = str(uuid.uuid4())
                
                # Inicia componentes ativos
                if not self._start_active_components():
                    return False
                
                # Inicia threads de processamento
                self._start_processing_threads()
                
                # Marca como em execução
                self.is_running = True
                self.current_state = SystemState.IDLE
                
                # Inicia sessão do dialogue manager
                if 'dialogue_manager' in self.components:
                    self.components['dialogue_manager'].start_session()
                
                # Executa callbacks
                self._execute_state_change_callbacks(SystemState.IDLE)
                
                self.logger.info(f"Sistema Sora iniciado - Sessão: {self.session_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Erro ao iniciar sistema: {e}")
            self.current_state = SystemState.ERROR
            return False
    
    def _start_active_components(self) -> bool:
        """Inicia componentes que precisam estar ativos."""
        try:
            # Inicia captura de vídeo
            if 'camera' in self.components:
                if not self.components['camera'].start_capture():
                    self.logger.error("Falha ao iniciar captura de vídeo")
                    return False
            
            # Inicia captura de áudio
            if 'microphone' in self.components:
                if not self.components['microphone'].start_listening():
                    self.logger.error("Falha ao iniciar captura de áudio")
                    return False
            
            # Inicia reconhecimento de fala
            if 'speech_recognition' in self.components:
                if not self.components['speech_recognition'].start_recognition():
                    self.logger.error("Falha ao iniciar reconhecimento de fala")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao iniciar componentes ativos: {e}")
            return False
    
    def _start_processing_threads(self):
        """Inicia threads de processamento."""
        try:
            # Thread principal de processamento
            self.component_threads['main_processor'] = threading.Thread(
                target=self._main_processing_loop,
                daemon=True
            )
            self.component_threads['main_processor'].start()
            
            # Thread de processamento de visão
            if self.config.enable_vision:
                self.component_threads['vision_processor'] = threading.Thread(
                    target=self._vision_processing_loop,
                    daemon=True
                )
                self.component_threads['vision_processor'].start()
            
            # Thread de processamento de áudio
            if self.config.enable_audio:
                self.component_threads['audio_processor'] = threading.Thread(
                    target=self._audio_processing_loop,
                    daemon=True
                )
                self.component_threads['audio_processor'].start()
            
            # Thread de geração de resposta
            if self.config.enable_response:
                self.component_threads['response_processor'] = threading.Thread(
                    target=self._response_processing_loop,
                    daemon=True
                )
                self.component_threads['response_processor'].start()
            
        except Exception as e:
            self.logger.error(f"Erro ao iniciar threads: {e}")
    
    def _main_processing_loop(self):
        """Loop principal de processamento."""
        self.logger.info("Iniciando loop principal de processamento...")
        
        while self.is_running and not self.shutdown_event.is_set():
            try:
                # Verifica se há dados para processar
                try:
                    data = self.processing_queue.get(timeout=1.0)
                    self._process_interaction_data(data)
                    self.processing_queue.task_done()
                except queue.Empty:
                    continue
                
            except Exception as e:
                self.logger.error(f"Erro no loop principal: {e}")
                time.sleep(0.1)
        
        self.logger.info("Loop principal de processamento finalizado")
    
    def _vision_processing_loop(self):
        """Loop de processamento de visão."""
        self.logger.debug("Iniciando loop de processamento de visão...")
        
        while self.is_running and not self.shutdown_event.is_set():
            try:
                # Captura frame da câmera
                if 'camera' in self.components:
                    frame = self.components['camera'].get_latest_frame()
                    
                    if frame is not None:
                        vision_data = self._process_vision_frame(frame)
                        
                        if vision_data:
                            # Adiciona à queue principal
                            try:
                                self.processing_queue.put_nowait({
                                    'type': 'vision',
                                    'data': vision_data,
                                    'timestamp': time.time()
                                })
                            except queue.Full:
                                # Remove item mais antigo se necessário
                                try:
                                    self.processing_queue.get_nowait()
                                    self.processing_queue.put_nowait({
                                        'type': 'vision',
                                        'data': vision_data,
                                        'timestamp': time.time()
                                    })
                                except queue.Empty:
                                    pass
                
                time.sleep(1.0 / 10.0)  # 10 FPS para visão
                
            except Exception as e:
                self.logger.error(f"Erro no loop de visão: {e}")
                time.sleep(0.1)
    
    def _audio_processing_loop(self):
        """Loop de processamento de áudio."""
        self.logger.debug("Iniciando loop de processamento de áudio...")
        
        # Configura callbacks do microfone
        if 'microphone' in self.components:
            self.components['microphone'].add_speech_chunk_callback(self._on_speech_chunk)
            self.components['microphone'].add_speech_end_callback(self._on_speech_end)
        
        while self.is_running and not self.shutdown_event.is_set():
            try:
                time.sleep(0.1)  # O processamento é feito via callbacks
                
            except Exception as e:
                self.logger.error(f"Erro no loop de áudio: {e}")
                time.sleep(0.1)
    
    def _response_processing_loop(self):
        """Loop de processamento de resposta."""
        self.logger.debug("Iniciando loop de processamento de resposta...")
        
        while self.is_running and not self.shutdown_event.is_set():
            try:
                # Processa respostas da queue
                try:
                    response_data = self.response_queue.get(timeout=1.0)
                    self._generate_and_deliver_response(response_data)
                    self.response_queue.task_done()
                except queue.Empty:
                    continue
                
            except Exception as e:
                self.logger.error(f"Erro no loop de resposta: {e}")
                time.sleep(0.1)
    
    def _process_vision_frame(self, frame) -> Optional[Dict]:
        """Processa frame de visão."""
        try:
            vision_data = {
                'timestamp': time.time(),
                'frame_shape': frame.shape if hasattr(frame, 'shape') else None
            }
            
            # Reconhecimento facial
            if 'facial_recognition' in self.components:
                faces = self.components['facial_recognition'].detect_faces(frame)
                vision_data['faces'] = faces
                
                # Análise de emoção se há faces
                if faces and 'emotion_analysis' in self.components:
                    emotions = self.components['emotion_analysis'].analyze_emotions(frame, faces)
                    vision_data['emotions'] = emotions
            
            # Estimativa de pose corporal
            if 'pose_estimation' in self.components:
                pose_data = self.components['pose_estimation'].estimate_pose(frame)
                vision_data['pose'] = pose_data
            
            return vision_data
            
        except Exception as e:
            self.logger.error(f"Erro no processamento de visão: {e}")
            return None
    
    def _on_speech_chunk(self, audio_chunk):
        """Callback para chunks de fala."""
        try:
            # Processa chunk de áudio
            if 'speech_recognition' in self.components:
                self.components['speech_recognition'].process_audio_chunk(audio_chunk)
            
            # Análise de áudio se disponível
            if 'audio_analysis' in self.components:
                audio_analysis = self.components['audio_analysis'].analyze_audio_chunk(audio_chunk)
                
                if audio_analysis:
                    # Adiciona análise à queue
                    try:
                        self.processing_queue.put_nowait({
                            'type': 'audio_analysis',
                            'data': audio_analysis,
                            'timestamp': time.time()
                        })
                    except queue.Full:
                        pass  # Descarta se queue estiver cheia
            
        except Exception as e:
            self.logger.error(f"Erro no callback de speech chunk: {e}")
    
    def _on_speech_end(self, voice_activity):
        """Callback para fim de fala."""
        try:
            # Obtém resultado do reconhecimento
            if 'speech_recognition' in self.components:
                recognition_result = self.components['speech_recognition'].get_latest_result()
                
                if recognition_result and recognition_result.full_text:
                    # Adiciona à queue de processamento
                    self.processing_queue.put_nowait({
                        'type': 'speech',
                        'data': recognition_result,
                        'timestamp': time.time()
                    })
            
        except Exception as e:
            self.logger.error(f"Erro no callback de fim de fala: {e}")
    
    def _process_interaction_data(self, data: Dict):
        """Processa dados de interação."""
        try:
            data_type = data.get('type')
            data_content = data.get('data')
            timestamp = data.get('timestamp', time.time())
            
            # Inicia nova interação se for fala
            if data_type == 'speech' and data_content:
                self._start_new_interaction(data_content, timestamp)
            
            # Adiciona dados à interação atual
            elif self.current_interaction:
                self._add_data_to_current_interaction(data_type, data_content, timestamp)
            
        except Exception as e:
            self.logger.error(f"Erro ao processar dados de interação: {e}")
    
    def _start_new_interaction(self, speech_data, timestamp: float):
        """Inicia nova interação baseada na fala."""
        try:
            with self.processing_lock:
                self.is_processing = True
                self.current_state = SystemState.PROCESSING
                
                # Cria nova interação
                interaction_id = str(uuid.uuid4())
                self.current_interaction = InteractionData(
                    interaction_id=interaction_id,
                    timestamp=timestamp,
                    duration=0.0,
                    speech_data=speech_data,
                    text_input=speech_data.full_text
                )
                
                self.logger.debug(f"Nova interação iniciada: {interaction_id}")
                
                # Processa NLP
                self._process_nlp_pipeline(speech_data.full_text)
                
        except Exception as e:
            self.logger.error(f"Erro ao iniciar interação: {e}")
            self.is_processing = False
            self.current_state = SystemState.IDLE
    
    def _process_nlp_pipeline(self, text: str):
        """Processa pipeline de NLP."""
        try:
            start_time = time.time()
            
            # Análise de sentimento
            sentiment_result = None
            if 'sentiment_analysis' in self.components:
                sentiment_result = self.components['sentiment_analysis'].analyze_text(text)
                self.current_interaction.sentiment_data = sentiment_result
                self.current_interaction.processing_times['sentiment'] = time.time() - start_time
            
            # Análise de intenção
            intent_result = None
            if 'intent_recognition' in self.components:
                intent_start = time.time()
                intent_result = self.components['intent_recognition'].analyze_intent(text)
                self.current_interaction.intent_data = intent_result
                self.current_interaction.processing_times['intent'] = time.time() - intent_start
            
            # Processamento do diálogo
            if 'dialogue_manager' in self.components:
                dialogue_start = time.time()
                
                # Prepara dados para o dialogue manager
                dialogue_input = {
                    'recognition_result': self.current_interaction.speech_data,
                    'sentiment_result': sentiment_result,
                    'intent_result': intent_result,
                    'emotion_result': getattr(self.current_interaction, 'emotion_data', None),
                    'audio_result': getattr(self.current_interaction, 'audio_analysis_data', None),
                    'pose_result': getattr(self.current_interaction, 'pose_data', None)
                }
                
                # Processa diálogo
                dialogue_response = self.components['dialogue_manager'].process_interaction(**dialogue_input)
                self.current_interaction.dialogue_response = dialogue_response
                self.current_interaction.processing_times['dialogue'] = time.time() - dialogue_start
                
                # Adiciona à queue de resposta
                self.response_queue.put_nowait({
                    'interaction': self.current_interaction,
                    'dialogue_response': dialogue_response
                })
            
        except Exception as e:
            self.logger.error(f"Erro no pipeline NLP: {e}")
            self.is_processing = False
            self.current_state = SystemState.IDLE
    
    def _add_data_to_current_interaction(self, data_type: str, data_content: Any, timestamp: float):
        """Adiciona dados à interação atual."""
        try:
            if not self.current_interaction:
                return
            
            # Adiciona dados baseado no tipo
            if data_type == 'vision':
                if 'faces' in data_content:
                    self.current_interaction.face_data = data_content['faces']
                if 'emotions' in data_content:
                    self.current_interaction.emotion_data = data_content['emotions']
                if 'pose' in data_content:
                    self.current_interaction.pose_data = data_content['pose']
            
            elif data_type == 'audio_analysis':
                self.current_interaction.audio_analysis_data = data_content
            
        except Exception as e:
            self.logger.error(f"Erro ao adicionar dados à interação: {e}")
    
    async def _generate_and_deliver_response(self, response_data: Dict):
        """Gera e entrega resposta completa."""
        try:
            interaction = response_data['interaction']
            dialogue_response = response_data['dialogue_response']
            
            self.current_state = SystemState.RESPONDING
            
            # Gera resposta com LLM
            llm_start = time.time()
            if 'llm_integration' in self.components:
                # Prepara contexto para LLM
                context = self._prepare_llm_context(interaction, dialogue_response)
                
                # Gera resposta
                llm_response = await self.components['llm_integration'].generate_response(context)
                interaction.llm_response = llm_response
                interaction.processing_times['llm'] = time.time() - llm_start
                
                # Síntese de fala
                if llm_response and 'speech_synthesis' in self.components:
                    synthesis_start = time.time()
                    synthesis_result = await self.components['speech_synthesis'].synthesize_from_llm_response(
                        llm_response,
                        dialogue_response.get('metadata', {}).get('emotion_to_display')
                    )
                    interaction.speech_synthesis = synthesis_result
                    interaction.processing_times['synthesis'] = time.time() - synthesis_start
                    
                    # Animação do avatar
                    if synthesis_result and 'avatar_animation' in self.components:
                        animation_start = time.time()
                        animation_sequence = self.components['avatar_animation'].generate_animation_from_speech(
                            synthesis_result,
                            llm_response,
                            dialogue_response.get('metadata', {}).get('emotion_to_display')
                        )
                        interaction.animation_sequence = animation_sequence
                        interaction.processing_times['animation'] = time.time() - animation_start
                        
                        # Reproduz resposta sincronizada
                        await self._deliver_synchronized_response(synthesis_result, animation_sequence)
                    
                    else:
                        # Reproduz apenas áudio
                        self.components['speech_synthesis'].play_synthesis_result(synthesis_result)
            
            # Finaliza interação
            self._finalize_interaction(interaction)
            
        except Exception as e:
            self.logger.error(f"Erro ao gerar resposta: {e}")
            self.current_state = SystemState.IDLE
            self.is_processing = False
    
    def _prepare_llm_context(self, interaction: InteractionData, dialogue_response: Dict) -> PromptContext:
        """Prepara contexto para o LLM."""
        try:
            # Determina emoção do usuário
            user_emotion = "neutral"
            if interaction.emotion_data:
                user_emotion = getattr(interaction.emotion_data, 'primary_emotion', 'neutral')
            
            # Determina sentimento
            user_sentiment = "neutral"
            if interaction.sentiment_data:
                user_sentiment = interaction.sentiment_data.overall_sentiment
            
            # Determina intenção
            user_intent = "unknown"
            if interaction.intent_data:
                user_intent = interaction.intent_data.primary_intent.intent
            
            # Prepara histórico de conversa
            conversation_history = []
            if 'dialogue_manager' in self.components:
                session_status = self.components['dialogue_manager'].get_session_status()
                # Simplifica histórico para o LLM
                # Em implementação real, recuperaria histórico formatado
            
            # Contexto do evento
            event_context = getattr(config, 'EVENT_CONTEXT', {})
            
            context = PromptContext(
                user_text=interaction.text_input or "",
                user_mood=dialogue_response.get('user_mood', 'neutral'),
                user_intent=user_intent,
                user_sentiment=user_sentiment,
                user_emotion=user_emotion,
                response_strategy=dialogue_response.get('response_strategy', 'informative'),
                dialogue_state=dialogue_response.get('dialogue_state', 'active_conversation'),
                conversation_history=conversation_history,
                current_topic=dialogue_response.get('metadata', {}).get('topic', ''),
                entities_extracted=getattr(interaction.intent_data, 'entities', {}),
                personality_traits=getattr(config, 'PERSONALITY_TRAITS', {}),
                event_context=event_context,
                session_duration=time.time() - self.startup_time,
                user_engagement=dialogue_response.get('metadata', {}).get('engagement', 0.5)
            )
            
            return context
            
        except Exception as e:
            self.logger.error(f"Erro ao preparar contexto LLM: {e}")
            # Contexto mínimo fallback
            return PromptContext(
                user_text=interaction.text_input or "Olá",
                response_strategy="informative"
            )
    
    async def _deliver_synchronized_response(self, synthesis_result, animation_sequence):
        """Entrega resposta sincronizada (áudio + animação)."""
        try:
            # Inicia animação
            if 'avatar_animation' in self.components:
                self.components['avatar_animation'].play_animation(animation_sequence, start_immediately=False)
            
            # Inicia reprodução de áudio
            if 'speech_synthesis' in self.components:
                # Sincroniza início
                animation_sequence.start_time = time.time()
                
                if 'avatar_animation' in self.components:
                    self.components['avatar_animation'].play_animation(animation_sequence, start_immediately=True)
                
                # Reproduz áudio
                self.components['speech_synthesis'].play_synthesis_result(synthesis_result, blocking=True)
            
        except Exception as e:
            self.logger.error(f"Erro na entrega sincronizada: {e}")
    
    def _finalize_interaction(self, interaction: InteractionData):
        """Finaliza interação atual."""
        try:
            # Calcula duração total
            interaction.duration = time.time() - interaction.timestamp
            
            # Calcula scores de qualidade
            self._calculate_quality_scores(interaction)
            
            # Adiciona ao histórico
            self.interaction_history.append(interaction)
            
            # Atualiza métricas
            self._update_system_metrics(interaction)
            
            # Executa callbacks
            self._execute_interaction_callbacks(interaction)
            
            # Salva interação se configurado
            if self.config.save_interactions:
                self._save_interaction(interaction)
            
            # Reset estado
            self.current_interaction = None
            self.is_processing = False
            self.current_state = SystemState.IDLE
            
            self.logger.debug(f"Interação finalizada: {interaction.interaction_id}, duração: {interaction.duration:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Erro ao finalizar interação: {e}")
    
    def _calculate_quality_scores(self, interaction: InteractionData):
        """Calcula scores de qualidade da interação."""
        try:
            scores = {}
            
            # Score de reconhecimento de fala
            if interaction.speech_data:
                scores['speech_recognition'] = getattr(interaction.speech_data, 'overall_confidence', 0.0)
            
            # Score de análise de sentimento
            if interaction.sentiment_data:
                scores['sentiment_analysis'] = interaction.sentiment_data.overall_confidence
            
            # Score de reconhecimento de intenção
            if interaction.intent_data:
                scores['intent_recognition'] = interaction.intent_data.primary_intent.confidence
            
            # Score de resposta LLM
            if interaction.llm_response:
                scores['llm_response'] = getattr(interaction.llm_response, 'confidence', 0.0)
            
            # Score de síntese de fala
            if interaction.speech_synthesis:
                scores['speech_synthesis'] = interaction.speech_synthesis.audio_quality
            
            # Score geral
            if scores:
                scores['overall'] = np.mean(list(scores.values()))
            
            interaction.quality_scores = scores
            
        except Exception as e:
            self.logger.error(f"Erro ao calcular scores de qualidade: {e}")
    
    def _update_system_metrics(self, interaction: InteractionData):
        """Atualiza métricas do sistema."""
        try:
            self.system_metrics['total_interactions'] += 1
            
            if interaction.quality_scores.get('overall', 0.0) > 0.5:
                self.system_metrics['successful_interactions'] += 1
            
            # Tempo de resposta
            response_time = interaction.duration
            self.system_metrics['response_times'].append(response_time)
            
            if self.system_metrics['response_times']:
                self.system_metrics['average_response_time'] = np.mean(
                    list(self.system_metrics['response_times'])
                )
            
            # Uptime
            self.system_metrics['uptime'] = time.time() - self.startup_time
            
        except Exception as e:
            self.logger.error(f"Erro ao atualizar métricas: {e}")
    
    def _save_interaction(self, interaction: InteractionData):
        """Salva dados da interação."""
        try:
            # Implementaria salvamento em arquivo/database
            # Por enquanto, apenas log
            self.logger.info(f"Interação salva: {interaction.interaction_id}")
            
        except Exception as e:
            self.logger.error(f"Erro ao salvar interação: {e}")
    
    def _resource_monitoring_loop(self):
        """Loop de monitoramento de recursos."""
        import psutil
        
        while self.is_running and not self.shutdown_event.is_set():
            try:
                # Coleta métricas de sistema
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                
                self.system_metrics['resource_usage'] = {
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_used_mb': memory.used / (1024 * 1024)
                }
                
                # Verifica saúde dos componentes
                self._update_component_health()
                
                time.sleep(5.0)  # Monitora a cada 5 segundos
                
            except Exception as e:
                self.logger.error(f"Erro no monitoramento de recursos: {e}")
                time.sleep(5.0)
    
    def _update_component_health(self):
        """Atualiza status de saúde dos componentes."""
        try:
            health = {}
            
            for component_name, component in self.components.items():
                try:
                    # Verifica se componente tem método de health check
                    if hasattr(component, 'is_healthy'):
                        health[component_name] = component.is_healthy()
                    elif hasattr(component, 'is_running'):
                        health[component_name] = component.is_running
                    else:
                        health[component_name] = True  # Assume saudável
                        
                except Exception as e:
                    health[component_name] = False
                    self.logger.error(f"Erro no health check de {component_name}: {e}")
            
            self.system_metrics['component_health'] = health
            
        except Exception as e:
            self.logger.error(f"Erro ao atualizar saúde dos componentes: {e}")
    
    def _health_check(self) -> bool:
        """Verifica saúde geral do sistema."""
        try:
            issues = []
            
            # Verifica componentes essenciais
            essential_components = ['dialogue_manager']
            
            if self.config.enable_audio:
                essential_components.extend(['microphone', 'speech_recognition'])
            
            if self.config.enable_response:
                essential_components.extend(['llm_integration', 'speech_synthesis'])
            
            for component_name in essential_components:
                if component_name not in self.components:
                    issues.append(f"Componente essencial ausente: {component_name}")
            
            if issues:
                for issue in issues:
                    self.logger.warning(issue)
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erro no health check: {e}")
            return False
    
    def stop_system(self):
        """Para o sistema completamente."""
        try:
            self.logger.info("Parando sistema Sora...")
            
            with self.main_lock:
                self.is_running = False
                self.current_state = SystemState.SHUTDOWN
                self.shutdown_event.set()
                
                # Para componentes ativos
                self._stop_active_components()
                
                # Aguarda threads terminarem
                self._stop_threads()
                
                # Finaliza sessão do dialogue manager
                if 'dialogue_manager' in self.components:
                    self.components['dialogue_manager'].end_session()
                
                self.logger.info("Sistema Sora parado")
                
        except Exception as e:
            self.logger.error(f"Erro ao parar sistema: {e}")
    
    def _stop_active_components(self):
        """Para componentes ativos."""
        try:
            # Para captura de vídeo
            if 'camera' in self.components:
                self.components['camera'].stop_capture()
            
            # Para captura de áudio
            if 'microphone' in self.components:
                self.components['microphone'].stop_listening()
            
            # Para reconhecimento de fala
            if 'speech_recognition' in self.components:
                self.components['speech_recognition'].stop_recognition()
            
            # Para animações
            if 'avatar_animation' in self.components:
                self.components['avatar_animation'].stop_current_animation()
            
        except Exception as e:
            self.logger.error(f"Erro ao parar componentes: {e}")
    
    def _stop_threads(self):
        """Para todas as threads."""
        try:
            for thread_name, thread in self.component_threads.items():
                if thread and thread.is_alive():
                    thread.join(timeout=3.0)
                    if thread.is_alive():
                        self.logger.warning(f"Thread {thread_name} não terminou no tempo esperado")
            
        except Exception as e:
            self.logger.error(f"Erro ao parar threads: {e}")
    
    def add_state_change_callback(self, callback: Callable):
        """Adiciona callback para mudanças de estado."""
        if callback not in self.state_change_callbacks:
            self.state_change_callbacks.append(callback)
    
    def add_interaction_callback(self, callback: Callable):
        """Adiciona callback para interações."""
        if callback not in self.interaction_callbacks:
            self.interaction_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable):
        """Adiciona callback para erros."""
        if callback not in self.error_callbacks:
            self.error_callbacks.append(callback)
    
    def _execute_state_change_callbacks(self, new_state: SystemState):
        """Executa callbacks de mudança de estado."""
        for callback in self.state_change_callbacks:
            try:
                callback(self.current_state, new_state)
            except Exception as e:
                self.logger.error(f"Erro ao executar callback de estado: {e}")
    
    def _execute_interaction_callbacks(self, interaction: InteractionData):
        """Executa callbacks de interação."""
        for callback in self.interaction_callbacks:
            try:
                callback(interaction)
            except Exception as e:
                self.logger.error(f"Erro ao executar callback de interação: {e}")
    
    def get_system_status(self) -> Dict:
        """Retorna status atual do sistema."""
        return {
            'state': self.current_state.value,
            'is_running': self.is_running,
            'is_processing': self.is_processing,
            'session_id': self.session_id,
            'uptime': time.time() - self.startup_time,
            'components_enabled': {
                'vision': self.config.enable_vision,
                'audio': self.config.enable_audio,
                'nlp': self.config.enable_nlp,
                'response': self.config.enable_response
            },
            'component_health': self.system_metrics.get('component_health', {}),
            'current_interaction': {
                'id': self.current_interaction.interaction_id if self.current_interaction else None,
                'duration': time.time() - self.current_interaction.timestamp if self.current_interaction else 0.0
            }
        }
    
    def get_system_metrics(self) -> Dict:
        """Retorna métricas do sistema."""
        return self.system_metrics.copy()
    
    def get_interaction_history(self, limit: int = 10) -> List[Dict]:
        """Retorna histórico de interações."""
        try:
            recent_interactions = list(self.interaction_history)[-limit:]
            
            # Simplifica dados para retorno
            simplified = []
            for interaction in recent_interactions:
                simplified.append({
                    'id': interaction.interaction_id,
                    'timestamp': interaction.timestamp,
                    'duration': interaction.duration,
                    'text_input': interaction.text_input,
                    'response_text': getattr(interaction.llm_response, 'text', None) if interaction.llm_response else None,
                    'quality_scores': interaction.quality_scores,
                    'processing_times': interaction.processing_times
                })
            
            return simplified
            
        except Exception as e:
            self.logger.error(f"Erro ao obter histórico: {e}")
            return []
    
    def __enter__(self):
        """Context manager entry."""
        self.start_system()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_system()
    
    def __del__(self):
        """Destructor para limpeza automática."""
        if self.is_running:
            self.stop_system()