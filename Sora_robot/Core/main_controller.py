# sora_robot/core/main_controller.py

import asyncio
import threading
import time
import queue
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import logging

from utils.logger import get_logger
from core.system_integration import SystemIntegration, SystemConfig, SystemState, InteractionData
import config

class OperationMode(Enum):
    """Modos de operação do robô."""
    INTERACTIVE = "interactive"      # Modo conversacional contínuo
    SINGLE_SHOT = "single_shot"     # Uma pergunta, uma resposta
    BATCH = "batch"                 # Processamento em lote
    API_MODE = "api_mode"           # Modo para integração via API
    DEMO_MODE = "demo_mode"         # Modo demonstração

class ControllerState(Enum):
    """Estados do controlador."""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    ACTIVE = "active"
    PAUSED = "paused"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"

@dataclass
class SoraResponse:
    """Resposta padronizada do Sora."""
    
    # Resposta principal
    text: str
    audio_available: bool = False
    animation_available: bool = False
    
    # Metadados
    response_id: str = ""
    timestamp: float = 0.0
    processing_time: float = 0.0
    
    # Contexto da resposta
    emotion_detected: str = "neutral"
    intent_detected: str = "unknown"
    confidence: float = 0.0
    
    # Dados adicionais
    suggested_followup: Optional[str] = None
    user_mood: str = "neutral"
    interaction_quality: float = 0.0
    
    # Status
    success: bool = True
    error_message: Optional[str] = None

@dataclass
class SoraConfig:
    """Configuração simplificada para o Sora."""
    
    # Configurações básicas
    name: str = "Sora"
    language: str = "pt-BR"
    personality: str = "friendly"  # friendly, professional, casual, empathetic
    
    # Modos de operação
    operation_mode: OperationMode = OperationMode.INTERACTIVE
    voice_enabled: bool = True
    vision_enabled: bool = True
    animation_enabled: bool = True
    
    # Configurações de resposta
    response_style: str = "balanced"  # concise, balanced, detailed
    max_response_time: float = 10.0
    enable_proactive_responses: bool = True
    
    # Configurações de qualidade
    speech_quality: str = "high"      # low, medium, high
    processing_quality: str = "high"  # low, medium, high
    
    # Configurações avançadas
    debug_mode: bool = False
    save_conversations: bool = False
    privacy_mode: bool = False

class MainController:
    """
    Controlador principal do robô Sora.
    Fornece interface de alto nível para interação com o sistema completo.
    """
    
    def __init__(self, sora_config: Optional[SoraConfig] = None):
        """
        Inicializa o controlador principal.
        
        Args:
            sora_config: Configuração do Sora
        """
        self.logger = get_logger(__name__)
        
        # Configuração
        self.sora_config = sora_config or SoraConfig()
        
        # Estado do controlador
        self.controller_state = ControllerState.UNINITIALIZED
        self.is_initialized = False
        self.is_active = False
        
        # Sistema de integração
        self.system_integration: Optional[SystemIntegration] = None
        
        # Threading
        self.controller_lock = threading.RLock()
        self.response_queue = queue.Queue()
        self.command_queue = queue.Queue()
        
        # Callbacks para eventos
        self.message_callbacks = []      # Callbacks para mensagens recebidas
        self.response_callbacks = []     # Callbacks para respostas geradas
        self.state_callbacks = []        # Callbacks para mudanças de estado
        self.error_callbacks = []        # Callbacks para erros
        
        # Estatísticas de uso
        self.usage_stats = {
            'total_interactions': 0,
            'successful_interactions': 0,
            'total_uptime': 0.0,
            'average_response_time': 0.0,
            'user_satisfaction': 0.0,
            'most_common_intents': {},
            'most_common_emotions': {}
        }
        
        # Histórico de conversas (simplificado)
        self.conversation_history = []
        
        # Cache de configurações
        self.runtime_config = {}
        
        self.logger.info("MainController inicializado")
    
    def initialize(self) -> bool:
        """
        Inicializa o sistema Sora.
        
        Returns:
            bool: True se inicialização bem-sucedida
        """
        try:
            with self.controller_lock:
                if self.is_initialized:
                    self.logger.warning("Sistema já inicializado")
                    return True
                
                self.controller_state = ControllerState.INITIALIZING
                self.logger.info("Inicializando sistema Sora...")
                
                # Converte configuração do Sora para configuração do sistema
                system_config = self._convert_to_system_config()
                
                # Inicializa sistema de integração
                self.system_integration = SystemIntegration(system_config)
                
                # Configura callbacks
                self._setup_system_callbacks()
                
                # Aplica configurações de personalidade
                self._apply_personality_config()
                
                # Verifica se tudo está funcionando
                if not self._verify_system_health():
                    raise Exception("Verificação de saúde do sistema falhou")
                
                self.is_initialized = True
                self.controller_state = ControllerState.READY
                
                self.logger.info("Sistema Sora inicializado com sucesso")
                return True
                
        except Exception as e:
            self.logger.error(f"Erro na inicialização: {e}")
            self.controller_state = ControllerState.ERROR
            self._execute_error_callbacks(str(e))
            return False
    
    def start(self) -> bool:
        """
        Inicia o sistema Sora.
        
        Returns:
            bool: True se iniciado com sucesso
        """
        try:
            with self.controller_lock:
                if not self.is_initialized:
                    if not self.initialize():
                        return False
                
                if self.is_active:
                    self.logger.warning("Sistema já está ativo")
                    return True
                
                self.logger.info("Iniciando sistema Sora...")
                
                # Inicia sistema de integração
                if not self.system_integration.start_system():
                    raise Exception("Falha ao iniciar sistema de integração")
                
                self.is_active = True
                self.controller_state = ControllerState.ACTIVE
                
                # Executa callbacks de mudança de estado
                self._execute_state_callbacks(ControllerState.ACTIVE)
                
                # Inicia monitoramento se configurado
                if self.sora_config.debug_mode:
                    self._start_monitoring()
                
                self.logger.info("Sistema Sora iniciado e pronto para interação")
                return True
                
        except Exception as e:
            self.logger.error(f"Erro ao iniciar sistema: {e}")
            self.controller_state = ControllerState.ERROR
            return False
    
    def stop(self) -> bool:
        """
        Para o sistema Sora.
        
        Returns:
            bool: True se parado com sucesso
        """
        try:
            with self.controller_lock:
                if not self.is_active:
                    self.logger.info("Sistema já está parado")
                    return True
                
                self.logger.info("Parando sistema Sora...")
                self.controller_state = ControllerState.SHUTTING_DOWN
                
                # Para sistema de integração
                if self.system_integration:
                    self.system_integration.stop_system()
                
                self.is_active = False
                self.controller_state = ControllerState.READY
                
                # Executa callbacks
                self._execute_state_callbacks(ControllerState.READY)
                
                self.logger.info("Sistema Sora parado")
                return True
                
        except Exception as e:
            self.logger.error(f"Erro ao parar sistema: {e}")
            return False
    
    def pause(self) -> bool:
        """
        Pausa o sistema temporariamente.
        
        Returns:
            bool: True se pausado com sucesso
        """
        try:
            with self.controller_lock:
                if not self.is_active:
                    return False
                
                self.logger.info("Pausando sistema Sora...")
                
                # Para componentes de entrada
                if self.system_integration:
                    # Para captura de áudio/vídeo temporariamente
                    if 'microphone' in self.system_integration.components:
                        self.system_integration.components['microphone'].stop_listening()
                    
                    if 'camera' in self.system_integration.components:
                        self.system_integration.components['camera'].stop_capture()
                
                self.controller_state = ControllerState.PAUSED
                self._execute_state_callbacks(ControllerState.PAUSED)
                
                return True
                
        except Exception as e:
            self.logger.error(f"Erro ao pausar sistema: {e}")
            return False
    
    def resume(self) -> bool:
        """
        Resume o sistema pausado.
        
        Returns:
            bool: True se resumed com sucesso
        """
        try:
            with self.controller_lock:
                if self.controller_state != ControllerState.PAUSED:
                    return False
                
                self.logger.info("Resumindo sistema Sora...")
                
                # Reinicia componentes de entrada
                if self.system_integration:
                    if 'microphone' in self.system_integration.components:
                        self.system_integration.components['microphone'].start_listening()
                    
                    if 'camera' in self.system_integration.components:
                        self.system_integration.components['camera'].start_capture()
                
                self.controller_state = ControllerState.ACTIVE
                self._execute_state_callbacks(ControllerState.ACTIVE)
                
                return True
                
        except Exception as e:
            self.logger.error(f"Erro ao resumir sistema: {e}")
            return False
    
    def send_message(self, message: str, wait_for_response: bool = True, 
                    timeout: float = None) -> Optional[SoraResponse]:
        """
        Envia mensagem de texto para o Sora.
        
        Args:
            message: Mensagem de texto
            wait_for_response: Se deve aguardar resposta
            timeout: Timeout para resposta (usa configuração padrão se None)
            
        Returns:
            Optional[SoraResponse]: Resposta do Sora se wait_for_response=True
        """
        try:
            if not self.is_active:
                return SoraResponse(
                    text="Sistema não está ativo",
                    success=False,
                    error_message="Sistema não inicializado ou não ativo"
                )
            
            if not message or not message.strip():
                return SoraResponse(
                    text="Mensagem vazia recebida",
                    success=False,
                    error_message="Mensagem não pode estar vazia"
                )
            
            # Executa callbacks de mensagem
            self._execute_message_callbacks(message)
            
            # Processa mensagem através do sistema
            if self.sora_config.operation_mode == OperationMode.SINGLE_SHOT:
                return self._process_single_shot_message(message, timeout)
            else:
                return self._process_interactive_message(message, wait_for_response, timeout)
            
        except Exception as e:
            self.logger.error(f"Erro ao processar mensagem: {e}")
            return SoraResponse(
                text="Desculpe, houve um erro interno",
                success=False,
                error_message=str(e)
            )
    
    def _process_single_shot_message(self, message: str, timeout: float = None) -> SoraResponse:
        """Processa mensagem em modo single-shot."""
        try:
            timeout = timeout or self.sora_config.max_response_time
            
            # Simula entrada de fala criando resultado de reconhecimento
            from audio_processing.speech_recognition import RecognitionResult, SpeechSegment
            
            recognition_result = RecognitionResult(
                segments=[],
                full_text=message,
                overall_confidence=1.0,
                total_duration=0.0,
                engine_used="text_input"
            )
            
            # Processa através do dialogue manager
            if 'dialogue_manager' in self.system_integration.components:
                dialogue_response = self.system_integration.components['dialogue_manager'].process_interaction(
                    recognition_result=recognition_result
                )
                
                return self._convert_dialogue_response_to_sora_response(dialogue_response)
            
            else:
                return SoraResponse(
                    text="Sistema de diálogo não disponível",
                    success=False,
                    error_message="Dialogue manager não inicializado"
                )
                
        except Exception as e:
            self.logger.error(f"Erro no processamento single-shot: {e}")
            return SoraResponse(
                text="Erro no processamento da mensagem",
                success=False,
                error_message=str(e)
            )
    
    def _process_interactive_message(self, message: str, wait_for_response: bool, 
                                   timeout: float = None) -> Optional[SoraResponse]:
        """Processa mensagem em modo interativo."""
        try:
            # Em modo interativo, injeta a mensagem no sistema como se fosse fala
            # Isso permite que o fluxo normal de processamento seja usado
            
            # Cria dados simulados de entrada
            interaction_data = {
                'type': 'text_input',
                'message': message,
                'timestamp': time.time(),
                'source': 'controller'
            }
            
            # Adiciona à queue de processamento
            if self.system_integration:
                try:
                    self.system_integration.processing_queue.put_nowait(interaction_data)
                except queue.Full:
                    return SoraResponse(
                        text="Sistema sobrecarregado, tente novamente",
                        success=False,
                        error_message="Queue de processamento cheia"
                    )
            
            if wait_for_response:
                # Aguarda resposta
                timeout = timeout or self.sora_config.max_response_time
                return self._wait_for_response(timeout)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Erro no processamento interativo: {e}")
            return SoraResponse(
                text="Erro no processamento",
                success=False,
                error_message=str(e)
            )
    
    def _wait_for_response(self, timeout: float) -> Optional[SoraResponse]:
        """Aguarda resposta do sistema."""
        try:
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                # Verifica se há resposta na queue
                try:
                    response_data = self.response_queue.get(timeout=0.1)
                    return self._process_response_data(response_data)
                except queue.Empty:
                    continue
            
            # Timeout
            return SoraResponse(
                text="Tempo limite excedido para resposta",
                success=False,
                error_message="Timeout na geração de resposta"
            )
            
        except Exception as e:
            self.logger.error(f"Erro ao aguardar resposta: {e}")
            return SoraResponse(
                text="Erro ao processar resposta",
                success=False,
                error_message=str(e)
            )
    
    def get_status(self) -> Dict:
        """
        Retorna status atual do sistema.
        
        Returns:
            Dict: Status detalhado
        """
        try:
            status = {
                'controller_state': self.controller_state.value,
                'is_initialized': self.is_initialized,
                'is_active': self.is_active,
                'operation_mode': self.sora_config.operation_mode.value,
                'configuration': {
                    'language': self.sora_config.language,
                    'personality': self.sora_config.personality,
                    'voice_enabled': self.sora_config.voice_enabled,
                    'vision_enabled': self.sora_config.vision_enabled,
                    'animation_enabled': self.sora_config.animation_enabled
                },
                'usage_stats': self.usage_stats.copy()
            }
            
            # Adiciona status do sistema de integração se disponível
            if self.system_integration:
                system_status = self.system_integration.get_system_status()
                status['system_integration'] = system_status
            
            return status
            
        except Exception as e:
            self.logger.error(f"Erro ao obter status: {e}")
            return {'error': str(e)}
    
    def get_conversation_history(self, limit: int = 10) -> List[Dict]:
        """
        Retorna histórico de conversas.
        
        Args:
            limit: Número máximo de itens
            
        Returns:
            List[Dict]: Histórico de conversas
        """
        try:
            if self.system_integration:
                return self.system_integration.get_interaction_history(limit)
            else:
                return self.conversation_history[-limit:] if self.conversation_history else []
                
        except Exception as e:
            self.logger.error(f"Erro ao obter histórico: {e}")
            return []
    
    def update_configuration(self, config_updates: Dict) -> bool:
        """
        Atualiza configuração em tempo de execução.
        
        Args:
            config_updates: Dicionário com atualizações
            
        Returns:
            bool: True se atualização bem-sucedida
        """
        try:
            with self.controller_lock:
                # Valida atualizações
                valid_updates = self._validate_config_updates(config_updates)
                
                if not valid_updates:
                    return False
                
                # Aplica atualizações
                for key, value in valid_updates.items():
                    if hasattr(self.sora_config, key):
                        setattr(self.sora_config, key, value)
                        self.logger.info(f"Configuração atualizada: {key} = {value}")
                
                # Aplica configurações específicas
                self._apply_runtime_config_updates(valid_updates)
                
                return True
                
        except Exception as e:
            self.logger.error(f"Erro ao atualizar configuração: {e}")
            return False
    
    def set_personality(self, personality: str) -> bool:
        """
        Muda personalidade do Sora.
        
        Args:
            personality: Nova personalidade (friendly, professional, casual, empathetic)
            
        Returns:
            bool: True se mudança bem-sucedida
        """
        try:
            valid_personalities = ['friendly', 'professional', 'casual', 'empathetic', 'energetic', 'calm']
            
            if personality not in valid_personalities:
                self.logger.error(f"Personalidade inválida: {personality}")
                return False
            
            self.sora_config.personality = personality
            
            # Aplica mudanças nos componentes
            if self.system_integration and 'dialogue_manager' in self.system_integration.components:
                personality_config = self._get_personality_config(personality)
                self.system_integration.components['dialogue_manager'].personality = personality_config
            
            self.logger.info(f"Personalidade alterada para: {personality}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao alterar personalidade: {e}")
            return False
    
    def set_language(self, language: str) -> bool:
        """
        Muda idioma do Sora.
        
        Args:
            language: Novo idioma (pt-BR, en-US, es-ES)
            
        Returns:
            bool: True se mudança bem-sucedida
        """
        try:
            valid_languages = ['pt-BR', 'en-US', 'es-ES']
            
            if language not in valid_languages:
                self.logger.error(f"Idioma inválido: {language}")
                return False
            
            self.sora_config.language = language
            
            # Atualiza componentes que dependem do idioma
            if self.system_integration:
                components_to_update = [
                    'speech_recognition', 'speech_synthesis', 
                    'sentiment_analysis', 'intent_recognition'
                ]
                
                for component_name in components_to_update:
                    if component_name in self.system_integration.components:
                        component = self.system_integration.components[component_name]
                        if hasattr(component, 'change_language'):
                            component.change_language(language)
            
            self.logger.info(f"Idioma alterado para: {language}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao alterar idioma: {e}")
            return False
    
    def enable_voice(self, enabled: bool = True) -> bool:
        """
        Habilita/desabilita síntese de voz.
        
        Args:
            enabled: True para habilitar, False para desabilitar
            
        Returns:
            bool: True se mudança bem-sucedida
        """
        try:
            self.sora_config.voice_enabled = enabled
            
            # Atualiza sistema de síntese
            if self.system_integration and 'speech_synthesis' in self.system_integration.components:
                synthesis = self.system_integration.components['speech_synthesis']
                if enabled:
                    # Reativa síntese se necessário
                    pass
                else:
                    # Para reprodução atual
                    synthesis.stop_current_playback()
            
            self.logger.info(f"Síntese de voz {'habilitada' if enabled else 'desabilitada'}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao alterar configuração de voz: {e}")
            return False
    
    def enable_vision(self, enabled: bool = True) -> bool:
        """
        Habilita/desabilita processamento de visão.
        
        Args:
            enabled: True para habilitar, False para desabilitar
            
        Returns:
            bool: True se mudança bem-sucedida
        """
        try:
            self.sora_config.vision_enabled = enabled
            
            # Atualiza captura de vídeo
            if self.system_integration and 'camera' in self.system_integration.components:
                camera = self.system_integration.components['camera']
                if enabled and not camera.is_capturing:
                    camera.start_capture()
                elif not enabled and camera.is_capturing:
                    camera.stop_capture()
            
            self.logger.info(f"Processamento de visão {'habilitado' if enabled else 'desabilitado'}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao alterar configuração de visão: {e}")
            return False
    
    def get_metrics(self) -> Dict:
        """
        Retorna métricas detalhadas do sistema.
        
        Returns:
            Dict: Métricas do sistema
        """
        try:
            metrics = {
                'controller_metrics': self.usage_stats.copy(),
                'uptime': time.time() - getattr(self, 'start_time', time.time()),
                'conversation_count': len(self.conversation_history)
            }
            
            # Adiciona métricas do sistema de integração
            if self.system_integration:
                system_metrics = self.system_integration.get_system_metrics()
                metrics['system_metrics'] = system_metrics
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Erro ao obter métricas: {e}")
            return {'error': str(e)}
    
    def reset_conversation(self) -> bool:
        """
        Reinicia conversa atual.
        
        Returns:
            bool: True se reset bem-sucedido
        """
        try:
            with self.controller_lock:
                # Limpa histórico local
                self.conversation_history.clear()
                
                # Reinicia sessão no dialogue manager
                if (self.system_integration and 
                    'dialogue_manager' in self.system_integration.components):
                    
                    dialogue_manager = self.system_integration.components['dialogue_manager']
                    dialogue_manager.end_session()
                    dialogue_manager.start_session()
                
                self.logger.info("Conversa reiniciada")
                return True
                
        except Exception as e:
            self.logger.error(f"Erro ao reiniciar conversa: {e}")
            return False
    
    # Métodos de callback
    def add_message_callback(self, callback: Callable[[str], None]):
        """Adiciona callback para mensagens recebidas."""
        if callback not in self.message_callbacks:
            self.message_callbacks.append(callback)
    
    def add_response_callback(self, callback: Callable[[SoraResponse], None]):
        """Adiciona callback para respostas geradas."""
        if callback not in self.response_callbacks:
            self.response_callbacks.append(callback)
    
    def add_state_callback(self, callback: Callable[[ControllerState], None]):
        """Adiciona callback para mudanças de estado."""
        if callback not in self.state_callbacks:
            self.state_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable[[str], None]):
        """Adiciona callback para erros."""
        if callback not in self.error_callbacks:
            self.error_callbacks.append(callback)
    
    # Métodos privados auxiliares
    def _convert_to_system_config(self) -> SystemConfig:
        """Converte configuração do Sora para configuração do sistema."""
        from core.system_integration import SystemConfig, PerformanceLevel, ProcessingMode
        
        # Mapeia qualidade para nível de performance
        quality_map = {
            'low': PerformanceLevel.LOW,
            'medium': PerformanceLevel.MEDIUM,
            'high': PerformanceLevel.HIGH
        }
        
        performance_level = quality_map.get(self.sora_config.processing_quality, PerformanceLevel.MEDIUM)
        
        return SystemConfig(
            enable_vision=self.sora_config.vision_enabled,
            enable_audio=self.sora_config.voice_enabled,
            enable_nlp=True,  # Sempre habilitado
            enable_response=True,  # Sempre habilitado
            performance_level=performance_level,
            processing_mode=ProcessingMode.REAL_TIME,
            debug_mode=self.sora_config.debug_mode,
            save_interactions=self.sora_config.save_conversations
        )
    
    def _setup_system_callbacks(self):
        """Configura callbacks do sistema de integração."""
        if self.system_integration:
            # Callback para mudanças de estado do sistema
            self.system_integration.add_state_change_callback(self._on_system_state_change)
            
            # Callback para interações completadas
            self.system_integration.add_interaction_callback(self._on_interaction_completed)
            
            # Callback para erros
            self.system_integration.add_error_callback(self._on_system_error)
    
    def _apply_personality_config(self):
        """Aplica configuração de personalidade."""
        personality_config = self._get_personality_config(self.sora_config.personality)
        
        if (self.system_integration and 
            'dialogue_manager' in self.system_integration.components):
            
            self.system_integration.components['dialogue_manager'].personality = personality_config
    
    def _get_personality_config(self, personality: str) -> Dict:
        """Retorna configuração para personalidade específica."""
        personalities = {
            'friendly': {
                'friendliness': 0.9,
                'helpfulness': 0.8,
                'formality': 0.4,
                'humor': 0.6,
                'empathy': 0.7,
                'enthusiasm': 0.7
            },
            'professional': {
                'friendliness': 0.6,
                'helpfulness': 0.9,
                'formality': 0.8,
                'humor': 0.3,
                'empathy': 0.6,
                'enthusiasm': 0.5
            },
            'casual': {
                'friendliness': 0.8,
                'helpfulness': 0.7,
                'formality': 0.2,
                'humor': 0.8,
                'empathy': 0.6,
                'enthusiasm': 0.6
            },
            'empathetic': {
                'friendliness': 0.8,
                'helpfulness': 0.9,
                'formality': 0.5,
                'humor': 0.4,
                'empathy': 0.9,
                'enthusiasm': 0.5
            }
        }
        
        return personalities.get(personality, personalities['friendly'])
    
    def _verify_system_health(self) -> bool:
        """Verifica saúde do sistema."""
        try:
            if not self.system_integration:
                return False
            
            # Verifica componentes essenciais
            essential_components = ['dialogue_manager']
            
            for component_name in essential_components:
                if component_name not in self.system_integration.components:
                    self.logger.error(f"Componente essencial ausente: {component_name}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erro na verificação de saúde: {e}")
            return False
    
    def _validate_config_updates(self, updates: Dict) -> Dict:
        """Valida atualizações de configuração."""
        valid_updates = {}
        
        # Lista de configurações que podem ser atualizadas
        updatable_configs = [
            'personality', 'language', 'voice_enabled', 'vision_enabled',
            'animation_enabled', 'response_style', 'max_response_time'
        ]
        
        for key, value in updates.items():
            if key in updatable_configs:
                # Validações específicas
                if key == 'personality' and value in ['friendly', 'professional', 'casual', 'empathetic']:
                    valid_updates[key] = value
                elif key == 'language' and value in ['pt-BR', 'en-US', 'es-ES']:
                    valid_updates[key] = value
                elif key in ['voice_enabled', 'vision_enabled', 'animation_enabled'] and isinstance(value, bool):
                    valid_updates[key] = value
                elif key == 'response_style' and value in ['concise', 'balanced', 'detailed']:
                    valid_updates[key] = value
                elif key == 'max_response_time' and isinstance(value, (int, float)) and value > 0:
                    valid_updates[key] = value
        
        return valid_updates
    
    def _apply_runtime_config_updates(self, updates: Dict):
        """Aplica atualizações de configuração em tempo de execução."""
        try:
            if 'personality' in updates:
                self.set_personality(updates['personality'])
            
            if 'language' in updates:
                self.set_language(updates['language'])
            
            if 'voice_enabled' in updates:
                self.enable_voice(updates['voice_enabled'])
            
            if 'vision_enabled' in updates:
                self.enable_vision(updates['vision_enabled'])
            
        except Exception as e:
            self.logger.error(f"Erro ao aplicar atualizações: {e}")
    
    def _start_monitoring(self):
        """Inicia monitoramento em modo debug."""
        def monitoring_loop():
            while self.is_active:
                try:
                    # Atualiza estatísticas
                    self._update_usage_stats()
                    time.sleep(10)  # Atualiza a cada 10 segundos
                except Exception as e:
                    self.logger.error(f"Erro no monitoramento: {e}")
        
        monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitoring_thread.start()
    
    def _update_usage_stats(self):
        """Atualiza estatísticas de uso."""
        try:
            if self.system_integration:
                system_metrics = self.system_integration.get_system_metrics()
                
                self.usage_stats.update({
                    'total_interactions': system_metrics.get('total_interactions', 0),
                    'successful_interactions': system_metrics.get('successful_interactions', 0),
                    'average_response_time': system_metrics.get('average_response_time', 0.0)
                })
        
        except Exception as e:
            self.logger.error(f"Erro ao atualizar estatísticas: {e}")
    
    # Callbacks do sistema
    def _on_system_state_change(self, old_state, new_state):
        """Callback para mudanças de estado do sistema."""
        self.logger.debug(f"Estado do sistema mudou: {old_state} -> {new_state}")
    
    def _on_interaction_completed(self, interaction: InteractionData):
        """Callback para interações completadas."""
        try:
            # Converte para formato simplificado
            sora_response = self._convert_interaction_to_sora_response(interaction)
            
            # Adiciona ao histórico
            self.conversation_history.append({
                'timestamp': interaction.timestamp,
                'user_input': interaction.text_input,
                'sora_response': sora_response.text,
                'quality': interaction.quality_scores.get('overall', 0.0)
            })
            
            # Executa callbacks de resposta
            self._execute_response_callbacks(sora_response)
            
        except Exception as e:
            self.logger.error(f"Erro ao processar interação completada: {e}")
    
    def _on_system_error(self, error: str):
        """Callback para erros do sistema."""
        self.logger.error(f"Erro do sistema: {error}")
        self._execute_error_callbacks(error)
    
    def _convert_interaction_to_sora_response(self, interaction: InteractionData) -> SoraResponse:
        """Converte InteractionData para SoraResponse."""
        try:
            response_text = ""
            if interaction.llm_response:
                response_text = getattr(interaction.llm_response, 'text', '')
            
            return SoraResponse(
                text=response_text,
                audio_available=interaction.speech_synthesis is not None,
                animation_available=interaction.animation_sequence is not None,
                response_id=interaction.interaction_id,
                timestamp=interaction.timestamp,
                processing_time=interaction.duration,
                emotion_detected=getattr(interaction.emotion_data, 'primary_emotion', 'neutral') if interaction.emotion_data else 'neutral',
                intent_detected=interaction.intent_data.primary_intent.intent if interaction.intent_data else 'unknown',
                confidence=interaction.quality_scores.get('overall', 0.0),
                interaction_quality=interaction.quality_scores.get('overall', 0.0),
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"Erro ao converter interação: {e}")
            return SoraResponse(
                text="Erro na conversão de resposta",
                success=False,
                error_message=str(e)
            )
    
    def _convert_dialogue_response_to_sora_response(self, dialogue_response: Dict) -> SoraResponse:
        """Converte resposta do dialogue manager para SoraResponse."""
        try:
            return SoraResponse(
                text=dialogue_response.get('response_text', ''),
                response_id=dialogue_response.get('turn_id', ''),
                timestamp=time.time(),
                emotion_detected=dialogue_response.get('metadata', {}).get('emotion_detected', 'neutral'),
                intent_detected=dialogue_response.get('metadata', {}).get('intent_detected', 'unknown'),
                confidence=dialogue_response.get('confidence', 0.0),
                user_mood=dialogue_response.get('user_mood', 'neutral'),
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"Erro ao converter resposta do diálogo: {e}")
            return SoraResponse(
                text="Erro na conversão",
                success=False,
                error_message=str(e)
            )
    
    # Execução de callbacks
    def _execute_message_callbacks(self, message: str):
        """Executa callbacks de mensagem."""
        for callback in self.message_callbacks:
            try:
                callback(message)
            except Exception as e:
                self.logger.error(f"Erro ao executar callback de mensagem: {e}")
    
    def _execute_response_callbacks(self, response: SoraResponse):
        """Executa callbacks de resposta."""
        for callback in self.response_callbacks:
            try:
                callback(response)
            except Exception as e:
                self.logger.error(f"Erro ao executar callback de resposta: {e}")
    
    def _execute_state_callbacks(self, state: ControllerState):
        """Executa callbacks de estado."""
        for callback in self.state_callbacks:
            try:
                callback(state)
            except Exception as e:
                self.logger.error(f"Erro ao executar callback de estado: {e}")
    
    def _execute_error_callbacks(self, error: str):
        """Executa callbacks de erro."""
        for callback in self.error_callbacks:
            try:
                callback(error)
            except Exception as e:
                self.logger.error(f"Erro ao executar callback de erro: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
    
    def __del__(self):
        """Destructor para limpeza automática."""
        if self.is_active:
            self.stop()


# Função de conveniência para criar instância do Sora
def create_sora(config: Optional[SoraConfig] = None) -> MainController:
    """
    Cria instância do robô Sora com configuração simplificada.
    
    Args:
        config: Configuração opcional
        
    Returns:
        MainController: Instância do controlador principal
    """
    return MainController(config)


# Exemplo de uso básico
if __name__ == "__main__":
    # Configuração personalizada
    config = SoraConfig(
        personality="friendly",
        language="pt-BR",
        voice_enabled=True,
        vision_enabled=True
    )
    
    # Cria e inicia o Sora
    sora = create_sora(config)
    
    try:
        if sora.start():
            print("Sora iniciado com sucesso!")
            
            # Exemplo de interação
            response = sora.send_message("Olá, como você está?")
            if response and response.success:
                print(f"Sora: {response.text}")
            
        else:
            print("Falha ao iniciar Sora")
            
    finally:
        sora.stop()