# sora_robot/api/api_interface.py

import asyncio
import json
import time
import uuid
from typing import Dict, List, Optional, Any
from dataclasses import asdict
import base64
import io
from pathlib import Path

# FastAPI e dependências
try:
    from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from fastapi.responses import JSONResponse, StreamingResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, Field
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    FastAPI = None

# WebSocket manager
from collections import defaultdict
import threading

from utils.logger import get_logger
from core.main_controller import MainController, SoraConfig, SoraResponse, OperationMode, ControllerState
import config

# Modelos Pydantic para API
class MessageRequest(BaseModel):
    """Modelo para requisição de mensagem."""
    message: str = Field(..., description="Mensagem de texto para o Sora")
    wait_for_response: bool = Field(True, description="Aguardar resposta")
    timeout: Optional[float] = Field(None, description="Timeout em segundos")
    user_id: Optional[str] = Field(None, description="ID do usuário")

class ConfigUpdateRequest(BaseModel):
    """Modelo para atualização de configuração."""
    personality: Optional[str] = Field(None, description="Nova personalidade")
    language: Optional[str] = Field(None, description="Novo idioma")
    voice_enabled: Optional[bool] = Field(None, description="Habilitar/desabilitar voz")
    vision_enabled: Optional[bool] = Field(None, description="Habilitar/desabilitar visão")
    animation_enabled: Optional[bool] = Field(None, description="Habilitar/desabilitar animação")
    response_style: Optional[str] = Field(None, description="Estilo de resposta")
    max_response_time: Optional[float] = Field(None, description="Tempo máximo de resposta")

class SoraConfigRequest(BaseModel):
    """Modelo para configuração inicial do Sora."""
    name: str = Field("Sora", description="Nome do robô")
    language: str = Field("pt-BR", description="Idioma")
    personality: str = Field("friendly", description="Personalidade")
    operation_mode: str = Field("interactive", description="Modo de operação")
    voice_enabled: bool = Field(True, description="Voz habilitada")
    vision_enabled: bool = Field(True, description="Visão habilitada")
    animation_enabled: bool = Field(True, description="Animação habilitada")
    debug_mode: bool = Field(False, description="Modo debug")

class AudioUploadRequest(BaseModel):
    """Modelo para upload de áudio."""
    audio_data: str = Field(..., description="Dados de áudio em base64")
    format: str = Field("wav", description="Formato do áudio")
    sample_rate: int = Field(44100, description="Taxa de amostragem")

class WebSocketMessage(BaseModel):
    """Modelo para mensagens WebSocket."""
    type: str = Field(..., description="Tipo da mensagem")
    data: Dict[str, Any] = Field(..., description="Dados da mensagem")
    timestamp: float = Field(default_factory=time.time, description="Timestamp")

class APIResponse(BaseModel):
    """Modelo base para respostas da API."""
    success: bool = Field(..., description="Sucesso da operação")
    message: Optional[str] = Field(None, description="Mensagem da resposta")
    data: Optional[Dict[str, Any]] = Field(None, description="Dados da resposta")
    timestamp: float = Field(default_factory=time.time, description="Timestamp")

class ConnectionManager:
    """Gerenciador de conexões WebSocket."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_connections: Dict[str, List[str]] = defaultdict(list)
        self.connection_lock = threading.Lock()
    
    async def connect(self, websocket: WebSocket, connection_id: str, user_id: Optional[str] = None):
        """Conecta um novo WebSocket."""
        await websocket.accept()
        
        with self.connection_lock:
            self.active_connections[connection_id] = websocket
            if user_id:
                self.user_connections[user_id].append(connection_id)
    
    def disconnect(self, connection_id: str, user_id: Optional[str] = None):
        """Desconecta WebSocket."""
        with self.connection_lock:
            if connection_id in self.active_connections:
                del self.active_connections[connection_id]
            
            if user_id and user_id in self.user_connections:
                if connection_id in self.user_connections[user_id]:
                    self.user_connections[user_id].remove(connection_id)
                if not self.user_connections[user_id]:
                    del self.user_connections[user_id]
    
    async def send_personal_message(self, message: str, connection_id: str):
        """Envia mensagem para conexão específica."""
        if connection_id in self.active_connections:
            try:
                await self.active_connections[connection_id].send_text(message)
            except Exception as e:
                # Remove conexão inválida
                self.disconnect(connection_id)
    
    async def send_to_user(self, message: str, user_id: str):
        """Envia mensagem para todas as conexões de um usuário."""
        if user_id in self.user_connections:
            for connection_id in self.user_connections[user_id][:]:  # Cópia para evitar modificação durante iteração
                await self.send_personal_message(message, connection_id)
    
    async def broadcast(self, message: str):
        """Envia mensagem para todas as conexões."""
        for connection_id in list(self.active_connections.keys()):
            await self.send_personal_message(message, connection_id)

class SoraAPI:
    """
    Interface API REST para o robô Sora.
    Fornece endpoints para todas as funcionalidades do sistema.
    """
    
    def __init__(self, sora_controller: Optional[MainController] = None):
        """
        Inicializa a API do Sora.
        
        Args:
            sora_controller: Controlador do Sora (será criado se None)
        """
        self.logger = get_logger(__name__)
        
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI não está disponível. Execute: pip install fastapi uvicorn")
        
        # Controlador principal
        self.sora_controller = sora_controller
        
        # Aplicação FastAPI
        self.app = FastAPI(
            title="Sora Robot API",
            description="API REST para interação com o robô assistente Sora",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Configurações
        self.api_config = {
            'host': getattr(config, 'API_HOST', '0.0.0.0'),
            'port': getattr(config, 'API_PORT', 8000),
            'reload': getattr(config, 'API_RELOAD', False),
            'cors_origins': getattr(config, 'CORS_ORIGINS', ["*"]),
            'api_key_required': getattr(config, 'API_KEY_REQUIRED', False),
            'api_key': getattr(config, 'API_KEY', 'sora-api-key-2024')
        }
        
        # Gerenciador de WebSocket
        self.connection_manager = ConnectionManager()
        
        # Sistema de autenticação
        self.security = HTTPBearer(auto_error=False) if self.api_config['api_key_required'] else None
        
        # Sessões ativas
        self.active_sessions = {}
        
        # Configurar middleware
        self._setup_middleware()
        
        # Configurar rotas
        self._setup_routes()
        
        # Callbacks do sistema
        self._setup_system_callbacks()
        
        self.logger.info("SoraAPI inicializada")
    
    def _setup_middleware(self):
        """Configura middleware da aplicação."""
        # CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.api_config['cors_origins'],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Middleware personalizado para logging
        @self.app.middleware("http")
        async def log_requests(request, call_next):
            start_time = time.time()
            response = await call_next(request)
            process_time = time.time() - start_time
            
            self.logger.info(
                f"{request.method} {request.url.path} - "
                f"Status: {response.status_code} - "
                f"Time: {process_time:.3f}s"
            )
            
            return response
    
    def _verify_api_key(self, credentials: HTTPAuthorizationCredentials = Depends(lambda: None)):
        """Verifica chave da API se autenticação estiver habilitada."""
        if not self.api_config['api_key_required']:
            return True
        
        if not credentials or credentials.credentials != self.api_config['api_key']:
            raise HTTPException(
                status_code=401,
                detail="Chave da API inválida",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return True
    
    def _setup_routes(self):
        """Configura todas as rotas da API."""
        
        # Rota de status
        @self.app.get("/", response_model=APIResponse, tags=["Status"])
        async def root():
            """Endpoint raiz - status da API."""
            return APIResponse(
                success=True,
                message="Sora Robot API está funcionando",
                data={
                    "version": "1.0.0",
                    "status": "online",
                    "timestamp": time.time()
                }
            )
        
        # Status do sistema
        @self.app.get("/status", response_model=APIResponse, tags=["Status"])
        async def get_status(authenticated: bool = Depends(self._verify_api_key)):
            """Obtém status detalhado do sistema."""
            try:
                if not self.sora_controller:
                    return APIResponse(
                        success=False,
                        message="Controlador não inicializado",
                        data={"controller_state": "uninitialized"}
                    )
                
                status = self.sora_controller.get_status()
                return APIResponse(
                    success=True,
                    message="Status obtido com sucesso",
                    data=status
                )
                
            except Exception as e:
                self.logger.error(f"Erro ao obter status: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Inicializar sistema
        @self.app.post("/initialize", response_model=APIResponse, tags=["Sistema"])
        async def initialize_system(
            config_request: Optional[SoraConfigRequest] = None,
            authenticated: bool = Depends(self._verify_api_key)
        ):
            """Inicializa o sistema Sora."""
            try:
                # Cria controlador se não existir
                if not self.sora_controller:
                    sora_config = SoraConfig()
                    
                    if config_request:
                        sora_config.name = config_request.name
                        sora_config.language = config_request.language
                        sora_config.personality = config_request.personality
                        sora_config.operation_mode = OperationMode(config_request.operation_mode)
                        sora_config.voice_enabled = config_request.voice_enabled
                        sora_config.vision_enabled = config_request.vision_enabled
                        sora_config.animation_enabled = config_request.animation_enabled
                        sora_config.debug_mode = config_request.debug_mode
                    
                    self.sora_controller = MainController(sora_config)
                    self._setup_system_callbacks()
                
                # Inicializa sistema
                success = self.sora_controller.initialize()
                
                return APIResponse(
                    success=success,
                    message="Sistema inicializado com sucesso" if success else "Falha na inicialização",
                    data={"initialized": success}
                )
                
            except Exception as e:
                self.logger.error(f"Erro na inicialização: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Iniciar sistema
        @self.app.post("/start", response_model=APIResponse, tags=["Sistema"])
        async def start_system(authenticated: bool = Depends(self._verify_api_key)):
            """Inicia o sistema Sora."""
            try:
                if not self.sora_controller:
                    raise HTTPException(status_code=400, detail="Sistema não inicializado")
                
                success = self.sora_controller.start()
                
                return APIResponse(
                    success=success,
                    message="Sistema iniciado com sucesso" if success else "Falha ao iniciar sistema",
                    data={"started": success}
                )
                
            except Exception as e:
                self.logger.error(f"Erro ao iniciar sistema: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Parar sistema
        @self.app.post("/stop", response_model=APIResponse, tags=["Sistema"])
        async def stop_system(authenticated: bool = Depends(self._verify_api_key)):
            """Para o sistema Sora."""
            try:
                if not self.sora_controller:
                    return APIResponse(success=True, message="Sistema já parado")
                
                success = self.sora_controller.stop()
                
                return APIResponse(
                    success=success,
                    message="Sistema parado com sucesso" if success else "Falha ao parar sistema",
                    data={"stopped": success}
                )
                
            except Exception as e:
                self.logger.error(f"Erro ao parar sistema: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Enviar mensagem
        @self.app.post("/message", response_model=APIResponse, tags=["Interação"])
        async def send_message(
            request: MessageRequest,
            authenticated: bool = Depends(self._verify_api_key)
        ):
            """Envia mensagem para o Sora."""
            try:
                if not self.sora_controller or not self.sora_controller.is_active:
                    raise HTTPException(status_code=400, detail="Sistema não está ativo")
                
                response = self.sora_controller.send_message(
                    request.message,
                    request.wait_for_response,
                    request.timeout
                )
                
                if response and response.success:
                    # Converte SoraResponse para dict
                    response_data = asdict(response)
                    
                    return APIResponse(
                        success=True,
                        message="Mensagem processada com sucesso",
                        data=response_data
                    )
                else:
                    error_msg = response.error_message if response else "Erro desconhecido"
                    return APIResponse(
                        success=False,
                        message=f"Erro ao processar mensagem: {error_msg}",
                        data={"error": error_msg}
                    )
                
            except Exception as e:
                self.logger.error(f"Erro ao processar mensagem: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Upload de áudio
        @self.app.post("/audio", response_model=APIResponse, tags=["Interação"])
        async def upload_audio(
            file: UploadFile = File(...),
            authenticated: bool = Depends(self._verify_api_key)
        ):
            """Faz upload de arquivo de áudio para processamento."""
            try:
                if not self.sora_controller or not self.sora_controller.is_active:
                    raise HTTPException(status_code=400, detail="Sistema não está ativo")
                
                # Lê arquivo de áudio
                audio_data = await file.read()
                
                # Processa áudio (implementação simplificada)
                # Em produção, integraria com o sistema de reconhecimento de fala
                
                return APIResponse(
                    success=True,
                    message="Áudio processado com sucesso",
                    data={
                        "filename": file.filename,
                        "size": len(audio_data),
                        "content_type": file.content_type
                    }
                )
                
            except Exception as e:
                self.logger.error(f"Erro ao processar áudio: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Configuração
        @self.app.put("/config", response_model=APIResponse, tags=["Configuração"])
        async def update_config(
            config_update: ConfigUpdateRequest,
            authenticated: bool = Depends(self._verify_api_key)
        ):
            """Atualiza configuração do sistema."""
            try:
                if not self.sora_controller:
                    raise HTTPException(status_code=400, detail="Sistema não inicializado")
                
                # Converte para dict removendo valores None
                updates = {k: v for k, v in config_update.dict().items() if v is not None}
                
                success = self.sora_controller.update_configuration(updates)
                
                return APIResponse(
                    success=success,
                    message="Configuração atualizada com sucesso" if success else "Falha na atualização",
                    data={"updated_fields": list(updates.keys())}
                )
                
            except Exception as e:
                self.logger.error(f"Erro ao atualizar configuração: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Obter configuração
        @self.app.get("/config", response_model=APIResponse, tags=["Configuração"])
        async def get_config(authenticated: bool = Depends(self._verify_api_key)):
            """Obtém configuração atual do sistema."""
            try:
                if not self.sora_controller:
                    raise HTTPException(status_code=400, detail="Sistema não inicializado")
                
                config_data = {
                    "name": self.sora_controller.sora_config.name,
                    "language": self.sora_controller.sora_config.language,
                    "personality": self.sora_controller.sora_config.personality,
                    "operation_mode": self.sora_controller.sora_config.operation_mode.value,
                    "voice_enabled": self.sora_controller.sora_config.voice_enabled,
                    "vision_enabled": self.sora_controller.sora_config.vision_enabled,
                    "animation_enabled": self.sora_controller.sora_config.animation_enabled,
                    "debug_mode": self.sora_controller.sora_config.debug_mode
                }
                
                return APIResponse(
                    success=True,
                    message="Configuração obtida com sucesso",
                    data=config_data
                )
                
            except Exception as e:
                self.logger.error(f"Erro ao obter configuração: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Histórico de conversas
        @self.app.get("/history", response_model=APIResponse, tags=["Histórico"])
        async def get_conversation_history(
            limit: int = 10,
            authenticated: bool = Depends(self._verify_api_key)
        ):
            """Obtém histórico de conversas."""
            try:
                if not self.sora_controller:
                    raise HTTPException(status_code=400, detail="Sistema não inicializado")
                
                history = self.sora_controller.get_conversation_history(limit)
                
                return APIResponse(
                    success=True,
                    message="Histórico obtido com sucesso",
                    data={"history": history, "count": len(history)}
                )
                
            except Exception as e:
                self.logger.error(f"Erro ao obter histórico: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Limpar histórico
        @self.app.delete("/history", response_model=APIResponse, tags=["Histórico"])
        async def clear_history(authenticated: bool = Depends(self._verify_api_key)):
            """Limpa histórico de conversas."""
            try:
                if not self.sora_controller:
                    raise HTTPException(status_code=400, detail="Sistema não inicializado")
                
                success = self.sora_controller.reset_conversation()
                
                return APIResponse(
                    success=success,
                    message="Histórico limpo com sucesso" if success else "Falha ao limpar histórico"
                )
                
            except Exception as e:
                self.logger.error(f"Erro ao limpar histórico: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Métricas
        @self.app.get("/metrics", response_model=APIResponse, tags=["Monitoramento"])
        async def get_metrics(authenticated: bool = Depends(self._verify_api_key)):
            """Obtém métricas do sistema."""
            try:
                if not self.sora_controller:
                    raise HTTPException(status_code=400, detail="Sistema não inicializado")
                
                metrics = self.sora_controller.get_metrics()
                
                return APIResponse(
                    success=True,
                    message="Métricas obtidas com sucesso",
                    data=metrics
                )
                
            except Exception as e:
                self.logger.error(f"Erro ao obter métricas: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # WebSocket para comunicação em tempo real
        @self.app.websocket("/ws/{client_id}")
        async def websocket_endpoint(websocket: WebSocket, client_id: str):
            """Endpoint WebSocket para comunicação em tempo real."""
            connection_id = str(uuid.uuid4())
            
            try:
                await self.connection_manager.connect(websocket, connection_id, client_id)
                self.logger.info(f"WebSocket conectado: {connection_id} (usuário: {client_id})")
                
                # Envia mensagem de boas-vindas
                await websocket.send_text(json.dumps({
                    "type": "connection",
                    "message": "Conectado ao Sora",
                    "connection_id": connection_id,
                    "timestamp": time.time()
                }))
                
                while True:
                    # Recebe mensagem do cliente
                    data = await websocket.receive_text()
                    message_data = json.loads(data)
                    
                    # Processa mensagem
                    await self._handle_websocket_message(websocket, connection_id, client_id, message_data)
                    
            except WebSocketDisconnect:
                self.connection_manager.disconnect(connection_id, client_id)
                self.logger.info(f"WebSocket desconectado: {connection_id}")
            except Exception as e:
                self.logger.error(f"Erro no WebSocket: {e}")
                self.connection_manager.disconnect(connection_id, client_id)
        
        # Health check
        @self.app.get("/health", response_model=APIResponse, tags=["Status"])
        async def health_check():
            """Health check para monitoramento."""
            try:
                health_data = {
                    "api_status": "healthy",
                    "timestamp": time.time(),
                    "uptime": time.time() - getattr(self, 'start_time', time.time())
                }
                
                if self.sora_controller:
                    status = self.sora_controller.get_status()
                    health_data["sora_status"] = status.get("controller_state", "unknown")
                    health_data["sora_active"] = status.get("is_active", False)
                
                return APIResponse(
                    success=True,
                    message="Sistema saudável",
                    data=health_data
                )
                
            except Exception as e:
                self.logger.error(f"Erro no health check: {e}")
                return APIResponse(
                    success=False,
                    message="Erro no sistema",
                    data={"error": str(e)}
                )
    
    async def _handle_websocket_message(self, websocket: WebSocket, connection_id: str, 
                                       client_id: str, message_data: Dict):
        """Processa mensagem recebida via WebSocket."""
        try:
            message_type = message_data.get("type", "message")
            
            if message_type == "message":
                # Mensagem de texto
                text = message_data.get("text", "")
                if text and self.sora_controller and self.sora_controller.is_active:
                    
                    response = self.sora_controller.send_message(text, wait_for_response=True)
                    
                    if response:
                        await websocket.send_text(json.dumps({
                            "type": "response",
                            "data": asdict(response),
                            "timestamp": time.time()
                        }))
                    else:
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "message": "Erro ao processar mensagem",
                            "timestamp": time.time()
                        }))
                        
            elif message_type == "ping":
                # Ping/pong para manter conexão
                await websocket.send_text(json.dumps({
                    "type": "pong",
                    "timestamp": time.time()
                }))
                
            elif message_type == "status":
                # Requisição de status
                if self.sora_controller:
                    status = self.sora_controller.get_status()
                    await websocket.send_text(json.dumps({
                        "type": "status",
                        "data": status,
                        "timestamp": time.time()
                    }))
        
        except Exception as e:
            self.logger.error(f"Erro ao processar mensagem WebSocket: {e}")
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": str(e),
                "timestamp": time.time()
            }))
    
    def _setup_system_callbacks(self):
        """Configura callbacks do sistema Sora."""
        if not self.sora_controller:
            return
        
        # Callback para respostas
        def on_response(response: SoraResponse):
            try:
                # Envia resposta via WebSocket para todos os clientes conectados
                message = json.dumps({
                    "type": "system_response",
                    "data": asdict(response),
                    "timestamp": time.time()
                })
                
                asyncio.create_task(self.connection_manager.broadcast(message))
                
            except Exception as e:
                self.logger.error(f"Erro ao enviar resposta via WebSocket: {e}")
        
        # Callback para mudanças de estado
        def on_state_change(state: ControllerState):
            try:
                message = json.dumps({
                    "type": "state_change",
                    "data": {"state": state.value},
                    "timestamp": time.time()
                })
                
                asyncio.create_task(self.connection_manager.broadcast(message))
                
            except Exception as e:
                self.logger.error(f"Erro ao enviar mudança de estado: {e}")
        
        # Callback para erros
        def on_error(error: str):
            try:
                message = json.dumps({
                    "type": "error",
                    "data": {"error": error},
                    "timestamp": time.time()
                })
                
                asyncio.create_task(self.connection_manager.broadcast(message))
                
            except Exception as e:
                self.logger.error(f"Erro ao enviar erro via WebSocket: {e}")
        
        # Adiciona callbacks
        self.sora_controller.add_response_callback(on_response)
        self.sora_controller.add_state_callback(on_state_change)
        self.sora_controller.add_error_callback(on_error)
    
    def run(self, **kwargs):
        """
        Executa o servidor da API.
        
        Args:
            **kwargs: Argumentos para uvicorn.run()
        """
        try:
            self.start_time = time.time()
            
            # Configurações padrão
            run_config = {
                'app': self.app,
                'host': self.api_config['host'],
                'port': self.api_config['port'],
                'reload': self.api_config['reload'],
                'log_level': 'info'
            }
            
            # Sobrescreve com argumentos fornecidos
            run_config.update(kwargs)
            
            self.logger.info(f"Iniciando servidor API em {run_config['host']}:{run_config['port']}")
            self.logger.info(f"Documentação disponível em: http://{run_config['host']}:{run_config['port']}/docs")
            
            uvicorn.run(**run_config)
            
        except Exception as e:
            self.logger.error(f"Erro ao executar servidor: {e}")
            raise
    
    def get_app(self):
        """Retorna aplicação FastAPI para uso com servidores ASGI."""
        return self.app


# Função de conveniência para criar API
def create_api(sora_controller: Optional[MainController] = None) -> SoraAPI:
    """
    Cria instância da API do Sora.
    
    Args:
        sora_controller: Controlador do Sora (opcional)
        
    Returns:
        SoraAPI: Instância da API
    """
    return SoraAPI(sora_controller)


# Servidor standalone
class SoraAPIServer:
    """Servidor standalone da API do Sora."""
    
    def __init__(self, sora_config: Optional[SoraConfig] = None, api_config: Optional[Dict] = None):
        """
        Inicializa servidor standalone.
        
        Args:
            sora_config: Configuração do Sora
            api_config: Configuração da API
        """
        self.logger = get_logger(__name__)
        
        # Cria controlador do Sora
        self.sora_controller = MainController(sora_config)
        
        # Cria API
        self.api = SoraAPI(self.sora_controller)
        
        # Aplica configurações da API se fornecidas
        if api_config:
            self.api.api_config.update(api_config)
    
    def start(self, auto_initialize_sora: bool = True):
        """
        Inicia servidor completo.
        
        Args:
            auto_initialize_sora: Se deve inicializar o Sora automaticamente
        """
        try:
            self.logger.info("Iniciando servidor Sora completo...")
            
            if auto_initialize_sora:
                # Inicializa e inicia o Sora
                if self.sora_controller.initialize():
                    if self.sora_controller.start():
                        self.logger.info("Sistema Sora iniciado com sucesso")
                    else:
                        self.logger.error("Falha ao iniciar sistema Sora")
                else:
                    self.logger.error("Falha ao inicializar sistema Sora")
            
            # Inicia servidor da API
            self.api.run()
            
        except KeyboardInterrupt:
            self.logger.info("Parando servidor...")
            self.stop()
        except Exception as e:
            self.logger.error(f"Erro no servidor: {e}")
            raise
    
    def stop(self):
        """Para servidor completo."""
        try:
            if self.sora_controller:
                self.sora_controller.stop()
            self.logger.info("Servidor parado")
        except Exception as e:
            self.logger.error(f"Erro ao parar servidor: {e}")


# Exemplo de uso
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Servidor API do Sora")
    parser.add_argument("--host", default="0.0.0.0", help="Host do servidor")
    parser.add_argument("--port", type=int, default=8000, help="Porta do servidor")
    parser.add_argument("--reload", action="store_true", help="Reload automático")
    parser.add_argument("--no-sora", action="store_true", help="Não inicializar Sora automaticamente")
    
    args = parser.parse_args()
    
    # Configuração personalizada
    sora_config = SoraConfig(
        personality="friendly",
        language="pt-BR",
        voice_enabled=True,
        vision_enabled=True,
        debug_mode=True
    )
    
    api_config = {
        'host': args.host,
        'port': args.port,
        'reload': args.reload
    }
    
    # Cria e inicia servidor
    server = SoraAPIServer(sora_config, api_config)
    server.start(auto_initialize_sora=not args.no_sora)