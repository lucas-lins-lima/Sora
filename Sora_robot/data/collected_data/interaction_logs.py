# =========================================================================
# SORA ROBOT - INTERACTION LOGGING SYSTEM
# Sistema avançado de logging e análise de interações usuário-Sora
# =========================================================================

import json
import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import sqlite3
import threading
from contextlib import contextmanager

from utils.logger import get_logger
from utils.helpers import validate_input, ensure_directory_exists
from utils.constants import SYSTEM_CONFIG

# =========================================================================
# CONFIGURAÇÕES E ENUMS
# =========================================================================

class InteractionType(Enum):
    """Tipos de interação possíveis"""
    VOICE_INPUT = "voice_input"
    TEXT_INPUT = "text_input"
    EMOTION_DETECTED = "emotion_detected"
    INTENT_RECOGNIZED = "intent_recognized"
    RESPONSE_GENERATED = "response_generated"
    VOICE_OUTPUT = "voice_output"
    AVATAR_ANIMATION = "avatar_animation"
    ERROR_OCCURRED = "error_occurred"
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    USER_FEEDBACK = "user_feedback"
    SYSTEM_EVENT = "system_event"

class InteractionStatus(Enum):
    """Status da interação"""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"
    PROCESSING = "processing"
    TIMEOUT = "timeout"

class SentimentCategory(Enum):
    """Categorias de sentimento"""
    VERY_POSITIVE = "very_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very_negative"

# =========================================================================
# ESTRUTURAS DE DADOS
# =========================================================================

@dataclass
class InteractionMetrics:
    """Métricas de performance da interação"""
    processing_time_ms: float
    response_time_ms: float
    confidence_score: float
    accuracy_score: Optional[float] = None
    tokens_used: Optional[int] = None
    api_calls_count: int = 0
    cache_hit: bool = False
    gpu_usage_percent: Optional[float] = None
    memory_usage_mb: Optional[float] = None

@dataclass
class UserContext:
    """Contexto do usuário durante a interação"""
    user_id: Optional[str] = None
    session_id: str = ""
    device_type: str = "unknown"
    location: Optional[str] = None
    language: str = "pt-BR"
    personality_preference: str = "friendly"
    previous_interactions_count: int = 0
    user_satisfaction_score: Optional[float] = None

@dataclass
class EmotionData:
    """Dados de emoção detectada"""
    primary_emotion: str
    emotion_confidence: float
    emotion_distribution: Dict[str, float]
    facial_landmarks_count: Optional[int] = None
    head_pose: Optional[Dict[str, float]] = None

@dataclass
class InteractionRecord:
    """Registro completo de uma interação"""
    # Identificação
    interaction_id: str
    timestamp: datetime
    interaction_type: InteractionType
    status: InteractionStatus
    
    # Conteúdo
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    
    # Análise
    sentiment_score: Optional[float] = None
    sentiment_category: Optional[SentimentCategory] = None
    intent: Optional[str] = None
    intent_confidence: Optional[float] = None
    
    # Contexto
    user_context: Optional[UserContext] = None
    emotion_data: Optional[EmotionData] = None
    
    # Métricas
    metrics: Optional[InteractionMetrics] = None
    
    # Metadados
    component: str = "unknown"
    version: str = "1.0.0"
    environment: str = "production"
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []

# =========================================================================
# SISTEMA DE LOGGING DE INTERAÇÕES
# =========================================================================

class InteractionLogger:
    """Sistema avançado de logging de interações"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.logger = get_logger(__name__)
        self.config = config or SYSTEM_CONFIG.get("interaction_logging", {})
        
        # Configurações
        self.data_dir = Path(self.config.get("data_directory", "data/collected_data"))
        self.db_path = self.data_dir / "interactions.db"
        self.json_log_path = self.data_dir / "interactions.jsonl"
        self.enable_db_storage = self.config.get("enable_database_storage", True)
        self.enable_file_storage = self.config.get("enable_file_storage", True)
        self.max_file_size_mb = self.config.get("max_file_size_mb", 100)
        self.retention_days = self.config.get("retention_days", 30)
        
        # Cache de interações em memória
        self._interaction_cache: List[InteractionRecord] = []
        self._cache_lock = threading.Lock()
        self._cache_max_size = self.config.get("cache_max_size", 1000)
        
        # Estatísticas
        self._stats = {
            "total_interactions": 0,
            "interactions_by_type": {},
            "interactions_by_status": {},
            "average_response_time": 0.0,
            "total_processing_time": 0.0
        }
        
        # Inicialização
        self._initialize_storage()
        self.logger.info("InteractionLogger inicializado")
    
    def _initialize_storage(self):
        """Inicializa sistemas de armazenamento"""
        try:
            # Criar diretórios
            ensure_directory_exists(self.data_dir)
            
            # Inicializar banco de dados
            if self.enable_db_storage:
                self._initialize_database()
            
            # Verificar arquivo de log
            if self.enable_file_storage:
                self._initialize_file_storage()
                
        except Exception as e:
            self.logger.error(f"Erro ao inicializar storage: {e}")
            raise
    
    def _initialize_database(self):
        """Inicializa banco de dados SQLite"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS interactions (
                        interaction_id TEXT PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        interaction_type TEXT NOT NULL,
                        status TEXT NOT NULL,
                        component TEXT,
                        user_id TEXT,
                        session_id TEXT,
                        input_data TEXT,
                        output_data TEXT,
                        sentiment_score REAL,
                        sentiment_category TEXT,
                        intent TEXT,
                        intent_confidence REAL,
                        processing_time_ms REAL,
                        response_time_ms REAL,
                        confidence_score REAL,
                        tokens_used INTEGER,
                        errors TEXT,
                        raw_data TEXT
                    )
                """)
                
                # Índices para performance
                conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON interactions(timestamp)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_type ON interactions(interaction_type)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_user ON interactions(user_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_session ON interactions(session_id)")
                
                conn.commit()
                self.logger.info("Banco de dados de interações inicializado")
                
        except Exception as e:
            self.logger.error(f"Erro ao inicializar banco de dados: {e}")
            raise
    
    def _initialize_file_storage(self):
        """Inicializa armazenamento em arquivo"""
        try:
            if not self.json_log_path.exists():
                self.json_log_path.touch()
                self.logger.info(f"Arquivo de log criado: {self.json_log_path}")
                
        except Exception as e:
            self.logger.error(f"Erro ao inicializar arquivo de log: {e}")
            raise
    
    @contextmanager
    def _get_db_connection(self):
        """Context manager para conexão com banco"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            yield conn
        finally:
            if conn:
                conn.close()
    
    def log_interaction(self, record: InteractionRecord) -> bool:
        """
        Registra uma interação completa
        
        Args:
            record: Registro da interação
            
        Returns:
            bool: True se registrado com sucesso
        """
        try:
            # Validar dados
            if not self._validate_record(record):
                return False
            
            # Adicionar ao cache
            self._add_to_cache(record)
            
            # Salvar no banco de dados
            if self.enable_db_storage:
                self._save_to_database(record)
            
            # Salvar em arquivo
            if self.enable_file_storage:
                self._save_to_file(record)
            
            # Atualizar estatísticas
            self._update_stats(record)
            
            self.logger.debug(f"Interação registrada: {record.interaction_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao registrar interação: {e}")
            return False
    
    def _validate_record(self, record: InteractionRecord) -> bool:
        """Valida um registro de interação"""
        try:
            if not record.interaction_id:
                self.logger.error("ID da interação é obrigatório")
                return False
            
            if not isinstance(record.timestamp, datetime):
                self.logger.error("Timestamp inválido")
                return False
            
            if not isinstance(record.interaction_type, InteractionType):
                self.logger.error("Tipo de interação inválido")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erro na validação: {e}")
            return False
    
    def _add_to_cache(self, record: InteractionRecord):
        """Adiciona registro ao cache em memória"""
        with self._cache_lock:
            self._interaction_cache.append(record)
            
            # Limitar tamanho do cache
            if len(self._interaction_cache) > self._cache_max_size:
                self._interaction_cache.pop(0)
    
    def _save_to_database(self, record: InteractionRecord):
        """Salva registro no banco de dados"""
        try:
            with self._get_db_connection() as conn:
                # Preparar dados
                data = {
                    'interaction_id': record.interaction_id,
                    'timestamp': record.timestamp.isoformat(),
                    'interaction_type': record.interaction_type.value,
                    'status': record.status.value,
                    'component': record.component,
                    'user_id': record.user_context.user_id if record.user_context else None,
                    'session_id': record.user_context.session_id if record.user_context else "",
                    'input_data': json.dumps(record.input_data),
                    'output_data': json.dumps(record.output_data),
                    'sentiment_score': record.sentiment_score,
                    'sentiment_category': record.sentiment_category.value if record.sentiment_category else None,
                    'intent': record.intent,
                    'intent_confidence': record.intent_confidence,
                    'processing_time_ms': record.metrics.processing_time_ms if record.metrics else None,
                    'response_time_ms': record.metrics.response_time_ms if record.metrics else None,
                    'confidence_score': record.metrics.confidence_score if record.metrics else None,
                    'tokens_used': record.metrics.tokens_used if record.metrics else None,
                    'errors': json.dumps(record.errors) if record.errors else None,
                    'raw_data': json.dumps(asdict(record))
                }
                
                # Inserir no banco
                conn.execute("""
                    INSERT OR REPLACE INTO interactions 
                    (interaction_id, timestamp, interaction_type, status, component,
                     user_id, session_id, input_data, output_data, sentiment_score,
                     sentiment_category, intent, intent_confidence, processing_time_ms,
                     response_time_ms, confidence_score, tokens_used, errors, raw_data)
                    VALUES 
                    (:interaction_id, :timestamp, :interaction_type, :status, :component,
                     :user_id, :session_id, :input_data, :output_data, :sentiment_score,
                     :sentiment_category, :intent, :intent_confidence, :processing_time_ms,
                     :response_time_ms, :confidence_score, :tokens_used, :errors, :raw_data)
                """, data)
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Erro ao salvar no banco: {e}")
            raise
    
    def _save_to_file(self, record: InteractionRecord):
        """Salva registro em arquivo JSONL"""
        try:
            # Converter para dict
            record_dict = asdict(record)
            
            # Converter datetime para string
            record_dict['timestamp'] = record.timestamp.isoformat()
            
            # Converter enums para strings
            record_dict['interaction_type'] = record.interaction_type.value
            record_dict['status'] = record.status.value
            if record.sentiment_category:
                record_dict['sentiment_category'] = record.sentiment_category.value
            
            # Escrever linha
            with open(self.json_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record_dict, ensure_ascii=False) + '\n')
                
        except Exception as e:
            self.logger.error(f"Erro ao salvar em arquivo: {e}")
            raise
    
    def _update_stats(self, record: InteractionRecord):
        """Atualiza estatísticas"""
        try:
            self._stats["total_interactions"] += 1
            
            # Por tipo
            interaction_type = record.interaction_type.value
            self._stats["interactions_by_type"][interaction_type] = \
                self._stats["interactions_by_type"].get(interaction_type, 0) + 1
            
            # Por status
            status = record.status.value
            self._stats["interactions_by_status"][status] = \
                self._stats["interactions_by_status"].get(status, 0) + 1
            
            # Tempos de resposta
            if record.metrics and record.metrics.response_time_ms:
                total_time = self._stats["total_processing_time"]
                total_interactions = self._stats["total_interactions"]
                
                self._stats["total_processing_time"] = total_time + record.metrics.response_time_ms
                self._stats["average_response_time"] = \
                    self._stats["total_processing_time"] / total_interactions
                    
        except Exception as e:
            self.logger.error(f"Erro ao atualizar estatísticas: {e}")
    
    def get_interactions(self, 
                        limit: int = 100,
                        interaction_type: Optional[InteractionType] = None,
                        user_id: Optional[str] = None,
                        session_id: Optional[str] = None,
                        start_time: Optional[datetime] = None,
                        end_time: Optional[datetime] = None) -> List[InteractionRecord]:
        """
        Recupera interações com filtros
        
        Args:
            limit: Número máximo de registros
            interaction_type: Filtrar por tipo
            user_id: Filtrar por usuário
            session_id: Filtrar por sessão
            start_time: Filtrar por data inicial
            end_time: Filtrar por data final
            
        Returns:
            Lista de registros de interação
        """
        try:
            if not self.enable_db_storage:
                # Retornar do cache se banco não disponível
                return self._get_from_cache(limit, interaction_type, user_id, session_id)
            
            # Construir query
            query = "SELECT raw_data FROM interactions WHERE 1=1"
            params = {}
            
            if interaction_type:
                query += " AND interaction_type = :interaction_type"
                params['interaction_type'] = interaction_type.value
            
            if user_id:
                query += " AND user_id = :user_id"
                params['user_id'] = user_id
            
            if session_id:
                query += " AND session_id = :session_id"
                params['session_id'] = session_id
            
            if start_time:
                query += " AND timestamp >= :start_time"
                params['start_time'] = start_time.isoformat()
            
            if end_time:
                query += " AND timestamp <= :end_time"
                params['end_time'] = end_time.isoformat()
            
            query += " ORDER BY timestamp DESC LIMIT :limit"
            params['limit'] = limit
            
            # Executar query
            with self._get_db_connection() as conn:
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
                
                # Converter de volta para objetos
                interactions = []
                for row in rows:
                    try:
                        data = json.loads(row['raw_data'])
                        # Reconstruir objeto (simplificado)
                        interaction = self._dict_to_interaction(data)
                        interactions.append(interaction)
                    except Exception as e:
                        self.logger.error(f"Erro ao deserializar interação: {e}")
                        continue
                
                return interactions
                
        except Exception as e:
            self.logger.error(f"Erro ao recuperar interações: {e}")
            return []
    
    def _get_from_cache(self, limit: int, interaction_type: Optional[InteractionType] = None,
                       user_id: Optional[str] = None, session_id: Optional[str] = None) -> List[InteractionRecord]:
        """Recupera interações do cache"""
        with self._cache_lock:
            filtered = self._interaction_cache.copy()
            
            # Aplicar filtros
            if interaction_type:
                filtered = [r for r in filtered if r.interaction_type == interaction_type]
            
            if user_id:
                filtered = [r for r in filtered 
                           if r.user_context and r.user_context.user_id == user_id]
            
            if session_id:
                filtered = [r for r in filtered 
                           if r.user_context and r.user_context.session_id == session_id]
            
            # Ordenar por timestamp (mais recente primeiro) e limitar
            filtered.sort(key=lambda x: x.timestamp, reverse=True)
            return filtered[:limit]
    
    def _dict_to_interaction(self, data: Dict[str, Any]) -> InteractionRecord:
        """Converte dict de volta para InteractionRecord (simplificado)"""
        # Esta é uma implementação simplificada
        # Em produção, você pode usar bibliotecas como pydantic ou dataclasses_json
        return InteractionRecord(
            interaction_id=data['interaction_id'],
            timestamp=datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00')),
            interaction_type=InteractionType(data['interaction_type']),
            status=InteractionStatus(data['status']),
            input_data=data.get('input_data', {}),
            output_data=data.get('output_data', {}),
            component=data.get('component', 'unknown')
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas de interações"""
        return self._stats.copy()
    
    def get_user_interaction_history(self, user_id: str, limit: int = 50) -> List[InteractionRecord]:
        """Recupera histórico de interações de um usuário específico"""
        return self.get_interactions(limit=limit, user_id=user_id)
    
    def get_session_interactions(self, session_id: str) -> List[InteractionRecord]:
        """Recupera todas as interações de uma sessão"""
        return self.get_interactions(session_id=session_id, limit=1000)
    
    def cleanup_old_interactions(self):
        """Remove interações antigas baseado na política de retenção"""
        try:
            if not self.enable_db_storage:
                return
            
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.retention_days)
            
            with self._get_db_connection() as conn:
                cursor = conn.execute(
                    "DELETE FROM interactions WHERE timestamp < ?",
                    (cutoff_date.isoformat(),)
                )
                deleted_count = cursor.rowcount
                conn.commit()
                
                if deleted_count > 0:
                    self.logger.info(f"Removidas {deleted_count} interações antigas")
                    
        except Exception as e:
            self.logger.error(f"Erro na limpeza de interações antigas: {e}")

# =========================================================================
# FUNÇÕES DE CONVENIÊNCIA
# =========================================================================

# Instância global do logger
_interaction_logger = None

def get_interaction_logger() -> InteractionLogger:
    """Retorna instância global do logger de interações"""
    global _interaction_logger
    if _interaction_logger is None:
        _interaction_logger = InteractionLogger()
    return _interaction_logger

def log_voice_input(user_input: str, user_context: UserContext, 
                   metrics: InteractionMetrics, interaction_id: str = None) -> bool:
    """Registra entrada de voz"""
    logger = get_interaction_logger()
    
    record = InteractionRecord(
        interaction_id=interaction_id or f"voice_{datetime.now().timestamp()}",
        timestamp=datetime.now(timezone.utc),
        interaction_type=InteractionType.VOICE_INPUT,
        status=InteractionStatus.SUCCESS,
        input_data={"text": user_input, "input_type": "voice"},
        output_data={},
        user_context=user_context,
        metrics=metrics,
        component="speech_recognition"
    )
    
    return logger.log_interaction(record)

def log_response_generated(response: str, intent: str, confidence: float,
                          user_context: UserContext, metrics: InteractionMetrics,
                          interaction_id: str = None) -> bool:
    """Registra resposta gerada"""
    logger = get_interaction_logger()
    
    record = InteractionRecord(
        interaction_id=interaction_id or f"response_{datetime.now().timestamp()}",
        timestamp=datetime.now(timezone.utc),
        interaction_type=InteractionType.RESPONSE_GENERATED,
        status=InteractionStatus.SUCCESS,
        input_data={},
        output_data={"response": response},
        intent=intent,
        intent_confidence=confidence,
        user_context=user_context,
        metrics=metrics,
        component="response_generation"
    )
    
    return logger.log_interaction(record)

def log_emotion_detected(emotion_data: EmotionData, user_context: UserContext,
                        interaction_id: str = None) -> bool:
    """Registra emoção detectada"""
    logger = get_interaction_logger()
    
    record = InteractionRecord(
        interaction_id=interaction_id or f"emotion_{datetime.now().timestamp()}",
        timestamp=datetime.now(timezone.utc),
        interaction_type=InteractionType.EMOTION_DETECTED,
        status=InteractionStatus.SUCCESS,
        input_data={},
        output_data={"primary_emotion": emotion_data.primary_emotion},
        user_context=user_context,
        emotion_data=emotion_data,
        component="emotion_analysis"
    )
    
    return logger.log_interaction(record)

# =========================================================================
# EXEMPLO DE USO
# =========================================================================

if __name__ == "__main__":
    # Exemplo de uso do sistema de logging
    
    # Configurar contexto de usuário
    user_ctx = UserContext(
        user_id="user_123",
        session_id="session_456",
        device_type="web",
        language="pt-BR",
        personality_preference="friendly"
    )
    
    # Configurar métricas
    metrics = InteractionMetrics(
        processing_time_ms=150.5,
        response_time_ms=200.3,
        confidence_score=0.85,
        tokens_used=45,
        api_calls_count=2
    )
    
    # Log de entrada de voz
    log_voice_input(
        user_input="Olá, como você está?",
        user_context=user_ctx,
        metrics=metrics
    )
    
    # Log de resposta gerada
    log_response_generated(
        response="Olá! Estou muito bem, obrigada por perguntar!",
        intent="greeting",
        confidence=0.92,
        user_context=user_ctx,
        metrics=metrics
    )
    
    # Recuperar estatísticas
    logger = get_interaction_logger()
    stats = logger.get_stats()
    print(f"Total de interações: {stats['total_interactions']}")