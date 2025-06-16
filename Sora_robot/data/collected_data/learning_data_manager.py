# =========================================================================
# SORA ROBOT - LEARNING DATA MANAGER
# Sistema avançado de coleta, processamento e gestão de dados de aprendizagem
# =========================================================================

import json
import asyncio
import hashlib
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import sqlite3
import threading
from contextlib import contextmanager
import pickle
import gzip
from collections import defaultdict, Counter

from utils.logger import get_logger
from utils.helpers import validate_input, ensure_directory_exists
from utils.constants import SYSTEM_CONFIG
from .interaction_logs import InteractionLogger, InteractionRecord, InteractionType

# =========================================================================
# CONFIGURAÇÕES E ENUMS
# =========================================================================

class DataType(Enum):
    """Tipos de dados de aprendizagem"""
    TEXT_PAIRS = "text_pairs"                    # Pares pergunta-resposta
    EMOTION_LABELS = "emotion_labels"            # Dados de emoção rotulados
    INTENT_SAMPLES = "intent_samples"            # Amostras de intenção
    CONVERSATION_FLOWS = "conversation_flows"    # Fluxos de conversa
    USER_FEEDBACK = "user_feedback"              # Feedback do usuário
    AUDIO_TRANSCRIPTS = "audio_transcripts"      # Transcrições de áudio
    FACIAL_EXPRESSIONS = "facial_expressions"   # Expressões faciais
    MULTIMODAL_PAIRS = "multimodal_pairs"       # Dados multimodais
    REINFORCEMENT_SIGNALS = "reinforcement_signals"  # Sinais de reforço
    PERSONALITY_ADAPTATIONS = "personality_adaptations"  # Adaptações de personalidade

class DataQuality(Enum):
    """Níveis de qualidade dos dados"""
    EXCELLENT = "excellent"      # 90-100% qualidade
    GOOD = "good"               # 70-89% qualidade
    ACCEPTABLE = "acceptable"    # 50-69% qualidade
    POOR = "poor"               # 30-49% qualidade
    UNUSABLE = "unusable"       # <30% qualidade

class LearningObjective(Enum):
    """Objetivos de aprendizagem"""
    CONVERSATION_IMPROVEMENT = "conversation_improvement"
    EMOTION_RECOGNITION = "emotion_recognition"
    INTENT_CLASSIFICATION = "intent_classification"
    PERSONALITY_MATCHING = "personality_matching"
    RESPONSE_GENERATION = "response_generation"
    MULTIMODAL_FUSION = "multimodal_fusion"
    USER_SATISFACTION = "user_satisfaction"
    CONTEXT_UNDERSTANDING = "context_understanding"

# =========================================================================
# ESTRUTURAS DE DADOS
# =========================================================================

@dataclass
class DataSample:
    """Amostra individual de dados de aprendizagem"""
    sample_id: str
    data_type: DataType
    timestamp: datetime
    
    # Dados de entrada
    input_data: Dict[str, Any]
    target_data: Dict[str, Any]
    
    # Metadados
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    interaction_id: Optional[str] = None
    
    # Qualidade e validação
    quality_score: float = 0.0
    quality_level: Optional[DataQuality] = None
    is_validated: bool = False
    validation_notes: List[str] = None
    
    # Contexto
    context_data: Dict[str, Any] = None
    modality: List[str] = None  # ['text', 'audio', 'video', 'emotion']
    
    # Processamento
    preprocessing_applied: List[str] = None
    augmentation_applied: List[str] = None
    
    def __post_init__(self):
        if self.validation_notes is None:
            self.validation_notes = []
        if self.context_data is None:
            self.context_data = {}
        if self.modality is None:
            self.modality = []
        if self.preprocessing_applied is None:
            self.preprocessing_applied = []
        if self.augmentation_applied is None:
            self.augmentation_applied = []

@dataclass
class DatasetMetrics:
    """Métricas de um conjunto de dados"""
    total_samples: int
    samples_by_type: Dict[str, int]
    samples_by_quality: Dict[str, int]
    average_quality_score: float
    date_range: Tuple[datetime, datetime]
    unique_users: int
    unique_sessions: int
    data_size_mb: float
    last_updated: datetime

@dataclass
class LearningDataset:
    """Conjunto de dados para aprendizagem"""
    dataset_id: str
    name: str
    description: str
    learning_objectives: List[LearningObjective]
    samples: List[DataSample]
    metrics: DatasetMetrics
    created_at: datetime
    updated_at: datetime
    version: str = "1.0.0"

# =========================================================================
# GERENCIADOR DE DADOS DE APRENDIZAGEM
# =========================================================================

class LearningDataManager:
    """Sistema avançado de gestão de dados de aprendizagem"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.logger = get_logger(__name__)
        self.config = config or SYSTEM_CONFIG.get("learning_data", {})
        
        # Configurações
        self.data_dir = Path(self.config.get("data_directory", "data/collected_data"))
        self.datasets_dir = self.data_dir / "datasets"
        self.db_path = self.data_dir / "learning_data.db"
        self.exports_dir = self.data_dir / "exports"
        
        # Configurações de qualidade
        self.min_quality_threshold = self.config.get("min_quality_threshold", 0.5)
        self.auto_validation = self.config.get("auto_validation", True)
        self.max_samples_per_dataset = self.config.get("max_samples_per_dataset", 10000)
        
        # Cache e threading
        self._datasets_cache: Dict[str, LearningDataset] = {}
        self._cache_lock = threading.Lock()
        
        # Estatísticas
        self._stats = {
            "total_datasets": 0,
            "total_samples": 0,
            "samples_by_type": defaultdict(int),
            "quality_distribution": defaultdict(int),
            "learning_objectives_count": defaultdict(int)
        }
        
        # Conectar com sistema de logging de interações
        try:
            self.interaction_logger = InteractionLogger()
        except Exception as e:
            self.logger.warning(f"Não foi possível conectar com InteractionLogger: {e}")
            self.interaction_logger = None
        
        # Inicialização
        self._initialize_storage()
        self._load_existing_datasets()
        self.logger.info("LearningDataManager inicializado")
    
    def _initialize_storage(self):
        """Inicializa sistemas de armazenamento"""
        try:
            # Criar diretórios
            ensure_directory_exists(self.data_dir)
            ensure_directory_exists(self.datasets_dir)
            ensure_directory_exists(self.exports_dir)
            
            # Inicializar banco de dados
            self._initialize_database()
            
        except Exception as e:
            self.logger.error(f"Erro ao inicializar storage: {e}")
            raise
    
    def _initialize_database(self):
        """Inicializa banco de dados SQLite"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Tabela de datasets
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS datasets (
                        dataset_id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        description TEXT,
                        learning_objectives TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        version TEXT,
                        total_samples INTEGER,
                        average_quality REAL,
                        data_size_mb REAL
                    )
                """)
                
                # Tabela de amostras
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS data_samples (
                        sample_id TEXT PRIMARY KEY,
                        dataset_id TEXT,
                        data_type TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        user_id TEXT,
                        session_id TEXT,
                        interaction_id TEXT,
                        quality_score REAL,
                        quality_level TEXT,
                        is_validated BOOLEAN,
                        modality TEXT,
                        input_data TEXT,
                        target_data TEXT,
                        context_data TEXT,
                        preprocessing_applied TEXT,
                        raw_data TEXT,
                        FOREIGN KEY (dataset_id) REFERENCES datasets (dataset_id)
                    )
                """)
                
                # Índices
                conn.execute("CREATE INDEX IF NOT EXISTS idx_dataset_id ON data_samples(dataset_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_data_type ON data_samples(data_type)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_quality ON data_samples(quality_score)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON data_samples(timestamp)")
                
                conn.commit()
                self.logger.info("Banco de dados de aprendizagem inicializado")
                
        except Exception as e:
            self.logger.error(f"Erro ao inicializar banco: {e}")
            raise
    
    def _load_existing_datasets(self):
        """Carrega datasets existentes"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("SELECT * FROM datasets")
                
                for row in cursor.fetchall():
                    dataset_id = row['dataset_id']
                    # Carregar samples do dataset
                    samples = self._load_dataset_samples(dataset_id)
                    
                    # Criar objeto dataset
                    dataset = LearningDataset(
                        dataset_id=dataset_id,
                        name=row['name'],
                        description=row['description'] or "",
                        learning_objectives=[LearningObjective(obj) for obj in 
                                           json.loads(row['learning_objectives'] or '[]')],
                        samples=samples,
                        metrics=self._calculate_metrics(samples),
                        created_at=datetime.fromisoformat(row['created_at']),
                        updated_at=datetime.fromisoformat(row['updated_at']),
                        version=row['version'] or "1.0.0"
                    )
                    
                    self._datasets_cache[dataset_id] = dataset
                    self._update_stats_for_dataset(dataset)
            
            self.logger.info(f"Carregados {len(self._datasets_cache)} datasets existentes")
            
        except Exception as e:
            self.logger.error(f"Erro ao carregar datasets: {e}")
    
    def _load_dataset_samples(self, dataset_id: str) -> List[DataSample]:
        """Carrega amostras de um dataset"""
        samples = []
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT * FROM data_samples WHERE dataset_id = ? ORDER BY timestamp",
                    (dataset_id,)
                )
                
                for row in cursor.fetchall():
                    sample = DataSample(
                        sample_id=row['sample_id'],
                        data_type=DataType(row['data_type']),
                        timestamp=datetime.fromisoformat(row['timestamp']),
                        input_data=json.loads(row['input_data']),
                        target_data=json.loads(row['target_data']),
                        user_id=row['user_id'],
                        session_id=row['session_id'],
                        interaction_id=row['interaction_id'],
                        quality_score=row['quality_score'] or 0.0,
                        quality_level=DataQuality(row['quality_level']) if row['quality_level'] else None,
                        is_validated=bool(row['is_validated']),
                        modality=json.loads(row['modality'] or '[]'),
                        context_data=json.loads(row['context_data'] or '{}'),
                        preprocessing_applied=json.loads(row['preprocessing_applied'] or '[]')
                    )
                    samples.append(sample)
            
        except Exception as e:
            self.logger.error(f"Erro ao carregar amostras do dataset {dataset_id}: {e}")
        
        return samples
    
    def create_dataset(self, 
                      name: str,
                      description: str,
                      learning_objectives: List[LearningObjective],
                      dataset_id: str = None) -> LearningDataset:
        """
        Cria um novo dataset de aprendizagem
        
        Args:
            name: Nome do dataset
            description: Descrição do dataset
            learning_objectives: Objetivos de aprendizagem
            dataset_id: ID customizado (opcional)
            
        Returns:
            Dataset criado
        """
        try:
            # Gerar ID se não fornecido
            if dataset_id is None:
                dataset_id = f"dataset_{int(datetime.now().timestamp())}"
            
            # Verificar se já existe
            if dataset_id in self._datasets_cache:
                raise ValueError(f"Dataset {dataset_id} já existe")
            
            # Criar dataset
            now = datetime.now(timezone.utc)
            dataset = LearningDataset(
                dataset_id=dataset_id,
                name=name,
                description=description,
                learning_objectives=learning_objectives,
                samples=[],
                metrics=DatasetMetrics(
                    total_samples=0,
                    samples_by_type={},
                    samples_by_quality={},
                    average_quality_score=0.0,
                    date_range=(now, now),
                    unique_users=0,
                    unique_sessions=0,
                    data_size_mb=0.0,
                    last_updated=now
                ),
                created_at=now,
                updated_at=now
            )
            
            # Salvar no banco
            self._save_dataset_to_db(dataset)
            
            # Adicionar ao cache
            with self._cache_lock:
                self._datasets_cache[dataset_id] = dataset
                self._update_stats_for_dataset(dataset)
            
            self.logger.info(f"Dataset criado: {name} ({dataset_id})")
            return dataset
            
        except Exception as e:
            self.logger.error(f"Erro ao criar dataset: {e}")
            raise
    
    def add_sample_to_dataset(self, 
                             dataset_id: str,
                             sample: DataSample,
                             auto_validate: bool = None) -> bool:
        """
        Adiciona amostra a um dataset
        
        Args:
            dataset_id: ID do dataset
            sample: Amostra a ser adicionada
            auto_validate: Validar automaticamente
            
        Returns:
            True se adicionado com sucesso
        """
        try:
            # Verificar se dataset existe
            if dataset_id not in self._datasets_cache:
                raise ValueError(f"Dataset {dataset_id} não encontrado")
            
            dataset = self._datasets_cache[dataset_id]
            
            # Verificar limite de amostras
            if len(dataset.samples) >= self.max_samples_per_dataset:
                self.logger.warning(f"Dataset {dataset_id} atingiu limite máximo de amostras")
                return False
            
            # Validar automaticamente se configurado
            if auto_validate or (auto_validate is None and self.auto_validation):
                self._validate_sample(sample)
            
            # Verificar qualidade mínima
            if sample.quality_score < self.min_quality_threshold:
                self.logger.warning(f"Amostra {sample.sample_id} rejeitada por baixa qualidade: {sample.quality_score}")
                return False
            
            # Adicionar ao dataset
            dataset.samples.append(sample)
            
            # Salvar no banco
            self._save_sample_to_db(dataset_id, sample)
            
            # Atualizar métricas
            dataset.metrics = self._calculate_metrics(dataset.samples)
            dataset.updated_at = datetime.now(timezone.utc)
            
            # Atualizar banco
            self._update_dataset_in_db(dataset)
            
            self.logger.debug(f"Amostra {sample.sample_id} adicionada ao dataset {dataset_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao adicionar amostra: {e}")
            return False
    
    def _validate_sample(self, sample: DataSample):
        """Valida uma amostra de dados"""
        try:
            quality_score = 0.0
            validation_notes = []
            
            # Validação básica
            if not sample.input_data:
                validation_notes.append("Dados de entrada vazios")
            else:
                quality_score += 0.3
            
            if not sample.target_data:
                validation_notes.append("Dados alvo vazios")
            else:
                quality_score += 0.3
            
            # Validação por tipo de dados
            if sample.data_type == DataType.TEXT_PAIRS:
                quality_score += self._validate_text_pairs(sample)
            elif sample.data_type == DataType.EMOTION_LABELS:
                quality_score += self._validate_emotion_labels(sample)
            elif sample.data_type == DataType.INTENT_SAMPLES:
                quality_score += self._validate_intent_samples(sample)
            else:
                quality_score += 0.2  # Score padrão para tipos não específicos
            
            # Validação de contexto
            if sample.context_data and len(sample.context_data) > 0:
                quality_score += 0.1
            
            # Validação de metadados
            if sample.user_id and sample.session_id:
                quality_score += 0.1
            
            # Determinar nível de qualidade
            if quality_score >= 0.9:
                quality_level = DataQuality.EXCELLENT
            elif quality_score >= 0.7:
                quality_level = DataQuality.GOOD
            elif quality_score >= 0.5:
                quality_level = DataQuality.ACCEPTABLE
            elif quality_score >= 0.3:
                quality_level = DataQuality.POOR
            else:
                quality_level = DataQuality.UNUSABLE
            
            # Atualizar amostra
            sample.quality_score = quality_score
            sample.quality_level = quality_level
            sample.is_validated = True
            sample.validation_notes = validation_notes
            
        except Exception as e:
            self.logger.error(f"Erro na validação: {e}")
            sample.quality_score = 0.0
            sample.quality_level = DataQuality.UNUSABLE
            sample.validation_notes = [f"Erro na validação: {e}"]
    
    def _validate_text_pairs(self, sample: DataSample) -> float:
        """Valida pares de texto (pergunta-resposta)"""
        score = 0.0
        
        try:
            input_text = sample.input_data.get('text', '')
            target_text = sample.target_data.get('text', '')
            
            # Verificar comprimento
            if 5 <= len(input_text) <= 500:
                score += 0.1
            if 5 <= len(target_text) <= 1000:
                score += 0.1
            
            # Verificar qualidade linguística básica
            if input_text.strip() and target_text.strip():
                score += 0.1
            
            # Verificar se há relação semântica (simplificado)
            common_words = set(input_text.lower().split()) & set(target_text.lower().split())
            if common_words:
                score += 0.1
            
        except Exception:
            score = 0.0
        
        return score
    
    def _validate_emotion_labels(self, sample: DataSample) -> float:
        """Valida dados de emoção rotulados"""
        score = 0.0
        
        try:
            emotion = sample.target_data.get('emotion', '')
            confidence = sample.target_data.get('confidence', 0.0)
            
            # Verificar emoções válidas
            valid_emotions = ['happy', 'sad', 'angry', 'surprised', 'neutral', 'fear', 'disgust']
            if emotion.lower() in valid_emotions:
                score += 0.2
            
            # Verificar confiança
            if 0.5 <= confidence <= 1.0:
                score += 0.2
            
        except Exception:
            score = 0.0
        
        return score
    
    def _validate_intent_samples(self, sample: DataSample) -> float:
        """Valida amostras de intenção"""
        score = 0.0
        
        try:
            text = sample.input_data.get('text', '')
            intent = sample.target_data.get('intent', '')
            confidence = sample.target_data.get('confidence', 0.0)
            
            # Verificar texto
            if 3 <= len(text.split()) <= 50:
                score += 0.1
            
            # Verificar intenção
            if intent and len(intent) > 0:
                score += 0.2
            
            # Verificar confiança
            if confidence > 0.7:
                score += 0.1
            
        except Exception:
            score = 0.0
        
        return score
    
    def collect_from_interactions(self, 
                                 dataset_id: str,
                                 hours_back: int = 24,
                                 min_quality: float = 0.5) -> int:
        """
        Coleta dados das interações recentes para um dataset
        
        Args:
            dataset_id: ID do dataset
            hours_back: Horas anteriores para coletar
            min_quality: Qualidade mínima das interações
            
        Returns:
            Número de amostras coletadas
        """
        if not self.interaction_logger:
            self.logger.warning("InteractionLogger não disponível")
            return 0
        
        try:
            # Definir período
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(hours=hours_back)
            
            # Recuperar interações
            interactions = self.interaction_logger.get_interactions(
                limit=1000,
                start_time=start_time,
                end_time=end_time
            )
            
            samples_added = 0
            
            for interaction in interactions:
                # Verificar qualidade mínima
                if (interaction.metrics and 
                    interaction.metrics.confidence_score and 
                    interaction.metrics.confidence_score < min_quality):
                    continue
                
                # Converter interação para amostra
                samples = self._interaction_to_samples(interaction)
                
                for sample in samples:
                    if self.add_sample_to_dataset(dataset_id, sample):
                        samples_added += 1
            
            self.logger.info(f"Coletadas {samples_added} amostras para dataset {dataset_id}")
            return samples_added
            
        except Exception as e:
            self.logger.error(f"Erro ao coletar dados das interações: {e}")
            return 0
    
    def _interaction_to_samples(self, interaction: InteractionRecord) -> List[DataSample]:
        """Converte uma interação em amostras de aprendizagem"""
        samples = []
        
        try:
            timestamp = interaction.timestamp
            user_context = interaction.user_context
            
            # Amostra de par texto (se disponível)
            if (interaction.interaction_type == InteractionType.RESPONSE_GENERATED and
                interaction.input_data and interaction.output_data):
                
                sample = DataSample(
                    sample_id=f"text_pair_{interaction.interaction_id}",
                    data_type=DataType.TEXT_PAIRS,
                    timestamp=timestamp,
                    input_data=interaction.input_data,
                    target_data=interaction.output_data,
                    user_id=user_context.user_id if user_context else None,
                    session_id=user_context.session_id if user_context else None,
                    interaction_id=interaction.interaction_id,
                    modality=['text'],
                    context_data={
                        'intent': interaction.intent,
                        'sentiment': interaction.sentiment_score,
                        'language': user_context.language if user_context else 'pt-BR'
                    }
                )
                samples.append(sample)
            
            # Amostra de emoção (se disponível)
            if (interaction.emotion_data and 
                interaction.interaction_type == InteractionType.EMOTION_DETECTED):
                
                sample = DataSample(
                    sample_id=f"emotion_{interaction.interaction_id}",
                    data_type=DataType.EMOTION_LABELS,
                    timestamp=timestamp,
                    input_data={}, # Dados de imagem não disponíveis neste contexto
                    target_data={
                        'emotion': interaction.emotion_data.primary_emotion,
                        'confidence': interaction.emotion_data.emotion_confidence,
                        'distribution': interaction.emotion_data.emotion_distribution
                    },
                    user_id=user_context.user_id if user_context else None,
                    session_id=user_context.session_id if user_context else None,
                    interaction_id=interaction.interaction_id,
                    modality=['emotion', 'video'],
                    context_data={}
                )
                samples.append(sample)
            
            # Amostra de intenção (se disponível)
            if (interaction.intent and interaction.intent_confidence and
                interaction.interaction_type == InteractionType.INTENT_RECOGNIZED):
                
                sample = DataSample(
                    sample_id=f"intent_{interaction.interaction_id}",
                    data_type=DataType.INTENT_SAMPLES,
                    timestamp=timestamp,
                    input_data=interaction.input_data,
                    target_data={
                        'intent': interaction.intent,
                        'confidence': interaction.intent_confidence
                    },
                    user_id=user_context.user_id if user_context else None,
                    session_id=user_context.session_id if user_context else None,
                    interaction_id=interaction.interaction_id,
                    modality=['text'],
                    context_data={
                        'sentiment': interaction.sentiment_score,
                        'language': user_context.language if user_context else 'pt-BR'
                    }
                )
                samples.append(sample)
        
        except Exception as e:
            self.logger.error(f"Erro ao converter interação {interaction.interaction_id}: {e}")
        
        return samples
    
    def export_dataset(self, 
                      dataset_id: str,
                      format: str = "json",
                      include_metadata: bool = True,
                      filter_quality: Optional[DataQuality] = None) -> Optional[Path]:
        """
        Exporta dataset para arquivo
        
        Args:
            dataset_id: ID do dataset
            format: Formato de exportação (json, csv, pickle, huggingface)
            include_metadata: Incluir metadados
            filter_quality: Filtrar por qualidade mínima
            
        Returns:
            Caminho do arquivo exportado
        """
        try:
            if dataset_id not in self._datasets_cache:
                raise ValueError(f"Dataset {dataset_id} não encontrado")
            
            dataset = self._datasets_cache[dataset_id]
            
            # Filtrar amostras por qualidade
            samples = dataset.samples
            if filter_quality:
                quality_levels = {
                    DataQuality.EXCELLENT: 4,
                    DataQuality.GOOD: 3,
                    DataQuality.ACCEPTABLE: 2,
                    DataQuality.POOR: 1,
                    DataQuality.UNUSABLE: 0
                }
                min_level = quality_levels[filter_quality]
                samples = [s for s in samples 
                          if s.quality_level and quality_levels.get(s.quality_level, 0) >= min_level]
            
            # Gerar nome do arquivo
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{dataset.name}_{timestamp}.{format}"
            export_path = self.exports_dir / filename
            
            # Exportar baseado no formato
            if format == "json":
                self._export_json(dataset, samples, export_path, include_metadata)
            elif format == "csv":
                self._export_csv(dataset, samples, export_path, include_metadata)
            elif format == "pickle":
                self._export_pickle(dataset, samples, export_path, include_metadata)
            elif format == "huggingface":
                self._export_huggingface(dataset, samples, export_path, include_metadata)
            else:
                raise ValueError(f"Formato não suportado: {format}")
            
            self.logger.info(f"Dataset {dataset_id} exportado para {export_path}")
            return export_path
            
        except Exception as e:
            self.logger.error(f"Erro ao exportar dataset: {e}")
            return None
    
    def _export_json(self, dataset: LearningDataset, samples: List[DataSample], 
                    path: Path, include_metadata: bool):
        """Exporta para formato JSON"""
        data = {
            "dataset_info": {
                "dataset_id": dataset.dataset_id,
                "name": dataset.name,
                "description": dataset.description,
                "learning_objectives": [obj.value for obj in dataset.learning_objectives],
                "version": dataset.version,
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "total_samples": len(samples)
            } if include_metadata else {},
            "samples": []
        }
        
        for sample in samples:
            sample_dict = asdict(sample)
            # Converter datetime para string
            sample_dict['timestamp'] = sample.timestamp.isoformat()
            sample_dict['data_type'] = sample.data_type.value
            if sample.quality_level:
                sample_dict['quality_level'] = sample.quality_level.value
            
            data["samples"].append(sample_dict)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def _export_csv(self, dataset: LearningDataset, samples: List[DataSample], 
                   path: Path, include_metadata: bool):
        """Exporta para formato CSV"""
        rows = []
        for sample in samples:
            row = {
                'sample_id': sample.sample_id,
                'data_type': sample.data_type.value,
                'timestamp': sample.timestamp.isoformat(),
                'user_id': sample.user_id,
                'session_id': sample.session_id,
                'quality_score': sample.quality_score,
                'quality_level': sample.quality_level.value if sample.quality_level else '',
                'is_validated': sample.is_validated,
                'input_data': json.dumps(sample.input_data),
                'target_data': json.dumps(sample.target_data),
                'context_data': json.dumps(sample.context_data),
                'modality': json.dumps(sample.modality)
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(path, index=False, encoding='utf-8')
    
    def _export_pickle(self, dataset: LearningDataset, samples: List[DataSample], 
                      path: Path, include_metadata: bool):
        """Exporta para formato Pickle"""
        data = {
            'dataset': dataset if include_metadata else None,
            'samples': samples
        }
        
        with gzip.open(path, 'wb') as f:
            pickle.dump(data, f)
    
    def _export_huggingface(self, dataset: LearningDataset, samples: List[DataSample], 
                           path: Path, include_metadata: bool):
        """Exporta para formato compatível com Hugging Face"""
        # Separar por tipo de dados
        text_pairs = []
        for sample in samples:
            if sample.data_type == DataType.TEXT_PAIRS:
                text_pairs.append({
                    'input': sample.input_data.get('text', ''),
                    'target': sample.target_data.get('text', ''),
                    'quality_score': sample.quality_score
                })
        
        # Criar estrutura Hugging Face
        hf_data = {
            'train': text_pairs,
            'metadata': {
                'dataset_name': dataset.name,
                'description': dataset.description,
                'total_samples': len(text_pairs)
            } if include_metadata else {}
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(hf_data, f, ensure_ascii=False, indent=2)
    
    def get_dataset_stats(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Retorna estatísticas detalhadas de um dataset"""
        if dataset_id not in self._datasets_cache:
            return None
        
        dataset = self._datasets_cache[dataset_id]
        samples = dataset.samples
        
        # Estatísticas básicas
        stats = {
            'dataset_info': {
                'id': dataset.dataset_id,
                'name': dataset.name,
                'description': dataset.description,
                'learning_objectives': [obj.value for obj in dataset.learning_objectives],
                'version': dataset.version,
                'created_at': dataset.created_at.isoformat(),
                'updated_at': dataset.updated_at.isoformat()
            },
            'sample_counts': {
                'total': len(samples),
                'validated': sum(1 for s in samples if s.is_validated),
                'by_type': {},
                'by_quality': {},
                'by_modality': {}
            },
            'quality_metrics': {
                'average_score': 0.0,
                'median_score': 0.0,
                'score_distribution': {},
                'validation_issues': []
            },
            'temporal_distribution': {
                'date_range': {},
                'samples_per_day': {},
                'recent_activity': {}
            },
            'user_distribution': {
                'unique_users': 0,
                'unique_sessions': 0,
                'samples_per_user': {}
            }
        }
        
        if not samples:
            return stats
        
        # Análise por tipo
        type_counts = Counter(s.data_type.value for s in samples)
        stats['sample_counts']['by_type'] = dict(type_counts)
        
        # Análise por qualidade
        quality_counts = Counter(s.quality_level.value for s in samples if s.quality_level)
        stats['sample_counts']['by_quality'] = dict(quality_counts)
        
        # Análise por modalidade
        modality_counts = defaultdict(int)
        for sample in samples:
            for mod in sample.modality:
                modality_counts[mod] += 1
        stats['sample_counts']['by_modality'] = dict(modality_counts)
        
        # Métricas de qualidade
        quality_scores = [s.quality_score for s in samples if s.quality_score > 0]
        if quality_scores:
            stats['quality_metrics']['average_score'] = np.mean(quality_scores)
            stats['quality_metrics']['median_score'] = np.median(quality_scores)
            
            # Distribuição de scores
            score_bins = np.histogram(quality_scores, bins=5, range=(0, 1))
            stats['quality_metrics']['score_distribution'] = {
                f"{score_bins[1][i]:.1f}-{score_bins[1][i+1]:.1f}": int(score_bins[0][i])
                for i in range(len(score_bins[0]))
            }
        
        # Problemas de validação
        validation_issues = []
        for sample in samples:
            if sample.validation_notes:
                validation_issues.extend(sample.validation_notes)
        issue_counts = Counter(validation_issues)
        stats['quality_metrics']['validation_issues'] = dict(issue_counts.most_common(5))
        
        # Distribuição temporal
        timestamps = [s.timestamp for s in samples]
        if timestamps:
            min_date = min(timestamps)
            max_date = max(timestamps)
            stats['temporal_distribution']['date_range'] = {
                'start': min_date.isoformat(),
                'end': max_date.isoformat(),
                'duration_days': (max_date - min_date).days
            }
            
            # Amostras por dia
            daily_counts = defaultdict(int)
            for ts in timestamps:
                day_key = ts.strftime('%Y-%m-%d')
                daily_counts[day_key] += 1
            stats['temporal_distribution']['samples_per_day'] = dict(daily_counts)
        
        # Distribuição de usuários
        user_ids = [s.user_id for s in samples if s.user_id]
        session_ids = [s.session_id for s in samples if s.session_id]
        
        stats['user_distribution']['unique_users'] = len(set(user_ids))
        stats['user_distribution']['unique_sessions'] = len(set(session_ids))
        
        if user_ids:
            user_counts = Counter(user_ids)
            stats['user_distribution']['samples_per_user'] = dict(user_counts.most_common(10))
        
        return stats
    
    def get_global_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas globais de todos os datasets"""
        return self._stats.copy()
    
    def _calculate_metrics(self, samples: List[DataSample]) -> DatasetMetrics:
        """Calcula métricas de um conjunto de amostras"""
        if not samples:
            return DatasetMetrics(
                total_samples=0,
                samples_by_type={},
                samples_by_quality={},
                average_quality_score=0.0,
                date_range=(datetime.now(timezone.utc), datetime.now(timezone.utc)),
                unique_users=0,
                unique_sessions=0,
                data_size_mb=0.0,
                last_updated=datetime.now(timezone.utc)
            )
        
        # Contagens por tipo
        type_counts = Counter(s.data_type.value for s in samples)
        
        # Contagens por qualidade
        quality_counts = Counter(s.quality_level.value for s in samples if s.quality_level)
        
        # Score médio de qualidade
        quality_scores = [s.quality_score for s in samples if s.quality_score > 0]
        avg_quality = np.mean(quality_scores) if quality_scores else 0.0
        
        # Range de datas
        timestamps = [s.timestamp for s in samples]
        date_range = (min(timestamps), max(timestamps)) if timestamps else (
            datetime.now(timezone.utc), datetime.now(timezone.utc)
        )
        
        # Usuários e sessões únicos
        unique_users = len(set(s.user_id for s in samples if s.user_id))
        unique_sessions = len(set(s.session_id for s in samples if s.session_id))
        
        # Estimativa de tamanho dos dados (simplificada)
        data_size_mb = len(samples) * 0.001  # Aproximação: 1KB por amostra
        
        return DatasetMetrics(
            total_samples=len(samples),
            samples_by_type=dict(type_counts),
            samples_by_quality=dict(quality_counts),
            average_quality_score=avg_quality,
            date_range=date_range,
            unique_users=unique_users,
            unique_sessions=unique_sessions,
            data_size_mb=data_size_mb,
            last_updated=datetime.now(timezone.utc)
        )
    
    def _save_dataset_to_db(self, dataset: LearningDataset):
        """Salva dataset no banco de dados"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                data = {
                    'dataset_id': dataset.dataset_id,
                    'name': dataset.name,
                    'description': dataset.description,
                    'learning_objectives': json.dumps([obj.value for obj in dataset.learning_objectives]),
                    'created_at': dataset.created_at.isoformat(),
                    'updated_at': dataset.updated_at.isoformat(),
                    'version': dataset.version,
                    'total_samples': dataset.metrics.total_samples,
                    'average_quality': dataset.metrics.average_quality_score,
                    'data_size_mb': dataset.metrics.data_size_mb
                }
                
                conn.execute("""
                    INSERT OR REPLACE INTO datasets 
                    (dataset_id, name, description, learning_objectives, created_at, 
                     updated_at, version, total_samples, average_quality, data_size_mb)
                    VALUES 
                    (:dataset_id, :name, :description, :learning_objectives, :created_at,
                     :updated_at, :version, :total_samples, :average_quality, :data_size_mb)
                """, data)
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Erro ao salvar dataset no banco: {e}")
            raise
    
    def _save_sample_to_db(self, dataset_id: str, sample: DataSample):
        """Salva amostra no banco de dados"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                data = {
                    'sample_id': sample.sample_id,
                    'dataset_id': dataset_id,
                    'data_type': sample.data_type.value,
                    'timestamp': sample.timestamp.isoformat(),
                    'user_id': sample.user_id,
                    'session_id': sample.session_id,
                    'interaction_id': sample.interaction_id,
                    'quality_score': sample.quality_score,
                    'quality_level': sample.quality_level.value if sample.quality_level else None,
                    'is_validated': sample.is_validated,
                    'modality': json.dumps(sample.modality),
                    'input_data': json.dumps(sample.input_data),
                    'target_data': json.dumps(sample.target_data),
                    'context_data': json.dumps(sample.context_data),
                    'preprocessing_applied': json.dumps(sample.preprocessing_applied),
                    'raw_data': json.dumps(asdict(sample))
                }
                
                conn.execute("""
                    INSERT OR REPLACE INTO data_samples 
                    (sample_id, dataset_id, data_type, timestamp, user_id, session_id,
                     interaction_id, quality_score, quality_level, is_validated, modality,
                     input_data, target_data, context_data, preprocessing_applied, raw_data)
                    VALUES 
                    (:sample_id, :dataset_id, :data_type, :timestamp, :user_id, :session_id,
                     :interaction_id, :quality_score, :quality_level, :is_validated, :modality,
                     :input_data, :target_data, :context_data, :preprocessing_applied, :raw_data)
                """, data)
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Erro ao salvar amostra no banco: {e}")
            raise
    
    def _update_dataset_in_db(self, dataset: LearningDataset):
        """Atualiza dataset no banco de dados"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE datasets SET 
                    updated_at = ?, total_samples = ?, average_quality = ?, data_size_mb = ?
                    WHERE dataset_id = ?
                """, (
                    dataset.updated_at.isoformat(),
                    dataset.metrics.total_samples,
                    dataset.metrics.average_quality_score,
                    dataset.metrics.data_size_mb,
                    dataset.dataset_id
                ))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Erro ao atualizar dataset no banco: {e}")
    
    def _update_stats_for_dataset(self, dataset: LearningDataset):
        """Atualiza estatísticas globais para um dataset"""
        self._stats["total_datasets"] += 1
        self._stats["total_samples"] += len(dataset.samples)
        
        for sample in dataset.samples:
            self._stats["samples_by_type"][sample.data_type.value] += 1
            if sample.quality_level:
                self._stats["quality_distribution"][sample.quality_level.value] += 1
        
        for objective in dataset.learning_objectives:
            self._stats["learning_objectives_count"][objective.value] += 1
    
    def cleanup_low_quality_data(self, min_quality: float = 0.3) -> int:
        """
        Remove dados de baixa qualidade de todos os datasets
        
        Args:
            min_quality: Score mínimo de qualidade
            
        Returns:
            Número de amostras removidas
        """
        removed_count = 0
        
        try:
            for dataset_id, dataset in self._datasets_cache.items():
                original_count = len(dataset.samples)
                
                # Filtrar amostras de qualidade adequada
                dataset.samples = [s for s in dataset.samples if s.quality_score >= min_quality]
                
                removed_from_dataset = original_count - len(dataset.samples)
                removed_count += removed_from_dataset
                
                if removed_from_dataset > 0:
                    # Atualizar métricas
                    dataset.metrics = self._calculate_metrics(dataset.samples)
                    dataset.updated_at = datetime.now(timezone.utc)
                    
                    # Atualizar banco de dados
                    self._update_dataset_in_db(dataset)
                    
                    # Remover amostras do banco
                    with sqlite3.connect(self.db_path) as conn:
                        conn.execute("""
                            DELETE FROM data_samples 
                            WHERE dataset_id = ? AND quality_score < ?
                        """, (dataset_id, min_quality))
                        conn.commit()
                    
                    self.logger.info(f"Removidas {removed_from_dataset} amostras de baixa qualidade do dataset {dataset_id}")
            
            self.logger.info(f"Total de {removed_count} amostras de baixa qualidade removidas")
            return removed_count
            
        except Exception as e:
            self.logger.error(f"Erro na limpeza de dados: {e}")
            return 0
    
    def backup_datasets(self, backup_path: Optional[Path] = None) -> Path:
        """
        Cria backup completo de todos os datasets
        
        Args:
            backup_path: Caminho customizado para backup
            
        Returns:
            Caminho do arquivo de backup
        """
        try:
            if backup_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = self.data_dir / f"backup_learning_data_{timestamp}.gz"
            
            # Dados para backup
            backup_data = {
                'version': '1.0.0',
                'created_at': datetime.now(timezone.utc).isoformat(),
                'datasets': {},
                'global_stats': self._stats
            }
            
            # Incluir todos os datasets
            for dataset_id, dataset in self._datasets_cache.items():
                backup_data['datasets'][dataset_id] = asdict(dataset)
            
            # Salvar backup comprimido
            with gzip.open(backup_path, 'wb') as f:
                pickle.dump(backup_data, f)
            
            self.logger.info(f"Backup criado: {backup_path}")
            return backup_path
            
        except Exception as e:
            self.logger.error(f"Erro ao criar backup: {e}")
            raise
    
    def list_datasets(self) -> List[Dict[str, Any]]:
        """Lista todos os datasets com informações básicas"""
        datasets_info = []
        
        for dataset_id, dataset in self._datasets_cache.items():
            info = {
                'dataset_id': dataset_id,
                'name': dataset.name,
                'description': dataset.description,
                'learning_objectives': [obj.value for obj in dataset.learning_objectives],
                'total_samples': len(dataset.samples),
                'average_quality': dataset.metrics.average_quality_score,
                'created_at': dataset.created_at.isoformat(),
                'updated_at': dataset.updated_at.isoformat()
            }
            datasets_info.append(info)
        
        # Ordenar por data de criação (mais recente primeiro)
        datasets_info.sort(key=lambda x: x['created_at'], reverse=True)
        return datasets_info

# =========================================================================
# FUNÇÕES DE CONVENIÊNCIA
# =========================================================================

# Instância global do gerenciador
_learning_data_manager = None

def get_learning_data_manager() -> LearningDataManager:
    """Retorna instância global do gerenciador de dados de aprendizagem"""
    global _learning_data_manager
    if _learning_data_manager is None:
        _learning_data_manager = LearningDataManager()
    return _learning_data_manager

def create_conversation_dataset(name: str, description: str = "") -> LearningDataset:
    """Cria dataset específico para dados de conversa"""
    manager = get_learning_data_manager()
    return manager.create_dataset(
        name=name,
        description=description or "Dataset para melhoria de conversação",
        learning_objectives=[
            LearningObjective.CONVERSATION_IMPROVEMENT,
            LearningObjective.RESPONSE_GENERATION,
            LearningObjective.CONTEXT_UNDERSTANDING
        ]
    )

def create_emotion_dataset(name: str, description: str = "") -> LearningDataset:
    """Cria dataset específico para reconhecimento de emoções"""
    manager = get_learning_data_manager()
    return manager.create_dataset(
        name=name,
        description=description or "Dataset para reconhecimento de emoções",
        learning_objectives=[
            LearningObjective.EMOTION_RECOGNITION,
            LearningObjective.MULTIMODAL_FUSION
        ]
    )

def create_intent_dataset(name: str, description: str = "") -> LearningDataset:
    """Cria dataset específico para classificação de intenções"""
    manager = get_learning_data_manager()
    return manager.create_dataset(
        name=name,
        description=description or "Dataset para classificação de intenções",
        learning_objectives=[
            LearningObjective.INTENT_CLASSIFICATION,
            LearningObjective.CONTEXT_UNDERSTANDING
        ]
    )

def add_conversation_sample(dataset_id: str, 
                           user_input: str, 
                           bot_response: str,
                           context: Dict[str, Any] = None,
                           user_id: str = None,
                           session_id: str = None) -> bool:
    """Adiciona amostra de conversa a um dataset"""
    manager = get_learning_data_manager()
    
    sample = DataSample(
        sample_id=f"conv_{int(datetime.now().timestamp())}{hash(user_input) % 1000}",
        data_type=DataType.TEXT_PAIRS,
        timestamp=datetime.now(timezone.utc),
        input_data={"text": user_input},
        target_data={"text": bot_response},
        user_id=user_id,
        session_id=session_id,
        context_data=context or {},
        modality=["text"]
    )
    
    return manager.add_sample_to_dataset(dataset_id, sample)

def add_emotion_sample(dataset_id: str,
                      emotion: str,
                      confidence: float,
                      user_id: str = None,
                      context: Dict[str, Any] = None) -> bool:
    """Adiciona amostra de emoção a um dataset"""
    manager = get_learning_data_manager()
    
    sample = DataSample(
        sample_id=f"emo_{int(datetime.now().timestamp())}{hash(emotion) % 1000}",
        data_type=DataType.EMOTION_LABELS,
        timestamp=datetime.now(timezone.utc),
        input_data={},  # Dados de imagem seriam adicionados aqui
        target_data={"emotion": emotion, "confidence": confidence},
        user_id=user_id,
        context_data=context or {},
        modality=["emotion", "video"]
    )
    
    return manager.add_sample_to_dataset(dataset_id, sample)

def add_intent_sample(dataset_id: str,
                     text: str,
                     intent: str,
                     confidence: float,
                     user_id: str = None,
                     context: Dict[str, Any] = None) -> bool:
    """Adiciona amostra de intenção a um dataset"""
    manager = get_learning_data_manager()
    
    sample = DataSample(
        sample_id=f"intent_{int(datetime.now().timestamp())}{hash(text) % 1000}",
        data_type=DataType.INTENT_SAMPLES,
        timestamp=datetime.now(timezone.utc),
        input_data={"text": text},
        target_data={"intent": intent, "confidence": confidence},
        user_id=user_id,
        context_data=context or {},
        modality=["text"]
    )
    
    return manager.add_sample_to_dataset(dataset_id, sample)

# =========================================================================
# EXEMPLO DE USO
# =========================================================================

if __name__ == "__main__":
    # Exemplo de uso do sistema de gestão de dados de aprendizagem
    
    # Criar instância do gerenciador
    manager = get_learning_data_manager()
    
    # Criar dataset de conversação
    conversation_dataset = create_conversation_dataset(
        name="Conversas Cotidianas",
        description="Dataset para melhorar conversas do dia a dia"
    )
    
    print(f"Dataset criado: {conversation_dataset.dataset_id}")
    
    # Adicionar algumas amostras de exemplo
    add_conversation_sample(
        dataset_id=conversation_dataset.dataset_id,
        user_input="Olá, como você está?",
        bot_response="Olá! Estou muito bem, obrigada por perguntar! Como posso ajudá-lo hoje?",
        context={"time_of_day": "morning", "user_mood": "friendly"},
        user_id="user_123"
    )
    
    add_conversation_sample(
        dataset_id=conversation_dataset.dataset_id,
        user_input="Que horas são?",
        bot_response="Agora são 14:30. Posso ajudá-lo com mais alguma coisa?",
        context={"time_of_day": "afternoon"},
        user_id="user_123"
    )
    
    # Coletar dados das interações recentes
    collected_samples = manager.collect_from_interactions(
        dataset_id=conversation_dataset.dataset_id,
        hours_back=24,
        min_quality=0.6
    )
    
    print(f"Coletadas {collected_samples} amostras das interações")
    
    # Obter estatísticas
    stats = manager.get_dataset_stats(conversation_dataset.dataset_id)
    print(f"Estatísticas do dataset:")
    print(f"- Total de amostras: {stats['sample_counts']['total']}")
    print(f"- Qualidade média: {stats['quality_metrics']['average_score']:.2f}")
    print(f"- Tipos de dados: {stats['sample_counts']['by_type']}")
    
    # Exportar dataset
    export_path = manager.export_dataset(
        dataset_id=conversation_dataset.dataset_id,
        format="json",
        filter_quality=DataQuality.ACCEPTABLE
    )
    
    if export_path:
        print(f"Dataset exportado para: {export_path}")
    
    # Listar todos os datasets
    all_datasets = manager.list_datasets()
    print(f"Total de datasets: {len(all_datasets)}")
    
    # Fazer backup
    backup_path = manager.backup_datasets()
    print(f"Backup criado: {backup_path}")