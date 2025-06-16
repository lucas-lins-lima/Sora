# sora_robot/nlp/intent_recognition.py

import re
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import deque, Counter
from enum import Enum
import threading
import time
import json

# Bibliotecas de NLP
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import RSLPStemmer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    nltk = None

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    TfidfVectorizer = None

from utils.logger import get_logger
from utils.constants import INTENTS
from audio_processing.speech_recognition import RecognitionResult
import config

class IntentConfidence(Enum):
    """Níveis de confiança na detecção de intenção."""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

class IntentCategory(Enum):
    """Categorias gerais de intenção."""
    INFORMATIONAL = "informational"  # Busca informações
    TRANSACTIONAL = "transactional"  # Quer realizar algo
    SOCIAL = "social"  # Interação social
    NAVIGATIONAL = "navigational"  # Navegação/localização
    SUPPORT = "support"  # Suporte/ajuda
    FEEDBACK = "feedback"  # Feedback/opinião

class IntentMethod(Enum):
    """Métodos de detecção de intenção disponíveis."""
    PATTERN_BASED = "pattern_based"
    KEYWORD_BASED = "keyword_based"
    ML_BASED = "ml_based"
    ENSEMBLE = "ensemble"

@dataclass
class IntentPattern:
    """Padrão para detecção de intenção."""
    intent: str
    patterns: List[str]
    keywords: Set[str]
    weight: float = 1.0
    requires_all: bool = False  # Se True, todas as keywords devem estar presentes

@dataclass
class IntentEvidence:
    """Evidência que suporta uma intenção específica."""
    evidence_type: str  # "keyword", "pattern", "context"
    text_matched: str
    confidence: float
    position: int  # Posição no texto
    weight: float = 1.0

@dataclass
class IntentResult:
    """Resultado de detecção de intenção."""
    intent: str
    confidence: float
    confidence_level: IntentConfidence
    category: IntentCategory
    
    # Evidências que suportam esta intenção
    evidences: List[IntentEvidence] = field(default_factory=list)
    
    # Parâmetros extraídos (entidades)
    parameters: Dict[str, str] = field(default_factory=dict)
    
    # Contexto adicional
    context_factors: Dict[str, float] = field(default_factory=dict)

@dataclass
class IntentAnalysisResult:
    """Resultado completo da análise de intenção."""
    
    # Intenção principal
    primary_intent: IntentResult
    
    # Intenções alternativas (com menor confiança)
    alternative_intents: List[IntentResult] = field(default_factory=list)
    
    # Distribuição de probabilidades
    intent_distribution: Dict[str, float] = field(default_factory=dict)
    
    # Análise contextual
    requires_clarification: bool = False
    ambiguity_score: float = 0.0
    
    # Entidades extraídas globalmente
    entities: Dict[str, List[str]] = field(default_factory=dict)
    
    # Características linguísticas
    question_indicators: List[str] = field(default_factory=list)
    command_indicators: List[str] = field(default_factory=list)
    emotion_indicators: List[str] = field(default_factory=list)
    
    # Metadados
    text_length: int = 0
    word_count: int = 0
    processing_time: float = 0.0
    method_used: str = ""
    timestamp: float = 0.0

class IntentRecognition:
    """
    Classe responsável pelo reconhecimento de intenções em texto.
    Utiliza múltiplas abordagens para identificar o que o usuário deseja fazer.
    """
    
    def __init__(self, language: str = "pt", method: IntentMethod = IntentMethod.ENSEMBLE):
        """
        Inicializa o sistema de reconhecimento de intenção.
        
        Args:
            language: Idioma para análise ("pt" ou "en")
            method: Método de análise a ser usado
        """
        self.logger = get_logger(__name__)
        
        # Configurações
        self.language = language
        self.method = method
        
        # Estado do sistema
        self.is_initialized = False
        
        # Threading
        self.analysis_lock = threading.Lock()
        
        # Cache de análises
        self.intent_cache = {}
        self.cache_ttl = 180  # 3 minutos
        
        # Histórico de intenções
        self.intent_history = deque(maxlen=50)
        
        # Padrões de intenção
        self.intent_patterns = {}
        self.intent_keywords = {}
        
        # Modelos de ML
        self.vectorizer = None
        self.intent_vectors = None
        
        # Ferramentas de NLP
        self.stemmer = None
        self.stopwords_set = set()
        self.spacy_model = None
        
        # Contexto de conversa
        self.conversation_context = {
            'previous_intents': deque(maxlen=5),
            'current_topic': None,
            'user_preferences': {},
            'session_start': time.time()
        }
        
        # Métricas de performance
        self.performance_metrics = {
            'total_analyses': 0,
            'successful_analyses': 0,
            'processing_times': deque(maxlen=100),
            'average_processing_time': 0.0,
            'intents_detected': {intent: 0 for intent in INTENTS.ALL_INTENTS},
            'confidence_distribution': deque(maxlen=100),
            'cache_hits': 0,
            'clarification_requests': 0
        }
        
        # Inicializa componentes
        self._initialize_components()
        
        self.logger.info(f"IntentRecognition inicializado - Idioma: {language}, Método: {method.value}")
    
    def _initialize_components(self):
        """Inicializa componentes de NLP necessários."""
        try:
            # Inicializa padrões de intenção
            self._load_intent_patterns()
            
            # Inicializa NLTK se disponível
            if NLTK_AVAILABLE:
                self._initialize_nltk()
            
            # Inicializa spaCy se disponível
            if SPACY_AVAILABLE:
                self._initialize_spacy()
            
            # Inicializa ML se disponível
            if SKLEARN_AVAILABLE:
                self._initialize_ml_components()
            
            self.is_initialized = True
            self.logger.info("Componentes de reconhecimento de intenção inicializados")
            
        except Exception as e:
            self.logger.error(f"Erro ao inicializar componentes: {e}")
            self.is_initialized = False
    
    def _load_intent_patterns(self):
        """Carrega padrões de intenção por idioma."""
        try:
            if self.language == "pt":
                self.intent_patterns = {
                    INTENTS.GREETING: IntentPattern(
                        intent=INTENTS.GREETING,
                        patterns=[
                            r'\b(oi|olá|bom dia|boa tarde|boa noite|e aí|salve)\b',
                            r'\b(como vai|tudo bem|como está)\b',
                            r'\b(prazer|legal te conhecer)\b'
                        ],
                        keywords={'oi', 'olá', 'bom', 'dia', 'tarde', 'noite', 'como', 'vai', 'tudo', 'bem', 'está', 'prazer'},
                        weight=1.0
                    ),
                    
                    INTENTS.QUESTION: IntentPattern(
                        intent=INTENTS.QUESTION,
                        patterns=[
                            r'\b(o que|que|qual|quando|onde|como|por que|porque|quem)\b',
                            r'\b(você sabe|me diga|pode me falar|gostaria de saber)\b',
                            r'^(.*)\?$'  # Termina com ?
                        ],
                        keywords={'que', 'qual', 'quando', 'onde', 'como', 'porque', 'quem', 'sabe', 'diga', 'falar', 'saber'},
                        weight=1.2
                    ),
                    
                    INTENTS.REQUEST: IntentPattern(
                        intent=INTENTS.REQUEST,
                        patterns=[
                            r'\b(pode|poderia|consegue|me ajuda|me ajude|preciso|quero|gostaria)\b',
                            r'\b(por favor|faz favor|se possível)\b',
                            r'\b(fazer|realizar|executar|criar|gerar)\b'
                        ],
                        keywords={'pode', 'poderia', 'consegue', 'ajuda', 'ajude', 'preciso', 'quero', 'gostaria', 'favor', 'fazer', 'realizar'},
                        weight=1.1
                    ),
                    
                    INTENTS.COMPLAINT: IntentPattern(
                        intent=INTENTS.COMPLAINT,
                        patterns=[
                            r'\b(não funciona|não está funcionando|problema|erro|falha)\b',
                            r'\b(reclamação|reclamar|insatisfeito|ruim|péssimo)\b',
                            r'\b(não consigo|não consegui|difícil|complicado)\b'
                        ],
                        keywords={'não', 'funciona', 'problema', 'erro', 'falha', 'reclamação', 'reclamar', 'insatisfeito', 'ruim', 'consigo'},
                        weight=1.0
                    ),
                    
                    INTENTS.COMPLIMENT: IntentPattern(
                        intent=INTENTS.COMPLIMENT,
                        patterns=[
                            r'\b(obrigado|obrigada|valeu|muito bom|excelente|ótimo)\b',
                            r'\b(parabéns|muito bem|legal|bacana|incrível)\b',
                            r'\b(gostei|adorei|perfeito|fantástico|maravilhoso)\b'
                        ],
                        keywords={'obrigado', 'obrigada', 'valeu', 'bom', 'excelente', 'ótimo', 'parabéns', 'legal', 'gostei', 'adorei', 'perfeito'},
                        weight=1.0
                    ),
                    
                    INTENTS.GOODBYE: IntentPattern(
                        intent=INTENTS.GOODBYE,
                        patterns=[
                            r'\b(tchau|adeus|até logo|até mais|falou|até)\b',
                            r'\b(vou embora|tenho que ir|preciso sair)\b',
                            r'\b(boa noite|bom fim de semana|até amanhã)\b'
                        ],
                        keywords={'tchau', 'adeus', 'até', 'logo', 'mais', 'falou', 'embora', 'ir', 'sair', 'noite', 'amanhã'},
                        weight=1.0
                    ),
                    
                    INTENTS.EVENT_INFO: IntentPattern(
                        intent=INTENTS.EVENT_INFO,
                        patterns=[
                            r'\b(evento|programação|agenda|atividade|palestra|apresentação)\b',
                            r'\b(que horas|horário|quando começa|quando termina)\b',
                            r'\b(informação|informações|detalhes|programa)\b'
                        ],
                        keywords={'evento', 'programação', 'agenda', 'atividade', 'palestra', 'horas', 'horário', 'começa', 'termina', 'informação'},
                        weight=1.1
                    ),
                    
                    INTENTS.LOCATION_REQUEST: IntentPattern(
                        intent=INTENTS.LOCATION_REQUEST,
                        patterns=[
                            r'\b(onde fica|onde está|localização|local|endereço)\b',
                            r'\b(como chegar|como ir|caminho|direção|rota)\b',
                            r'\b(estande|sala|auditório|banheiro|saída)\b'
                        ],
                        keywords={'onde', 'fica', 'está', 'localização', 'local', 'endereço', 'chegar', 'ir', 'caminho', 'direção', 'estande', 'sala'},
                        weight=1.1
                    ),
                    
                    INTENTS.HELP: IntentPattern(
                        intent=INTENTS.HELP,
                        patterns=[
                            r'\b(ajuda|socorro|help|auxílio|suporte)\b',
                            r'\b(não sei|não entendo|não compreendo|confuso)\b',
                            r'\b(como funciona|como usar|como faço|tutorial)\b'
                        ],
                        keywords={'ajuda', 'socorro', 'help', 'auxílio', 'suporte', 'sei', 'entendo', 'compreendo', 'funciona', 'usar', 'faço'},
                        weight=1.2
                    ),
                    
                    INTENTS.REPEAT: IntentPattern(
                        intent=INTENTS.REPEAT,
                        patterns=[
                            r'\b(repete|repetir|de novo|novamente|outra vez)\b',
                            r'\b(não ouvi|não escutei|pode repetir|falar de novo)\b',
                            r'\b(como|o que você disse|não entendi)\b'
                        ],
                        keywords={'repete', 'repetir', 'novo', 'novamente', 'vez', 'ouvi', 'escutei', 'repetir', 'disse', 'entendi'},
                        weight=1.0
                    )
                }
                
            else:  # English
                self.intent_patterns = {
                    INTENTS.GREETING: IntentPattern(
                        intent=INTENTS.GREETING,
                        patterns=[
                            r'\b(hi|hello|hey|good morning|good afternoon|good evening)\b',
                            r'\b(how are you|how do you do|nice to meet you)\b'
                        ],
                        keywords={'hi', 'hello', 'hey', 'good', 'morning', 'afternoon', 'evening', 'how', 'are', 'you', 'nice', 'meet'},
                        weight=1.0
                    ),
                    
                    INTENTS.QUESTION: IntentPattern(
                        intent=INTENTS.QUESTION,
                        patterns=[
                            r'\b(what|which|when|where|how|why|who|whose|whom)\b',
                            r'\b(do you know|can you tell|could you explain)\b',
                            r'^(.*)\?$'
                        ],
                        keywords={'what', 'which', 'when', 'where', 'how', 'why', 'who', 'know', 'tell', 'explain'},
                        weight=1.2
                    ),
                    
                    INTENTS.REQUEST: IntentPattern(
                        intent=INTENTS.REQUEST,
                        patterns=[
                            r'\b(can|could|would|please|help me|i need|i want|i would like)\b',
                            r'\b(do|make|create|generate|perform)\b'
                        ],
                        keywords={'can', 'could', 'would', 'please', 'help', 'need', 'want', 'like', 'do', 'make', 'create'},
                        weight=1.1
                    ),
                    
                    INTENTS.COMPLAINT: IntentPattern(
                        intent=INTENTS.COMPLAINT,
                        patterns=[
                            r'\b(not working|broken|problem|error|issue|bug)\b',
                            r'\b(complaint|complain|dissatisfied|bad|terrible)\b',
                            r'\b(can\'t|cannot|unable|difficult|hard)\b'
                        ],
                        keywords={'not', 'working', 'broken', 'problem', 'error', 'issue', 'complaint', 'bad', 'terrible', 'difficult'},
                        weight=1.0
                    ),
                    
                    INTENTS.COMPLIMENT: IntentPattern(
                        intent=INTENTS.COMPLIMENT,
                        patterns=[
                            r'\b(thank you|thanks|great|excellent|awesome|amazing)\b',
                            r'\b(good job|well done|perfect|fantastic|wonderful)\b',
                            r'\b(love it|like it|impressive|brilliant)\b'
                        ],
                        keywords={'thank', 'thanks', 'great', 'excellent', 'awesome', 'good', 'perfect', 'love', 'like', 'brilliant'},
                        weight=1.0
                    ),
                    
                    INTENTS.GOODBYE: IntentPattern(
                        intent=INTENTS.GOODBYE,
                        patterns=[
                            r'\b(bye|goodbye|see you|farewell|take care)\b',
                            r'\b(have to go|need to leave|going now)\b',
                            r'\b(good night|have a good day|until tomorrow)\b'
                        ],
                        keywords={'bye', 'goodbye', 'see', 'you', 'farewell', 'take', 'care', 'go', 'leave', 'night', 'day'},
                        weight=1.0
                    )
                }
            
            # Mapeia intenções para categorias
            self.intent_categories = {
                INTENTS.GREETING: IntentCategory.SOCIAL,
                INTENTS.GOODBYE: IntentCategory.SOCIAL,
                INTENTS.COMPLIMENT: IntentCategory.SOCIAL,
                INTENTS.QUESTION: IntentCategory.INFORMATIONAL,
                INTENTS.EVENT_INFO: IntentCategory.INFORMATIONAL,
                INTENTS.REQUEST: IntentCategory.TRANSACTIONAL,
                INTENTS.COMPLAINT: IntentCategory.FEEDBACK,
                INTENTS.LOCATION_REQUEST: IntentCategory.NAVIGATIONAL,
                INTENTS.HELP: IntentCategory.SUPPORT,
                INTENTS.REPEAT: IntentCategory.SUPPORT,
                INTENTS.CLARIFICATION: IntentCategory.SUPPORT
            }
            
            self.logger.info(f"Padrões de intenção carregados: {len(self.intent_patterns)} intenções")
            
        except Exception as e:
            self.logger.error(f"Erro ao carregar padrões de intenção: {e}")
    
    def _initialize_nltk(self):
        """Inicializa componentes do NLTK."""
        try:
            # Configura stemmer e stopwords
            if self.language == "pt":
                self.stemmer = RSLPStemmer()
                self.stopwords_set = set(stopwords.words('portuguese'))
            else:
                from nltk.stem import PorterStemmer
                self.stemmer = PorterStemmer()
                self.stopwords_set = set(stopwords.words('english'))
            
            self.logger.info("NLTK inicializado para reconhecimento de intenção")
            
        except Exception as e:
            self.logger.error(f"Erro ao inicializar NLTK: {e}")
    
    def _initialize_spacy(self):
        """Inicializa modelo do spaCy."""
        try:
            model_name = "pt_core_news_sm" if self.language == "pt" else "en_core_web_sm"
            
            try:
                self.spacy_model = spacy.load(model_name)
                self.logger.info(f"Modelo spaCy {model_name} carregado para intenção")
            except OSError:
                self.logger.warning(f"Modelo spaCy {model_name} não encontrado")
                
        except Exception as e:
            self.logger.error(f"Erro ao inicializar spaCy: {e}")
    
    def _initialize_ml_components(self):
        """Inicializa componentes de machine learning."""
        try:
            # Prepara dados de treinamento básicos
            training_texts = []
            training_labels = []
            
            for intent, pattern in self.intent_patterns.items():
                # Gera exemplos sintéticos baseados nos padrões
                examples = self._generate_training_examples(pattern)
                training_texts.extend(examples)
                training_labels.extend([intent] * len(examples))
            
            if training_texts:
                # Treina vectorizer TF-IDF
                self.vectorizer = TfidfVectorizer(
                    max_features=1000,
                    stop_words=list(self.stopwords_set) if self.stopwords_set else None,
                    ngram_range=(1, 2)
                )
                
                self.intent_vectors = self.vectorizer.fit_transform(training_texts)
                
                self.logger.info(f"Componentes ML inicializados com {len(training_texts)} exemplos")
            
        except Exception as e:
            self.logger.error(f"Erro ao inicializar componentes ML: {e}")
    
    def _generate_training_examples(self, pattern: IntentPattern) -> List[str]:
        """Gera exemplos de treinamento baseados nos padrões."""
        examples = []
        
        # Exemplos baseados em keywords (implementação simplificada)
        if self.language == "pt":
            templates = {
                INTENTS.GREETING: ["oi", "olá", "bom dia", "como vai"],
                INTENTS.QUESTION: ["o que é isso", "qual é", "como funciona", "onde fica"],
                INTENTS.REQUEST: ["pode me ajudar", "preciso de", "gostaria de", "me ajude"],
                INTENTS.COMPLAINT: ["não funciona", "problema com", "erro no", "não consigo"],
                INTENTS.COMPLIMENT: ["muito bom", "obrigado", "excelente", "parabéns"],
                INTENTS.GOODBYE: ["tchau", "até logo", "vou embora", "até mais"]
            }
        else:
            templates = {
                INTENTS.GREETING: ["hi", "hello", "good morning", "how are you"],
                INTENTS.QUESTION: ["what is", "which one", "how does", "where is"],
                INTENTS.REQUEST: ["can you help", "i need", "please do", "help me"],
                INTENTS.COMPLAINT: ["not working", "problem with", "error in", "can't do"],
                INTENTS.COMPLIMENT: ["very good", "thank you", "excellent", "great job"],
                INTENTS.GOODBYE: ["bye", "see you", "goodbye", "take care"]
            }
        
        examples = templates.get(pattern.intent, [])
        return examples[:5]  # Máximo 5 exemplos por intenção
    
    def analyze_intent(self, text: str) -> Optional[IntentAnalysisResult]:
        """
        Analisa intenção de um texto.
        
        Args:
            text: Texto para análise
            
        Returns:
            Optional[IntentAnalysisResult]: Resultado da análise ou None se falhou
        """
        if not text or not text.strip():
            return None
        
        start_time = time.time()
        
        try:
            with self.analysis_lock:
                # Normaliza texto
                normalized_text = self._normalize_text(text)
                
                # Verifica cache
                cache_key = self._generate_cache_key(normalized_text)
                if cache_key in self.intent_cache:
                    cached_result, cache_time = self.intent_cache[cache_key]
                    if time.time() - cache_time < self.cache_ttl:
                        self.performance_metrics['cache_hits'] += 1
                        return cached_result
                
                # Análise baseada no método selecionado
                result = None
                
                if self.method == IntentMethod.ENSEMBLE:
                    result = self._analyze_ensemble(normalized_text)
                elif self.method == IntentMethod.PATTERN_BASED:
                    result = self._analyze_pattern_based(normalized_text)
                elif self.method == IntentMethod.KEYWORD_BASED:
                    result = self._analyze_keyword_based(normalized_text)
                elif self.method == IntentMethod.ML_BASED and self.vectorizer:
                    result = self._analyze_ml_based(normalized_text)
                else:
                    # Fallback para análise baseada em padrões
                    result = self._analyze_pattern_based(normalized_text)
                
                if result:
                    result.timestamp = start_time
                    result.processing_time = time.time() - start_time
                    result.text_length = len(text)
                    result.word_count = len(normalized_text.split())
                    result.method_used = self.method.value
                    
                    # Extrai entidades
                    result.entities = self._extract_entities(normalized_text)
                    
                    # Analisa características linguísticas
                    result.question_indicators = self._find_question_indicators(normalized_text)
                    result.command_indicators = self._find_command_indicators(normalized_text)
                    
                    # Verifica se precisa de clarificação
                    result.requires_clarification = self._needs_clarification(result)
                    result.ambiguity_score = self._calculate_ambiguity_score(result)
                    
                    # Atualiza contexto de conversa
                    self._update_conversation_context(result)
                    
                    # Atualiza cache
                    self.intent_cache[cache_key] = (result, time.time())
                    
                    # Atualiza histórico
                    self.intent_history.append(result)
                    
                    # Limpa cache antigo
                    self._cleanup_cache()
                    
                    # Atualiza métricas
                    self._update_performance_metrics(result, True)
                    
                    self.logger.debug(f"Intenção analisada: {result.primary_intent.intent} (confiança: {result.primary_intent.confidence:.2f})")
                    
                    return result
                else:
                    self._update_performance_metrics(None, False)
                    return None
                    
        except Exception as e:
            self.logger.error(f"Erro na análise de intenção: {e}")
            self._update_performance_metrics(None, False)
            return None
    
    def analyze_recognition_result(self, recognition_result: RecognitionResult) -> Optional[IntentAnalysisResult]:
        """
        Analisa intenção de um resultado de reconhecimento de fala.
        
        Args:
            recognition_result: Resultado do reconhecimento de fala
            
        Returns:
            Optional[IntentAnalysisResult]: Resultado da análise de intenção
        """
        if not recognition_result or not recognition_result.full_text:
            return None
        
        return self.analyze_intent(recognition_result.full_text)
    
    def _normalize_text(self, text: str) -> str:
        """Normaliza texto para análise."""
        try:
            # Remove caracteres especiais excessivos
            text = re.sub(r'[^\w\s\.\!\?\,\;\:]', ' ', text)
            
            # Normaliza espaços
            text = re.sub(r'\s+', ' ', text)
            
            # Remove espaços extras
            text = text.strip()
            
            # Converte para minúsculas
            text = text.lower()
            
            return text
            
        except Exception as e:
            self.logger.error(f"Erro na normalização de texto: {e}")
            return text
    
    def _analyze_ensemble(self, text: str) -> Optional[IntentAnalysisResult]:
        """Análise ensemble combinando múltiplos métodos."""
        try:
            results = []
            weights = []
            
            # Análise baseada em padrões (sempre disponível)
            pattern_result = self._analyze_pattern_based(text)
            if pattern_result:
                results.append(pattern_result)
                weights.append(0.4)
            
            # Análise baseada em keywords
            keyword_result = self._analyze_keyword_based(text)
            if keyword_result:
                results.append(keyword_result)
                weights.append(0.3)
            
            # Análise ML se disponível
            if self.vectorizer:
                ml_result = self._analyze_ml_based(text)
                if ml_result:
                    results.append(ml_result)
                    weights.append(0.3)
            
            if not results:
                return None
            
            # Normaliza pesos
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
            
            # Combina resultados
            combined_scores = {}
            all_evidences = []
            
            for result, weight in zip(results, weights):
                intent = result.primary_intent.intent
                confidence = result.primary_intent.confidence * weight
                
                if intent in combined_scores:
                    combined_scores[intent] += confidence
                else:
                    combined_scores[intent] = confidence
                
                # Coleta evidências
                all_evidences.extend(result.primary_intent.evidences)
            
            # Encontra intenção principal
            primary_intent_name = max(combined_scores, key=combined_scores.get)
            primary_confidence = combined_scores[primary_intent_name]
            
            # Cria resultado combinado
            primary_intent = IntentResult(
                intent=primary_intent_name,
                confidence=primary_confidence,
                confidence_level=self._confidence_to_level(primary_confidence),
                category=self.intent_categories.get(primary_intent_name, IntentCategory.INFORMATIONAL),
                evidences=all_evidences
            )
            
            # Intenções alternativas
            alternative_intents = []
            for intent, score in sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[1:3]:
                if score > 0.2:  # Threshold mínimo
                    alt_intent = IntentResult(
                        intent=intent,
                        confidence=score,
                        confidence_level=self._confidence_to_level(score),
                        category=self.intent_categories.get(intent, IntentCategory.INFORMATIONAL)
                    )
                    alternative_intents.append(alt_intent)
            
            result = IntentAnalysisResult(
                primary_intent=primary_intent,
                alternative_intents=alternative_intents,
                intent_distribution=combined_scores,
                method_used="ensemble"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Erro na análise ensemble: {e}")
            return None
    
    def _analyze_pattern_based(self, text: str) -> Optional[IntentAnalysisResult]:
        """Análise baseada em padrões regex."""
        try:
            intent_scores = {}
            all_evidences = {}
            
            for intent, pattern in self.intent_patterns.items():
                score = 0.0
                evidences = []
                
                # Verifica cada padrão
                for regex_pattern in pattern.patterns:
                    matches = re.finditer(regex_pattern, text, re.IGNORECASE)
                    for match in matches:
                        evidence = IntentEvidence(
                            evidence_type="pattern",
                            text_matched=match.group(),
                            confidence=0.8,
                            position=match.start(),
                            weight=pattern.weight
                        )
                        evidences.append(evidence)
                        score += 0.8 * pattern.weight
                
                if evidences:
                    intent_scores[intent] = min(1.0, score)
                    all_evidences[intent] = evidences
            
            if not intent_scores:
                return None
            
            # Encontra intenção principal
            primary_intent_name = max(intent_scores, key=intent_scores.get)
            primary_confidence = intent_scores[primary_intent_name]
            
            primary_intent = IntentResult(
                intent=primary_intent_name,
                confidence=primary_confidence,
                confidence_level=self._confidence_to_level(primary_confidence),
                category=self.intent_categories.get(primary_intent_name, IntentCategory.INFORMATIONAL),
                evidences=all_evidences.get(primary_intent_name, [])
            )
            
            # Intenções alternativas
            alternative_intents = []
            for intent, score in sorted(intent_scores.items(), key=lambda x: x[1], reverse=True)[1:3]:
                if score > 0.3:
                    alt_intent = IntentResult(
                        intent=intent,
                        confidence=score,
                        confidence_level=self._confidence_to_level(score),
                        category=self.intent_categories.get(intent, IntentCategory.INFORMATIONAL),
                        evidences=all_evidences.get(intent, [])
                    )
                    alternative_intents.append(alt_intent)
            
            result = IntentAnalysisResult(
                primary_intent=primary_intent,
                alternative_intents=alternative_intents,
                intent_distribution=intent_scores,
                method_used="pattern_based"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Erro na análise baseada em padrões: {e}")
            return None
    
    def _analyze_keyword_based(self, text: str) -> Optional[IntentAnalysisResult]:
        """Análise baseada em palavras-chave."""
        try:
            words = self._tokenize_words(text)
            if not words:
                return None
            
            intent_scores = {}
            all_evidences = {}
            
            for intent, pattern in self.intent_patterns.items():
                score = 0.0
                evidences = []
                matched_keywords = 0
                
                for word in words:
                    if word in self.stopwords_set:
                        continue
                    
                    # Stemming se disponível
                    stemmed_word = self.stemmer.stem(word) if self.stemmer else word
                    
                    # Verifica se palavra está nas keywords
                    if word in pattern.keywords or stemmed_word in pattern.keywords:
                        evidence = IntentEvidence(
                            evidence_type="keyword",
                            text_matched=word,
                            confidence=0.7,
                            position=text.find(word),
                            weight=pattern.weight
                        )
                        evidences.append(evidence)
                        score += 0.7 * pattern.weight
                        matched_keywords += 1
                
                # Bonus se várias keywords foram encontradas
                if matched_keywords > 1:
                    score *= 1.2
                
                if evidences:
                    intent_scores[intent] = min(1.0, score / len(pattern.keywords))  # Normaliza pelo número de keywords
                    all_evidences[intent] = evidences
            
            if not intent_scores:
                return None
            
            # Encontra intenção principal
            primary_intent_name = max(intent_scores, key=intent_scores.get)
            primary_confidence = intent_scores[primary_intent_name]
            
            primary_intent = IntentResult(
                intent=primary_intent_name,
                confidence=primary_confidence,
                confidence_level=self._confidence_to_level(primary_confidence),
                category=self.intent_categories.get(primary_intent_name, IntentCategory.INFORMATIONAL),
                evidences=all_evidences.get(primary_intent_name, [])
            )
            
            result = IntentAnalysisResult(
                primary_intent=primary_intent,
                intent_distribution=intent_scores,
                method_used="keyword_based"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Erro na análise baseada em keywords: {e}")
            return None
    
    def _analyze_ml_based(self, text: str) -> Optional[IntentAnalysisResult]:
        """Análise baseada em machine learning."""
        try:
            if not self.vectorizer or self.intent_vectors is None:
                return None
            
            # Vectoriza o texto
            text_vector = self.vectorizer.transform([text])
            
            # Calcula similaridade com exemplos de treinamento
            similarities = cosine_similarity(text_vector, self.intent_vectors).flatten()
            
            # Encontra exemplos mais similares
            top_indices = np.argsort(similarities)[-5:][::-1]  # Top 5
            
            # Calcula scores por intenção
            intent_scores = {}
            for idx in top_indices:
                if similarities[idx] > 0.1:  # Threshold mínimo
                    # Aqui precisaríamos mapear índice para intenção
                    # Implementação simplificada
                    pass
            
            # Por enquanto, retorna resultado baseado em maior similaridade
            if len(similarities) > 0 and np.max(similarities) > 0.3:
                confidence = float(np.max(similarities))
                
                # Mapeia para intenção (implementação simplificada)
                # Em implementação real, usaríamos labels de treinamento
                primary_intent = IntentResult(
                    intent=INTENTS.QUESTION,  # Placeholder
                    confidence=confidence,
                    confidence_level=self._confidence_to_level(confidence),
                    category=IntentCategory.INFORMATIONAL
                )
                
                result = IntentAnalysisResult(
                    primary_intent=primary_intent,
                    intent_distribution={INTENTS.QUESTION: confidence},
                    method_used="ml_based"
                )
                
                return result
            
            return None
            
        except Exception as e:
            self.logger.error(f"Erro na análise ML: {e}")
            return None
    
    def _tokenize_words(self, text: str) -> List[str]:
        """Tokeniza texto em palavras."""
        try:
            if NLTK_AVAILABLE:
                return word_tokenize(text, language='portuguese' if self.language == 'pt' else 'english')
            else:
                words = re.findall(r'\b\w+\b', text.lower())
                return words
                
        except Exception as e:
            self.logger.error(f"Erro na tokenização: {e}")
            return text.split()
    
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extrai entidades nomeadas do texto."""
        entities = {
            'time': [],
            'location': [],
            'person': [],
            'organization': [],
            'number': []
        }
        
        try:
            # Entidades de tempo
            time_patterns = [
                r'\b(\d{1,2}:\d{2})\b',  # HH:MM
                r'\b(\d{1,2}h\d{0,2})\b',  # 14h30
                r'\b(amanhã|hoje|ontem|semana|mês|ano)\b',
                r'\b(manhã|tarde|noite|madrugada)\b'
            ]
            
            for pattern in time_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                entities['time'].extend([match if isinstance(match, str) else match[0] for match in matches])
            
            # Entidades de localização
            location_patterns = [
                r'\b(sala|auditório|estande|pavilhão|andar|bloco)\s+\w+\b',
                r'\b(térreo|primeiro|segundo|terceiro)\s+andar\b'
            ]
            
            for pattern in location_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                entities['location'].extend(matches)
            
            # Números
            number_patterns = [
                r'\b(\d+)\b'
            ]
            
            for pattern in number_patterns:
                matches = re.findall(pattern, text)
                entities['number'].extend(matches)
            
            # Remove duplicatas
            for key in entities:
                entities[key] = list(set(entities[key]))
            
            # Remove entidades vazias
            entities = {k: v for k, v in entities.items() if v}
            
        except Exception as e:
            self.logger.error(f"Erro na extração de entidades: {e}")
        
        return entities
    
    def _find_question_indicators(self, text: str) -> List[str]:
        """Encontra indicadores de pergunta no texto."""
        indicators = []
        
        if self.language == "pt":
            question_words = ['o que', 'que', 'qual', 'quando', 'onde', 'como', 'por que', 'porque', 'quem']
        else:
            question_words = ['what', 'which', 'when', 'where', 'how', 'why', 'who', 'whose', 'whom']
        
        for word in question_words:
            if word in text:
                indicators.append(word)
        
        # Verifica se termina com ?
        if text.strip().endswith('?'):
            indicators.append('question_mark')
        
        return indicators
    
    def _find_command_indicators(self, text: str) -> List[str]:
        """Encontra indicadores de comando no texto."""
        indicators = []
        
        if self.language == "pt":
            command_words = ['faça', 'execute', 'realize', 'crie', 'gere', 'mostre', 'abra', 'feche']
        else:
            command_words = ['do', 'execute', 'perform', 'create', 'generate', 'show', 'open', 'close']
        
        for word in command_words:
            if word in text:
                indicators.append(word)
        
        return indicators
    
    def _confidence_to_level(self, confidence: float) -> IntentConfidence:
        """Converte score de confiança para nível."""
        if confidence >= 0.9:
            return IntentConfidence.VERY_HIGH
        elif confidence >= 0.7:
            return IntentConfidence.HIGH
        elif confidence >= 0.5:
            return IntentConfidence.MEDIUM
        elif confidence >= 0.3:
            return IntentConfidence.LOW
        else:
            return IntentConfidence.VERY_LOW
    
    def _needs_clarification(self, result: IntentAnalysisResult) -> bool:
        """Determina se a análise precisa de clarificação."""
        # Precisa de clarificação se:
        # 1. Confiança baixa
        # 2. Múltiplas intenções com confiança similar
        # 3. Intenção muito genérica
        
        if result.primary_intent.confidence < 0.5:
            return True
        
        if len(result.alternative_intents) > 0:
            diff = result.primary_intent.confidence - result.alternative_intents[0].confidence
            if diff < 0.2:  # Diferença pequena
                return True
        
        return False
    
    def _calculate_ambiguity_score(self, result: IntentAnalysisResult) -> float:
        """Calcula score de ambiguidade da análise."""
        if not result.alternative_intents:
            return 0.0
        
        # Ambiguidade baseada na diferença entre primeira e segunda opções
        primary_conf = result.primary_intent.confidence
        secondary_conf = result.alternative_intents[0].confidence
        
        ambiguity = 1.0 - (primary_conf - secondary_conf)
        return max(0.0, min(1.0, ambiguity))
    
    def _update_conversation_context(self, result: IntentAnalysisResult):
        """Atualiza contexto da conversa."""
        try:
            # Adiciona intenção ao histórico
            self.conversation_context['previous_intents'].append({
                'intent': result.primary_intent.intent,
                'confidence': result.primary_intent.confidence,
                'timestamp': time.time()
            })
            
            # Atualiza tópico atual baseado na categoria
            self.conversation_context['current_topic'] = result.primary_intent.category.value
            
        except Exception as e:
            self.logger.error(f"Erro ao atualizar contexto: {e}")
    
    def _generate_cache_key(self, text: str) -> str:
        """Gera chave de cache baseada no texto."""
        text_hash = hash(text)
        return f"intent_{text_hash}_{len(text)}"
    
    def _cleanup_cache(self):
        """Remove entradas antigas do cache."""
        current_time = time.time()
        expired_keys = []
        
        for cache_key, (_, cache_time) in self.intent_cache.items():
            if current_time - cache_time > self.cache_ttl:
                expired_keys.append(cache_key)
        
        for key in expired_keys:
            del self.intent_cache[key]
    
    def _update_performance_metrics(self, result: Optional[IntentAnalysisResult], success: bool):
        """Atualiza métricas de performance."""
        self.performance_metrics['total_analyses'] += 1
        
        if success and result:
            self.performance_metrics['successful_analyses'] += 1
            
            # Atualiza tempos de processamento
            self.performance_metrics['processing_times'].append(result.processing_time)
            
            # Conta intenções detectadas
            intent = result.primary_intent.intent
            if intent in self.performance_metrics['intents_detected']:
                self.performance_metrics['intents_detected'][intent] += 1
            
            # Distribui-ção de confiança
            self.performance_metrics['confidence_distribution'].append(result.primary_intent.confidence)
            
            # Conta pedidos de clarificação
            if result.requires_clarification:
                self.performance_metrics['clarification_requests'] += 1
    
    def get_conversation_context(self) -> Dict:
        """Retorna contexto atual da conversa."""
        return self.conversation_context.copy()
    
    def get_intent_trend(self, window_size: int = 10) -> Dict:
        """
        Analisa tendência de intenções nas últimas análises.
        
        Args:
            window_size: Número de análises a considerar
            
        Returns:
            Dict: Dados de tendência
        """
        try:
            if len(self.intent_history) < 2:
                return {'trend': 'insufficient_data'}
            
            # Pega últimas análises
            recent_analyses = list(self.intent_history)[-window_size:]
            
            # Conta intenções
            intents = [analysis.primary_intent.intent for analysis in recent_analyses]
            intent_counts = Counter(intents)
            
            # Intenção predominante
            dominant_intent = intent_counts.most_common(1)[0][0] if intent_counts else None
            
            # Confiança média
            avg_confidence = np.mean([analysis.primary_intent.confidence for analysis in recent_analyses])
            
            # Taxa de clarificação
            clarification_rate = sum(1 for analysis in recent_analyses if analysis.requires_clarification) / len(recent_analyses)
            
            return {
                'dominant_intent': dominant_intent,
                'intent_distribution': dict(intent_counts),
                'average_confidence': avg_confidence,
                'clarification_rate': clarification_rate,
                'sample_size': len(recent_analyses),
                'session_duration': time.time() - self.conversation_context['session_start']
            }
            
        except Exception as e:
            self.logger.error(f"Erro ao calcular tendência de intenção: {e}")
            return {'error': str(e)}
    
    def get_performance_metrics(self) -> Dict:
        """Retorna métricas de performance."""
        metrics = self.performance_metrics.copy()
        
        # Adiciona métricas calculadas
        if metrics['total_analyses'] > 0:
            metrics['success_rate'] = metrics['successful_analyses'] / metrics['total_analyses']
            metrics['cache_hit_rate'] = metrics['cache_hits'] / metrics['total_analyses']
            metrics['clarification_rate'] = metrics['clarification_requests'] / metrics['total_analyses']
        else:
            metrics['success_rate'] = 0.0
            metrics['cache_hit_rate'] = 0.0
            metrics['clarification_rate'] = 0.0
        
        if metrics['confidence_distribution']:
            metrics['average_confidence'] = np.mean(metrics['confidence_distribution'])
        else:
            metrics['average_confidence'] = 0.0
        
        if metrics['processing_times']:
            metrics['average_processing_time'] = np.mean(metrics['processing_times'])
        else:
            metrics['average_processing_time'] = 0.0
        
        return metrics
    
    def clear_cache(self):
        """Limpa cache de análises."""
        self.intent_cache.clear()
        self.logger.info("Cache de reconhecimento de intenção limpo")
    
    def clear_history(self):
        """Limpa histórico de análises."""
        self.intent_history.clear()
        self.conversation_context['previous_intents'].clear()
        self.logger.info("Histórico de reconhecimento de intenção limpo")
    
    def reset_conversation(self):
        """Reinicia contexto da conversa."""
        self.conversation_context = {
            'previous_intents': deque(maxlen=5),
            'current_topic': None,
            'user_preferences': {},
            'session_start': time.time()
        }
        self.logger.info("Contexto de conversa reiniciado")