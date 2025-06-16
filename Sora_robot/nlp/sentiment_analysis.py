# sora_robot/nlp/sentiment_analysis.py

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
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import RSLPStemmer  # Para português
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
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    TextBlob = None

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    pipeline = None

from utils.logger import get_logger
from utils.constants import SENTIMENTS, EMOTIONS
from audio_processing.speech_recognition import RecognitionResult
import config

class SentimentIntensity(Enum):
    """Intensidade do sentimento."""
    VERY_NEGATIVE = "very_negative"
    NEGATIVE = "negative"
    SLIGHTLY_NEGATIVE = "slightly_negative"
    NEUTRAL = "neutral"
    SLIGHTLY_POSITIVE = "slightly_positive"
    POSITIVE = "positive"
    VERY_POSITIVE = "very_positive"

class SentimentMethod(Enum):
    """Métodos de análise de sentimento disponíveis."""
    LEXICON_BASED = "lexicon_based"
    TEXTBLOB = "textblob"
    TRANSFORMERS = "transformers"
    SPACY = "spacy"
    ENSEMBLE = "ensemble"

@dataclass
class WordSentiment:
    """Sentimento de uma palavra específica."""
    word: str
    sentiment: str
    score: float
    confidence: float
    context_modifier: float = 1.0  # Modificador baseado no contexto

@dataclass
class SentenceSentiment:
    """Sentimento de uma sentença."""
    text: str
    sentiment: str
    score: float
    confidence: float
    intensity: SentimentIntensity
    
    # Análise detalhada
    positive_words: List[WordSentiment] = field(default_factory=list)
    negative_words: List[WordSentiment] = field(default_factory=list)
    neutral_words: List[WordSentiment] = field(default_factory=list)
    
    # Modificadores contextuais
    negation_detected: bool = False
    intensifiers_detected: List[str] = field(default_factory=list)
    diminishers_detected: List[str] = field(default_factory=list)

@dataclass
class SentimentAnalysisResult:
    """Resultado completo da análise de sentimento."""
    
    # Resultado geral
    overall_sentiment: str
    overall_score: float  # -1.0 (muito negativo) a 1.0 (muito positivo)
    overall_confidence: float
    intensity: SentimentIntensity
    
    # Análise por sentença
    sentences: List[SentenceSentiment] = field(default_factory=list)
    
    # Distribuição de sentimentos
    sentiment_distribution: Dict[str, float] = field(default_factory=dict)
    
    # Características linguísticas
    emotional_indicators: List[str] = field(default_factory=list)
    sentiment_trends: List[float] = field(default_factory=list)  # Evolução do sentimento ao longo do texto
    
    # Contexto e modificadores
    context_factors: Dict[str, float] = field(default_factory=dict)
    linguistic_patterns: List[str] = field(default_factory=list)
    
    # Metadados
    text_length: int = 0
    word_count: int = 0
    sentence_count: int = 0
    method_used: str = ""
    processing_time: float = 0.0
    timestamp: float = 0.0
    
    # Comparação com outros tipos de análise emocional
    facial_emotion_correlation: float = 0.0
    vocal_emotion_correlation: float = 0.0

class SentimentAnalysis:
    """
    Classe responsável pela análise de sentimento em texto.
    Utiliza múltiplas abordagens para análise robusta e contextual.
    """
    
    def __init__(self, language: str = "pt", method: SentimentMethod = SentimentMethod.ENSEMBLE):
        """
        Inicializa o sistema de análise de sentimento.
        
        Args:
            language: Idioma para análise ("pt" ou "en")
            method: Método de análise a ser usado
        """
        self.logger = get_logger(__name__)
        
        # Configurações
        self.language = language
        self.method = method
        
        # Estados do sistema
        self.is_initialized = False
        
        # Threading
        self.analysis_lock = threading.Lock()
        
        # Cache de análises
        self.sentiment_cache = {}
        self.cache_ttl = 300  # 5 minutos
        
        # Histórico de sentimentos
        self.sentiment_history = deque(maxlen=100)
        
        # Lexicons de sentimento
        self.positive_words = set()
        self.negative_words = set()
        self.intensifiers = set()
        self.diminishers = set()
        self.negation_words = set()
        
        # Modelos de ML
        self.sentiment_model = None
        self.spacy_model = None
        
        # Ferramentas de NLP
        self.stemmer = None
        self.stopwords_set = set()
        
        # Métricas de performance
        self.performance_metrics = {
            'total_analyses': 0,
            'successful_analyses': 0,
            'processing_times': deque(maxlen=100),
            'average_processing_time': 0.0,
            'sentiments_detected': {sentiment: 0 for sentiment in SENTIMENTS.ALL_SENTIMENTS},
            'method_usage': {},
            'cache_hits': 0
        }
        
        # Inicializa componentes
        self._initialize_components()
        
        self.logger.info(f"SentimentAnalysis inicializado - Idioma: {language}, Método: {method.value}")
    
    def _initialize_components(self):
        """Inicializa componentes de NLP necessários."""
        try:
            # Inicializa lexicons
            self._load_sentiment_lexicons()
            
            # Inicializa NLTK se disponível
            if NLTK_AVAILABLE:
                self._initialize_nltk()
            
            # Inicializa spaCy se disponível
            if SPACY_AVAILABLE:
                self._initialize_spacy()
            
            # Inicializa Transformers se disponível
            if TRANSFORMERS_AVAILABLE:
                self._initialize_transformers()
            
            self.is_initialized = True
            self.logger.info("Componentes de análise de sentimento inicializados")
            
        except Exception as e:
            self.logger.error(f"Erro ao inicializar componentes: {e}")
            self.is_initialized = False
    
    def _load_sentiment_lexicons(self):
        """Carrega lexicons de sentimento."""
        try:
            if self.language == "pt":
                # Lexicon em português
                self.positive_words = {
                    'alegre', 'feliz', 'bom', 'ótimo', 'excelente', 'maravilhoso', 'fantástico',
                    'amor', 'carinho', 'gosto', 'adoro', 'amo', 'perfeito', 'incrível',
                    'positivo', 'agradável', 'satisfeito', 'contente', 'animado', 'empolgado',
                    'sucesso', 'vitória', 'conquista', 'realização', 'progresso', 'melhoria',
                    'esperança', 'confiança', 'otimismo', 'entusiasmo', 'paixão', 'prazer',
                    'beleza', 'harmonia', 'paz', 'calma', 'tranquilidade', 'serenidade',
                    'sorte', 'fortuna', 'benção', 'gratidão', 'reconhecimento', 'valorização'
                }
                
                self.negative_words = {
                    'triste', 'ruim', 'péssimo', 'terrível', 'horrível', 'odioso', 'ódio',
                    'raiva', 'irritado', 'chateado', 'frustrado', 'decepcionado', 'furioso',
                    'negativo', 'desagradável', 'insatisfeito', 'descontente', 'deprimido',
                    'fracasso', 'derrota', 'perda', 'problema', 'dificuldade', 'obstáculo',
                    'medo', 'receio', 'ansiedade', 'preocupação', 'stress', 'tensão',
                    'dor', 'sofrimento', 'tristeza', 'melancolia', 'desespero', 'desânimo',
                    'azar', 'infelicidade', 'mal', 'cruel', 'injusto', 'injustiça'
                }
                
                self.intensifiers = {
                    'muito', 'extremamente', 'super', 'ultra', 'mega', 'totalmente',
                    'completamente', 'absolutamente', 'realmente', 'verdadeiramente',
                    'profundamente', 'intensamente', 'fortemente', 'bastante', 'bem'
                }
                
                self.diminishers = {
                    'pouco', 'levemente', 'ligeiramente', 'meio', 'um pouco', 'apenas',
                    'somente', 'só', 'quase', 'praticamente', 'raramente', 'dificilmente'
                }
                
                self.negation_words = {
                    'não', 'nunca', 'jamais', 'nada', 'nenhum', 'nenhuma', 'nem',
                    'tampouco', 'sequer', 'sem', 'impossível', 'incapaz'
                }
                
            else:  # English
                self.positive_words = {
                    'good', 'great', 'excellent', 'wonderful', 'fantastic', 'amazing',
                    'love', 'like', 'enjoy', 'perfect', 'incredible', 'awesome',
                    'positive', 'pleasant', 'satisfied', 'happy', 'excited', 'enthusiastic',
                    'success', 'victory', 'achievement', 'progress', 'improvement',
                    'hope', 'confidence', 'optimism', 'enthusiasm', 'passion', 'pleasure',
                    'beauty', 'harmony', 'peace', 'calm', 'tranquility', 'serenity'
                }
                
                self.negative_words = {
                    'bad', 'terrible', 'horrible', 'awful', 'hate', 'dislike',
                    'angry', 'upset', 'frustrated', 'disappointed', 'furious',
                    'negative', 'unpleasant', 'dissatisfied', 'sad', 'depressed',
                    'failure', 'defeat', 'loss', 'problem', 'difficulty', 'obstacle',
                    'fear', 'anxiety', 'worry', 'stress', 'tension', 'concern',
                    'pain', 'suffering', 'sadness', 'despair', 'hopeless'
                }
                
                self.intensifiers = {
                    'very', 'extremely', 'super', 'ultra', 'totally', 'completely',
                    'absolutely', 'really', 'truly', 'deeply', 'intensely', 'strongly'
                }
                
                self.diminishers = {
                    'little', 'slightly', 'somewhat', 'kind of', 'sort of', 'barely',
                    'hardly', 'scarcely', 'rarely', 'seldom', 'almost'
                }
                
                self.negation_words = {
                    'not', 'never', 'no', 'none', 'neither', 'nor', 'nothing',
                    'nobody', 'nowhere', 'without', 'impossible', 'unable'
                }
            
            self.logger.info(f"Lexicons carregados: {len(self.positive_words)} palavras positivas, {len(self.negative_words)} negativas")
            
        except Exception as e:
            self.logger.error(f"Erro ao carregar lexicons: {e}")
    
    def _initialize_nltk(self):
        """Inicializa componentes do NLTK."""
        try:
            # Download de recursos necessários
            required_nltk_data = ['punkt', 'stopwords', 'rslp']
            
            for data in required_nltk_data:
                try:
                    nltk.data.find(f'tokenizers/{data}' if data == 'punkt' else 
                                  f'corpora/{data}' if data in ['stopwords'] else f'stemmers/{data}')
                except LookupError:
                    self.logger.info(f"Baixando {data}...")
                    nltk.download(data, quiet=True)
            
            # Inicializa stemmer
            if self.language == "pt":
                self.stemmer = RSLPStemmer()
                self.stopwords_set = set(stopwords.words('portuguese'))
            else:
                from nltk.stem import PorterStemmer
                self.stemmer = PorterStemmer()
                self.stopwords_set = set(stopwords.words('english'))
            
            self.logger.info("NLTK inicializado com sucesso")
            
        except Exception as e:
            self.logger.error(f"Erro ao inicializar NLTK: {e}")
    
    def _initialize_spacy(self):
        """Inicializa modelo do spaCy."""
        try:
            model_name = "pt_core_news_sm" if self.language == "pt" else "en_core_web_sm"
            
            try:
                self.spacy_model = spacy.load(model_name)
                self.logger.info(f"Modelo spaCy {model_name} carregado")
            except OSError:
                self.logger.warning(f"Modelo spaCy {model_name} não encontrado. Use: python -m spacy download {model_name}")
                
        except Exception as e:
            self.logger.error(f"Erro ao inicializar spaCy: {e}")
    
    def _initialize_transformers(self):
        """Inicializa modelo Transformers para análise de sentimento."""
        try:
            if self.language == "pt":
                model_name = "neuralmind/bert-base-portuguese-cased"
            else:
                model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
            
            try:
                self.sentiment_model = pipeline(
                    "sentiment-analysis", 
                    model=model_name,
                    device=-1  # CPU, use 0 para GPU
                )
                self.logger.info(f"Modelo Transformers {model_name} carregado")
            except Exception as e:
                self.logger.warning(f"Não foi possível carregar modelo específico: {e}")
                # Fallback para modelo genérico
                self.sentiment_model = pipeline("sentiment-analysis", device=-1)
                
        except Exception as e:
            self.logger.error(f"Erro ao inicializar Transformers: {e}")
    
    def analyze_text(self, text: str) -> Optional[SentimentAnalysisResult]:
        """
        Analisa sentimento de um texto.
        
        Args:
            text: Texto para análise
            
        Returns:
            Optional[SentimentAnalysisResult]: Resultado da análise ou None se falhou
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
                if cache_key in self.sentiment_cache:
                    cached_result, cache_time = self.sentiment_cache[cache_key]
                    if time.time() - cache_time < self.cache_ttl:
                        self.performance_metrics['cache_hits'] += 1
                        return cached_result
                
                # Análise baseada no método selecionado
                result = None
                
                if self.method == SentimentMethod.ENSEMBLE:
                    result = self._analyze_ensemble(normalized_text)
                elif self.method == SentimentMethod.LEXICON_BASED:
                    result = self._analyze_lexicon_based(normalized_text)
                elif self.method == SentimentMethod.TEXTBLOB and TEXTBLOB_AVAILABLE:
                    result = self._analyze_textblob(normalized_text)
                elif self.method == SentimentMethod.TRANSFORMERS and self.sentiment_model:
                    result = self._analyze_transformers(normalized_text)
                elif self.method == SentimentMethod.SPACY and self.spacy_model:
                    result = self._analyze_spacy(normalized_text)
                else:
                    # Fallback para análise baseada em lexicon
                    result = self._analyze_lexicon_based(normalized_text)
                
                if result:
                    result.timestamp = start_time
                    result.processing_time = time.time() - start_time
                    result.text_length = len(text)
                    result.word_count = len(normalized_text.split())
                    result.sentence_count = len([s for s in normalized_text.split('.') if s.strip()])
                    result.method_used = self.method.value
                    
                    # Atualiza cache
                    self.sentiment_cache[cache_key] = (result, time.time())
                    
                    # Atualiza histórico
                    self.sentiment_history.append(result)
                    
                    # Limpa cache antigo
                    self._cleanup_cache()
                    
                    # Atualiza métricas
                    self._update_performance_metrics(result, True)
                    
                    self.logger.debug(f"Sentimento analisado: {result.overall_sentiment} (score: {result.overall_score:.2f})")
                    
                    return result
                else:
                    self._update_performance_metrics(None, False)
                    return None
                    
        except Exception as e:
            self.logger.error(f"Erro na análise de sentimento: {e}")
            self._update_performance_metrics(None, False)
            return None
    
    def analyze_recognition_result(self, recognition_result: RecognitionResult) -> Optional[SentimentAnalysisResult]:
        """
        Analisa sentimento de um resultado de reconhecimento de fala.
        
        Args:
            recognition_result: Resultado do reconhecimento de fala
            
        Returns:
            Optional[SentimentAnalysisResult]: Resultado da análise de sentimento
        """
        if not recognition_result or not recognition_result.full_text:
            return None
        
        return self.analyze_text(recognition_result.full_text)
    
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
    
    def _analyze_ensemble(self, text: str) -> Optional[SentimentAnalysisResult]:
        """Análise ensemble combinando múltiplos métodos."""
        try:
            results = []
            weights = []
            
            # Análise baseada em lexicon (sempre disponível)
            lexicon_result = self._analyze_lexicon_based(text)
            if lexicon_result:
                results.append(lexicon_result)
                weights.append(0.3)
            
            # TextBlob se disponível
            if TEXTBLOB_AVAILABLE:
                textblob_result = self._analyze_textblob(text)
                if textblob_result:
                    results.append(textblob_result)
                    weights.append(0.2)
            
            # Transformers se disponível
            if self.sentiment_model:
                transformers_result = self._analyze_transformers(text)
                if transformers_result:
                    results.append(transformers_result)
                    weights.append(0.4)
            
            # spaCy se disponível
            if self.spacy_model:
                spacy_result = self._analyze_spacy(text)
                if spacy_result:
                    results.append(spacy_result)
                    weights.append(0.1)
            
            if not results:
                return None
            
            # Normaliza pesos
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
            
            # Combina resultados
            combined_score = sum(result.overall_score * weight for result, weight in zip(results, weights))
            combined_confidence = sum(result.overall_confidence * weight for result, weight in zip(results, weights))
            
            # Determina sentimento combinado
            overall_sentiment = self._score_to_sentiment(combined_score)
            intensity = self._score_to_intensity(combined_score)
            
            # Cria resultado combinado
            result = SentimentAnalysisResult(
                overall_sentiment=overall_sentiment,
                overall_score=combined_score,
                overall_confidence=combined_confidence,
                intensity=intensity,
                sentences=lexicon_result.sentences if lexicon_result else [],
                sentiment_distribution=self._calculate_combined_distribution(results, weights),
                emotional_indicators=list(set(sum([r.emotional_indicators for r in results], []))),
                method_used="ensemble"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Erro na análise ensemble: {e}")
            return None
    
    def _analyze_lexicon_based(self, text: str) -> Optional[SentimentAnalysisResult]:
        """Análise baseada em lexicon de sentimentos."""
        try:
            # Tokeniza o texto
            sentences = self._split_sentences(text)
            sentence_results = []
            
            for sentence in sentences:
                sentence_result = self._analyze_sentence_lexicon(sentence)
                if sentence_result:
                    sentence_results.append(sentence_result)
            
            if not sentence_results:
                return None
            
            # Calcula sentimento geral
            overall_score = np.mean([s.score for s in sentence_results])
            overall_confidence = np.mean([s.confidence for s in sentence_results])
            overall_sentiment = self._score_to_sentiment(overall_score)
            intensity = self._score_to_intensity(overall_score)
            
            # Coleta indicadores emocionais
            emotional_indicators = []
            for sentence in sentence_results:
                emotional_indicators.extend([w.word for w in sentence.positive_words + sentence.negative_words])
            
            # Calcula distribuição de sentimentos
            sentiment_counts = Counter([s.sentiment for s in sentence_results])
            total_sentences = len(sentence_results)
            sentiment_distribution = {
                sentiment: count / total_sentences 
                for sentiment, count in sentiment_counts.items()
            }
            
            # Tendência de sentimento ao longo do texto
            sentiment_trends = [s.score for s in sentence_results]
            
            result = SentimentAnalysisResult(
                overall_sentiment=overall_sentiment,
                overall_score=overall_score,
                overall_confidence=overall_confidence,
                intensity=intensity,
                sentences=sentence_results,
                sentiment_distribution=sentiment_distribution,
                emotional_indicators=list(set(emotional_indicators)),
                sentiment_trends=sentiment_trends,
                method_used="lexicon_based"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Erro na análise baseada em lexicon: {e}")
            return None
    
    def _analyze_sentence_lexicon(self, sentence: str) -> Optional[SentenceSentiment]:
        """Analisa sentimento de uma sentença usando lexicon."""
        try:
            words = self._tokenize_words(sentence)
            if not words:
                return None
            
            positive_words = []
            negative_words = []
            neutral_words = []
            
            # Análise palavra por palavra
            for i, word in enumerate(words):
                if word in self.stopwords_set:
                    continue
                
                # Stemming se disponível
                stemmed_word = self.stemmer.stem(word) if self.stemmer else word
                
                # Determina sentimento da palavra
                word_sentiment = self._get_word_sentiment(word, stemmed_word)
                
                if word_sentiment:
                    # Verifica modificadores contextuais
                    context_modifier = self._calculate_context_modifier(words, i)
                    word_sentiment.context_modifier = context_modifier
                    
                    if word_sentiment.sentiment == SENTIMENTS.POSITIVE:
                        positive_words.append(word_sentiment)
                    elif word_sentiment.sentiment == SENTIMENTS.NEGATIVE:
                        negative_words.append(word_sentiment)
                    else:
                        neutral_words.append(word_sentiment)
            
            # Calcula score da sentença
            pos_score = sum(w.score * w.context_modifier for w in positive_words)
            neg_score = sum(w.score * w.context_modifier for w in negative_words)
            
            total_score = pos_score - neg_score
            
            # Normaliza score (-1 a 1)
            max_possible = len(words)
            if max_possible > 0:
                normalized_score = max(-1.0, min(1.0, total_score / max_possible))
            else:
                normalized_score = 0.0
            
            # Determina sentimento da sentença
            sentence_sentiment = self._score_to_sentiment(normalized_score)
            intensity = self._score_to_intensity(normalized_score)
            
            # Calcula confiança
            confidence = self._calculate_sentence_confidence(positive_words, negative_words, neutral_words, words)
            
            # Detecta padrões linguísticos
            negation_detected = self._detect_negation(words)
            intensifiers = self._detect_intensifiers(words)
            diminishers = self._detect_diminishers(words)
            
            result = SentenceSentiment(
                text=sentence,
                sentiment=sentence_sentiment,
                score=normalized_score,
                confidence=confidence,
                intensity=intensity,
                positive_words=positive_words,
                negative_words=negative_words,
                neutral_words=neutral_words,
                negation_detected=negation_detected,
                intensifiers_detected=intensifiers,
                diminishers_detected=diminishers
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Erro na análise de sentença: {e}")
            return None
    
    def _get_word_sentiment(self, word: str, stemmed_word: str) -> Optional[WordSentiment]:
        """Determina sentimento de uma palavra."""
        
        # Verifica palavra original e stemmed
        words_to_check = [word, stemmed_word]
        
        for check_word in words_to_check:
            if check_word in self.positive_words:
                return WordSentiment(
                    word=word,
                    sentiment=SENTIMENTS.POSITIVE,
                    score=1.0,
                    confidence=0.8
                )
            elif check_word in self.negative_words:
                return WordSentiment(
                    word=word,
                    sentiment=SENTIMENTS.NEGATIVE,
                    score=1.0,
                    confidence=0.8
                )
        
        return None
    
    def _calculate_context_modifier(self, words: List[str], word_index: int) -> float:
        """Calcula modificador contextual baseado em palavras próximas."""
        modifier = 1.0
        
        try:
            # Verifica palavras anteriores (janela de 3 palavras)
            start_idx = max(0, word_index - 3)
            context_words = words[start_idx:word_index]
            
            # Negação
            if any(neg_word in context_words for neg_word in self.negation_words):
                modifier *= -1.0
            
            # Intensificadores
            if any(int_word in context_words for int_word in self.intensifiers):
                modifier *= 1.5
            
            # Diminutivos
            if any(dim_word in context_words for dim_word in self.diminishers):
                modifier *= 0.5
            
            return modifier
            
        except Exception as e:
            self.logger.error(f"Erro no cálculo de modificador contextual: {e}")
            return 1.0
    
    def _calculate_sentence_confidence(self, positive_words: List[WordSentiment], 
                                     negative_words: List[WordSentiment], 
                                     neutral_words: List[WordSentiment],
                                     all_words: List[str]) -> float:
        """Calcula confiança da análise de sentimento da sentença."""
        try:
            # Fatores de confiança
            sentiment_word_ratio = (len(positive_words) + len(negative_words)) / max(1, len(all_words))
            
            # Confiança individual das palavras
            if positive_words or negative_words:
                avg_word_confidence = np.mean([w.confidence for w in positive_words + negative_words])
            else:
                avg_word_confidence = 0.5
            
            # Consenso (se há mistura de sentimentos)
            if positive_words and negative_words:
                consensus_penalty = 0.3
            else:
                consensus_penalty = 0.0
            
            # Calcula confiança final
            confidence = (sentiment_word_ratio * 0.4 + avg_word_confidence * 0.6) - consensus_penalty
            
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            self.logger.error(f"Erro no cálculo de confiança: {e}")
            return 0.5
    
    def _detect_negation(self, words: List[str]) -> bool:
        """Detecta presença de negação na sentença."""
        return any(word in self.negation_words for word in words)
    
    def _detect_intensifiers(self, words: List[str]) -> List[str]:
        """Detecta intensificadores na sentença."""
        return [word for word in words if word in self.intensifiers]
    
    def _detect_diminishers(self, words: List[str]) -> List[str]:
        """Detecta diminutivos na sentença."""
        return [word for word in words if word in self.diminishers]
    
    def _analyze_textblob(self, text: str) -> Optional[SentimentAnalysisResult]:
        """Análise usando TextBlob."""
        try:
            blob = TextBlob(text)
            
            # Análise geral
            polarity = blob.sentiment.polarity  # -1 a 1
            subjectivity = blob.sentiment.subjectivity  # 0 a 1
            
            # Converte para nosso formato
            overall_sentiment = self._score_to_sentiment(polarity)
            intensity = self._score_to_intensity(polarity)
            confidence = (1.0 - abs(0.5 - subjectivity)) * 2  # Converte subjetividade em confiança
            
            result = SentimentAnalysisResult(
                overall_sentiment=overall_sentiment,
                overall_score=polarity,
                overall_confidence=confidence,
                intensity=intensity,
                sentiment_distribution={overall_sentiment: 1.0},
                method_used="textblob"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Erro na análise TextBlob: {e}")
            return None
    
    def _analyze_transformers(self, text: str) -> Optional[SentimentAnalysisResult]:
        """Análise usando modelo Transformers."""
        try:
            # Limita tamanho do texto (modelos têm limite de tokens)
            max_length = 512
            if len(text) > max_length:
                text = text[:max_length]
            
            results = self.sentiment_model(text)
            
            if results and len(results) > 0:
                result = results[0]
                label = result['label'].lower()
                score = result['score']
                
                # Mapeia labels para nosso formato
                if 'pos' in label or 'good' in label:
                    sentiment = SENTIMENTS.POSITIVE
                    sentiment_score = score
                elif 'neg' in label or 'bad' in label:
                    sentiment = SENTIMENTS.NEGATIVE
                    sentiment_score = -score
                else:
                    sentiment = SENTIMENTS.NEUTRAL
                    sentiment_score = 0.0
                
                intensity = self._score_to_intensity(sentiment_score)
                
                result = SentimentAnalysisResult(
                    overall_sentiment=sentiment,
                    overall_score=sentiment_score,
                    overall_confidence=score,
                    intensity=intensity,
                    sentiment_distribution={sentiment: 1.0},
                    method_used="transformers"
                )
                
                return result
            
            return None
            
        except Exception as e:
            self.logger.error(f"Erro na análise Transformers: {e}")
            return None
    
    def _analyze_spacy(self, text: str) -> Optional[SentimentAnalysisResult]:
        """Análise usando spaCy (implementação básica)."""
        try:
            doc = self.spacy_model(text)
            
            # spaCy não tem análise de sentimento built-in por padrão
            # Esta é uma implementação básica usando nosso lexicon
            sentiment_score = 0.0
            word_count = 0
            
            for token in doc:
                if not token.is_stop and not token.is_punct:
                    lemma = token.lemma_.lower()
                    
                    if lemma in self.positive_words:
                        sentiment_score += 1.0
                        word_count += 1
                    elif lemma in self.negative_words:
                        sentiment_score -= 1.0
                        word_count += 1
            
            if word_count > 0:
                normalized_score = sentiment_score / word_count
            else:
                normalized_score = 0.0
            
            overall_sentiment = self._score_to_sentiment(normalized_score)
            intensity = self._score_to_intensity(normalized_score)
            confidence = min(1.0, word_count / len([token for token in doc if not token.is_stop]))
            
            result = SentimentAnalysisResult(
                overall_sentiment=overall_sentiment,
                overall_score=normalized_score,
                overall_confidence=confidence,
                intensity=intensity,
                sentiment_distribution={overall_sentiment: 1.0},
                method_used="spacy"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Erro na análise spaCy: {e}")
            return None
    
    def _score_to_sentiment(self, score: float) -> str:
        """Converte score numérico para sentimento."""
        if score > SENTIMENTS.POSITIVE_THRESHOLD:
            return SENTIMENTS.POSITIVE
        elif score < SENTIMENTS.NEGATIVE_THRESHOLD:
            return SENTIMENTS.NEGATIVE
        else:
            return SENTIMENTS.NEUTRAL
    
    def _score_to_intensity(self, score: float) -> SentimentIntensity:
        """Converte score numérico para intensidade."""
        if score >= 0.7:
            return SentimentIntensity.VERY_POSITIVE
        elif score >= 0.3:
            return SentimentIntensity.POSITIVE
        elif score >= 0.1:
            return SentimentIntensity.SLIGHTLY_POSITIVE
        elif score <= -0.7:
            return SentimentIntensity.VERY_NEGATIVE
        elif score <= -0.3:
            return SentimentIntensity.NEGATIVE
        elif score <= -0.1:
            return SentimentIntensity.SLIGHTLY_NEGATIVE
        else:
            return SentimentIntensity.NEUTRAL
    
    def _split_sentences(self, text: str) -> List[str]:
        """Divide texto em sentenças."""
        try:
            if NLTK_AVAILABLE:
                return sent_tokenize(text, language='portuguese' if self.language == 'pt' else 'english')
            else:
                # Fallback simples
                sentences = re.split(r'[.!?]+', text)
                return [s.strip() for s in sentences if s.strip()]
                
        except Exception as e:
            self.logger.error(f"Erro ao dividir sentenças: {e}")
            return [text]
    
    def _tokenize_words(self, text: str) -> List[str]:
        """Tokeniza texto em palavras."""
        try:
            if NLTK_AVAILABLE:
                return word_tokenize(text, language='portuguese' if self.language == 'pt' else 'english')
            else:
                # Fallback simples
                words = re.findall(r'\b\w+\b', text.lower())
                return words
                
        except Exception as e:
            self.logger.error(f"Erro na tokenização: {e}")
            return text.split()
    
    def _calculate_combined_distribution(self, results: List[SentimentAnalysisResult], 
                                       weights: List[float]) -> Dict[str, float]:
        """Calcula distribuição combinada de sentimentos."""
        try:
            combined_dist = {}
            
            for result, weight in zip(results, weights):
                for sentiment, prob in result.sentiment_distribution.items():
                    if sentiment in combined_dist:
                        combined_dist[sentiment] += prob * weight
                    else:
                        combined_dist[sentiment] = prob * weight
            
            # Normaliza
            total = sum(combined_dist.values())
            if total > 0:
                combined_dist = {k: v/total for k, v in combined_dist.items()}
            
            return combined_dist
            
        except Exception as e:
            self.logger.error(f"Erro no cálculo de distribuição combinada: {e}")
            return {}
    
    def _generate_cache_key(self, text: str) -> str:
        """Gera chave de cache baseada no texto."""
        text_hash = hash(text)
        return f"sentiment_{text_hash}_{len(text)}"
    
    def _cleanup_cache(self):
        """Remove entradas antigas do cache."""
        current_time = time.time()
        expired_keys = []
        
        for cache_key, (_, cache_time) in self.sentiment_cache.items():
            if current_time - cache_time > self.cache_ttl:
                expired_keys.append(cache_key)
        
        for key in expired_keys:
            del self.sentiment_cache[key]
    
    def _update_performance_metrics(self, result: Optional[SentimentAnalysisResult], success: bool):
        """Atualiza métricas de performance."""
        self.performance_metrics['total_analyses'] += 1
        
        if success and result:
            self.performance_metrics['successful_analyses'] += 1
            
            # Atualiza tempos de processamento
            self.performance_metrics['processing_times'].append(result.processing_time)
            if self.performance_metrics['processing_times']:
                self.performance_metrics['average_processing_time'] = np.mean(
                    self.performance_metrics['processing_times']
                )
            
            # Conta sentimentos detectados
            sentiment = result.overall_sentiment
            if sentiment in self.performance_metrics['sentiments_detected']:
                self.performance_metrics['sentiments_detected'][sentiment] += 1
            
            # Conta uso de métodos
            method = result.method_used
            if method in self.performance_metrics['method_usage']:
                self.performance_metrics['method_usage'][method] += 1
            else:
                self.performance_metrics['method_usage'][method] = 1
    
    def get_sentiment_trend(self, window_size: int = 10) -> Dict:
        """
        Analisa tendência de sentimento nas últimas análises.
        
        Args:
            window_size: Número de análises a considerar
            
        Returns:
            Dict: Dados de tendência
        """
        try:
            if len(self.sentiment_history) < 2:
                return {'trend': 'insufficient_data'}
            
            # Pega últimas análises
            recent_analyses = list(self.sentiment_history)[-window_size:]
            
            # Extrai scores
            scores = [analysis.overall_score for analysis in recent_analyses]
            
            # Calcula tendência
            if len(scores) >= 2:
                # Regressão linear simples
                x = np.arange(len(scores))
                slope = np.polyfit(x, scores, 1)[0]
                
                if slope > 0.1:
                    trend = 'improving'
                elif slope < -0.1:
                    trend = 'declining'
                else:
                    trend = 'stable'
            else:
                trend = 'stable'
            
            # Calcula variabilidade
            variability = np.std(scores) if len(scores) > 1 else 0.0
            
            # Sentimento predominante
            sentiments = [analysis.overall_sentiment for analysis in recent_analyses]
            sentiment_counts = Counter(sentiments)
            dominant_sentiment = sentiment_counts.most_common(1)[0][0] if sentiment_counts else 'neutral'
            
            return {
                'trend': trend,
                'slope': slope if len(scores) >= 2 else 0.0,
                'variability': variability,
                'dominant_sentiment': dominant_sentiment,
                'score_range': (min(scores), max(scores)) if scores else (0, 0),
                'average_score': np.mean(scores) if scores else 0.0,
                'sample_size': len(recent_analyses)
            }
            
        except Exception as e:
            self.logger.error(f"Erro ao calcular tendência de sentimento: {e}")
            return {'error': str(e)}
    
    def correlate_with_emotions(self, facial_emotion: str = None, vocal_emotion: str = None) -> Dict:
        """
        Correlaciona sentimento textual com emoções faciais e vocais.
        
        Args:
            facial_emotion: Emoção facial detectada
            vocal_emotion: Emoção vocal detectada
            
        Returns:
            Dict: Correlações e análise integrada
        """
        try:
            if not self.sentiment_history:
                return {'message': 'Nenhum dado de sentimento disponível'}
            
            latest_sentiment = self.sentiment_history[-1]
            
            # Mapeamento de sentimentos para emoções
            sentiment_emotion_map = {
                SENTIMENTS.POSITIVE: ['happy', 'excited', 'confident'],
                SENTIMENTS.NEGATIVE: ['sad', 'angry', 'stressed'],
                SENTIMENTS.NEUTRAL: ['neutral', 'calm']
            }
            
            correlations = {}
            
            # Correlação facial
            if facial_emotion:
                expected_emotions = sentiment_emotion_map.get(latest_sentiment.overall_sentiment, [])
                facial_correlation = 1.0 if facial_emotion in expected_emotions else 0.0
                correlations['facial'] = {
                    'correlation': facial_correlation,
                    'expected': expected_emotions,
                    'detected': facial_emotion,
                    'match': facial_emotion in expected_emotions
                }
            
            # Correlação vocal
            if vocal_emotion:
                expected_emotions = sentiment_emotion_map.get(latest_sentiment.overall_sentiment, [])
                vocal_correlation = 1.0 if vocal_emotion in expected_emotions else 0.0
                correlations['vocal'] = {
                    'correlation': vocal_correlation,
                    'expected': expected_emotions,
                    'detected': vocal_emotion,
                    'match': vocal_emotion in expected_emotions
                }
            
            # Análise integrada
            consistency_score = np.mean([corr['correlation'] for corr in correlations.values()])
            
            return {
                'text_sentiment': latest_sentiment.overall_sentiment,
                'sentiment_score': latest_sentiment.overall_score,
                'correlations': correlations,
                'consistency_score': consistency_score,
                'is_consistent': consistency_score > 0.5,
                'analysis_confidence': latest_sentiment.overall_confidence
            }
            
        except Exception as e:
            self.logger.error(f"Erro na correlação com emoções: {e}")
            return {'error': str(e)}
    
    def get_performance_metrics(self) -> Dict:
        """Retorna métricas de performance."""
        metrics = self.performance_metrics.copy()
        
        # Adiciona métricas calculadas
        if metrics['total_analyses'] > 0:
            metrics['success_rate'] = metrics['successful_analyses'] / metrics['total_analyses']
            metrics['cache_hit_rate'] = metrics['cache_hits'] / metrics['total_analyses']
        else:
            metrics['success_rate'] = 0.0
            metrics['cache_hit_rate'] = 0.0
        
        return metrics
    
    def clear_cache(self):
        """Limpa cache de análises."""
        self.sentiment_cache.clear()
        self.logger.info("Cache de análise de sentimento limpo")
    
    def clear_history(self):
        """Limpa histórico de análises."""
        self.sentiment_history.clear()
        self.logger.info("Histórico de análise de sentimento limpo")