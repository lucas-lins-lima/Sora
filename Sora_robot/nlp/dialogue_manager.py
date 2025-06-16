# sora_robot/nlp/dialogue_manager.py

import time
import threading
import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, NamedTuple
from dataclasses import dataclass, field
from collections import deque, Counter
from enum import Enum
import uuid

from utils.logger import get_logger
from utils.constants import INTENTS, SENTIMENTS, EMOTIONS, DEFAULT_MESSAGES
from vision_processing.facial_recognition import FaceData
from vision_processing.emotion_analysis import EmotionData
from vision_processing.body_pose_estimation import BodyPoseData
from audio_processing.speech_recognition import RecognitionResult
from audio_processing.audio_analysis import AudioAnalysisResult
from nlp.sentiment_analysis import SentimentAnalysisResult
from nlp.intent_recognition import IntentAnalysisResult
import config

class DialogueState(Enum):
    """Estados possíveis do diálogo."""
    GREETING = "greeting"
    ACTIVE_CONVERSATION = "active_conversation"
    WAITING_CLARIFICATION = "waiting_clarification"
    PROVIDING_INFO = "providing_info"
    HANDLING_REQUEST = "handling_request"
    CLOSING = "closing"
    IDLE = "idle"

class ResponseStrategy(Enum):
    """Estratégias de resposta."""
    INFORMATIVE = "informative"
    EMPATHETIC = "empathetic"
    ENCOURAGING = "encouraging"
    CLARIFYING = "clarifying"
    DIRECTIVE = "directive"
    CASUAL = "casual"
    FORMAL = "formal"

class UserMood(Enum):
    """Estados de humor do usuário."""
    VERY_POSITIVE = "very_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very_negative"
    CONFUSED = "confused"
    FRUSTRATED = "frustrated"
    EXCITED = "excited"

@dataclass
class UserState:
    """Estado atual do usuário baseado em todas as análises."""
    
    # Estados emocionais
    facial_emotion: str = "neutral"
    facial_confidence: float = 0.0
    vocal_emotion: str = "neutral"
    vocal_confidence: float = 0.0
    text_sentiment: str = "neutral"
    sentiment_confidence: float = 0.0
    
    # Características comportamentais
    engagement_level: float = 0.5  # 0.0 a 1.0
    energy_level: float = 0.5
    stress_level: float = 0.0
    confidence_level: float = 0.5
    
    # Intenção e contexto
    current_intent: str = "unknown"
    intent_confidence: float = 0.0
    needs_clarification: bool = False
    
    # Estado integrado
    overall_mood: UserMood = UserMood.NEUTRAL
    consistency_score: float = 0.0  # Consistência entre diferentes análises
    
    # Timestamp
    timestamp: float = 0.0

@dataclass
class ConversationTurn:
    """Representa uma volta na conversa."""
    
    turn_id: str
    timestamp: float
    
    # Input do usuário
    user_text: str = ""
    user_state: Optional[UserState] = None
    
    # Análises realizadas
    recognition_result: Optional[RecognitionResult] = None
    sentiment_result: Optional[SentimentAnalysisResult] = None
    intent_result: Optional[IntentAnalysisResult] = None
    emotion_result: Optional[EmotionData] = None
    audio_result: Optional[AudioAnalysisResult] = None
    pose_result: Optional[BodyPoseData] = None
    
    # Resposta gerada
    bot_response: str = ""
    response_strategy: Optional[ResponseStrategy] = None
    response_metadata: Dict = field(default_factory=dict)
    
    # Contexto da volta
    dialogue_state: DialogueState = DialogueState.IDLE
    topic: str = ""
    entities_extracted: Dict = field(default_factory=dict)

@dataclass
class DialogueSession:
    """Sessão completa de diálogo."""
    
    session_id: str
    start_time: float
    last_activity: float
    
    # Histórico da conversa
    turns: List[ConversationTurn] = field(default_factory=list)
    
    # Estado atual
    current_state: DialogueState = DialogueState.IDLE
    current_topic: str = ""
    
    # Perfil do usuário na sessão
    user_profile: Dict = field(default_factory=dict)
    user_preferences: Dict = field(default_factory=dict)
    
    # Estatísticas da sessão
    total_turns: int = 0
    avg_response_time: float = 0.0
    dominant_emotions: List[str] = field(default_factory=list)
    main_topics: List[str] = field(default_factory=list)

class DialogueManager:
    """
    Gerenciador central de diálogo que integra todas as análises e coordena respostas.
    Mantém contexto, estado e estratégias de conversa.
    """
    
    def __init__(self, personality_profile: Dict = None):
        """
        Inicializa o gerenciador de diálogo.
        
        Args:
            personality_profile: Perfil de personalidade do robô
        """
        self.logger = get_logger(__name__)
        
        # Configurações de personalidade
        self.personality = personality_profile or self._default_personality()
        
        # Estado atual
        self.current_session: Optional[DialogueSession] = None
        self.current_state = DialogueState.IDLE
        self.current_user_state: Optional[UserState] = None
        
        # Threading
        self.manager_lock = threading.Lock()
        
        # Histórico de sessões
        self.session_history = deque(maxlen=10)
        
        # Contexto de conhecimento
        self.knowledge_base = self._load_knowledge_base()
        self.conversation_patterns = self._load_conversation_patterns()
        
        # Estratégias de resposta
        self.response_strategies = self._initialize_response_strategies()
        
        # Cache para otimização
        self.analysis_cache = {}
        self.response_cache = {}
        
        # Métricas de performance
        self.performance_metrics = {
            'total_sessions': 0,
            'total_turns': 0,
            'average_session_duration': 0.0,
            'response_strategies_used': Counter(),
            'user_satisfaction_indicators': deque(maxlen=100),
            'state_transitions': Counter(),
            'processing_times': deque(maxlen=100)
        }
        
        # Configurações de tempo
        self.session_timeout = 300  # 5 minutos de inatividade
        self.max_session_duration = 3600  # 1 hora máxima
        
        self.logger.info("DialogueManager inicializado")
    
    def _default_personality(self) -> Dict:
        """Define personalidade padrão do robô."""
        return {
            'name': 'Sora',
            'characteristics': {
                'friendliness': 0.8,
                'helpfulness': 0.9,
                'formality': 0.6,
                'humor': 0.5,
                'empathy': 0.8,
                'patience': 0.9,
                'curiosity': 0.7
            },
            'response_style': {
                'verbosity': 0.7,  # Quão detalhada são as respostas
                'enthusiasm': 0.6,
                'directness': 0.7,
                'encouragement': 0.8
            },
            'behavioral_traits': {
                'adapts_to_user_mood': True,
                'remembers_context': True,
                'asks_follow_up_questions': True,
                'provides_examples': True,
                'shows_emotions': True
            }
        }
    
    def _load_knowledge_base(self) -> Dict:
        """Carrega base de conhecimento do arquivo."""
        try:
            # Carrega conhecimento do evento
            knowledge = {
                'event_info': {},
                'general_knowledge': {},
                'user_profiles': {},
                'conversation_topics': {}
            }
            
            # Tenta carregar do arquivo
            try:
                with open('data/knowledge_base/event_info.json', 'r', encoding='utf-8') as f:
                    knowledge['event_info'] = json.load(f)
            except FileNotFoundError:
                # Conhecimento padrão se arquivo não existir
                knowledge['event_info'] = {
                    'name': 'Evento Sora',
                    'description': 'Um evento incrível com tecnologia avançada',
                    'location': 'Centro de Convenções',
                    'schedule': 'Consulte a programação oficial',
                    'contacts': 'Procure nossa equipe de suporte'
                }
            
            return knowledge
            
        except Exception as e:
            self.logger.error(f"Erro ao carregar base de conhecimento: {e}")
            return {}
    
    def _load_conversation_patterns(self) -> Dict:
        """Carrega padrões de conversa."""
        return {
            'greeting_responses': [
                "Olá! Eu sou a Sora. Como posso ajudá-lo hoje?",
                "Oi! Prazer em conhecê-lo! Em que posso ser útil?",
                "Bem-vindo! Sou a Sora, sua assistente virtual. O que você gostaria de saber?"
            ],
            'clarification_requests': [
                "Desculpe, não entendi completamente. Pode me explicar melhor?",
                "Poderia ser mais específico sobre o que você precisa?",
                "Não tenho certeza se compreendi. Pode reformular sua pergunta?"
            ],
            'empathy_responses': [
                "Entendo como você se sente.",
                "Isso deve ser frustrante mesmo.",
                "Compreendo sua situação."
            ],
            'encouragement_responses': [
                "Você está no caminho certo!",
                "Ótima pergunta!",
                "Que bom que você perguntou!"
            ]
        }
    
    def _initialize_response_strategies(self) -> Dict:
        """Inicializa estratégias de resposta baseadas em contexto."""
        return {
            ResponseStrategy.INFORMATIVE: {
                'description': 'Fornece informações detalhadas e precisas',
                'triggers': ['question', 'event_info', 'location_request'],
                'mood_compatibility': [UserMood.NEUTRAL, UserMood.POSITIVE],
                'response_patterns': [
                    "Com base nas informações que tenho, {info}",
                    "Posso te informar que {info}",
                    "Segundo nossos dados, {info}"
                ]
            },
            
            ResponseStrategy.EMPATHETIC: {
                'description': 'Responde com empatia e compreensão',
                'triggers': ['complaint', 'negative_sentiment'],
                'mood_compatibility': [UserMood.NEGATIVE, UserMood.FRUSTRATED, UserMood.CONFUSED],
                'response_patterns': [
                    "Entendo sua preocupação. {response}",
                    "Compreendo como isso deve ser {emotion}. {response}",
                    "Você tem razão em se sentir assim. {response}"
                ]
            },
            
            ResponseStrategy.ENCOURAGING: {
                'description': 'Oferece encorajamento e suporte',
                'triggers': ['help_request', 'uncertain_user'],
                'mood_compatibility': [UserMood.CONFUSED, UserMood.NEUTRAL],
                'response_patterns': [
                    "Não se preocupe, vou te ajudar! {response}",
                    "Ótima pergunta! {response}",
                    "Estou aqui para ajudar. {response}"
                ]
            },
            
            ResponseStrategy.CLARIFYING: {
                'description': 'Busca clarificação quando necessário',
                'triggers': ['ambiguous_intent', 'low_confidence'],
                'mood_compatibility': [UserMood.NEUTRAL, UserMood.CONFUSED],
                'response_patterns': [
                    "Para te ajudar melhor, {clarification_question}",
                    "Só para confirmar, você está perguntando sobre {topic}?",
                    "Posso esclarecer melhor se você me disser {specific_info}"
                ]
            },
            
            ResponseStrategy.DIRECTIVE: {
                'description': 'Fornece instruções claras e diretas',
                'triggers': ['request', 'help_needed'],
                'mood_compatibility': [UserMood.NEUTRAL, UserMood.POSITIVE],
                'response_patterns': [
                    "Para fazer isso, siga estes passos: {instructions}",
                    "Você pode {action} fazendo o seguinte: {steps}",
                    "A melhor maneira é: {method}"
                ]
            },
            
            ResponseStrategy.CASUAL: {
                'description': 'Conversa casual e descontraída',
                'triggers': ['greeting', 'compliment', 'social_interaction'],
                'mood_compatibility': [UserMood.POSITIVE, UserMood.EXCITED, UserMood.NEUTRAL],
                'response_patterns': [
                    "{casual_response}! {main_content}",
                    "Que legal! {response}",
                    "Bacana! {content}"
                ]
            }
        }
    
    def start_session(self, user_id: str = None) -> str:
        """
        Inicia uma nova sessão de diálogo.
        
        Args:
            user_id: ID do usuário (opcional)
            
        Returns:
            str: ID da sessão criada
        """
        try:
            with self.manager_lock:
                # Finaliza sessão anterior se existir
                if self.current_session:
                    self.end_session()
                
                # Cria nova sessão
                session_id = str(uuid.uuid4())
                current_time = time.time()
                
                self.current_session = DialogueSession(
                    session_id=session_id,
                    start_time=current_time,
                    last_activity=current_time,
                    current_state=DialogueState.GREETING
                )
                
                self.current_state = DialogueState.GREETING
                
                # Atualiza métricas
                self.performance_metrics['total_sessions'] += 1
                
                self.logger.info(f"Nova sessão iniciada: {session_id}")
                return session_id
                
        except Exception as e:
            self.logger.error(f"Erro ao iniciar sessão: {e}")
            return ""
    
    def process_interaction(self, **kwargs) -> Dict:
        """
        Processa uma interação completa do usuário.
        
        Args:
            **kwargs: Pode incluir recognition_result, sentiment_result, intent_result,
                     emotion_result, audio_result, pose_result
                     
        Returns:
            Dict: Resposta processada e metadados
        """
        start_time = time.time()
        
        try:
            with self.manager_lock:
                # Verifica se há sessão ativa
                if not self.current_session:
                    session_id = self.start_session()
                else:
                    session_id = self.current_session.session_id
                
                # Atualiza atividade da sessão
                self.current_session.last_activity = time.time()
                
                # Cria novo turn
                turn = ConversationTurn(
                    turn_id=str(uuid.uuid4()),
                    timestamp=start_time
                )
                
                # Integra todas as análises
                user_state = self._integrate_analyses(kwargs)
                turn.user_state = user_state
                self.current_user_state = user_state
                
                # Extrai dados específicos
                recognition_result = kwargs.get('recognition_result')
                if recognition_result:
                    turn.user_text = recognition_result.full_text
                    turn.recognition_result = recognition_result
                
                turn.sentiment_result = kwargs.get('sentiment_result')
                turn.intent_result = kwargs.get('intent_result')
                turn.emotion_result = kwargs.get('emotion_result')
                turn.audio_result = kwargs.get('audio_result')
                turn.pose_result = kwargs.get('pose_result')
                
                # Determina estado do diálogo
                new_state = self._determine_dialogue_state(turn)
                self._transition_state(new_state)
                turn.dialogue_state = new_state
                
                # Extrai entidades e tópico
                turn.entities_extracted = self._extract_entities(turn)
                turn.topic = self._determine_topic(turn)
                
                # Seleciona estratégia de resposta
                strategy = self._select_response_strategy(turn)
                turn.response_strategy = strategy
                
                # Gera resposta
                response_data = self._generate_response(turn, strategy)
                turn.bot_response = response_data.get('text', '')
                turn.response_metadata = response_data.get('metadata', {})
                
                # Adiciona turn à sessão
                self.current_session.turns.append(turn)
                self.current_session.total_turns += 1
                
                # Atualiza contexto da sessão
                self._update_session_context(turn)
                
                # Atualiza métricas
                processing_time = time.time() - start_time
                self._update_performance_metrics(turn, processing_time)
                
                # Prepara resposta final
                response = {
                    'session_id': session_id,
                    'turn_id': turn.turn_id,
                    'response_text': turn.bot_response,
                    'response_strategy': strategy.value,
                    'dialogue_state': new_state.value,
                    'user_mood': user_state.overall_mood.value,
                    'confidence': user_state.consistency_score,
                    'requires_clarification': user_state.needs_clarification,
                    'metadata': turn.response_metadata,
                    'processing_time': processing_time
                }
                
                self.logger.info(f"Interação processada - Estado: {new_state.value}, Estratégia: {strategy.value}")
                
                return response
                
        except Exception as e:
            self.logger.error(f"Erro ao processar interação: {e}")
            return {
                'error': str(e),
                'response_text': "Desculpe, houve um problema. Pode tentar novamente?",
                'dialogue_state': DialogueState.IDLE.value
            }
    
    def _integrate_analyses(self, analyses: Dict) -> UserState:
        """Integra todas as análises em um estado unificado do usuário."""
        user_state = UserState(timestamp=time.time())
        
        try:
            # Análise de emoção facial
            emotion_result = analyses.get('emotion_result')
            if emotion_result:
                user_state.facial_emotion = emotion_result.primary_emotion
                user_state.facial_confidence = emotion_result.emotion_confidence
            
            # Análise de áudio
            audio_result = analyses.get('audio_result')
            if audio_result:
                user_state.vocal_emotion = audio_result.vocal_emotion.value
                user_state.vocal_confidence = audio_result.emotion_confidence
                user_state.energy_level = audio_result.energy_level
                user_state.stress_level = audio_result.stress_level
                user_state.confidence_level = audio_result.confidence_level
            
            # Análise de sentimento
            sentiment_result = analyses.get('sentiment_result')
            if sentiment_result:
                user_state.text_sentiment = sentiment_result.overall_sentiment
                user_state.sentiment_confidence = sentiment_result.overall_confidence
            
            # Análise de intenção
            intent_result = analyses.get('intent_result')
            if intent_result:
                user_state.current_intent = intent_result.primary_intent.intent
                user_state.intent_confidence = intent_result.primary_intent.confidence
                user_state.needs_clarification = intent_result.requires_clarification
            
            # Análise de pose corporal
            pose_result = analyses.get('pose_result')
            if pose_result:
                user_state.engagement_level = pose_result.engagement_level
                if hasattr(pose_result, 'energy_level'):
                    user_state.energy_level = max(user_state.energy_level, pose_result.energy_level)
            
            # Calcula estado integrado
            user_state.overall_mood = self._calculate_overall_mood(user_state)
            user_state.consistency_score = self._calculate_consistency_score(user_state)
            
            return user_state
            
        except Exception as e:
            self.logger.error(f"Erro ao integrar análises: {e}")
            return user_state
    
    def _calculate_overall_mood(self, user_state: UserState) -> UserMood:
        """Calcula humor geral baseado em todas as análises."""
        try:
            # Mapeia emoções/sentimentos para scores numéricos
            emotion_score = 0.0
            sentiment_score = 0.0
            
            # Score da emoção facial
            emotion_map = {
                'happy': 0.8, 'excited': 0.9, 'calm': 0.3, 'confident': 0.5,
                'neutral': 0.0, 'sad': -0.7, 'angry': -0.8, 'stressed': -0.6,
                'surprised': 0.2, 'confused': -0.3, 'frustrated': -0.7
            }
            
            if user_state.facial_emotion in emotion_map:
                emotion_score = emotion_map[user_state.facial_emotion] * user_state.facial_confidence
            
            # Score do sentimento textual
            sentiment_map = {
                'positive': 0.6, 'negative': -0.6, 'neutral': 0.0
            }
            
            if user_state.text_sentiment in sentiment_map:
                sentiment_score = sentiment_map[user_state.text_sentiment] * user_state.sentiment_confidence
            
            # Fatores adicionais
            stress_penalty = -user_state.stress_level * 0.5
            energy_bonus = (user_state.energy_level - 0.5) * 0.3
            
            # Score combinado
            combined_score = (emotion_score * 0.4 + sentiment_score * 0.4 + 
                            stress_penalty * 0.1 + energy_bonus * 0.1)
            
            # Mapeia para enum
            if combined_score >= 0.6:
                return UserMood.VERY_POSITIVE
            elif combined_score >= 0.2:
                return UserMood.POSITIVE
            elif combined_score >= -0.2:
                return UserMood.NEUTRAL
            elif combined_score >= -0.6:
                return UserMood.NEGATIVE
            else:
                return UserMood.VERY_NEGATIVE
                
        except Exception as e:
            self.logger.error(f"Erro ao calcular humor geral: {e}")
            return UserMood.NEUTRAL
    
    def _calculate_consistency_score(self, user_state: UserState) -> float:
        """Calcula consistência entre diferentes análises."""
        try:
            # Compara emoção facial com sentimento textual
            emotion_sentiment_consistency = 0.5
            
            positive_emotions = ['happy', 'excited', 'confident', 'calm']
            negative_emotions = ['sad', 'angry', 'stressed', 'frustrated']
            
            if (user_state.facial_emotion in positive_emotions and user_state.text_sentiment == 'positive') or \
               (user_state.facial_emotion in negative_emotions and user_state.text_sentiment == 'negative') or \
               (user_state.facial_emotion == 'neutral' and user_state.text_sentiment == 'neutral'):
                emotion_sentiment_consistency = 1.0
            elif (user_state.facial_emotion in positive_emotions and user_state.text_sentiment == 'negative') or \
                 (user_state.facial_emotion in negative_emotions and user_state.text_sentiment == 'positive'):
                emotion_sentiment_consistency = 0.0
            
            # Compara confiança das análises
            confidence_consistency = np.mean([
                user_state.facial_confidence,
                user_state.vocal_confidence,
                user_state.sentiment_confidence,
                user_state.intent_confidence
            ])
            
            # Score final
            consistency = (emotion_sentiment_consistency * 0.6 + confidence_consistency * 0.4)
            
            return min(1.0, max(0.0, consistency))
            
        except Exception as e:
            self.logger.error(f"Erro ao calcular consistência: {e}")
            return 0.5
    
    def _determine_dialogue_state(self, turn: ConversationTurn) -> DialogueState:
        """Determina o estado do diálogo baseado no contexto."""
        try:
            # Primeira interação - sempre greeting
            if not self.current_session.turns:
                return DialogueState.GREETING
            
            # Baseado na intenção
            if turn.intent_result:
                intent = turn.intent_result.primary_intent.intent
                
                if intent == INTENTS.GREETING:
                    return DialogueState.GREETING
                elif intent == INTENTS.GOODBYE:
                    return DialogueState.CLOSING
                elif intent in [INTENTS.QUESTION, INTENTS.EVENT_INFO, INTENTS.LOCATION_REQUEST]:
                    return DialogueState.PROVIDING_INFO
                elif intent in [INTENTS.REQUEST, INTENTS.HELP]:
                    return DialogueState.HANDLING_REQUEST
                elif intent == INTENTS.CLARIFICATION or turn.user_state.needs_clarification:
                    return DialogueState.WAITING_CLARIFICATION
            
            # Se não conseguiu determinar, mantém conversa ativa
            return DialogueState.ACTIVE_CONVERSATION
            
        except Exception as e:
            self.logger.error(f"Erro ao determinar estado do diálogo: {e}")
            return DialogueState.ACTIVE_CONVERSATION
    
    def _transition_state(self, new_state: DialogueState):
        """Realiza transição de estado do diálogo."""
        old_state = self.current_state
        
        if old_state != new_state:
            self.current_state = new_state
            self.current_session.current_state = new_state
            
            # Atualiza métricas
            transition = f"{old_state.value}->{new_state.value}"
            self.performance_metrics['state_transitions'][transition] += 1
            
            self.logger.debug(f"Transição de estado: {old_state.value} -> {new_state.value}")
    
    def _extract_entities(self, turn: ConversationTurn) -> Dict:
        """Extrai entidades do turn atual."""
        entities = {}
        
        try:
            # Entidades da análise de intenção
            if turn.intent_result and turn.intent_result.entities:
                entities.update(turn.intent_result.entities)
            
            # Entidades do texto reconhecido
            if turn.recognition_result:
                # Implementação básica - em produção usaria NER mais avançado
                text = turn.recognition_result.full_text.lower()
                
                # Extrai horários
                import re
                times = re.findall(r'\b(\d{1,2}:\d{2}|\d{1,2}h\d{0,2})\b', text)
                if times:
                    entities['time'] = times
                
                # Extrai localizações
                location_keywords = ['sala', 'auditório', 'estande', 'pavilhão', 'andar']
                locations = [word for word in text.split() 
                           if any(keyword in word for keyword in location_keywords)]
                if locations:
                    entities['location'] = locations
            
            return entities
            
        except Exception as e:
            self.logger.error(f"Erro ao extrair entidades: {e}")
            return {}
    
    def _determine_topic(self, turn: ConversationTurn) -> str:
        """Determina o tópico da conversa."""
        try:
            # Baseado na intenção
            if turn.intent_result:
                intent = turn.intent_result.primary_intent.intent
                category = turn.intent_result.primary_intent.category.value
                
                if intent == INTENTS.EVENT_INFO:
                    return "event_information"
                elif intent == INTENTS.LOCATION_REQUEST:
                    return "navigation"
                elif intent in [INTENTS.HELP, INTENTS.CLARIFICATION]:
                    return "support"
                elif intent == INTENTS.COMPLAINT:
                    return "issue_resolution"
                else:
                    return category
            
            return "general"
            
        except Exception as e:
            self.logger.error(f"Erro ao determinar tópico: {e}")
            return "general"
    
    def _select_response_strategy(self, turn: ConversationTurn) -> ResponseStrategy:
        """Seleciona estratégia de resposta baseada no contexto."""
        try:
            user_mood = turn.user_state.overall_mood if turn.user_state else UserMood.NEUTRAL
            intent = turn.intent_result.primary_intent.intent if turn.intent_result else "unknown"
            
            # Estratégia baseada no humor do usuário
            if user_mood in [UserMood.NEGATIVE, UserMood.VERY_NEGATIVE, UserMood.FRUSTRATED]:
                return ResponseStrategy.EMPATHETIC
            
            # Estratégia baseada na intenção
            if intent == INTENTS.QUESTION or intent == INTENTS.EVENT_INFO:
                return ResponseStrategy.INFORMATIVE
            elif intent == INTENTS.REQUEST:
                return ResponseStrategy.DIRECTIVE
            elif intent in [INTENTS.HELP, INTENTS.CLARIFICATION]:
                return ResponseStrategy.ENCOURAGING
            elif intent == INTENTS.GREETING or intent == INTENTS.COMPLIMENT:
                return ResponseStrategy.CASUAL
            
            # Verifica se precisa de clarificação
            if turn.user_state and turn.user_state.needs_clarification:
                return ResponseStrategy.CLARIFYING
            
            # Padrão baseado no estado do diálogo
            if self.current_state == DialogueState.GREETING:
                return ResponseStrategy.CASUAL
            elif self.current_state == DialogueState.PROVIDING_INFO:
                return ResponseStrategy.INFORMATIVE
            else:
                return ResponseStrategy.INFORMATIVE
                
        except Exception as e:
            self.logger.error(f"Erro ao selecionar estratégia: {e}")
            return ResponseStrategy.INFORMATIVE
    
    def _generate_response(self, turn: ConversationTurn, strategy: ResponseStrategy) -> Dict:
        """Gera resposta baseada no turn e estratégia."""
        try:
            response_text = ""
            metadata = {
                'strategy': strategy.value,
                'confidence': 0.8,
                'requires_followup': False,
                'emotion_to_display': 'neutral',
                'animation_suggestions': []
            }
            
            # Gera resposta baseada na estratégia
            if strategy == ResponseStrategy.EMPATHETIC:
                response_text = self._generate_empathetic_response(turn)
                metadata['emotion_to_display'] = 'caring'
                
            elif strategy == ResponseStrategy.INFORMATIVE:
                response_text = self._generate_informative_response(turn)
                metadata['emotion_to_display'] = 'confident'
                
            elif strategy == ResponseStrategy.ENCOURAGING:
                response_text = self._generate_encouraging_response(turn)
                metadata['emotion_to_display'] = 'supportive'
                
            elif strategy == ResponseStrategy.CLARIFYING:
                response_text = self._generate_clarifying_response(turn)
                metadata['requires_followup'] = True
                
            elif strategy == ResponseStrategy.DIRECTIVE:
                response_text = self._generate_directive_response(turn)
                metadata['emotion_to_display'] = 'helpful'
                
            elif strategy == ResponseStrategy.CASUAL:
                response_text = self._generate_casual_response(turn)
                metadata['emotion_to_display'] = 'friendly'
            
            # Fallback se não gerou resposta
            if not response_text:
                response_text = self._generate_fallback_response(turn)
            
            # Adapta resposta baseada na personalidade
            response_text = self._adapt_response_to_personality(response_text, turn)
            
            # Sugere animações
            metadata['animation_suggestions'] = self._suggest_animations(strategy, turn)
            
            return {
                'text': response_text,
                'metadata': metadata
            }
            
        except Exception as e:
            self.logger.error(f"Erro ao gerar resposta: {e}")
            return {
                'text': "Desculpe, houve um problema. Como posso ajudá-lo?",
                'metadata': {'strategy': 'fallback', 'confidence': 0.5}
            }
    
    def _generate_empathetic_response(self, turn: ConversationTurn) -> str:
        """Gera resposta empática."""
        try:
            empathy_phrases = [
                "Entendo como você se sente.",
                "Isso deve ser realmente frustrante.",
                "Compreendo sua preocupação.",
                "Você tem razão em se sentir assim."
            ]
            
            base_phrase = np.random.choice(empathy_phrases)
            
            # Adiciona solução se possível
            if turn.intent_result:
                intent = turn.intent_result.primary_intent.intent
                if intent == INTENTS.COMPLAINT:
                    return f"{base_phrase} Vou fazer o meu melhor para te ajudar a resolver isso. Pode me contar mais detalhes sobre o problema?"
                elif intent == INTENTS.HELP:
                    return f"{base_phrase} Estou aqui para te ajudar. Vamos resolver isso juntos!"
            
            return f"{base_phrase} Como posso te ajudar a melhorar essa situação?"
            
        except Exception as e:
            self.logger.error(f"Erro na resposta empática: {e}")
            return "Entendo sua situação. Como posso ajudar?"
    
    def _generate_informative_response(self, turn: ConversationTurn) -> str:
        """Gera resposta informativa."""
        try:
            if turn.intent_result:
                intent = turn.intent_result.primary_intent.intent
                
                if intent == INTENTS.EVENT_INFO:
                    info = self.knowledge_base.get('event_info', {})
                    return f"Sobre o evento: {info.get('description', 'Consulte nossa equipe para mais informações.')} A programação completa está disponível no site oficial."
                
                elif intent == INTENTS.LOCATION_REQUEST:
                    return "Para informações sobre localização, posso te ajudar! Os estandes principais ficam no Pavilhão Azul. Se você está procurando algo específico, me diga o que precisa encontrar."
                
                elif intent == INTENTS.QUESTION:
                    return "Baseado em sua pergunta, vou te fornecer as informações mais precisas que tenho. Se precisar de mais detalhes, não hesite em perguntar!"
            
            return "Posso te fornecer informações sobre o evento, localização dos estandes, programação e muito mais. O que você gostaria de saber?"
            
        except Exception as e:
            self.logger.error(f"Erro na resposta informativa: {e}")
            return "Posso te ajudar com informações. O que você gostaria de saber?"
    
    def _generate_encouraging_response(self, turn: ConversationTurn) -> str:
        """Gera resposta encorajadora."""
        encouraging_phrases = [
            "Ótima pergunta!",
            "Que bom que você perguntou!",
            "Vou te ajudar com prazer!",
            "Não se preocupe, vamos resolver isso juntos!"
        ]
        
        base_phrase = np.random.choice(encouraging_phrases)
        
        if turn.intent_result and turn.intent_result.primary_intent.intent == INTENTS.HELP:
            return f"{base_phrase} Estou aqui exatamente para isso. Me diga como posso te ajudar e vamos encontrar a melhor solução."
        
        return f"{base_phrase} Como posso te auxiliar?"
    
    def _generate_clarifying_response(self, turn: ConversationTurn) -> str:
        """Gera resposta para clarificação."""
        clarification_phrases = [
            "Para te ajudar melhor, preciso entender um pouco mais.",
            "Só para confirmar que entendi corretamente,",
            "Pode me dar mais detalhes sobre"
        ]
        
        base_phrase = np.random.choice(clarification_phrases)
        
        if turn.topic:
            return f"{base_phrase} você está perguntando sobre {turn.topic}?"
        
        return f"{base_phrase} o que você está procurando especificamente?"
    
    def _generate_directive_response(self, turn: ConversationTurn) -> str:
        """Gera resposta diretiva com instruções."""
        if turn.intent_result:
            intent = turn.intent_result.primary_intent.intent
            
            if intent == INTENTS.REQUEST:
                return "Para atender sua solicitação, vou te guiar pelos passos necessários. Primeiro, me diga exatamente o que você precisa fazer."
        
        return "Vou te ajudar com instruções claras. Me diga o que você precisa fazer e te guio passo a passo."
    
    def _generate_casual_response(self, turn: ConversationTurn) -> str:
        """Gera resposta casual e amigável."""
        if turn.intent_result:
            intent = turn.intent_result.primary_intent.intent
            
            if intent == INTENTS.GREETING:
                greetings = self.conversation_patterns.get('greeting_responses', DEFAULT_MESSAGES.GREETING_MESSAGES)
                return np.random.choice(greetings)
            
            elif intent == INTENTS.COMPLIMENT:
                return "Que bom saber! Fico feliz em poder ajudar. Há mais alguma coisa que você gostaria de saber?"
        
        return "Oi! Como posso te ajudar hoje?"
    
    def _generate_fallback_response(self, turn: ConversationTurn) -> str:
        """Gera resposta de fallback."""
        fallbacks = DEFAULT_MESSAGES.FALLBACK_RESPONSES
        return np.random.choice(fallbacks)
    
    def _adapt_response_to_personality(self, response: str, turn: ConversationTurn) -> str:
        """Adapta resposta baseada na personalidade configurada."""
        try:
            # Ajusta formalidade
            formality = self.personality['characteristics']['formality']
            if formality < 0.4 and 'você' in response:
                response = response.replace('você', 'você')  # Mantém informal
            
            # Adiciona entusiasmo se configurado
            enthusiasm = self.personality['response_style']['enthusiasm']
            if enthusiasm > 0.7 and not response.endswith('!'):
                if turn.user_state and turn.user_state.overall_mood in [UserMood.POSITIVE, UserMood.VERY_POSITIVE]:
                    response += "!"
            
            # Adiciona empatia se usuário está negativo
            empathy = self.personality['characteristics']['empathy']
            if empathy > 0.7 and turn.user_state and turn.user_state.overall_mood in [UserMood.NEGATIVE, UserMood.VERY_NEGATIVE]:
                if not any(phrase in response.lower() for phrase in ['entendo', 'compreendo', 'sei como']):
                    response = "Entendo sua situação. " + response
            
            return response
            
        except Exception as e:
            self.logger.error(f"Erro ao adaptar resposta: {e}")
            return response
    
    def _suggest_animations(self, strategy: ResponseStrategy, turn: ConversationTurn) -> List[str]:
        """Sugere animações baseadas na estratégia e contexto."""
        animations = []
        
        try:
            if strategy == ResponseStrategy.EMPATHETIC:
                animations = ['caring_expression', 'nodding', 'gentle_gesture']
            elif strategy == ResponseStrategy.ENCOURAGING:
                animations = ['thumbs_up', 'encouraging_smile', 'open_arms']
            elif strategy == ResponseStrategy.INFORMATIVE:
                animations = ['confident_posture', 'pointing_gesture', 'explaining_hands']
            elif strategy == ResponseStrategy.CASUAL:
                animations = ['friendly_wave', 'casual_smile', 'relaxed_posture']
            
            # Adiciona baseado no humor do usuário
            if turn.user_state:
                if turn.user_state.overall_mood == UserMood.VERY_POSITIVE:
                    animations.append('excited_expression')
                elif turn.user_state.overall_mood in [UserMood.NEGATIVE, UserMood.VERY_NEGATIVE]:
                    animations.append('sympathetic_expression')
            
            return animations
            
        except Exception as e:
            self.logger.error(f"Erro ao sugerir animações: {e}")
            return ['neutral_expression']
    
    def _update_session_context(self, turn: ConversationTurn):
        """Atualiza contexto da sessão atual."""
        try:
            if not self.current_session:
                return
            
            # Atualiza tópico atual
            if turn.topic:
                self.current_session.current_topic = turn.topic
                if turn.topic not in self.current_session.main_topics:
                    self.current_session.main_topics.append(turn.topic)
            
            # Atualiza emoções dominantes
            if turn.user_state and turn.user_state.facial_emotion:
                if turn.user_state.facial_emotion not in self.current_session.dominant_emotions:
                    self.current_session.dominant_emotions.append(turn.user_state.facial_emotion)
            
            # Atualiza perfil do usuário
            if turn.user_state:
                profile = self.current_session.user_profile
                
                # Preferências inferidas
                if turn.user_state.overall_mood == UserMood.POSITIVE:
                    profile['responds_well_to'] = profile.get('responds_well_to', [])
                    if turn.response_strategy.value not in profile['responds_well_to']:
                        profile['responds_well_to'].append(turn.response_strategy.value)
                
                # Padrões de comunicação
                if turn.user_state.engagement_level > 0.7:
                    profile['high_engagement_topics'] = profile.get('high_engagement_topics', [])
                    if turn.topic not in profile['high_engagement_topics']:
                        profile['high_engagement_topics'].append(turn.topic)
            
        except Exception as e:
            self.logger.error(f"Erro ao atualizar contexto da sessão: {e}")
    
    def _update_performance_metrics(self, turn: ConversationTurn, processing_time: float):
        """Atualiza métricas de performance."""
        try:
            self.performance_metrics['total_turns'] += 1
            self.performance_metrics['processing_times'].append(processing_time)
            
            if turn.response_strategy:
                self.performance_metrics['response_strategies_used'][turn.response_strategy.value] += 1
            
            # Indicador de satisfação baseado no humor do usuário
            if turn.user_state:
                satisfaction_score = 0.5  # Neutro
                
                if turn.user_state.overall_mood in [UserMood.POSITIVE, UserMood.VERY_POSITIVE]:
                    satisfaction_score = 0.8
                elif turn.user_state.overall_mood in [UserMood.NEGATIVE, UserMood.VERY_NEGATIVE]:
                    satisfaction_score = 0.2
                
                self.performance_metrics['user_satisfaction_indicators'].append(satisfaction_score)
            
        except Exception as e:
            self.logger.error(f"Erro ao atualizar métricas: {e}")
    
    def end_session(self, reason: str = "normal") -> Dict:
        """
        Finaliza a sessão atual.
        
        Args:
            reason: Motivo do fim da sessão
            
        Returns:
            Dict: Resumo da sessão
        """
        try:
            if not self.current_session:
                return {'message': 'Nenhuma sessão ativa'}
            
            # Finaliza sessão
            end_time = time.time()
            duration = end_time - self.current_session.start_time
            
            # Calcula métricas da sessão
            if self.current_session.total_turns > 0:
                avg_response_time = np.mean([
                    turn.response_metadata.get('processing_time', 0)
                    for turn in self.current_session.turns
                ])
                self.current_session.avg_response_time = avg_response_time
            
            # Adiciona ao histórico
            self.session_history.append(self.current_session)
            
            # Prepara resumo
            summary = {
                'session_id': self.current_session.session_id,
                'duration': duration,
                'total_turns': self.current_session.total_turns,
                'main_topics': self.current_session.main_topics,
                'dominant_emotions': self.current_session.dominant_emotions,
                'avg_response_time': self.current_session.avg_response_time,
                'end_reason': reason
            }
            
            # Atualiza métricas globais
            self.performance_metrics['average_session_duration'] = np.mean([
                session.last_activity - session.start_time
                for session in self.session_history
            ])
            
            # Limpa sessão atual
            self.current_session = None
            self.current_state = DialogueState.IDLE
            self.current_user_state = None
            
            self.logger.info(f"Sessão finalizada - Duração: {duration:.1f}s, Turns: {summary['total_turns']}")
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Erro ao finalizar sessão: {e}")
            return {'error': str(e)}
    
    def get_session_status(self) -> Dict:
        """Retorna status da sessão atual."""
        if not self.current_session:
            return {'status': 'no_active_session'}
        
        current_time = time.time()
        duration = current_time - self.current_session.start_time
        inactive_time = current_time - self.current_session.last_activity
        
        return {
            'session_id': self.current_session.session_id,
            'current_state': self.current_state.value,
            'duration': duration,
            'inactive_time': inactive_time,
            'total_turns': self.current_session.total_turns,
            'current_topic': self.current_session.current_topic,
            'user_mood': self.current_user_state.overall_mood.value if self.current_user_state else 'unknown'
        }
    
    def get_performance_metrics(self) -> Dict:
        """Retorna métricas de performance do gerenciador."""
        metrics = self.performance_metrics.copy()
        
        # Adiciona métricas calculadas
        if metrics['processing_times']:
            metrics['average_processing_time'] = np.mean(metrics['processing_times'])
        
        if metrics['user_satisfaction_indicators']:
            metrics['average_satisfaction'] = np.mean(metrics['user_satisfaction_indicators'])
        
        return metrics
    
    def clear_history(self):
        """Limpa histórico de sessões."""
        self.session_history.clear()
        self.logger.info("Histórico de sessões limpo")