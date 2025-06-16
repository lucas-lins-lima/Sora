# sora_robot/response_generation/llm_integration.py

import asyncio
import threading
import time
import json
import re
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import numpy as np

# OpenAI
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None

# Google AI (Gemini)
try:
    import google.generativeai as genai
    GOOGLE_AI_AVAILABLE = True
except ImportError:
    GOOGLE_AI_AVAILABLE = False
    genai = None

# Anthropic Claude
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    anthropic = None

# Transformers para modelos locais
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from utils.logger import get_logger
from utils.constants import DEFAULT_MESSAGES
from nlp.dialogue_manager import DialogueState, ResponseStrategy, UserMood
import config

class LLMProvider(Enum):
    """Provedores de LLM disponíveis."""
    OPENAI_GPT = "openai_gpt"
    GOOGLE_GEMINI = "google_gemini"
    ANTHROPIC_CLAUDE = "anthropic_claude"
    LOCAL_MODEL = "local_model"
    AUTO = "auto"

class ResponseQuality(Enum):
    """Qualidade da resposta gerada."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    FAILED = "failed"

@dataclass
class LLMConfig:
    """Configuração para um provedor de LLM."""
    provider: LLMProvider
    model_name: str
    api_key: Optional[str] = None
    max_tokens: int = 500
    temperature: float = 0.7
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    timeout: float = 10.0
    enabled: bool = True

@dataclass
class PromptContext:
    """Contexto para geração de prompt."""
    
    # Dados do usuário
    user_text: str
    user_mood: str = "neutral"
    user_intent: str = "unknown"
    user_sentiment: str = "neutral"
    user_emotion: str = "neutral"
    
    # Estratégia de resposta
    response_strategy: str = "informative"
    dialogue_state: str = "active_conversation"
    
    # Contexto da conversa
    conversation_history: List[Dict] = field(default_factory=list)
    current_topic: str = ""
    entities_extracted: Dict = field(default_factory=dict)
    
    # Personalidade do robô
    personality_traits: Dict = field(default_factory=dict)
    
    # Contexto do evento
    event_context: Dict = field(default_factory=dict)
    
    # Metadados
    session_duration: float = 0.0
    user_engagement: float = 0.5

@dataclass
class LLMResponse:
    """Resposta gerada por LLM."""
    
    text: str
    provider: LLMProvider
    model_used: str
    
    # Métricas de qualidade
    quality_score: float = 0.0
    confidence: float = 0.0
    relevance: float = 0.0
    coherence: float = 0.0
    
    # Metadados de geração
    tokens_used: int = 0
    generation_time: float = 0.0
    temperature_used: float = 0.7
    
    # Análise do conteúdo
    contains_question: bool = False
    emotional_tone: str = "neutral"
    suggested_followup: Optional[str] = None
    
    # Informações de erro
    error: Optional[str] = None
    fallback_used: bool = False

class LLMIntegration:
    """
    Classe responsável pela integração com Large Language Models.
    Gerencia múltiplos provedores e gera respostas contextualmente apropriadas.
    """
    
    def __init__(self):
        """Inicializa o sistema de integração LLM."""
        self.logger = get_logger(__name__)
        
        # Configurações de provedores
        self.providers_config = {}
        self.active_provider = None
        self.fallback_providers = []
        
        # Estado do sistema
        self.is_initialized = False
        
        # Threading
        self.generation_lock = threading.Lock()
        
        # Cache de respostas
        self.response_cache = {}
        self.cache_ttl = 300  # 5 minutos
        
        # Histórico de respostas
        self.response_history = deque(maxlen=100)
        
        # Templates de prompt
        self.prompt_templates = self._load_prompt_templates()
        
        # Modelos locais
        self.local_models = {}
        
        # Métricas de performance
        self.performance_metrics = {
            'total_requests': 0,
            'successful_responses': 0,
            'failed_responses': 0,
            'average_response_time': 0.0,
            'response_times': deque(maxlen=100),
            'provider_usage': {},
            'quality_scores': deque(maxlen=100),
            'cache_hits': 0,
            'token_usage': {
                'total_tokens': 0,
                'input_tokens': 0,
                'output_tokens': 0
            }
        }
        
        # Inicializa componentes
        self._initialize_providers()
        
        self.logger.info("LLMIntegration inicializado")
    
    def _initialize_providers(self):
        """Inicializa provedores de LLM disponíveis."""
        try:
            # OpenAI GPT
            if OPENAI_AVAILABLE and hasattr(config, 'API_KEYS') and config.API_KEYS.get('openai'):
                self.providers_config[LLMProvider.OPENAI_GPT] = LLMConfig(
                    provider=LLMProvider.OPENAI_GPT,
                    model_name="gpt-3.5-turbo",
                    api_key=config.API_KEYS['openai'],
                    max_tokens=400,
                    temperature=0.7
                )
                openai.api_key = config.API_KEYS['openai']
                self.logger.info("OpenAI GPT configurado")
            
            # Google Gemini
            if GOOGLE_AI_AVAILABLE and hasattr(config, 'API_KEYS') and config.API_KEYS.get('gemini'):
                self.providers_config[LLMProvider.GOOGLE_GEMINI] = LLMConfig(
                    provider=LLMProvider.GOOGLE_GEMINI,
                    model_name="gemini-pro",
                    api_key=config.API_KEYS['gemini'],
                    max_tokens=400,
                    temperature=0.7
                )
                genai.configure(api_key=config.API_KEYS['gemini'])
                self.logger.info("Google Gemini configurado")
            
            # Anthropic Claude
            if ANTHROPIC_AVAILABLE and hasattr(config, 'API_KEYS') and config.API_KEYS.get('anthropic'):
                self.providers_config[LLMProvider.ANTHROPIC_CLAUDE] = LLMConfig(
                    provider=LLMProvider.ANTHROPIC_CLAUDE,
                    model_name="claude-3-sonnet-20240229",
                    api_key=config.API_KEYS['anthropic'],
                    max_tokens=400,
                    temperature=0.7
                )
                self.logger.info("Anthropic Claude configurado")
            
            # Seleciona provedor principal
            if self.providers_config:
                # Prioridade: OpenAI > Gemini > Claude
                if LLMProvider.OPENAI_GPT in self.providers_config:
                    self.active_provider = LLMProvider.OPENAI_GPT
                elif LLMProvider.GOOGLE_GEMINI in self.providers_config:
                    self.active_provider = LLMProvider.GOOGLE_GEMINI
                elif LLMProvider.ANTHROPIC_CLAUDE in self.providers_config:
                    self.active_provider = LLMProvider.ANTHROPIC_CLAUDE
                
                # Define fallbacks
                self.fallback_providers = [p for p in self.providers_config.keys() if p != self.active_provider]
                
                self.is_initialized = True
                self.logger.info(f"Provedor principal: {self.active_provider.value}")
            else:
                self.logger.warning("Nenhum provedor LLM configurado - usando fallbacks")
                self.is_initialized = False
                
        except Exception as e:
            self.logger.error(f"Erro ao inicializar provedores: {e}")
            self.is_initialized = False
    
    def _load_prompt_templates(self) -> Dict:
        """Carrega templates de prompt para diferentes contextos."""
        return {
            'system_message': """Você é a Sora, uma assistente virtual empática e inteligente em um evento tecnológico. Suas características:

PERSONALIDADE:
- Amigável, prestativa e empática
- Adapta-se ao humor e estado emocional do usuário
- Mantém conversas naturais e envolventes
- Demonstra interesse genuíno pelas pessoas

COMPORTAMENTO:
- Use linguagem natural e acessível
- Seja concisa mas informativa
- Demonstre empatia quando o usuário estiver negativo
- Celebre quando o usuário estiver positivo
- Faça perguntas de acompanhamento quando apropriado

CONTEXTO:
- Você está em um evento tecnológico
- Pode fornecer informações sobre programação, localização e atividades
- Quando não souber algo específico, seja honesta e ofereça ajuda alternativa

INSTRUÇÕES:
- Sempre responda em português brasileiro
- Mantenha respostas entre 1-3 frases
- Adapte o tom ao humor do usuário
- Seja natural e conversacional""",

            'response_templates': {
                'empathetic': """O usuário está se sentindo {user_mood} e disse: "{user_text}"

Estratégia: Resposta empática e compreensiva
Humor do usuário: {user_mood}
Contexto: {context}

Responda de forma empática, reconhecendo os sentimentos do usuário:""",

                'informative': """O usuário perguntou: "{user_text}"

Estratégia: Resposta informativa e útil
Intenção: {user_intent}
Tópico: {current_topic}
Contexto: {context}

Forneça uma resposta informativa e útil:""",

                'encouraging': """O usuário disse: "{user_text}"

Estratégia: Resposta encorajadora e motivacional
Situação: Usuário precisa de apoio
Contexto: {context}

Responda de forma encorajadora e oferecendo suporte:""",

                'casual': """O usuário disse: "{user_text}"

Estratégia: Conversa casual e amigável
Humor: {user_mood}
Contexto: {context}

Responda de forma amigável e descontraída:""",

                'clarifying': """O usuário disse: "{user_text}"

Estratégia: Buscar clarificação
Problema: Intenção não está clara
Contexto: {context}

Faça uma pergunta para esclarecer o que o usuário precisa:"""
            },

            'context_enhancers': {
                'conversation_history': "Conversa anterior: {history}",
                'user_emotion': "Estado emocional: {emotion} (confiança: {confidence})",
                'event_context': "Contexto do evento: {event_info}",
                'session_info': "Duração da conversa: {duration}, Engajamento: {engagement}"
            }
        }
    
    async def generate_response(self, context: PromptContext) -> LLMResponse:
        """
        Gera resposta usando LLM baseado no contexto fornecido.
        
        Args:
            context: Contexto para geração da resposta
            
        Returns:
            LLMResponse: Resposta gerada
        """
        start_time = time.time()
        
        try:
            # Verifica cache primeiro
            cache_key = self._generate_cache_key(context)
            cached_response = self._get_cached_response(cache_key)
            if cached_response:
                self.performance_metrics['cache_hits'] += 1
                return cached_response
            
            # Gera prompt
            prompt = self._build_prompt(context)
            
            # Tenta gerar resposta com provedor principal
            response = None
            if self.active_provider and self.is_initialized:
                response = await self._generate_with_provider(prompt, context, self.active_provider)
            
            # Fallback para outros provedores se necessário
            if not response or response.error:
                for fallback_provider in self.fallback_providers:
                    try:
                        response = await self._generate_with_provider(prompt, context, fallback_provider)
                        if response and not response.error:
                            break
                    except Exception as e:
                        self.logger.warning(f"Fallback {fallback_provider.value} falhou: {e}")
                        continue
            
            # Fallback final para resposta pré-definida
            if not response or response.error:
                response = self._generate_fallback_response(context)
                response.fallback_used = True
            
            # Pós-processamento
            if response:
                response.generation_time = time.time() - start_time
                response = self._post_process_response(response, context)
                
                # Atualiza cache
                self._cache_response(cache_key, response)
                
                # Atualiza histórico
                self.response_history.append(response)
                
                # Atualiza métricas
                self._update_performance_metrics(response, True)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Erro na geração de resposta: {e}")
            # Retorna resposta de erro
            error_response = LLMResponse(
                text="Desculpe, houve um problema técnico. Como posso ajudá-lo?",
                provider=LLMProvider.AUTO,
                model_used="fallback",
                error=str(e),
                fallback_used=True,
                generation_time=time.time() - start_time
            )
            self._update_performance_metrics(error_response, False)
            return error_response
    
    def generate_response_sync(self, context: PromptContext) -> LLMResponse:
        """Versão síncrona da geração de resposta."""
        try:
            # Tenta usar loop de eventos existente
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Se já há um loop rodando, cria uma task
                future = asyncio.create_task(self.generate_response(context))
                # Aguarda a conclusão usando threading
                return asyncio.run_coroutine_threadsafe(self.generate_response(context), loop).result(timeout=30)
            else:
                # Se não há loop, cria um novo
                return asyncio.run(self.generate_response(context))
        except Exception as e:
            # Fallback para geração síncrona simples
            self.logger.warning(f"Fallback para geração síncrona: {e}")
            return self._generate_simple_response(context)
    
    def _build_prompt(self, context: PromptContext) -> str:
        """Constrói prompt baseado no contexto."""
        try:
            # Seleciona template baseado na estratégia
            strategy_template = self.prompt_templates['response_templates'].get(
                context.response_strategy, 
                self.prompt_templates['response_templates']['informative']
            )
            
            # Constrói contexto adicional
            context_parts = []
            
            # Adiciona histórico se disponível
            if context.conversation_history:
                history_text = " | ".join([
                    f"Usuário: {turn.get('user', '')} -> Sora: {turn.get('bot', '')}"
                    for turn in context.conversation_history[-3:]  # Últimas 3 interações
                ])
                context_parts.append(f"Histórico recente: {history_text}")
            
            # Adiciona contexto emocional
            if context.user_emotion != "neutral":
                context_parts.append(f"Emoção do usuário: {context.user_emotion}")
            
            # Adiciona entidades extraídas
            if context.entities_extracted:
                entities_text = ", ".join([
                    f"{key}: {', '.join(values) if isinstance(values, list) else values}"
                    for key, values in context.entities_extracted.items()
                ])
                context_parts.append(f"Entidades mencionadas: {entities_text}")
            
            # Adiciona contexto do evento
            if context.event_context:
                event_info = context.event_context.get('description', 'Evento tecnológico')
                context_parts.append(f"Evento: {event_info}")
            
            combined_context = " | ".join(context_parts) if context_parts else "Conversa geral"
            
            # Formata o prompt final
            prompt = strategy_template.format(
                user_text=context.user_text,
                user_mood=context.user_mood,
                user_intent=context.user_intent,
                current_topic=context.current_topic or "geral",
                context=combined_context
            )
            
            return prompt
            
        except Exception as e:
            self.logger.error(f"Erro ao construir prompt: {e}")
            return f"O usuário disse: '{context.user_text}'. Responda de forma útil e empática."
    
    async def _generate_with_provider(self, prompt: str, context: PromptContext, provider: LLMProvider) -> Optional[LLMResponse]:
        """Gera resposta com provedor específico."""
        try:
            config = self.providers_config.get(provider)
            if not config or not config.enabled:
                return None
            
            start_time = time.time()
            
            if provider == LLMProvider.OPENAI_GPT:
                return await self._generate_openai(prompt, context, config)
            elif provider == LLMProvider.GOOGLE_GEMINI:
                return await self._generate_gemini(prompt, context, config)
            elif provider == LLMProvider.ANTHROPIC_CLAUDE:
                return await self._generate_claude(prompt, context, config)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Erro com provedor {provider.value}: {e}")
            return LLMResponse(
                text="",
                provider=provider,
                model_used=config.model_name if config else "unknown",
                error=str(e)
            )
    
    async def _generate_openai(self, prompt: str, context: PromptContext, config: LLMConfig) -> LLMResponse:
        """Gera resposta usando OpenAI GPT."""
        try:
            messages = [
                {"role": "system", "content": self.prompt_templates['system_message']},
                {"role": "user", "content": prompt}
            ]
            
            # Adiciona histórico da conversa se disponível
            if context.conversation_history:
                for turn in context.conversation_history[-2:]:  # Últimas 2 interações
                    if turn.get('user'):
                        messages.insert(-1, {"role": "user", "content": turn['user']})
                    if turn.get('bot'):
                        messages.insert(-1, {"role": "assistant", "content": turn['bot']})
            
            response = await openai.ChatCompletion.acreate(
                model=config.model_name,
                messages=messages,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                frequency_penalty=config.frequency_penalty,
                presence_penalty=config.presence_penalty,
                timeout=config.timeout
            )
            
            generated_text = response.choices[0].message.content.strip()
            tokens_used = response.usage.total_tokens
            
            return LLMResponse(
                text=generated_text,
                provider=LLMProvider.OPENAI_GPT,
                model_used=config.model_name,
                tokens_used=tokens_used,
                temperature_used=config.temperature
            )
            
        except Exception as e:
            self.logger.error(f"Erro OpenAI: {e}")
            return LLMResponse(
                text="",
                provider=LLMProvider.OPENAI_GPT,
                model_used=config.model_name,
                error=str(e)
            )
    
    async def _generate_gemini(self, prompt: str, context: PromptContext, config: LLMConfig) -> LLMResponse:
        """Gera resposta usando Google Gemini."""
        try:
            model = genai.GenerativeModel(config.model_name)
            
            # Constrói prompt completo
            full_prompt = f"{self.prompt_templates['system_message']}\n\n{prompt}"
            
            # Configura geração
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p
            )
            
            response = await model.generate_content_async(
                full_prompt,
                generation_config=generation_config
            )
            
            generated_text = response.text.strip()
            
            return LLMResponse(
                text=generated_text,
                provider=LLMProvider.GOOGLE_GEMINI,
                model_used=config.model_name,
                temperature_used=config.temperature
            )
            
        except Exception as e:
            self.logger.error(f"Erro Gemini: {e}")
            return LLMResponse(
                text="",
                provider=LLMProvider.GOOGLE_GEMINI,
                model_used=config.model_name,
                error=str(e)
            )
    
    async def _generate_claude(self, prompt: str, context: PromptContext, config: LLMConfig) -> LLMResponse:
        """Gera resposta usando Anthropic Claude."""
        try:
            client = anthropic.AsyncAnthropic(api_key=config.api_key)
            
            # Constrói mensagens
            messages = [
                {"role": "user", "content": f"{self.prompt_templates['system_message']}\n\n{prompt}"}
            ]
            
            response = await client.messages.create(
                model=config.model_name,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                messages=messages
            )
            
            generated_text = response.content[0].text.strip()
            tokens_used = response.usage.input_tokens + response.usage.output_tokens
            
            return LLMResponse(
                text=generated_text,
                provider=LLMProvider.ANTHROPIC_CLAUDE,
                model_used=config.model_name,
                tokens_used=tokens_used,
                temperature_used=config.temperature
            )
            
        except Exception as e:
            self.logger.error(f"Erro Claude: {e}")
            return LLMResponse(
                text="",
                provider=LLMProvider.ANTHROPIC_CLAUDE,
                model_used=config.model_name,
                error=str(e)
            )
    
    def _generate_simple_response(self, context: PromptContext) -> LLMResponse:
        """Gera resposta simples sem LLM (fallback)."""
        try:
            # Resposta baseada na estratégia e contexto
            if context.response_strategy == "empathetic":
                if context.user_mood in ["negative", "very_negative", "frustrated"]:
                    text = f"Entendo como você se sente. {self._get_empathetic_followup(context)}"
                else:
                    text = "Compreendo sua situação. Como posso ajudá-lo melhor?"
            
            elif context.response_strategy == "informative":
                if context.user_intent == "question":
                    text = "Vou te ajudar com essa informação. Com base no que sei sobre o evento, posso te orientar."
                else:
                    text = "Posso fornecer informações sobre o evento. O que você gostaria de saber?"
            
            elif context.response_strategy == "encouraging":
                text = "Que ótima pergunta! Vou te ajudar com prazer. Em que posso ser útil?"
            
            elif context.response_strategy == "clarifying":
                text = f"Para te ajudar melhor, você está perguntando sobre {context.current_topic or 'o evento'}?"
            
            else:
                # Casual ou padrão
                if context.user_intent == "greeting":
                    text = np.random.choice(DEFAULT_MESSAGES.GREETING_MESSAGES)
                else:
                    text = np.random.choice(DEFAULT_MESSAGES.FALLBACK_RESPONSES)
            
            return LLMResponse(
                text=text,
                provider=LLMProvider.AUTO,
                model_used="rule_based",
                quality_score=0.6,
                confidence=0.7,
                fallback_used=True
            )
            
        except Exception as e:
            self.logger.error(f"Erro na resposta simples: {e}")
            return LLMResponse(
                text="Como posso ajudá-lo hoje?",
                provider=LLMProvider.AUTO,
                model_used="static",
                error=str(e),
                fallback_used=True
            )
    
    def _get_empathetic_followup(self, context: PromptContext) -> str:
        """Gera follow-up empático baseado no contexto."""
        if context.user_intent == "complaint":
            return "Vou fazer o meu melhor para resolver isso. Pode me contar mais sobre o problema?"
        elif context.user_intent == "help":
            return "Estou aqui para te ajudar. Vamos resolver isso juntos!"
        else:
            return "Como posso te ajudar a melhorar essa situação?"
    
    def _generate_fallback_response(self, context: PromptContext) -> LLMResponse:
        """Gera resposta de fallback quando todos os provedores falham."""
        fallback_messages = [
            "Desculpe, estou com um pequeno problema técnico. Como posso ajudá-lo?",
            "Houve uma dificuldade momentânea. Pode repetir sua pergunta?",
            "Estou processando sua solicitação. Pode me dar mais detalhes?",
            "Desculpe pela demora. Em que posso ser útil?"
        ]
        
        return LLMResponse(
            text=np.random.choice(fallback_messages),
            provider=LLMProvider.AUTO,
            model_used="fallback",
            quality_score=0.5,
            confidence=0.6,
            fallback_used=True
        )
    
    def _post_process_response(self, response: LLMResponse, context: PromptContext) -> LLMResponse:
        """Pós-processa a resposta gerada."""
        try:
            # Limpa e formata texto
            text = response.text.strip()
            
            # Remove aspas desnecessárias
            text = re.sub(r'^["\']|["\']$', '', text)
            
            # Garante que termina com pontuação
            if text and not text.endswith(('.', '!', '?')):
                text += '.'
            
            # Analisa características da resposta
            response.contains_question = '?' in text
            response.emotional_tone = self._analyze_emotional_tone(text)
            response.suggested_followup = self._suggest_followup(text, context)
            
            # Calcula métricas de qualidade
            response.quality_score = self._calculate_quality_score(text, context)
            response.confidence = self._calculate_confidence_score(response)
            response.relevance = self._calculate_relevance_score(text, context)
            response.coherence = self._calculate_coherence_score(text)
            
            response.text = text
            return response
            
        except Exception as e:
            self.logger.error(f"Erro no pós-processamento: {e}")
            return response
    
    def _analyze_emotional_tone(self, text: str) -> str:
        """Analisa tom emocional da resposta."""
        text_lower = text.lower()
        
        # Palavras indicativas de diferentes tons
        positive_words = ['ótimo', 'excelente', 'fantástico', 'maravilhoso', 'legal', 'bom']
        empathetic_words = ['entendo', 'compreendo', 'sei como', 'sinto muito']
        encouraging_words = ['você consegue', 'vai dar certo', 'não se preocupe', 'vamos resolver']
        neutral_words = ['informação', 'dados', 'detalhes', 'processo']
        
        if any(word in text_lower for word in empathetic_words):
            return 'empathetic'
        elif any(word in text_lower for word in encouraging_words):
            return 'encouraging'
        elif any(word in text_lower for word in positive_words):
            return 'positive'
        elif any(word in text_lower for word in neutral_words):
            return 'informative'
        else:
            return 'neutral'
    
    def _suggest_followup(self, text: str, context: PromptContext) -> Optional[str]:
        """Sugere pergunta de acompanhamento."""
        if context.response_strategy == "informative" and not text.endswith('?'):
            return "Há mais alguma coisa sobre isso que você gostaria de saber?"
        elif context.response_strategy == "empathetic":
            return "Como posso te ajudar melhor com isso?"
        elif context.user_mood in ["confused", "negative"]:
            return "Posso esclarecer alguma coisa para você?"
        
        return None
    
    def _calculate_quality_score(self, text: str, context: PromptContext) -> float:
        """Calcula score de qualidade da resposta."""
        score = 0.5  # Base
        
        # Comprimento apropriado
        word_count = len(text.split())
        if 10 <= word_count <= 50:
            score += 0.2
        elif word_count < 5:
            score -= 0.2
        
        # Relevância ao contexto
        if context.current_topic and context.current_topic.lower() in text.lower():
            score += 0.1
        
        # Adequação à estratégia
        strategy_keywords = {
            'empathetic': ['entendo', 'compreendo', 'sinto'],
            'informative': ['informação', 'sobre', 'posso'],
            'encouraging': ['ótimo', 'você consegue', 'vamos'],
            'clarifying': ['esclarecer', 'confirmar', 'entender']
        }
        
        if context.response_strategy in strategy_keywords:
            keywords = strategy_keywords[context.response_strategy]
            if any(keyword in text.lower() for keyword in keywords):
                score += 0.2
        
        return min(1.0, max(0.0, score))
    
    def _calculate_confidence_score(self, response: LLMResponse) -> float:
        """Calcula score de confiança da resposta."""
        confidence = 0.5
        
        # Resposta sem erro
        if not response.error:
            confidence += 0.3
        
        # Modelo de qualidade usado
        if response.provider in [LLMProvider.OPENAI_GPT, LLMProvider.GOOGLE_GEMINI, LLMProvider.ANTHROPIC_CLAUDE]:
            confidence += 0.2
        
        # Tamanho apropriado
        if 20 <= len(response.text) <= 200:
            confidence += 0.1
        
        return min(1.0, max(0.0, confidence))
    
    def _calculate_relevance_score(self, text: str, context: PromptContext) -> float:
        """Calcula relevância da resposta ao contexto."""
        relevance = 0.5
        
        # Menciona elementos do contexto
        if context.user_text:
            user_words = set(context.user_text.lower().split())
            response_words = set(text.lower().split())
            overlap = len(user_words.intersection(response_words))
            relevance += min(0.3, overlap * 0.05)
        
        # Adequação ao tópico
        if context.current_topic and context.current_topic.lower() in text.lower():
            relevance += 0.2
        
        return min(1.0, max(0.0, relevance))
    
    def _calculate_coherence_score(self, text: str) -> float:
        """Calcula coerência da resposta."""
        coherence = 0.8  # Base alta para respostas de LLM
        
        # Verifica se há repetições excessivas
        words = text.lower().split()
        unique_words = set(words)
        repetition_ratio = len(unique_words) / len(words) if words else 1
        
        if repetition_ratio < 0.7:
            coherence -= 0.2
        
        # Verifica pontuação apropriada
        if text.count('.') + text.count('!') + text.count('?') == 0 and len(words) > 5:
            coherence -= 0.1
        
        return min(1.0, max(0.0, coherence))
    
    def _generate_cache_key(self, context: PromptContext) -> str:
        """Gera chave de cache baseada no contexto."""
        key_components = [
            context.user_text,
            context.response_strategy,
            context.user_mood,
            context.current_topic or ""
        ]
        combined = "|".join(key_components)
        return str(hash(combined))
    
    def _get_cached_response(self, cache_key: str) -> Optional[LLMResponse]:
        """Recupera resposta do cache se válida."""
        if cache_key in self.response_cache:
            cached_response, cache_time = self.response_cache[cache_key]
            if time.time() - cache_time < self.cache_ttl:
                return cached_response
        return None
    
    def _cache_response(self, cache_key: str, response: LLMResponse):
        """Armazena resposta no cache."""
        self.response_cache[cache_key] = (response, time.time())
        
        # Limpa cache antigo
        current_time = time.time()
        expired_keys = [
            key for key, (_, cache_time) in self.response_cache.items()
            if current_time - cache_time > self.cache_ttl
        ]
        for key in expired_keys:
            del self.response_cache[key]
    
    def _update_performance_metrics(self, response: LLMResponse, success: bool):
        """Atualiza métricas de performance."""
        self.performance_metrics['total_requests'] += 1
        
        if success and not response.error:
            self.performance_metrics['successful_responses'] += 1
        else:
            self.performance_metrics['failed_responses'] += 1
        
        # Tempo de resposta
        if response.generation_time > 0:
            self.performance_metrics['response_times'].append(response.generation_time)
            if self.performance_metrics['response_times']:
                self.performance_metrics['average_response_time'] = np.mean(
                    self.performance_metrics['response_times']
                )
        
        # Uso de provedores
        provider_name = response.provider.value
        if provider_name in self.performance_metrics['provider_usage']:
            self.performance_metrics['provider_usage'][provider_name] += 1
        else:
            self.performance_metrics['provider_usage'][provider_name] = 1
        
        # Score de qualidade
        if response.quality_score > 0:
            self.performance_metrics['quality_scores'].append(response.quality_score)
        
        # Tokens usados
        if response.tokens_used > 0:
            self.performance_metrics['token_usage']['total_tokens'] += response.tokens_used
    
    def get_performance_metrics(self) -> Dict:
        """Retorna métricas de performance."""
        metrics = self.performance_metrics.copy()
        
        # Adiciona métricas calculadas
        if metrics['total_requests'] > 0:
            metrics['success_rate'] = metrics['successful_responses'] / metrics['total_requests']
        else:
            metrics['success_rate'] = 0.0
        
        if metrics['quality_scores']:
            metrics['average_quality'] = np.mean(metrics['quality_scores'])
        else:
            metrics['average_quality'] = 0.0
        
        if metrics['total_requests'] > 0:
            metrics['cache_hit_rate'] = metrics['cache_hits'] / metrics['total_requests']
        else:
            metrics['cache_hit_rate'] = 0.0
        
        return metrics
    
    def get_provider_status(self) -> Dict:
        """Retorna status dos provedores configurados."""
        status = {}
        
        for provider, config in self.providers_config.items():
            status[provider.value] = {
                'enabled': config.enabled,
                'model': config.model_name,
                'is_active': provider == self.active_provider,
                'has_api_key': bool(config.api_key)
            }
        
        return status
    
    def switch_provider(self, new_provider: LLMProvider) -> bool:
        """
        Troca o provedor ativo.
        
        Args:
            new_provider: Novo provedor a ser usado
            
        Returns:
            bool: True se a troca foi bem-sucedida
        """
        try:
            if new_provider in self.providers_config and self.providers_config[new_provider].enabled:
                old_provider = self.active_provider
                self.active_provider = new_provider
                
                # Atualiza fallbacks
                self.fallback_providers = [p for p in self.providers_config.keys() if p != new_provider]
                
                self.logger.info(f"Provedor alterado: {old_provider.value if old_provider else 'None'} -> {new_provider.value}")
                return True
            else:
                self.logger.error(f"Provedor {new_provider.value} não está disponível")
                return False
                
        except Exception as e:
            self.logger.error(f"Erro ao trocar provedor: {e}")
            return False
    
    def clear_cache(self):
        """Limpa cache de respostas."""
        self.response_cache.clear()
        self.logger.info("Cache de respostas LLM limpo")
    
    def clear_history(self):
        """Limpa histórico de respostas."""
        self.response_history.clear()
        self.logger.info("Histórico de respostas LLM limpo")