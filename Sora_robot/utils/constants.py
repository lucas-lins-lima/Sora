# sora_robot/utils/constants.py

"""
Constantes globais utilizadas em todo o projeto Sora.
Define configurações, limiares e mapeamentos utilizados pelos diferentes módulos.
"""

# =============================================================================
# CONFIGURAÇÕES DA CÂMERA
# =============================================================================

class CAMERA_SETTINGS:
    """Configurações relacionadas ao processamento de vídeo/câmera."""
    
    # Resolução padrão da câmera
    DEFAULT_RESOLUTION = (1280, 720)  # HD
    FALLBACK_RESOLUTION = (640, 480)  # VGA como fallback
    
    # Taxa de frames
    TARGET_FPS = 30
    ENABLE_FPS_LIMIT = True
    
    # Processamento de imagem
    AUTO_BRIGHTNESS = True
    AUTO_CONTRAST = True
    MIRROR_MODE = True  # Efeito espelho para interação natural
    
    # Buffer e performance
    FRAME_BUFFER_SIZE = 1
    MAX_FRAME_DROP_RATE = 0.1  # 10% máximo de frames perdidos

# =============================================================================
# CONFIGURAÇÕES DE ÁUDIO
# =============================================================================

class AUDIO_SETTINGS:
    """Configurações relacionadas ao processamento de áudio."""
    
    # Parâmetros de gravação
    SAMPLE_RATE = 44100
    CHANNELS = 1  # Mono
    CHUNK_SIZE = 1024
    FORMAT = 'int16'
    
    # Detecção de fala
    SILENCE_THRESHOLD = 500
    SILENCE_DURATION = 2.0  # segundos
    
    # Processamento
    NOISE_REDUCTION = True
    AUTO_GAIN = True

# =============================================================================
# EMOÇÕES E SENTIMENTOS
# =============================================================================

class EMOTIONS:
    """Definições de emoções reconhecidas pelo sistema."""
    
    # Emoções básicas (Ekman)
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    SURPRISED = "surprised"
    FEAR = "fear"
    DISGUST = "disgust"
    
    # Lista de todas as emoções
    ALL_EMOTIONS = [NEUTRAL, HAPPY, SAD, ANGRY, SURPRISED, FEAR, DISGUST]
    
    # Mapeamento de emoções para valores numéricos
    EMOTION_TO_VALUE = {
        NEUTRAL: 0,
        HAPPY: 1,
        SAD: 2,
        ANGRY: 3,
        SURPRISED: 4,
        FEAR: 5,
        DISGUST: 6
    }
    
    # Mapeamento inverso
    VALUE_TO_EMOTION = {v: k for k, v in EMOTION_TO_VALUE.items()}
    
    # Grupos de emoções
    POSITIVE_EMOTIONS = [HAPPY, SURPRISED]
    NEGATIVE_EMOTIONS = [SAD, ANGRY, FEAR, DISGUST]
    NEUTRAL_EMOTIONS = [NEUTRAL]

class SENTIMENTS:
    """Definições de sentimentos para análise de texto."""
    
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    
    ALL_SENTIMENTS = [POSITIVE, NEGATIVE, NEUTRAL]
    
    # Limiares para classificação
    POSITIVE_THRESHOLD = 0.1
    NEGATIVE_THRESHOLD = -0.1

# =============================================================================
# INTENÇÕES DE DIÁLOGO
# =============================================================================

class INTENTS:
    """Definições de intenções identificadas na fala do usuário."""
    
    # Intenções básicas
    GREETING = "greeting"
    QUESTION = "question"
    REQUEST = "request"
    COMPLAINT = "complaint"
    COMPLIMENT = "compliment"
    GOODBYE = "goodbye"
    
    # Intenções específicas do evento
    EVENT_INFO = "event_info"
    LOCATION_REQUEST = "location_request"
    SCHEDULE_REQUEST = "schedule_request"
    CONTACT_REQUEST = "contact_request"
    
    # Intenções de controle
    REPEAT = "repeat"
    CLARIFICATION = "clarification"
    HELP = "help"
    
    ALL_INTENTS = [
        GREETING, QUESTION, REQUEST, COMPLAINT, COMPLIMENT, GOODBYE,
        EVENT_INFO, LOCATION_REQUEST, SCHEDULE_REQUEST, CONTACT_REQUEST,
        REPEAT, CLARIFICATION, HELP
    ]

# =============================================================================
# CONFIGURAÇÕES DE POSE CORPORAL
# =============================================================================

class BODY_POSE:
    """Configurações para estimativa de pose corporal."""
    
    # Confiança mínima para detecção de keypoints
    MIN_DETECTION_CONFIDENCE = 0.5
    MIN_TRACKING_CONFIDENCE = 0.5
    
    # Keypoints principais do MediaPipe
    KEYPOINTS = {
        'NOSE': 0,
        'LEFT_EYE_INNER': 1,
        'LEFT_EYE': 2,
        'LEFT_EYE_OUTER': 3,
        'RIGHT_EYE_INNER': 4,
        'RIGHT_EYE': 5,
        'RIGHT_EYE_OUTER': 6,
        'LEFT_EAR': 7,
        'RIGHT_EAR': 8,
        'MOUTH_LEFT': 9,
        'MOUTH_RIGHT': 10,
        'LEFT_SHOULDER': 11,
        'RIGHT_SHOULDER': 12,
        'LEFT_ELBOW': 13,
        'RIGHT_ELBOW': 14,
        'LEFT_WRIST': 15,
        'RIGHT_WRIST': 16,
        'LEFT_PINKY': 17,
        'RIGHT_PINKY': 18,
        'LEFT_INDEX': 19,
        'RIGHT_INDEX': 20,
        'LEFT_THUMB': 21,
        'RIGHT_THUMB': 22,
        'LEFT_HIP': 23,
        'RIGHT_HIP': 24,
        'LEFT_KNEE': 25,
        'RIGHT_KNEE': 26,
        'LEFT_ANKLE': 27,
        'RIGHT_ANKLE': 28,
        'LEFT_HEEL': 29,
        'RIGHT_HEEL': 30,
        'LEFT_FOOT_INDEX': 31,
        'RIGHT_FOOT_INDEX': 32
    }
    
    # Gestos detectáveis
    GESTURES = {
        'WAVE': 'wave',
        'POINTING': 'pointing',
        'THUMBS_UP': 'thumbs_up',
        'OPEN_ARMS': 'open_arms',
        'CROSSED_ARMS': 'crossed_arms',
        'HAND_ON_HIP': 'hand_on_hip'
    }

# =============================================================================
# CONFIGURAÇÕES DE FACE
# =============================================================================

class FACE_DETECTION:
    """Configurações para detecção e reconhecimento facial."""
    
    # Confiança mínima para detecção
    MIN_DETECTION_CONFIDENCE = 0.5
    MIN_TRACKING_CONFIDENCE = 0.5
    
    # Tamanho mínimo de face para processar (largura x altura em pixels)
    MIN_FACE_SIZE = (100, 100)
    
    # Máximo número de faces para processar simultaneamente
    MAX_FACES = 5
    
    # Landmarks faciais importantes
    FACIAL_LANDMARKS = {
        'LEFT_EYE': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
        'RIGHT_EYE': [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
        'MOUTH': [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 78],
        'NOSE': [1, 2, 5, 4, 6, 168, 8, 9, 10, 151, 195, 197, 196, 3, 51, 48, 115, 131, 134, 102, 49, 220, 305, 279, 331, 294, 303, 332, 284, 298],
        'EYEBROWS': [70, 63, 105, 66, 107, 55, 65, 52, 53, 46, 296, 334, 293, 300, 276, 283, 282, 295, 285, 336, 296, 334]
    }

# =============================================================================
# CONFIGURAÇÕES DO AVATAR
# =============================================================================

class AVATAR_SETTINGS:
    """Configurações para animação do avatar."""
    
    # Mapeamento de emoções para arquivos de animação
    EMOTION_TO_ANIMATION = {
        EMOTIONS.NEUTRAL: "neutral.mp4",
        EMOTIONS.HAPPY: "happy.mp4",
        EMOTIONS.SAD: "sad.mp4",
        EMOTIONS.ANGRY: "angry.mp4",
        EMOTIONS.SURPRISED: "surprised.mp4",
        EMOTIONS.FEAR: "fear.mp4",
        EMOTIONS.DISGUST: "disgust.mp4"
    }
    
    # Animações para diferentes estados
    STATE_ANIMATIONS = {
        'SPEAKING': "speaking_loop.mp4",
        'LISTENING': "attentive_pose.mp4",
        'THINKING': "thinking.mp4",
        'GREETING': "wave_gesture.mp4",
        'IDLE': "neutral.mp4"
    }
    
    # Configurações de reprodução
    DEFAULT_FPS = 30
    LOOP_ANIMATIONS = True
    SMOOTH_TRANSITIONS = True

# =============================================================================
# CONFIGURAÇÕES DE LOG
# =============================================================================

class LOG_SETTINGS:
    """Configurações para sistema de logs."""
    
    # Níveis de log
    LEVELS = {
        'DEBUG': 10,
        'INFO': 20,
        'WARNING': 30,
        'ERROR': 40,
        'CRITICAL': 50
    }
    
    # Formato padrão
    FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    
    # Arquivo de log
    LOG_FILE = "sora_robot.log"
    MAX_LOG_SIZE = 10 * 1024 * 1024  # 10MB
    BACKUP_COUNT = 5

# =============================================================================
# CONFIGURAÇÕES DE PERFORMANCE
# =============================================================================

class PERFORMANCE:
    """Configurações relacionadas à performance do sistema."""
    
    # Limiares de performance
    MAX_PROCESSING_TIME = {
        'FRAME_PROCESSING': 0.033,  # ~30 FPS
        'AUDIO_PROCESSING': 0.1,    # 100ms
        'LLM_RESPONSE': 5.0,        # 5 segundos
        'EMOTION_ANALYSIS': 0.05    # 50ms
    }
    
    # Configurações de cache
    ENABLE_CACHING = True
    CACHE_SIZE = 1000
    CACHE_TTL = 300  # 5 minutos
    
    # Threading
    MAX_WORKER_THREADS = 4
    THREAD_POOL_SIZE = 8

# =============================================================================
# CONFIGURAÇÕES DA API
# =============================================================================

class API_SETTINGS:
    """Configurações para a API web."""
    
    # Endpoints
    ENDPOINTS = {
        'HEALTH': '/health',
        'CAMERA_STREAM': '/camera/stream',
        'CHAT': '/chat',
        'EMOTION_STATUS': '/emotion/status',
        'METRICS': '/metrics'
    }
    
    # WebSocket
    WEBSOCKET_SETTINGS = {
        'PING_INTERVAL': 30,
        'PING_TIMEOUT': 10,
        'MAX_MESSAGE_SIZE': 1024 * 1024,  # 1MB
        'COMPRESSION': True
    }
    
    # CORS
    CORS_ORIGINS = ['http://localhost:3000', 'http://127.0.0.1:5000']

# =============================================================================
# MENSAGENS PADRÃO
# =============================================================================

class DEFAULT_MESSAGES:
    """Mensagens padrão utilizadas pelo sistema."""
    
    GREETING_MESSAGES = [
        "Olá! Eu sou a Sora, como posso ajudá-lo hoje?",
        "Bem-vindo! Sou a Sora, sua assistente virtual. Em que posso ajudar?",
        "Oi! Prazer em conhecê-lo! Eu sou a Sora."
    ]
    
    ERROR_MESSAGES = {
        'CAMERA_ERROR': "Desculpe, estou com problemas na câmera no momento.",
        'AUDIO_ERROR': "Não consegui processar o áudio. Pode repetir?",
        'PROCESSING_ERROR': "Desculpe, estou processando sua solicitação. Um momento...",
        'UNKNOWN_ERROR': "Ops! Algo deu errado. Pode tentar novamente?"
    }
    
    FALLBACK_RESPONSES = [
        "Interessante! Conte-me mais sobre isso.",
        "Entendo. Como posso ajudá-lo com isso?",
        "Hmm, deixe-me pensar sobre isso...",
        "Pode me dar mais detalhes sobre sua pergunta?"
    ]

# =============================================================================
# CONFIGURAÇÕES DE DADOS
# =============================================================================

class DATA_SETTINGS:
    """Configurações para persistência de dados."""
    
    # Formatos de arquivo
    SUPPORTED_FORMATS = {
        'PROFILES': '.json',
        'LOGS': '.jsonl',
        'CACHE': '.pkl'
    }
    
    # Limites de armazenamento
    MAX_INTERACTION_LOGS = 10000
    MAX_USER_PROFILES = 1000
    DATA_RETENTION_DAYS = 30
    
    # Backup
    ENABLE_BACKUP = True
    BACKUP_INTERVAL = 24 * 3600  # 24 horas em segundos

# sora_robot/utils/constants.py (adicionando seção de áudio)

# =============================================================================
# CONFIGURAÇÕES DE ÁUDIO (adicionando à seção existente)
# =============================================================================

class AUDIO_SETTINGS:
    """Configurações relacionadas ao processamento de áudio."""
    
    # Parâmetros de gravação
    SAMPLE_RATE = 44100  # Hz
    CHANNELS = 1  # Mono
    CHUNK_SIZE = 1024  # Samples per chunk
    FORMAT = 'int16'  # 16-bit PCM
    
    # Detecção de fala
    SILENCE_THRESHOLD = 500  # Amplitude mínima para considerar som
    SILENCE_DURATION = 2.0  # Segundos de silêncio para finalizar fala
    
    # Processamento
    NOISE_REDUCTION = True
    AUTO_GAIN = True
    VOLUME_NORMALIZATION = True
    
    # VAD (Voice Activity Detection)
    VAD_SENSITIVITY = 0.5  # 0.0 a 1.0 (menor = mais sensível)
    MIN_SPEECH_DURATION = 0.3  # Mínimo de segundos para considerar fala
    MAX_SPEECH_DURATION = 30.0  # Máximo de segundos de fala contínua
    
    # Qualidade de áudio
    MIN_SIGNAL_QUALITY = 0.3  # Qualidade mínima para processar
    TARGET_VOLUME_LEVEL = 0.7  # Nível alvo para normalização
    
    # Buffer settings
    AUDIO_BUFFER_SECONDS = 10  # Segundos de áudio em buffer
    SPEECH_BUFFER_SECONDS = 30  # Buffer para gravação de fala

# =============================================================================
# CONFIGURAÇÕES DE RECONHECIMENTO DE FALA  
# =============================================================================

class SPEECH_RECOGNITION:
    """Configurações para reconhecimento de fala."""
    
    # Idiomas suportados
    SUPPORTED_LANGUAGES = ['pt-BR', 'en-US', 'es-ES']
    DEFAULT_LANGUAGE = 'pt-BR'
    
    # Configurações do Google Speech-to-Text
    GOOGLE_STT_CONFIG = {
        'encoding': 'LINEAR16',
        'sample_rate_hertz': 44100,
        'language_code': 'pt-BR',
        'alternative_language_codes': ['en-US'],
        'max_alternatives': 3,
        'profanity_filter': True,
        'enable_word_time_offsets': True,
        'enable_word_confidence': True,
        'enable_automatic_punctuation': True,
        'enable_spoken_punctuation': False,
        'enable_spoken_emojis': False
    }
    
    # Configurações de streaming
    STREAMING_CONFIG = {
        'interim_results': True,
        'single_utterance': False,
        'max_streaming_duration': 305  # segundos (Google limit)
    }
    
    # Thresholds
    CONFIDENCE_THRESHOLD = 0.7  # Confiança mínima para aceitar transcrição
    MIN_SPEECH_LENGTH = 0.5  # Mínimo em segundos
    MAX_SPEECH_LENGTH = 60.0  # Máximo em segundos

from enum import Enum
from typing import Dict, List, Set, Any

# =============================================================================
# INTENÇÕES (INTENTS) - Usadas pelo sistema de reconhecimento de intenção
# =============================================================================

class INTENTS:
    """Constantes para intenções do usuário."""
    
    # Intenções sociais
    GREETING = "greeting"
    GOODBYE = "goodbye"
    COMPLIMENT = "compliment"
    COMPLAINT = "complaint"
    
    # Intenções informacionais
    QUESTION = "question"
    EVENT_INFO = "event_info"
    LOCATION_REQUEST = "location_request"
    TIME_REQUEST = "time_request"
    SCHEDULE_REQUEST = "schedule_request"
    
    # Intenções de ação
    REQUEST = "request"
    HELP = "help"
    REPEAT = "repeat"
    CLARIFICATION = "clarification"
    
    # Intenções específicas do evento
    SPEAKER_INFO = "speaker_info"
    WORKSHOP_INFO = "workshop_info"
    REGISTRATION = "registration"
    NETWORKING = "networking"
    
    # Intenções técnicas
    DEMO_REQUEST = "demo_request"
    FEEDBACK = "feedback"
    SUGGESTION = "suggestion"
    
    # Meta-intenções
    UNKNOWN = "unknown"
    AMBIGUOUS = "ambiguous"
    
    # Lista de todas as intenções
    ALL_INTENTS = [
        GREETING, GOODBYE, COMPLIMENT, COMPLAINT,
        QUESTION, EVENT_INFO, LOCATION_REQUEST, TIME_REQUEST, SCHEDULE_REQUEST,
        REQUEST, HELP, REPEAT, CLARIFICATION,
        SPEAKER_INFO, WORKSHOP_INFO, REGISTRATION, NETWORKING,
        DEMO_REQUEST, FEEDBACK, SUGGESTION,
        UNKNOWN, AMBIGUOUS
    ]
    
    # Intenções por categoria
    SOCIAL_INTENTS = [GREETING, GOODBYE, COMPLIMENT, COMPLAINT]
    INFORMATIONAL_INTENTS = [QUESTION, EVENT_INFO, LOCATION_REQUEST, TIME_REQUEST, SCHEDULE_REQUEST]
    ACTION_INTENTS = [REQUEST, HELP, REPEAT, CLARIFICATION]
    EVENT_INTENTS = [SPEAKER_INFO, WORKSHOP_INFO, REGISTRATION, NETWORKING]

# =============================================================================
# SENTIMENTOS (SENTIMENTS) - Usadas pelo sistema de análise de sentimento
# =============================================================================

class SENTIMENTS:
    """Constantes para sentimentos detectados no texto."""
    
    # Sentimentos básicos
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    
    # Sentimentos específicos
    VERY_POSITIVE = "very_positive"
    SLIGHTLY_POSITIVE = "slightly_positive"
    SLIGHTLY_NEGATIVE = "slightly_negative"
    VERY_NEGATIVE = "very_negative"
    
    # Sentimentos complexos
    MIXED = "mixed"
    UNCERTAIN = "uncertain"
    
    # Lista de todos os sentimentos
    ALL_SENTIMENTS = [
        POSITIVE, NEGATIVE, NEUTRAL,
        VERY_POSITIVE, SLIGHTLY_POSITIVE, SLIGHTLY_NEGATIVE, VERY_NEGATIVE,
        MIXED, UNCERTAIN
    ]
    
    # Agrupamentos
    POSITIVE_SENTIMENTS = [POSITIVE, VERY_POSITIVE, SLIGHTLY_POSITIVE]
    NEGATIVE_SENTIMENTS = [NEGATIVE, VERY_NEGATIVE, SLIGHTLY_NEGATIVE]
    NEUTRAL_SENTIMENTS = [NEUTRAL, MIXED, UNCERTAIN]

# =============================================================================
# EMOÇÕES (EMOTIONS) - Usadas pelo sistema de análise de emoções faciais
# =============================================================================

class EMOTIONS:
    """Constantes para emoções detectadas em faces."""
    
    # Emoções básicas (Paul Ekman)
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    NEUTRAL = "neutral"
    
    # Emoções estendidas
    EXCITED = "excited"
    CALM = "calm"
    CONFUSED = "confused"
    FRUSTRATED = "frustrated"
    CONFIDENT = "confident"
    ANXIOUS = "anxious"
    CONTEMPT = "contempt"
    
    # Estados específicos
    THINKING = "thinking"
    LISTENING = "listening"
    SPEAKING = "speaking"
    CONCENTRATING = "concentrating"
    
    # Lista de todas as emoções
    ALL_EMOTIONS = [
        HAPPY, SAD, ANGRY, FEAR, SURPRISE, DISGUST, NEUTRAL,
        EXCITED, CALM, CONFUSED, FRUSTRATED, CONFIDENT, ANXIOUS, CONTEMPT,
        THINKING, LISTENING, SPEAKING, CONCENTRATING
    ]
    
    # Emoções por valência
    POSITIVE_EMOTIONS = [HAPPY, EXCITED, CALM, CONFIDENT]
    NEGATIVE_EMOTIONS = [SAD, ANGRY, FEAR, FRUSTRATED, ANXIOUS, DISGUST, CONTEMPT]
    NEUTRAL_EMOTIONS = [NEUTRAL, SURPRISE, CONFUSED, THINKING, LISTENING, SPEAKING, CONCENTRATING]

# =============================================================================
# ESTADOS DO SISTEMA - Usados pelos módulos de controle
# =============================================================================

class SYSTEM_STATES:
    """Estados possíveis do sistema."""
    
    INITIALIZING = "initializing"
    READY = "ready"
    ACTIVE = "active"
    LISTENING = "listening"
    PROCESSING = "processing"
    RESPONDING = "responding"
    PAUSED = "paused"
    ERROR = "error"
    SHUTDOWN = "shutdown"
    
    ALL_STATES = [
        INITIALIZING, READY, ACTIVE, LISTENING,
        PROCESSING, RESPONDING, PAUSED, ERROR, SHUTDOWN
    ]

# =============================================================================
# CONFIGURAÇÕES DE ÁUDIO - Usadas pelos módulos de processamento de áudio
# =============================================================================

class AUDIO_SETTINGS:
    """Configurações padrão para processamento de áudio."""
    
    # Formatos suportados
    SUPPORTED_FORMATS = ["wav", "mp3", "flac", "ogg", "m4a"]
    
    # Taxas de amostragem padrão
    SAMPLE_RATES = {
        "low": 16000,
        "medium": 24000,
        "high": 44100,
        "ultra": 48000
    }
    
    # Configurações de qualidade
    QUALITY_SETTINGS = {
        "low": {"sample_rate": 16000, "channels": 1, "bit_depth": 16},
        "medium": {"sample_rate": 24000, "channels": 1, "bit_depth": 16},
        "high": {"sample_rate": 44100, "channels": 2, "bit_depth": 24},
        "ultra": {"sample_rate": 48000, "channels": 2, "bit_depth": 32}
    }
    
    # Configurações de VAD (Voice Activity Detection)
    VAD_SETTINGS = {
        "silence_threshold": 0.01,
        "silence_duration": 2.0,
        "min_speech_duration": 0.5,
        "max_speech_duration": 30.0
    }

# =============================================================================
# CONFIGURAÇÕES DE VISÃO - Usadas pelos módulos de processamento de visão
# =============================================================================

class VISION_SETTINGS:
    """Configurações para processamento de visão."""
    
    # Resoluções suportadas
    RESOLUTIONS = {
        "low": (640, 480),
        "medium": (1280, 720),
        "high": (1920, 1080),
        "ultra": (3840, 2160)
    }
    
    # Taxas de quadros
    FPS_OPTIONS = [15, 24, 30, 60]
    
    # Configurações de detecção
    DETECTION_SETTINGS = {
        "face_min_size": (30, 30),
        "face_max_size": (300, 300),
        "confidence_threshold": 0.7,
        "overlap_threshold": 0.3
    }
    
    # Cores para visualização (BGR)
    COLORS = {
        "red": (0, 0, 255),
        "green": (0, 255, 0),
        "blue": (255, 0, 0),
        "yellow": (0, 255, 255),
        "purple": (255, 0, 255),
        "cyan": (255, 255, 0),
        "white": (255, 255, 255),
        "black": (0, 0, 0)
    }

# =============================================================================
# MENSAGENS PADRÃO - Usadas para respostas fallback
# =============================================================================

class DEFAULT_MESSAGES:
    """Mensagens padrão do sistema."""
    
    # Mensagens de saudação
    GREETING_MESSAGES = [
        "Olá! Eu sou a Sora. Como posso ajudá-lo hoje?",
        "Oi! Prazer em conhecê-lo! Em que posso ser útil?",
        "Bem-vindo! Sou a Sora, sua assistente virtual. O que você gostaria de saber?",
        "Olá! É um prazer estar aqui com você. Como posso ajudar?"
    ]
    
    # Mensagens de despedida
    GOODBYE_MESSAGES = [
        "Foi um prazer conversar com você! Até logo!",
        "Obrigada pela conversa! Tenha um ótimo dia!",
        "Até mais! Estarei aqui sempre que precisar.",
        "Tchau! Espero ter ajudado. Volte sempre!"
    ]
    
    # Mensagens de fallback
    FALLBACK_RESPONSES = [
        "Desculpe, não entendi completamente. Pode reformular sua pergunta?",
        "Hmm, não tenho certeza sobre isso. Pode me dar mais detalhes?",
        "Interessante! Pode me explicar melhor o que você precisa?",
        "Não compreendi totalmente. Vamos tentar de novo?"
    ]
    
    # Mensagens de erro
    ERROR_MESSAGES = [
        "Desculpe, houve um problema técnico. Pode tentar novamente?",
        "Ops! Algo deu errado. Vamos tentar mais uma vez?",
        "Encontrei uma dificuldade técnica. Por favor, tente novamente.",
        "Houve um erro temporário. Pode repetir sua solicitação?"
    ]
    
    # Mensagens de processamento
    PROCESSING_MESSAGES = [
        "Deixe-me pensar...",
        "Processando...",
        "Um momento, por favor...",
        "Analisando..."
    ]
    
    # Mensagens de clarificação
    CLARIFICATION_MESSAGES = [
        "Para te ajudar melhor, você está perguntando sobre:",
        "Só para confirmar, você quer saber sobre:",
        "Posso esclarecer melhor se você me disser:",
        "Para dar uma resposta mais precisa, você se refere a:"
    ]

# =============================================================================
# CÓDIGOS DE ERRO - Usados para tratamento de erros
# =============================================================================

class ERROR_CODES:
    """Códigos de erro padronizados."""
    
    # Erros de sistema
    SYSTEM_INIT_ERROR = "SYS_001"
    SYSTEM_SHUTDOWN_ERROR = "SYS_002"
    MEMORY_ERROR = "SYS_003"
    THREAD_ERROR = "SYS_004"
    
    # Erros de dispositivos
    CAMERA_ERROR = "DEV_001"
    MICROPHONE_ERROR = "DEV_002"
    SPEAKER_ERROR = "DEV_003"
    GPU_ERROR = "DEV_004"
    
    # Erros de processamento
    VISION_PROCESSING_ERROR = "PROC_001"
    AUDIO_PROCESSING_ERROR = "PROC_002"
    NLP_PROCESSING_ERROR = "PROC_003"
    ANIMATION_ERROR = "PROC_004"
    
    # Erros de rede
    API_CONNECTION_ERROR = "NET_001"
    API_AUTH_ERROR = "NET_002"
    API_TIMEOUT_ERROR = "NET_003"
    API_RATE_LIMIT_ERROR = "NET_004"
    
    # Erros de dados
    DATA_VALIDATION_ERROR = "DATA_001"
    FILE_NOT_FOUND_ERROR = "DATA_002"
    PARSING_ERROR = "DATA_003"
    ENCODING_ERROR = "DATA_004"
    
    # Mapeamento de códigos para mensagens
    ERROR_MESSAGES = {
        SYSTEM_INIT_ERROR: "Falha na inicialização do sistema",
        SYSTEM_SHUTDOWN_ERROR: "Erro durante o shutdown",
        MEMORY_ERROR: "Erro de memória insuficiente",
        THREAD_ERROR: "Erro no gerenciamento de threads",
        
        CAMERA_ERROR: "Erro de acesso à câmera",
        MICROPHONE_ERROR: "Erro de acesso ao microfone",
        SPEAKER_ERROR: "Erro de acesso ao alto-falante",
        GPU_ERROR: "Erro de acesso à GPU",
        
        VISION_PROCESSING_ERROR: "Erro no processamento de visão",
        AUDIO_PROCESSING_ERROR: "Erro no processamento de áudio",
        NLP_PROCESSING_ERROR: "Erro no processamento de linguagem",
        ANIMATION_ERROR: "Erro na geração de animação",
        
        API_CONNECTION_ERROR: "Erro de conexão com API",
        API_AUTH_ERROR: "Erro de autenticação na API",
        API_TIMEOUT_ERROR: "Timeout na requisição da API",
        API_RATE_LIMIT_ERROR: "Limite de taxa da API excedido",
        
        DATA_VALIDATION_ERROR: "Erro de validação de dados",
        FILE_NOT_FOUND_ERROR: "Arquivo não encontrado",
        PARSING_ERROR: "Erro ao analisar dados",
        ENCODING_ERROR: "Erro de codificação"
    }

# =============================================================================
# CONFIGURAÇÕES DE PERFORMANCE - Usadas para otimização
# =============================================================================

class PERFORMANCE:
    """Constantes para configurações de performance."""
    
    # Limites de recursos
    MAX_MEMORY_MB = 2048
    MAX_CPU_PERCENT = 80
    MAX_THREADS = 16
    MAX_QUEUE_SIZE = 100
    
    # Timeouts (em segundos)
    TIMEOUTS = {
        "camera_init": 5.0,
        "microphone_init": 3.0,
        "speech_recognition": 10.0,
        "llm_generation": 30.0,
        "tts_synthesis": 15.0,
        "api_request": 20.0,
        "file_operation": 5.0
    }
    
    # Intervalos de processamento (em segundos)
    INTERVALS = {
        "face_detection": 0.1,  # 10 FPS
        "emotion_analysis": 0.2,  # 5 FPS
        "pose_estimation": 0.1,  # 10 FPS
        "audio_analysis": 0.05,  # 20 FPS
        "system_monitoring": 1.0,  # 1 Hz
        "cache_cleanup": 60.0  # 1 minuto
    }
    
    # Tamanhos de buffer
    BUFFER_SIZES = {
        "audio_chunk": 1024,
        "video_frame": 3,
        "processing_queue": 50,
        "response_cache": 100
    }

# =============================================================================
# CONFIGURAÇÕES DE REDE - Usadas para comunicação
# =============================================================================

class NETWORK:
    """Constantes para configurações de rede."""
    
    # Portas padrão
    DEFAULT_PORTS = {
        "api": 8000,
        "websocket": 8001,
        "streaming": 8002,
        "debug": 8003
    }
    
    # Headers HTTP
    DEFAULT_HEADERS = {
        "Content-Type": "application/json",
        "User-Agent": "Sora-Robot/1.0",
        "Accept": "application/json"
    }
    
    # Status codes específicos
    STATUS_CODES = {
        "PROCESSING": 202,
        "RATE_LIMITED": 429,
        "AI_ERROR": 500,
        "DEVICE_ERROR": 503
    }

# =============================================================================
# FORMATAÇÃO E LOCALIZAÇÃO - Usadas para apresentação
# =============================================================================

class FORMATTING:
    """Constantes para formatação de dados."""
    
    # Formatos de data e hora
    DATE_FORMATS = {
        "short": "%d/%m/%Y",
        "long": "%d de %B de %Y",
        "time": "%H:%M:%S",
        "datetime": "%d/%m/%Y %H:%M:%S",
        "iso": "%Y-%m-%dT%H:%M:%S"
    }
    
    # Formatos de números
    NUMBER_FORMATS = {
        "percentage": "{:.1%}",
        "decimal": "{:.2f}",
        "integer": "{:,}",
        "currency": "R$ {:.2f}"
    }
    
    # Idiomas suportados
    SUPPORTED_LANGUAGES = {
        "pt-BR": "Português (Brasil)",
        "en-US": "English (United States)",
        "es-ES": "Español (España)"
    }

# =============================================================================
# EXPRESSÕES REGULARES - Usadas para validação e parsing
# =============================================================================

class REGEX_PATTERNS:
    """Padrões de expressões regulares comuns."""
    
    # Validação de dados
    EMAIL = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    PHONE = r'^\+?1?\d{9,15}$'
    URL = r'https?://(?:[-\w.])+(?:\:[0-9]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:\#(?:[\w.])*)?)?'
    
    # Extração de informações
    TIME_24H = r'\b([01]?[0-9]|2[0-3]):([0-5][0-9])\b'
    TIME_12H = r'\b(1[0-2]|[1-9]):([0-5][0-9])\s?(AM|PM|am|pm)\b'
    DATE_BR = r'\b(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{2,4})\b'
    
    # Limpeza de texto
    REMOVE_SPECIAL_CHARS = r'[^\w\s\.\!\?\,\;\:]'
    NORMALIZE_SPACES = r'\s+'
    
    # Detecção de elementos
    QUESTION_INDICATORS = r'\b(o que|que|qual|quando|onde|como|por que|porque|quem)\b'
    COMMAND_INDICATORS = r'\b(faça|execute|realize|crie|gere|mostre|abra|feche)\b'

# =============================================================================
# CONFIGURAÇÕES DE CACHE - Usadas para otimização de memória
# =============================================================================

class CACHE_SETTINGS:
    """Configurações para sistema de cache."""
    
    # TTL (Time To Live) em segundos
    TTL_SETTINGS = {
        "face_recognition": 30,
        "emotion_analysis": 10,
        "speech_recognition": 60,
        "sentiment_analysis": 180,
        "intent_recognition": 180,
        "llm_response": 300,
        "tts_audio": 600,
        "animation_sequence": 120
    }
    
    # Tamanhos máximos de cache
    MAX_CACHE_SIZES = {
        "vision_cache": 50,  # número de frames
        "audio_cache": 20,   # número de clips
        "text_cache": 100,   # número de análises
        "response_cache": 50, # número de respostas
        "file_cache": 100    # MB
    }
    
    # Estratégias de limpeza
    CLEANUP_STRATEGIES = {
        "lru": "Least Recently Used",
        "fifo": "First In, First Out",
        "ttl": "Time To Live",
        "size": "Size Based"
    }

# =============================================================================
# METADADOS DE QUALIDADE - Usados para avaliação
# =============================================================================

class QUALITY_METRICS:
    """Métricas e thresholds de qualidade."""
    
    # Thresholds de confiança
    CONFIDENCE_THRESHOLDS = {
        "very_low": 0.3,
        "low": 0.5,
        "medium": 0.7,
        "high": 0.8,
        "very_high": 0.9
    }
    
    # Métricas de performance
    PERFORMANCE_TARGETS = {
        "response_time": 3.0,      # segundos
        "processing_accuracy": 0.85, # 85%
        "user_satisfaction": 0.8,   # 80%
        "system_uptime": 0.99       # 99%
    }
    
    # Indicadores de saúde
    HEALTH_INDICATORS = {
        "memory_usage": 80,         # % máximo
        "cpu_usage": 70,           # % máximo
        "error_rate": 5,           # % máximo
        "response_time": 5.0       # segundos máximo
    }

# =============================================================================
# CONFIGURAÇÕES DE EVENTOS - Específicas para o contexto
# =============================================================================

class EVENT_CONFIG:
    """Configurações específicas para eventos."""
    
    # Tipos de eventos suportados
    EVENT_TYPES = [
        "conference", "workshop", "presentation", 
        "networking", "demo", "exhibition"
    ]
    
    # Salas e localizações típicas
    COMMON_LOCATIONS = [
        "auditório principal", "sala de workshops", "área de networking",
        "estandes", "recepção", "cafeteria", "banheiros", "estacionamento"
    ]
    
    # Horários típicos
    TYPICAL_SCHEDULES = {
        "morning": "08:00-12:00",
        "afternoon": "13:00-17:00",
        "evening": "18:00-22:00"
    }

# =============================================================================
# UTILITÁRIOS DE VALIDAÇÃO
# =============================================================================

def validate_emotion(emotion: str) -> bool:
    """Valida se uma emoção é válida."""
    return emotion in EMOTIONS.ALL_EMOTIONS

def validate_intent(intent: str) -> bool:
    """Valida se uma intenção é válida."""
    return intent in INTENTS.ALL_INTENTS

def validate_sentiment(sentiment: str) -> bool:
    """Valida se um sentimento é válido."""
    return sentiment in SENTIMENTS.ALL_SENTIMENTS

def get_error_message(error_code: str) -> str:
    """Obtém mensagem de erro para um código."""
    return ERROR_CODES.ERROR_MESSAGES.get(error_code, "Erro desconhecido")

def get_confidence_level(confidence: float) -> str:
    """Converte valor de confiança para nível textual."""
    if confidence >= QUALITY_METRICS.CONFIDENCE_THRESHOLDS["very_high"]:
        return "very_high"
    elif confidence >= QUALITY_METRICS.CONFIDENCE_THRESHOLDS["high"]:
        return "high"
    elif confidence >= QUALITY_METRICS.CONFIDENCE_THRESHOLDS["medium"]:
        return "medium"
    elif confidence >= QUALITY_METRICS.CONFIDENCE_THRESHOLDS["low"]:
        return "low"
    else:
        return "very_low"

# =============================================================================
# EXPORTAÇÕES
# =============================================================================

__all__ = [
    # Classes principais
    "INTENTS", "SENTIMENTS", "EMOTIONS", "SYSTEM_STATES",
    "AUDIO_SETTINGS", "VISION_SETTINGS", "DEFAULT_MESSAGES",
    "ERROR_CODES", "PERFORMANCE", "NETWORK", "FORMATTING",
    "REGEX_PATTERNS", "CACHE_SETTINGS", "QUALITY_METRICS",
    "EVENT_CONFIG",
    
    # Funções utilitárias
    "validate_emotion", "validate_intent", "validate_sentiment",
    "get_error_message", "get_confidence_level"
]