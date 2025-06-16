# sora_robot/config.py

"""
Configurações centralizadas do sistema Sora Robot.
Este arquivo centraliza todas as configurações, chaves de API e parâmetros do sistema.
"""

import os
from pathlib import Path
from typing import Dict, List, Any, Optional

# =============================================================================
# CONFIGURAÇÕES GERAIS DO SISTEMA
# =============================================================================

# Informações do projeto
PROJECT_NAME = "Sora Robot"
VERSION = "1.0.0"
DESCRIPTION = "Assistente Virtual Inteligente com IA Multimodal"
AUTHOR = "Equipe Sora"

# Diretórios do projeto
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
LOGS_DIR = ROOT_DIR / "logs"
MODELS_DIR = ROOT_DIR / "models"
TEMP_DIR = ROOT_DIR / "temp"
CACHE_DIR = ROOT_DIR / "cache"
CONFIG_DIR = ROOT_DIR / "config"

# Criar diretórios se não existirem
for directory in [DATA_DIR, LOGS_DIR, MODELS_DIR, TEMP_DIR, CACHE_DIR, CONFIG_DIR]:
    directory.mkdir(exist_ok=True)

# =============================================================================
# CHAVES DE API E AUTENTICAÇÃO
# =============================================================================

API_KEYS = {
    # OpenAI (GPT)
    "openai": os.getenv("OPENAI_API_KEY", ""),
    
    # Google Cloud (Gemini, TTS, Vision)
    "google_cloud": os.getenv("GOOGLE_CLOUD_API_KEY", ""),
    "gemini": os.getenv("GOOGLE_AI_API_KEY", ""),
    
    # Anthropic (Claude)
    "anthropic": os.getenv("ANTHROPIC_API_KEY", ""),
    
    # Microsoft Azure (Speech, Cognitive Services)
    "azure_speech": os.getenv("AZURE_SPEECH_KEY", ""),
    "azure_region": os.getenv("AZURE_SPEECH_REGION", "eastus"),
    
    # AWS (Polly, Rekognition)
    "aws_access_key": os.getenv("AWS_ACCESS_KEY_ID", ""),
    "aws_secret_key": os.getenv("AWS_SECRET_ACCESS_KEY", ""),
    "aws_region": os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    
    # Outras APIs
    "elevenlabs": os.getenv("ELEVENLABS_API_KEY", ""),
    "deepgram": os.getenv("DEEPGRAM_API_KEY", ""),
    "assemblyai": os.getenv("ASSEMBLYAI_API_KEY", "")
}

# =============================================================================
# CONFIGURAÇÕES DE DISPOSITIVOS
# =============================================================================

# Configurações da câmera
CAMERA_CONFIG = {
    "device_id": 0,  # ID da câmera (0 = padrão)
    "width": 1280,
    "height": 720,
    "fps": 30,
    "buffer_size": 1,
    "auto_focus": True,
    "auto_exposure": True,
    
    # Configurações de processamento
    "face_detection_confidence": 0.7,
    "face_recognition_threshold": 0.6,
    "emotion_detection_frequency": 5,  # FPS para detecção de emoção
    "pose_estimation_frequency": 10,   # FPS para estimativa de pose
}

# Configurações do microfone
MICROPHONE_CONFIG = {
    "device_id": None,  # None = dispositivo padrão
    "sample_rate": 44100,
    "channels": 1,
    "chunk_size": 1024,
    "buffer_duration": 5.0,  # segundos
    
    # Configurações de processamento
    "noise_reduction": True,
    "automatic_gain_control": True,
    "voice_activity_detection": True,
    "silence_threshold": 0.01,
    "silence_duration": 2.0,  # segundos de silêncio para fim de fala
}

# Configurações de alto-falante
SPEAKER_CONFIG = {
    "device_id": None,  # None = dispositivo padrão
    "sample_rate": 24000,
    "channels": 1,
    "buffer_size": 1024,
    "volume": 0.8,  # 0.0 a 1.0
}

# =============================================================================
# CONFIGURAÇÕES DE PROCESSAMENTO
# =============================================================================

# Configurações de visão
VISION_CONFIG = {
    "face_detection_model": "mtcnn",  # "mtcnn", "opencv", "mediapipe"
    "emotion_model": "fer2013",       # "fer2013", "affectnet"
    "pose_model": "mediapipe",        # "mediapipe", "openpose"
    
    # Configurações de qualidade
    "image_quality": "medium",        # "low", "medium", "high"
    "processing_fps": 10,             # FPS para processamento
    "detection_confidence": 0.7,
    "tracking_enabled": True,
    "face_detection_model": "mtcnn",
    "processing_fps": 10,
    "detection_confidence": 0.7
    
    # Cache e otimização
    "enable_gpu": True,
    "cache_enabled": True,
    "cache_size": 100,  # MB
}

# Configurações de áudio
AUDIO_CONFIG = {
    "speech_recognition_engine": "whisper",  # "whisper", "google", "azure"
    "language": "pt-BR",
    "recognition_timeout": 10.0,
    
    # Configurações de qualidade
    "audio_quality": "high",         # "low", "medium", "high"
    "noise_suppression": True,
    "echo_cancellation": True,
    
    # Cache e otimização
    "cache_enabled": True,
    "cache_duration": 300,  # segundos
}

# Configurações de NLP
NLP_CONFIG = {
    "sentiment_method": "ensemble",    # "lexicon", "textblob", "transformers", "ensemble"
    "intent_method": "ensemble",       # "pattern", "keyword", "ml", "ensemble"
    "language": "pt",

    # Configurações de qualidade
    "processing_quality": "high",     # "low", "medium", "high"
    "confidence_threshold": 0.6,
    
    # Cache e otimização
    "cache_enabled": True,
    "cache_size": 1000,  # número de análises
}

# =============================================================================
# CONFIGURAÇÕES DE LLM
# =============================================================================

LLM_CONFIG = {
    "primary_provider": "openai",     # "openai", "google", "anthropic"
    "fallback_providers": ["google", "anthropic"],
    "primary_provider": "openai",
    "max_tokens": 500,
    "temperature": 0.7
    
    # Configurações de geração
    "max_tokens": 500,
    "temperature": 0.7,
    "top_p": 0.9,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    
    # Timeouts e retry
    "timeout": 15.0,
    "max_retries": 3,
    "retry_delay": 1.0,
    
    # Cache
    "cache_enabled": True,
    "cache_ttl": 300,  # segundos
}

# Modelos específicos por provedor
LLM_MODELS = {
    "openai": {
        "gpt-3.5-turbo": {"max_tokens": 4096, "cost_per_1k": 0.002},
        "gpt-4": {"max_tokens": 8192, "cost_per_1k": 0.03},
        "gpt-4-turbo": {"max_tokens": 128000, "cost_per_1k": 0.01}
    },
    "google": {
        "gemini-pro": {"max_tokens": 2048, "cost_per_1k": 0.0005},
        "gemini-pro-vision": {"max_tokens": 2048, "cost_per_1k": 0.0025}
    },
    "anthropic": {
        "claude-3-sonnet": {"max_tokens": 4096, "cost_per_1k": 0.015},
        "claude-3-opus": {"max_tokens": 4096, "cost_per_1k": 0.075}
    }
}

# =============================================================================
# CONFIGURAÇÕES DE SÍNTESE DE FALA
# =============================================================================

TTS_CONFIG = {
    "primary_provider": "google_cloud",  # "google_cloud", "azure", "aws", "elevenlabs"
    "fallback_providers": ["azure", "aws"],
    
    # Configurações de voz
    "default_voice": {
        "language": "pt-BR",
        "gender": "female",
        "name": "pt-BR-Wavenet-A",
        "speaking_rate": 1.1,
        "pitch": 2.0,
        "volume": 0.8
    },
    
    # Configurações de qualidade
    "audio_format": "wav",
    "sample_rate": 24000,
    "audio_quality": "high",
    
    # Cache e otimização
    "cache_enabled": True,
    "cache_size": 50,  # MB
    "cache_ttl": 600,  # segundos
}

# Vozes disponíveis por emoção
EMOTION_VOICES = {
    "happy": {
        "speaking_rate": 1.2,
        "pitch": 3.0,
        "volume": 0.9
    },
    "sad": {
        "speaking_rate": 0.8,
        "pitch": -2.0,
        "volume": 0.7
    },
    "excited": {
        "speaking_rate": 1.3,
        "pitch": 5.0,
        "volume": 1.0
    },
    "calm": {
        "speaking_rate": 0.9,
        "pitch": -1.0,
        "volume": 0.8
    },
    "empathetic": {
        "speaking_rate": 0.9,
        "pitch": -0.5,
        "volume": 0.8
    }
}

# =============================================================================
# CONFIGURAÇÕES DE ANIMAÇÃO
# =============================================================================

ANIMATION_CONFIG = {
    "renderer": "pygame",            # "pygame", "opencv", "none"
    "fps": 30,
    "resolution": (800, 600),
    
    # Configurações de qualidade
    "animation_quality": "medium",   # "low", "medium", "high"
    "lip_sync_precision": "high",    # "low", "medium", "high"
    "facial_expression_intensity": 0.7,
    "gesture_frequency": 0.6,
    
    # Características do avatar
    "avatar_style": "realistic",     # "realistic", "cartoon", "minimal"
    "idle_animations": True,
    "micro_expressions": True,
    "breathing_animation": True,
}

# =============================================================================
# CONFIGURAÇÕES DE PERSONALIDADE
# =============================================================================

PERSONALITY_PROFILES = {
    "friendly": {
        "characteristics": {
            "friendliness": 0.9,
            "helpfulness": 0.8,
            "formality": 0.4,
            "humor": 0.6,
            "empathy": 0.7,
            "patience": 0.8,
            "enthusiasm": 0.7
        },
        "response_style": {
            "verbosity": 0.7,
            "enthusiasm": 0.6,
            "directness": 0.6,
            "encouragement": 0.8
        }
        "friendly": {
        "characteristics": {
            "friendliness": 0.9,
            "helpfulness": 0.8,
            # ...
        }
    }
},
    
    "professional": {
        "characteristics": {
            "friendliness": 0.6,
            "helpfulness": 0.9,
            "formality": 0.8,
            "humor": 0.3,
            "empathy": 0.6,
            "patience": 0.9,
            "enthusiasm": 0.5
        },
        "response_style": {
            "verbosity": 0.8,
            "enthusiasm": 0.4,
            "directness": 0.8,
            "encouragement": 0.6
        }
    },
    
    "casual": {
        "characteristics": {
            "friendliness": 0.8,
            "helpfulness": 0.7,
            "formality": 0.2,
            "humor": 0.8,
            "empathy": 0.6,
            "patience": 0.7,
            "enthusiasm": 0.6
        },
        "response_style": {
            "verbosity": 0.5,
            "enthusiasm": 0.6,
            "directness": 0.5,
            "encouragement": 0.7
        }
    },
    
    "empathetic": {
        "characteristics": {
            "friendliness": 0.8,
            "helpfulness": 0.9,
            "formality": 0.5,
            "humor": 0.4,
            "empathy": 0.9,
            "patience": 0.9,
            "enthusiasm": 0.5
        },
        "response_style": {
            "verbosity": 0.8,
            "enthusiasm": 0.4,
            "directness": 0.5,
            "encouragement": 0.9
        }
    }
}

# =============================================================================
# CONFIGURAÇÕES DE REDE E API
# =============================================================================

API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "reload": False,
    "workers": 1,
    
    # Segurança
    "api_key_required": False,
    "api_key": "sora-api-key-2024",
    "cors_origins": ["*"],
    "rate_limiting": True,
    "max_requests_per_minute": 60,
    
    # Timeouts
    "request_timeout": 30.0,
    "response_timeout": 45.0,
    
    # WebSocket
    "websocket_enabled": True,
    "max_connections": 100,
    "heartbeat_interval": 30,
}

# =============================================================================
# CONFIGURAÇÕES DE PERFORMANCE
# =============================================================================

PERFORMANCE_CONFIG = {
    "max_concurrent_processes": 4,
    "thread_pool_size": 8,
    "max_queue_size": 50,
    
    # Cache global
    "global_cache_enabled": True,
    "cache_size_mb": 100,
    "cache_cleanup_interval": 300,  # segundos
    
    # Monitoramento
    "monitoring_enabled": True,
    "metrics_interval": 10,  # segundos
    "log_performance": True,
    
    # Otimizações
    "gpu_acceleration": True,
    "parallel_processing": True,
    "async_operations": True,
}

# =============================================================================
# CONFIGURAÇÕES DE LOGGING
# =============================================================================

LOGGING_CONFIG = {
    "level": "INFO",  # "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "date_format": "%Y-%m-%d %H:%M:%S",
    
    # Arquivos de log
    "file_logging": True,
    "log_rotation": True,
    "max_file_size": "10MB",
    "backup_count": 5,
    
    # Logs por módulo
    "module_logs": {
        "vision": "INFO",
        "audio": "INFO",
        "nlp": "INFO",
        "dialogue": "INFO",
        "response": "INFO",
        "api": "INFO",
        "system": "DEBUG"
    },
    
    # Filtros
    "exclude_patterns": [
        "heartbeat",
        "health_check"
    ]
}

# =============================================================================
# CONFIGURAÇÕES DE DADOS E MODELOS
# =============================================================================

MODEL_CONFIG = {
    "auto_download": True,
    "model_cache_dir": MODELS_DIR,
    "check_updates": True,
    
    # Modelos de visão
    "vision_models": {
        "face_detection": "mtcnn_weights.pkl",
        "emotion_recognition": "fer2013_weights.h5",
        "pose_estimation": "pose_model.pb"
    },
    
    # Modelos de áudio
    "audio_models": {
        "speech_recognition": "whisper-base.pt",
        "voice_activity": "vad_model.onnx",
        "noise_reduction": "rnnoise_model.bin"
    },
    
    # Modelos de NLP
    "nlp_models": {
        "sentiment": "sentiment_model.pkl",
        "intent": "intent_model.pkl",
        "embeddings": "embeddings.bin"
    }
}

# =============================================================================
# CONFIGURAÇÕES DE CONTEXTO DO EVENTO
# =============================================================================

EVENT_CONTEXT = {
    "name": "Evento Tecnológico 2024",
    "description": "Evento de inovação e tecnologia com palestras, workshops e networking",
    "location": "Centro de Convenções",
    "dates": "15-17 de Dezembro de 2024",
    
    # Informações específicas
    "schedule": {
        "day1": "Palestras principais e keynotes",
        "day2": "Workshops técnicos e demonstrações",
        "day3": "Networking e encerramento"
    },
    
    "locations": {
        "auditorio_principal": "Auditório A - Térreo",
        "salas_workshop": "Salas B1-B5 - Primeiro andar",
        "area_networking": "Pavilhão C - Térreo",
        "estandes": "Pavilhão D - Térreo"
    },
    
    "services": {
        "wifi": "EventoTech2024",
        "catering": "Pavilhão E",
        "parking": "Estacionamento Sul",
        "support": "Balcão de informações - Entrada principal"
    }
}

# =============================================================================
# CONFIGURAÇÕES DE DESENVOLVIMENTO
# =============================================================================

DEBUG_CONFIG = {
    "debug_mode": False,
    "verbose_logging": False,
    "save_interactions": False,
    "save_audio_files": False,
    "save_video_frames": False,
    
    # Desenvolvimento
    "mock_external_apis": False,
    "simulate_hardware": False,
    "test_mode": False,
    
    # Profiling
    "enable_profiling": False,
    "profile_output_dir": TEMP_DIR / "profiles"
}

# =============================================================================
# CONFIGURAÇÕES DE SEGURANÇA
# =============================================================================

SECURITY_CONFIG = {
    "privacy_mode": False,
    "data_retention_days": 30,
    "anonymize_data": True,
    
    # Criptografia
    "encrypt_stored_data": False,
    "encryption_key": os.getenv("SORA_ENCRYPTION_KEY", ""),
    
    # Validação
    "input_validation": True,
    "max_input_length": 1000,
    "blocked_patterns": [],
    
    # Auditoria
    "audit_logging": True,
    "audit_log_file": LOGS_DIR / "audit.log"
}

# =============================================================================
# CONFIGURAÇÕES ESPECÍFICAS POR AMBIENTE
# =============================================================================

ENVIRONMENT = os.getenv("SORA_ENVIRONMENT", "development")

if ENVIRONMENT == "production":
    # Configurações para produção
    LOGGING_CONFIG["level"] = "WARNING"
    DEBUG_CONFIG["debug_mode"] = False
    API_CONFIG["reload"] = False
    PERFORMANCE_CONFIG["monitoring_enabled"] = True
    
elif ENVIRONMENT == "development":
    # Configurações para desenvolvimento
    LOGGING_CONFIG["level"] = "DEBUG"
    DEBUG_CONFIG["debug_mode"] = True
    API_CONFIG["reload"] = True
    PERFORMANCE_CONFIG["monitoring_enabled"] = False

elif ENVIRONMENT == "testing":
    # Configurações para testes
    LOGGING_CONFIG["level"] = "ERROR"
    DEBUG_CONFIG["test_mode"] = True
    DEBUG_CONFIG["mock_external_apis"] = True
    PERFORMANCE_CONFIG["monitoring_enabled"] = False

# =============================================================================
# VALIDAÇÃO DE CONFIGURAÇÕES
# =============================================================================

def validate_config():
    """Valida as configurações do sistema."""
    errors = []
    warnings = []
    
    # Verifica chaves de API obrigatórias
    required_apis = ["openai", "google_cloud"]
    for api in required_apis:
        if not API_KEYS.get(api):
            warnings.append(f"Chave da API {api} não configurada")
    
    # Verifica diretórios
    for directory in [DATA_DIR, LOGS_DIR, MODELS_DIR]:
        if not directory.exists():
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"Não foi possível criar diretório {directory}: {e}")
    
    # Verifica configurações de dispositivos
    if CAMERA_CONFIG["device_id"] < 0:
        warnings.append("ID da câmera inválido")
    
    if MICROPHONE_CONFIG["sample_rate"] < 8000:
        warnings.append("Taxa de amostragem do microfone muito baixa")
    
    # Verifica configurações de performance
    if PERFORMANCE_CONFIG["max_concurrent_processes"] < 1:
        errors.append("Número de processos deve ser >= 1")
    
    return errors, warnings

def get_config_summary():
    """Retorna resumo das configurações principais."""
    return {
        "project": PROJECT_NAME,
        "version": VERSION,
        "environment": ENVIRONMENT,
        "apis_configured": [k for k, v in API_KEYS.items() if v],
        "components_enabled": {
            "vision": bool(VISION_CONFIG.get("enabled", True)),
            "audio": bool(AUDIO_CONFIG.get("enabled", True)),
            "nlp": bool(NLP_CONFIG.get("enabled", True)),
            "animation": bool(ANIMATION_CONFIG.get("enabled", True))
        },
        "debug_mode": DEBUG_CONFIG["debug_mode"],
        "api_port": API_CONFIG["port"]
    }

# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def get_api_key(service: str) -> Optional[str]:
    """Obtém chave da API para um serviço específico."""
    return API_KEYS.get(service)

def get_model_path(model_type: str, model_name: str) -> Path:
    """Obtém caminho para um modelo específico."""
    return MODELS_DIR / model_type / model_name

def get_temp_file(filename: str) -> Path:
    """Obtém caminho para arquivo temporário."""
    return TEMP_DIR / filename

def get_log_file(module: str) -> Path:
    """Obtém caminho para arquivo de log de um módulo."""
    return LOGS_DIR / f"{module}.log"

# =============================================================================
# INICIALIZAÇÃO
# =============================================================================

# Valida configurações na importação
_errors, _warnings = validate_config()

if _errors:
    print("ERROS de configuração encontrados:")
    for error in _errors:
        print(f"  ❌ {error}")

if _warnings:
    print("AVISOS de configuração:")
    for warning in _warnings:
        print(f"  ⚠️  {warning}")

# Exporta configurações principais para fácil acesso
__all__ = [
    "API_KEYS", "CAMERA_CONFIG", "MICROPHONE_CONFIG", "SPEAKER_CONFIG",
    "VISION_CONFIG", "AUDIO_CONFIG", "NLP_CONFIG", "LLM_CONFIG",
    "TTS_CONFIG", "ANIMATION_CONFIG", "PERSONALITY_PROFILES",
    "API_CONFIG", "PERFORMANCE_CONFIG", "LOGGING_CONFIG",
    "EVENT_CONTEXT", "DEBUG_CONFIG", "SECURITY_CONFIG",
    "validate_config", "get_config_summary", "get_api_key"
]