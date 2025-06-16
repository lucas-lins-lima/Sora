# sora_robot/utils/logger.py

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional
import config
from utils.constants import LOG_SETTINGS

def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: str = "INFO",
    console_output: bool = True
) -> logging.Logger:
    """
    Configura e retorna um logger personalizado.
    
    Args:
        name: Nome do logger
        log_file: Arquivo de log (opcional)
        level: Nível de log
        console_output: Se deve mostrar logs no console
        
    Returns:
        logging.Logger: Logger configurado
    """
    logger = logging.getLogger(name)
    
    # Evita duplicação de handlers
    if logger.handlers:
        return logger
    
    # Configura nível
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    
    # Formato personalizado
    formatter = logging.Formatter(
        LOG_SETTINGS.FORMAT,
        datefmt=LOG_SETTINGS.DATE_FORMAT
    )
    
    # Handler para arquivo se especificado
    if log_file:
        # Cria diretório se não existir
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=LOG_SETTINGS.MAX_LOG_SIZE,
            backupCount=LOG_SETTINGS.BACKUP_COUNT,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Handler para console
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """
    Retorna logger com configuração padrão do projeto.
    
    Args:
        name: Nome do logger (geralmente __name__)
        
    Returns:
        logging.Logger: Logger configurado
    """
    return setup_logger(
        name=name,
        log_file=config.LOG_FILE if hasattr(config, 'LOG_FILE') else LOG_SETTINGS.LOG_FILE,
        level=config.LOG_LEVEL if hasattr(config, 'LOG_LEVEL') else "INFO",
        console_output=True
    )

import logging
import logging.handlers
import sys
import os
import threading
import time
import traceback
from pathlib import Path
from typing import Dict, Optional, List, Any, Union
from datetime import datetime, timedelta
import json

# Importa configurações
try:
    from config import LOGGING_CONFIG, LOGS_DIR, DEBUG_CONFIG
except ImportError:
    # Fallback se config não estiver disponível
    LOGS_DIR = Path("logs")
    LOGS_DIR.mkdir(exist_ok=True)
    
    LOGGING_CONFIG = {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "date_format": "%Y-%m-%d %H:%M:%S",
        "file_logging": True,
        "log_rotation": True,
        "max_file_size": "10MB",
        "backup_count": 5,
        "module_logs": {},
        "exclude_patterns": []
    }
    
    DEBUG_CONFIG = {"debug_mode": False}

class ColoredFormatter(logging.Formatter):
    """Formatter que adiciona cores ao output do console."""
    
    # Códigos ANSI para cores
    COLORS = {
        'DEBUG': '\033[36m',      # Ciano
        'INFO': '\033[32m',       # Verde
        'WARNING': '\033[33m',    # Amarelo
        'ERROR': '\033[31m',      # Vermelho
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        """Formata record com cores."""
        # Adiciona cor baseada no nível
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        # Formata mensagem original
        original_format = super().format(record)
        
        # Adiciona cores apenas se for um terminal
        if sys.stderr.isatty():
            # Colore o nível
            colored_level = f"{color}{record.levelname}{reset}"
            original_format = original_format.replace(record.levelname, colored_level, 1)
        
        return original_format

class StructuredFormatter(logging.Formatter):
    """Formatter que gera logs em formato JSON estruturado."""
    
    def format(self, record):
        """Formata record como JSON."""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread': threading.current_thread().name,
            'process': os.getpid()
        }
        
        # Adiciona informações de exceção se disponível
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Adiciona campos extras se disponível
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        return json.dumps(log_entry, ensure_ascii=False)

class PerformanceFilter(logging.Filter):
    """Filtro que adiciona métricas de performance."""
    
    def __init__(self):
        super().__init__()
        self.start_times = {}
    
    def filter(self, record):
        """Adiciona informações de performance ao record."""
        thread_id = threading.get_ident()
        
        # Marca tempo de início se for uma operação
        if hasattr(record, 'operation_start'):
            self.start_times[thread_id] = time.time()
        
        # Calcula duração se for fim de operação
        elif hasattr(record, 'operation_end') and thread_id in self.start_times:
            duration = time.time() - self.start_times[thread_id]
            record.msg = f"{record.msg} [Duration: {duration:.3f}s]"
            del self.start_times[thread_id]
        
        return True

class ExcludePatternFilter(logging.Filter):
    """Filtro que exclui mensagens baseado em padrões."""
    
    def __init__(self, patterns: List[str]):
        super().__init__()
        self.patterns = patterns
    
    def filter(self, record):
        """Filtra records baseado nos padrões de exclusão."""
        message = record.getMessage().lower()
        
        for pattern in self.patterns:
            if pattern.lower() in message:
                return False
        
        return True

class SoraLogger:
    """
    Sistema de logging personalizado para o Sora Robot.
    Gerencia múltiplos loggers, handlers e configurações.
    """
    
    def __init__(self):
        """Inicializa o sistema de logging."""
        self._loggers = {}
        self._handlers = {}
        self._lock = threading.Lock()
        self._initialized = False
        
        # Configurações
        self.config = LOGGING_CONFIG.copy()
        self.logs_dir = Path(LOGS_DIR)
        
        # Cria diretório de logs se não existir
        self.logs_dir.mkdir(exist_ok=True)
        
        # Inicializa sistema
        self._setup_logging()
    
    def _setup_logging(self):
        """Configura sistema de logging."""
        try:
            with self._lock:
                if self._initialized:
                    return
                
                # Configura logging root
                self._setup_root_logger()
                
                # Cria handlers
                self._create_handlers()
                
                # Configura loggers de módulos específicos
                self._setup_module_loggers()
                
                self._initialized = True
                
        except Exception as e:
            print(f"Erro ao configurar logging: {e}")
            # Fallback para logging básico
            logging.basicConfig(level=logging.INFO)
    
    def _setup_root_logger(self):
        """Configura logger root."""
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.config['level']))
        
        # Remove handlers existentes
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
    
    def _create_handlers(self):
        """Cria handlers para diferentes tipos de output."""
        
        # Handler para console
        console_handler = self._create_console_handler()
        if console_handler:
            self._handlers['console'] = console_handler
        
        # Handler para arquivo principal
        if self.config.get('file_logging', True):
            file_handler = self._create_file_handler('main.log')
            if file_handler:
                self._handlers['file'] = file_handler
        
        # Handler para erros
        error_handler = self._create_error_handler()
        if error_handler:
            self._handlers['error'] = error_handler
        
        # Handler estruturado (JSON)
        if DEBUG_CONFIG.get('debug_mode', False):
            structured_handler = self._create_structured_handler()
            if structured_handler:
                self._handlers['structured'] = structured_handler
    
    def _create_console_handler(self) -> Optional[logging.Handler]:
        """Cria handler para output no console."""
        try:
            handler = logging.StreamHandler(sys.stdout)
            
            # Formatter com cores
            formatter = ColoredFormatter(
                fmt=self.config['format'],
                datefmt=self.config['date_format']
            )
            handler.setFormatter(formatter)
            
            # Filtros
            self._add_filters_to_handler(handler)
            
            return handler
            
        except Exception as e:
            print(f"Erro ao criar console handler: {e}")
            return None
    
    def _create_file_handler(self, filename: str) -> Optional[logging.Handler]:
        """Cria handler para arquivo com rotação."""
        try:
            filepath = self.logs_dir / filename
            
            if self.config.get('log_rotation', True):
                # Handler com rotação de arquivos
                max_bytes = self._parse_size(self.config.get('max_file_size', '10MB'))
                backup_count = self.config.get('backup_count', 5)
                
                handler = logging.handlers.RotatingFileHandler(
                    filepath,
                    maxBytes=max_bytes,
                    backupCount=backup_count,
                    encoding='utf-8'
                )
            else:
                # Handler simples
                handler = logging.FileHandler(filepath, encoding='utf-8')
            
            # Formatter padrão
            formatter = logging.Formatter(
                fmt=self.config['format'],
                datefmt=self.config['date_format']
            )
            handler.setFormatter(formatter)
            
            # Filtros
            self._add_filters_to_handler(handler)
            
            return handler
            
        except Exception as e:
            print(f"Erro ao criar file handler: {e}")
            return None
    
    def _create_error_handler(self) -> Optional[logging.Handler]:
        """Cria handler específico para erros."""
        try:
            filepath = self.logs_dir / 'errors.log'
            
            handler = logging.handlers.RotatingFileHandler(
                filepath,
                maxBytes=self._parse_size('5MB'),
                backupCount=10,
                encoding='utf-8'
            )
            
            # Apenas logs de erro e críticos
            handler.setLevel(logging.ERROR)
            
            # Formatter detalhado para erros
            formatter = logging.Formatter(
                fmt='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s\n'
                    'Thread: %(thread)d - Process: %(process)d\n'
                    '%(pathname)s in %(funcName)s()\n',
                datefmt=self.config['date_format']
            )
            handler.setFormatter(formatter)
            
            return handler
            
        except Exception as e:
            print(f"Erro ao criar error handler: {e}")
            return None
    
    def _create_structured_handler(self) -> Optional[logging.Handler]:
        """Cria handler para logs estruturados (JSON)."""
        try:
            filepath = self.logs_dir / 'structured.jsonl'
            
            handler = logging.handlers.RotatingFileHandler(
                filepath,
                maxBytes=self._parse_size('20MB'),
                backupCount=3,
                encoding='utf-8'
            )
            
            # Formatter JSON
            formatter = StructuredFormatter()
            handler.setFormatter(formatter)
            
            return handler
            
        except Exception as e:
            print(f"Erro ao criar structured handler: {e}")
            return None
    
    def _add_filters_to_handler(self, handler: logging.Handler):
        """Adiciona filtros a um handler."""
        try:
            # Filtro de exclusão de padrões
            exclude_patterns = self.config.get('exclude_patterns', [])
            if exclude_patterns:
                exclude_filter = ExcludePatternFilter(exclude_patterns)
                handler.addFilter(exclude_filter)
            
            # Filtro de performance (apenas em debug)
            if DEBUG_CONFIG.get('debug_mode', False):
                performance_filter = PerformanceFilter()
                handler.addFilter(performance_filter)
                
        except Exception as e:
            print(f"Erro ao adicionar filtros: {e}")
    
    def _setup_module_loggers(self):
        """Configura loggers para módulos específicos."""
        module_configs = self.config.get('module_logs', {})
        
        for module_name, level in module_configs.items():
            try:
                logger = logging.getLogger(module_name)
                logger.setLevel(getattr(logging, level.upper()))
                
                # Cria handler específico se necessário
                if module_name in ['api', 'system', 'performance']:
                    module_handler = self._create_module_handler(module_name)
                    if module_handler:
                        logger.addHandler(module_handler)
                
            except Exception as e:
                print(f"Erro ao configurar logger do módulo {module_name}: {e}")
    
    def _create_module_handler(self, module_name: str) -> Optional[logging.Handler]:
        """Cria handler específico para um módulo."""
        try:
            filepath = self.logs_dir / f'{module_name}.log'
            
            handler = logging.handlers.RotatingFileHandler(
                filepath,
                maxBytes=self._parse_size('5MB'),
                backupCount=3,
                encoding='utf-8'
            )
            
            # Formatter específico
            formatter = logging.Formatter(
                fmt=f'%(asctime)s - {module_name.upper()} - %(levelname)s - %(message)s',
                datefmt=self.config['date_format']
            )
            handler.setFormatter(formatter)
            
            return handler
            
        except Exception as e:
            print(f"Erro ao criar handler do módulo {module_name}: {e}")
            return None
    
    def _parse_size(self, size_str: str) -> int:
        """Converte string de tamanho para bytes."""
        size_str = size_str.upper()
        
        if size_str.endswith('KB'):
            return int(size_str[:-2]) * 1024
        elif size_str.endswith('MB'):
            return int(size_str[:-2]) * 1024 * 1024
        elif size_str.endswith('GB'):
            return int(size_str[:-2]) * 1024 * 1024 * 1024
        else:
            return int(size_str)
    
    def get_logger(self, name: str) -> logging.Logger:
        """
        Obtém logger para um módulo específico.
        
        Args:
            name: Nome do módulo/logger
            
        Returns:
            logging.Logger: Logger configurado
        """
        try:
            with self._lock:
                if name in self._loggers:
                    return self._loggers[name]
                
                # Cria novo logger
                logger = logging.getLogger(name)
                
                # Adiciona handlers se logger não tiver
                if not logger.handlers:
                    for handler in self._handlers.values():
                        logger.addHandler(handler)
                
                # Evita propagação dupla
                logger.propagate = False
                
                # Armazena referência
                self._loggers[name] = logger
                
                return logger
                
        except Exception as e:
            print(f"Erro ao criar logger {name}: {e}")
            # Fallback para logger básico
            return logging.getLogger(name)
    
    def log_performance(self, logger_name: str, operation: str, duration: float, **kwargs):
        """
        Log específico para métricas de performance.
        
        Args:
            logger_name: Nome do logger
            operation: Nome da operação
            duration: Duração em segundos
            **kwargs: Métricas adicionais
        """
        try:
            logger = self.get_logger(f"{logger_name}.performance")
            
            # Prepara mensagem
            metrics = {
                'operation': operation,
                'duration': f"{duration:.3f}s",
                **kwargs
            }
            
            metrics_str = " | ".join([f"{k}={v}" for k, v in metrics.items()])
            logger.info(f"PERFORMANCE: {metrics_str}")
            
        except Exception as e:
            print(f"Erro ao logar performance: {e}")
    
    def log_error_with_context(self, logger_name: str, error: Exception, context: Dict[str, Any]):
        """
        Log de erro com contexto adicional.
        
        Args:
            logger_name: Nome do logger
            error: Exceção ocorrida
            context: Contexto adicional
        """
        try:
            logger = self.get_logger(logger_name)
            
            # Prepara contexto
            context_str = " | ".join([f"{k}={v}" for k, v in context.items()])
            
            logger.error(
                f"ERRO: {type(error).__name__}: {str(error)} | CONTEXTO: {context_str}",
                exc_info=True,
                extra={'extra_fields': context}
            )
            
        except Exception as e:
            print(f"Erro ao logar erro com contexto: {e}")
    
    def log_system_event(self, event_type: str, message: str, **data):
        """
        Log de eventos do sistema.
        
        Args:
            event_type: Tipo do evento
            message: Mensagem do evento
            **data: Dados adicionais
        """
        try:
            logger = self.get_logger('system.events')
            
            event_data = {
                'event_type': event_type,
                'timestamp': datetime.now().isoformat(),
                **data
            }
            
            logger.info(
                f"SYSTEM_EVENT: {event_type} - {message}",
                extra={'extra_fields': event_data}
            )
            
        except Exception as e:
            print(f"Erro ao logar evento do sistema: {e}")
    
    def get_log_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas dos logs."""
        try:
            stats = {
                'loggers_created': len(self._loggers),
                'handlers_active': len(self._handlers),
                'log_files': [],
                'total_size': 0
            }
            
            # Verifica arquivos de log
            for log_file in self.logs_dir.glob('*.log'):
                file_size = log_file.stat().st_size
                stats['log_files'].append({
                    'name': log_file.name,
                    'size': file_size,
                    'size_mb': f"{file_size / (1024*1024):.2f}",
                    'modified': datetime.fromtimestamp(log_file.stat().st_mtime).isoformat()
                })
                stats['total_size'] += file_size
            
            stats['total_size_mb'] = f"{stats['total_size'] / (1024*1024):.2f}"
            
            return stats
            
        except Exception as e:
            return {'error': str(e)}
    
    def cleanup_old_logs(self, days: int = 30):
        """
        Remove logs antigos.
        
        Args:
            days: Número de dias para manter
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            cutoff_timestamp = cutoff_date.timestamp()
            
            removed_files = []
            
            for log_file in self.logs_dir.glob('*.log*'):
                if log_file.stat().st_mtime < cutoff_timestamp:
                    log_file.unlink()
                    removed_files.append(log_file.name)
            
            if removed_files:
                logger = self.get_logger('system.cleanup')
                logger.info(f"Logs antigos removidos: {removed_files}")
            
            return removed_files
            
        except Exception as e:
            print(f"Erro ao limpar logs antigos: {e}")
            return []
    
    def set_log_level(self, logger_name: str, level: str):
        """
        Define nível de log para um logger específico.
        
        Args:
            logger_name: Nome do logger
            level: Nível de log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        try:
            logger = self.get_logger(logger_name)
            logger.setLevel(getattr(logging, level.upper()))
            
            self.log_system_event(
                'log_level_changed',
                f"Nível do logger {logger_name} alterado para {level}",
                logger=logger_name,
                new_level=level
            )
            
        except Exception as e:
            print(f"Erro ao definir nível de log: {e}")

# =============================================================================
# INSTÂNCIA GLOBAL DO SISTEMA DE LOGGING
# =============================================================================

# Cria instância global
_sora_logger = SoraLogger()

def get_logger(name: str) -> logging.Logger:
    """
    Função principal para obter loggers.
    Esta é a função que todos os módulos do Sora usam.
    
    Args:
        name: Nome do módulo/logger
        
    Returns:
        logging.Logger: Logger configurado
    """
    return _sora_logger.get_logger(name)

def setup_logging():
    """Inicializa sistema de logging (chamada opcional)."""
    global _sora_logger
    if not _sora_logger._initialized:
        _sora_logger._setup_logging()

def log_performance(logger_name: str, operation: str, duration: float, **kwargs):
    """Log de performance (função de conveniência)."""
    _sora_logger.log_performance(logger_name, operation, duration, **kwargs)

def log_error_with_context(logger_name: str, error: Exception, context: Dict[str, Any]):
    """Log de erro com contexto (função de conveniência)."""
    _sora_logger.log_error_with_context(logger_name, error, context)

def log_system_event(event_type: str, message: str, **data):
    """Log de evento do sistema (função de conveniência)."""
    _sora_logger.log_system_event(event_type, message, **data)

def get_log_stats() -> Dict[str, Any]:
    """Retorna estatísticas dos logs."""
    return _sora_logger.get_log_stats()

def cleanup_old_logs(days: int = 30) -> List[str]:
    """Remove logs antigos."""
    return _sora_logger.cleanup_old_logs(days)

def set_log_level(logger_name: str, level: str):
    """Define nível de log."""
    _sora_logger.set_log_level(logger_name, level)

# =============================================================================
# DECORATORS UTILITÁRIOS
# =============================================================================

def log_execution_time(logger_name: str = None):
    """
    Decorator para logar tempo de execução de funções.
    
    Args:
        logger_name: Nome do logger (usa nome do módulo se None)
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            # Determina nome do logger
            actual_logger_name = logger_name or func.__module__
            logger = get_logger(actual_logger_name)
            
            try:
                logger.debug(f"Iniciando execução: {func.__name__}")
                result = func(*args, **kwargs)
                
                duration = time.time() - start_time
                logger.debug(f"Execução concluída: {func.__name__} [{duration:.3f}s]")
                
                # Log de performance se duração for significativa
                if duration > 1.0:
                    log_performance(actual_logger_name, func.__name__, duration)
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"Erro na execução: {func.__name__} [{duration:.3f}s]: {e}")
                raise
        
        return wrapper
    return decorator

def log_exceptions(logger_name: str = None):
    """
    Decorator para logar exceções automaticamente.
    
    Args:
        logger_name: Nome do logger (usa nome do módulo se None)
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Determina nome do logger
            actual_logger_name = logger_name or func.__module__
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                log_error_with_context(
                    actual_logger_name,
                    e,
                    {
                        'function': func.__name__,
                        'args': str(args)[:200],  # Limita tamanho
                        'kwargs': str(kwargs)[:200]
                    }
                )
                raise
        
        return wrapper
    return decorator

# =============================================================================
# CONTEXTMANAGER PARA LOGGING
# =============================================================================

from contextlib import contextmanager

@contextmanager
def log_context(logger_name: str, operation: str, level: str = 'INFO'):
    """
    Context manager para logar início e fim de operações.
    
    Args:
        logger_name: Nome do logger
        operation: Nome da operação
        level: Nível de log
    """
    logger = get_logger(logger_name)
    start_time = time.time()
    
    try:
        getattr(logger, level.lower())(f"Iniciando: {operation}")
        yield logger
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Erro em {operation} [{duration:.3f}s]: {e}")
        raise
    else:
        duration = time.time() - start_time
        getattr(logger, level.lower())(f"Concluído: {operation} [{duration:.3f}s]")

# =============================================================================
# EXEMPLO DE USO
# =============================================================================

if __name__ == "__main__":
    # Exemplo de uso do sistema de logging
    
    # Obtém logger
    logger = get_logger("test_module")
    
    # Logs básicos
    logger.debug("Mensagem de debug")
    logger.info("Sistema inicializado")
    logger.warning("Aviso sobre algo")
    logger.error("Erro encontrado")
    
    # Log com decorator
    @log_execution_time("test_module")
    def test_function():
        time.sleep(1)
        return "resultado"
    
    result = test_function()
    
    # Log com context manager
    with log_context("test_module", "operacao_complexa"):
        time.sleep(0.5)
        logger.info("Processando...")
    
    # Log de performance
    log_performance("test_module", "test_operation", 1.5, accuracy=0.95, items=100)
    
    # Log de evento do sistema
    log_system_event("test_event", "Sistema de teste executado", module="test")
    
    # Estatísticas
    stats = get_log_stats()
    print(f"Estatísticas dos logs: {stats}")