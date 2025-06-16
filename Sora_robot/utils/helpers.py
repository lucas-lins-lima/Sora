# sora_robot/utils/helpers.py

"""
Funções utilitárias centralizadas para o Sora Robot.
Fornece funcionalidades comuns usadas por todos os módulos do sistema.
"""

import os
import re
import json
import time
import uuid
import hashlib
import base64
import threading
import asyncio
import inspect
import pickle
import gzip
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, Tuple, Type
from datetime import datetime, timedelta
from functools import wraps
from collections import defaultdict, deque
import math
import statistics

# Imports opcionais
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    cv2 = None

from utils.logger import get_logger

# =============================================================================
# VALIDAÇÕES E VERIFICAÇÕES
# =============================================================================

def validate_email(email: str) -> bool:
    """
    Valida endereço de email.
    
    Args:
        email: Endereço de email a validar
        
    Returns:
        bool: True se email é válido
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def validate_url(url: str) -> bool:
    """
    Valida URL.
    
    Args:
        url: URL a validar
        
    Returns:
        bool: True se URL é válida
    """
    pattern = r'https?://(?:[-\w.])+(?:\:[0-9]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:\#(?:[\w.])*)?)?'
    return bool(re.match(pattern, url))

def validate_json(json_string: str) -> bool:
    """
    Valida string JSON.
    
    Args:
        json_string: String JSON a validar
        
    Returns:
        bool: True se JSON é válido
    """
    try:
        json.loads(json_string)
        return True
    except (json.JSONDecodeError, TypeError):
        return False

def is_numeric(value: Any) -> bool:
    """
    Verifica se valor é numérico.
    
    Args:
        value: Valor a verificar
        
    Returns:
        bool: True se é numérico
    """
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False

def validate_confidence_score(score: float) -> bool:
    """
    Valida score de confiança (0.0 a 1.0).
    
    Args:
        score: Score a validar
        
    Returns:
        bool: True se válido
    """
    return isinstance(score, (int, float)) and 0.0 <= score <= 1.0

def validate_coordinates(x: float, y: float, width: int = None, height: int = None) -> bool:
    """
    Valida coordenadas.
    
    Args:
        x: Coordenada X
        y: Coordenada Y
        width: Largura máxima (opcional)
        height: Altura máxima (opcional)
        
    Returns:
        bool: True se coordenadas são válidas
    """
    if not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
        return False
    
    if x < 0 or y < 0:
        return False
    
    if width is not None and x > width:
        return False
    
    if height is not None and y > height:
        return False
    
    return True

# =============================================================================
# CONVERSÕES E FORMATAÇÃO
# =============================================================================

def safe_float(value: Any, default: float = 0.0) -> float:
    """
    Converte valor para float de forma segura.
    
    Args:
        value: Valor a converter
        default: Valor padrão se conversão falhar
        
    Returns:
        float: Valor convertido ou padrão
    """
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_int(value: Any, default: int = 0) -> int:
    """
    Converte valor para int de forma segura.
    
    Args:
        value: Valor a converter
        default: Valor padrão se conversão falhar
        
    Returns:
        int: Valor convertido ou padrão
    """
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return default

def format_duration(seconds: float) -> str:
    """
    Formata duração em segundos para string legível.
    
    Args:
        seconds: Duração em segundos
        
    Returns:
        str: Duração formatada (ex: "1h 30m 45s")
    """
    if seconds < 0:
        return "0s"
    
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or not parts:
        parts.append(f"{secs}s")
    
    return " ".join(parts)

def format_timestamp(timestamp: float = None, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Formata timestamp para string.
    
    Args:
        timestamp: Timestamp (usa atual se None)
        format_str: Formato de saída
        
    Returns:
        str: Timestamp formatado
    """
    if timestamp is None:
        timestamp = time.time()
    
    return datetime.fromtimestamp(timestamp).strftime(format_str)

def format_bytes(bytes_value: int) -> str:
    """
    Formata bytes para string legível.
    
    Args:
        bytes_value: Valor em bytes
        
    Returns:
        str: Tamanho formatado (ex: "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024
    return f"{bytes_value:.1f} PB"

def format_percentage(value: float, decimal_places: int = 1) -> str:
    """
    Formata valor como porcentagem.
    
    Args:
        value: Valor (0.0 a 1.0)
        decimal_places: Casas decimais
        
    Returns:
        str: Porcentagem formatada
    """
    percentage = value * 100
    return f"{percentage:.{decimal_places}f}%"

def slugify(text: str) -> str:
    """
    Converte texto para slug (URL-friendly).
    
    Args:
        text: Texto a converter
        
    Returns:
        str: Slug gerado
    """
    # Remove acentos e converte para minúsculas
    text = text.lower()
    
    # Substitui caracteres especiais
    text = re.sub(r'[àáâãäå]', 'a', text)
    text = re.sub(r'[èéêë]', 'e', text)
    text = re.sub(r'[ìíîï]', 'i', text)
    text = re.sub(r'[òóôõö]', 'o', text)
    text = re.sub(r'[ùúûü]', 'u', text)
    text = re.sub(r'[ç]', 'c', text)
    text = re.sub(r'[ñ]', 'n', text)
    
    # Remove caracteres não alfanuméricos
    text = re.sub(r'[^a-z0-9\s-]', '', text)
    
    # Substitui espaços e múltiplos hífens
    text = re.sub(r'[-\s]+', '-', text)
    
    return text.strip('-')

# =============================================================================
# MANIPULAÇÃO DE ARQUIVOS E DIRETÓRIOS
# =============================================================================

def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Garante que diretório existe, criando se necessário.
    
    Args:
        path: Caminho do diretório
        
    Returns:
        Path: Caminho do diretório
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def safe_filename(filename: str) -> str:
    """
    Torna nome de arquivo seguro para o sistema.
    
    Args:
        filename: Nome do arquivo
        
    Returns:
        str: Nome seguro
    """
    # Remove/substitui caracteres problemáticos
    filename = re.sub(r'[<>:"/\|?*]', '_', filename)
    
    # Remove espaços extras
    filename = re.sub(r'\s+', '_', filename)
    
    # Limita tamanho
    name, ext = os.path.splitext(filename)
    if len(name) > 200:
        name = name[:200]
    
    return f"{name}{ext}"

def get_file_hash(filepath: Union[str, Path], algorithm: str = 'md5') -> str:
    """
    Calcula hash de arquivo.
    
    Args:
        filepath: Caminho do arquivo
        algorithm: Algoritmo de hash ('md5', 'sha1', 'sha256')
        
    Returns:
        str: Hash do arquivo
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")
    
    hash_obj = hashlib.new(algorithm)
    
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_obj.update(chunk)
    
    return hash_obj.hexdigest()

def copy_file_safe(src: Union[str, Path], dst: Union[str, Path], overwrite: bool = False) -> bool:
    """
    Copia arquivo de forma segura.
    
    Args:
        src: Arquivo origem
        dst: Arquivo destino
        overwrite: Se deve sobrescrever arquivo existente
        
    Returns:
        bool: True se cópia foi bem-sucedida
    """
    try:
        src = Path(src)
        dst = Path(dst)
        
        if not src.exists():
            return False
        
        if dst.exists() and not overwrite:
            return False
        
        # Garante que diretório de destino existe
        ensure_directory(dst.parent)
        
        import shutil
        shutil.copy2(src, dst)
        return True
        
    except Exception:
        return False

def read_file_safe(filepath: Union[str, Path], encoding: str = 'utf-8', default: str = None) -> Optional[str]:
    """
    Lê arquivo de forma segura.
    
    Args:
        filepath: Caminho do arquivo
        encoding: Codificação do arquivo
        default: Valor padrão se leitura falhar
        
    Returns:
        Optional[str]: Conteúdo do arquivo ou None
    """
    try:
        with open(filepath, 'r', encoding=encoding) as f:
            return f.read()
    except Exception:
        return default

def write_file_safe(filepath: Union[str, Path], content: str, encoding: str = 'utf-8') -> bool:
    """
    Escreve arquivo de forma segura.
    
    Args:
        filepath: Caminho do arquivo
        content: Conteúdo a escrever
        encoding: Codificação do arquivo
        
    Returns:
        bool: True se escrita foi bem-sucedida
    """
    try:
        filepath = Path(filepath)
        ensure_directory(filepath.parent)
        
        with open(filepath, 'w', encoding=encoding) as f:
            f.write(content)
        return True
    except Exception:
        return False

# =============================================================================
# OPERAÇÕES COM JSON E PICKLE
# =============================================================================

def load_json_safe(filepath: Union[str, Path], default: Any = None) -> Any:
    """
    Carrega JSON de forma segura.
    
    Args:
        filepath: Caminho do arquivo JSON
        default: Valor padrão se carregar falhar
        
    Returns:
        Any: Dados carregados ou valor padrão
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return default

def save_json_safe(data: Any, filepath: Union[str, Path], indent: int = 2) -> bool:
    """
    Salva JSON de forma segura.
    
    Args:
        data: Dados a salvar
        filepath: Caminho do arquivo
        indent: Indentação para formatação
        
    Returns:
        bool: True se salvamento foi bem-sucedido
    """
    try:
        filepath = Path(filepath)
        ensure_directory(filepath.parent)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        return True
    except Exception:
        return False

def load_pickle_safe(filepath: Union[str, Path], default: Any = None) -> Any:
    """
    Carrega arquivo pickle de forma segura.
    
    Args:
        filepath: Caminho do arquivo
        default: Valor padrão se carregar falhar
        
    Returns:
        Any: Dados carregados ou valor padrão
    """
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return default

def save_pickle_safe(data: Any, filepath: Union[str, Path], compress: bool = False) -> bool:
    """
    Salva dados em pickle de forma segura.
    
    Args:
        data: Dados a salvar
        filepath: Caminho do arquivo
        compress: Se deve comprimir arquivo
        
    Returns:
        bool: True se salvamento foi bem-sucedido
    """
    try:
        filepath = Path(filepath)
        ensure_directory(filepath.parent)
        
        if compress:
            with gzip.open(filepath, 'wb') as f:
                pickle.dump(data, f)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
        return True
    except Exception:
        return False

# =============================================================================
# UTILITÁRIOS DE THREADING E ASYNC
# =============================================================================

class ThreadSafeCounter:
    """Contador thread-safe."""
    
    def __init__(self, initial_value: int = 0):
        self._value = initial_value
        self._lock = threading.Lock()
    
    def increment(self, amount: int = 1) -> int:
        """Incrementa contador."""
        with self._lock:
            self._value += amount
            return self._value
    
    def decrement(self, amount: int = 1) -> int:
        """Decrementa contador."""
        with self._lock:
            self._value -= amount
            return self._value
    
    def get(self) -> int:
        """Obtém valor atual."""
        with self._lock:
            return self._value
    
    def set(self, value: int) -> int:
        """Define novo valor."""
        with self._lock:
            self._value = value
            return self._value

class ThreadSafeDict:
    """Dicionário thread-safe."""
    
    def __init__(self):
        self._data = {}
        self._lock = threading.RLock()
    
    def get(self, key: Any, default: Any = None) -> Any:
        """Obtém valor."""
        with self._lock:
            return self._data.get(key, default)
    
    def set(self, key: Any, value: Any) -> Any:
        """Define valor."""
        with self._lock:
            self._data[key] = value
            return value
    
    def pop(self, key: Any, default: Any = None) -> Any:
        """Remove e retorna valor."""
        with self._lock:
            return self._data.pop(key, default)
    
    def keys(self) -> List[Any]:
        """Retorna chaves."""
        with self._lock:
            return list(self._data.keys())
    
    def values(self) -> List[Any]:
        """Retorna valores."""
        with self._lock:
            return list(self._data.values())
    
    def items(self) -> List[Tuple[Any, Any]]:
        """Retorna itens."""
        with self._lock:
            return list(self._data.items())
    
    def clear(self):
        """Limpa dicionário."""
        with self._lock:
            self._data.clear()

def run_in_thread(func: Callable, *args, daemon: bool = True, **kwargs) -> threading.Thread:
    """
    Executa função em thread separada.
    
    Args:
        func: Função a executar
        *args: Argumentos da função
        daemon: Se thread deve ser daemon
        **kwargs: Argumentos nomeados da função
        
    Returns:
        threading.Thread: Thread criada
    """
    thread = threading.Thread(target=func, args=args, kwargs=kwargs, daemon=daemon)
    thread.start()
    return thread

def timeout_function(timeout_seconds: float):
    """
    Decorator para adicionar timeout a funções.
    
    Args:
        timeout_seconds: Timeout em segundos
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = [None]
            exception = [None]
            
            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e
            
            thread = threading.Thread(target=target, daemon=True)
            thread.start()
            thread.join(timeout_seconds)
            
            if thread.is_alive():
                raise TimeoutError(f"Função {func.__name__} excedeu timeout de {timeout_seconds}s")
            
            if exception[0]:
                raise exception[0]
            
            return result[0]
        
        return wrapper
    return decorator

def retry_on_exception(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Decorator para retry automático em caso de exceção.
    
    Args:
        max_retries: Número máximo de tentativas
        delay: Delay inicial entre tentativas
        backoff: Multiplicador do delay a cada tentativa
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries:
                        raise e
                    
                    time.sleep(current_delay)
                    current_delay *= backoff
            
        return wrapper
    return decorator

# =============================================================================
# UTILITÁRIOS MATEMÁTICOS E ESTATÍSTICOS
# =============================================================================

def clamp(value: float, min_value: float, max_value: float) -> float:
    """
    Limita valor entre mínimo e máximo.
    
    Args:
        value: Valor a limitar
        min_value: Valor mínimo
        max_value: Valor máximo
        
    Returns:
        float: Valor limitado
    """
    return max(min_value, min(value, max_value))

def normalize(value: float, min_value: float, max_value: float) -> float:
    """
    Normaliza valor entre 0 e 1.
    
    Args:
        value: Valor a normalizar
        min_value: Valor mínimo da escala
        max_value: Valor máximo da escala
        
    Returns:
        float: Valor normalizado
    """
    if max_value == min_value:
        return 0.0
    return (value - min_value) / (max_value - min_value)

def lerp(start: float, end: float, t: float) -> float:
    """
    Interpolação linear entre dois valores.
    
    Args:
        start: Valor inicial
        end: Valor final
        t: Fator de interpolação (0.0 a 1.0)
        
    Returns:
        float: Valor interpolado
    """
    return start + (end - start) * clamp(t, 0.0, 1.0)

def moving_average(values: List[float], window_size: int) -> List[float]:
    """
    Calcula média móvel.
    
    Args:
        values: Lista de valores
        window_size: Tamanho da janela
        
    Returns:
        List[float]: Médias móveis
    """
    if len(values) < window_size:
        return values
    
    averages = []
    for i in range(len(values) - window_size + 1):
        window = values[i:i + window_size]
        averages.append(sum(window) / window_size)
    
    return averages

def calculate_confidence_interval(values: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    """
    Calcula intervalo de confiança.
    
    Args:
        values: Lista de valores
        confidence: Nível de confiança (0.0 a 1.0)
        
    Returns:
        Tuple[float, float]: (limite_inferior, limite_superior)
    """
    if not values:
        return (0.0, 0.0)
    
    n = len(values)
    mean = statistics.mean(values)
    
    if n == 1:
        return (mean, mean)
    
    std_err = statistics.stdev(values) / math.sqrt(n)
    
    # Aproximação usando distribuição normal
    z_score = 1.96 if confidence == 0.95 else 2.576  # Para 99%
    margin = z_score * std_err
    
    return (mean - margin, mean + margin)

def euclidean_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """
    Calcula distância euclidiana entre dois pontos.
    
    Args:
        point1: Primeiro ponto (x, y)
        point2: Segundo ponto (x, y)
        
    Returns:
        float: Distância euclidiana
    """
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def cosine_similarity(vector1: List[float], vector2: List[float]) -> float:
    """
    Calcula similaridade de cosseno entre dois vetores.
    
    Args:
        vector1: Primeiro vetor
        vector2: Segundo vetor
        
    Returns:
        float: Similaridade de cosseno (-1 a 1)
    """
    if NUMPY_AVAILABLE:
        v1 = np.array(vector1)
        v2 = np.array(vector2)
        
        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    else:
        # Implementação sem numpy
        dot_product = sum(a * b for a, b in zip(vector1, vector2))
        norm1 = math.sqrt(sum(a * a for a in vector1))
        norm2 = math.sqrt(sum(b * b for b in vector2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)

# =============================================================================
# UTILITÁRIOS DE CACHE E MEMOIZAÇÃO
# =============================================================================

class LRUCache:
    """Cache LRU (Least Recently Used) thread-safe."""
    
    def __init__(self, max_size: int = 128):
        self.max_size = max_size
        self._cache = {}
        self._order = deque()
        self._lock = threading.Lock()
    
    def get(self, key: Any, default: Any = None) -> Any:
        """Obtém valor do cache."""
        with self._lock:
            if key in self._cache:
                # Move para final (mais recente)
                self._order.remove(key)
                self._order.append(key)
                return self._cache[key]
            return default
    
    def set(self, key: Any, value: Any):
        """Define valor no cache."""
        with self._lock:
            if key in self._cache:
                # Atualiza valor existente
                self._cache[key] = value
                self._order.remove(key)
                self._order.append(key)
            else:
                # Adiciona novo valor
                if len(self._cache) >= self.max_size:
                    # Remove item mais antigo
                    oldest_key = self._order.popleft()
                    del self._cache[oldest_key]
                
                self._cache[key] = value
                self._order.append(key)
    
    def clear(self):
        """Limpa cache."""
        with self._lock:
            self._cache.clear()
            self._order.clear()
    
    def size(self) -> int:
        """Retorna tamanho atual do cache."""
        with self._lock:
            return len(self._cache)

def memoize(max_size: int = 128):
    """
    Decorator para memoização de funções.
    
    Args:
        max_size: Tamanho máximo do cache
    """
    def decorator(func):
        cache = LRUCache(max_size)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Cria chave do cache
            key = (args, tuple(sorted(kwargs.items())))
            
            # Verifica se está no cache
            result = cache.get(key)
            if result is not None:
                return result
            
            # Calcula resultado e armazena
            result = func(*args, **kwargs)
            cache.set(key, result)
            
            return result
        
        wrapper.cache_clear = cache.clear
        wrapper.cache_size = cache.size
        
        return wrapper
    return decorator

# =============================================================================
# UTILITÁRIOS DE REDE E URL
# =============================================================================

def is_port_open(host: str, port: int, timeout: float = 3.0) -> bool:
    """
    Verifica se porta está aberta.
    
    Args:
        host: Endereço do host
        port: Número da porta
        timeout: Timeout da conexão
        
    Returns:
        bool: True se porta está aberta
    """
    try:
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            return result == 0
    except Exception:
        return False

def get_local_ip() -> str:
    """
    Obtém IP local da máquina.
    
    Returns:
        str: Endereço IP local
    """
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

def parse_url_params(url: str) -> Dict[str, str]:
    """
    Extrai parâmetros de URL.
    
    Args:
        url: URL a analisar
        
    Returns:
        Dict[str, str]: Parâmetros extraídos
    """
    try:
        from urllib.parse import urlparse, parse_qs
        parsed = urlparse(url)
        params = parse_qs(parsed.query)
        
        # Converte listas de um elemento para strings
        result = {}
        for key, values in params.items():
            if len(values) == 1:
                result[key] = values[0]
            else:
                result[key] = values
        
        return result
    except Exception:
        return {}

# =============================================================================
# UTILITÁRIOS DE PROCESSAMENTO DE IMAGEM
# =============================================================================

def resize_image_safe(image: Any, target_size: Tuple[int, int], maintain_aspect: bool = True) -> Optional[Any]:
    """
    Redimensiona imagem de forma segura.
    
    Args:
        image: Imagem (numpy array ou similar)
        target_size: Tamanho de destino (width, height)
        maintain_aspect: Se deve manter proporção
        
    Returns:
        Optional[Any]: Imagem redimensionada ou None
    """
    if not OPENCV_AVAILABLE:
        return None
    
    try:
        if maintain_aspect:
            h, w = image.shape[:2]
            target_w, target_h = target_size
            
            # Calcula nova dimensão mantendo proporção
            aspect = w / h
            if target_w / target_h > aspect:
                new_h = target_h
                new_w = int(aspect * new_h)
            else:
                new_w = target_w
                new_h = int(new_w / aspect)
            
            resized = cv2.resize(image, (new_w, new_h))
            
            # Adiciona padding se necessário
            if new_w != target_w or new_h != target_h:
                delta_w = target_w - new_w
                delta_h = target_h - new_h
                top, bottom = delta_h // 2, delta_h - (delta_h // 2)
                left, right = delta_w // 2, delta_w - (delta_w // 2)
                
                resized = cv2.copyMakeBorder(
                    resized, top, bottom, left, right,
                    cv2.BORDER_CONSTANT, value=[0, 0, 0]
                )
            
            return resized
        else:
            return cv2.resize(image, target_size)
            
    except Exception:
        return None

def crop_image_safe(image: Any, x: int, y: int, width: int, height: int) -> Optional[Any]:
    """
    Recorta imagem de forma segura.
    
    Args:
        image: Imagem a recortar
        x: Coordenada X do recorte
        y: Coordenada Y do recorte
        width: Largura do recorte
        height: Altura do recorte
        
    Returns:
        Optional[Any]: Imagem recortada ou None
    """
    try:
        h, w = image.shape[:2]
        
        # Valida coordenadas
        x = max(0, min(x, w))
        y = max(0, min(y, h))
        width = max(1, min(width, w - x))
        height = max(1, min(height, h - y))
        
        return image[y:y+height, x:x+width]
        
    except Exception:
        return None

# =============================================================================
# UTILITÁRIOS DE TEMPO E DATA
# =============================================================================

def get_current_timestamp() -> float:
    """Retorna timestamp atual."""
    return time.time()

def timestamp_to_datetime(timestamp: float) -> datetime:
    """Converte timestamp para datetime."""
    return datetime.fromtimestamp(timestamp)

def datetime_to_timestamp(dt: datetime) -> float:
    """Converte datetime para timestamp."""
    return dt.timestamp()

def format_relative_time(timestamp: float) -> str:
    """
    Formata tempo relativo (ex: "há 5 minutos").
    
    Args:
        timestamp: Timestamp de referência
        
    Returns:
        str: Tempo relativo formatado
    """
    now = time.time()
    diff = now - timestamp
    
    if diff < 60:
        return "há poucos segundos"
    elif diff < 3600:
        minutes = int(diff // 60)
        return f"há {minutes} minuto{'s' if minutes != 1 else ''}"
    elif diff < 86400:
        hours = int(diff // 3600)
        return f"há {hours} hora{'s' if hours != 1 else ''}"
    else:
        days = int(diff // 86400)
        return f"há {days} dia{'s' if days != 1 else ''}"

# =============================================================================
# UTILITÁRIOS DE GERAÇÃO
# =============================================================================

def generate_uuid() -> str:
    """Gera UUID único."""
    return str(uuid.uuid4())

def generate_short_id(length: int = 8) -> str:
    """
    Gera ID curto aleatório.
    
    Args:
        length: Comprimento do ID
        
    Returns:
        str: ID gerado
    """
    import string
    import random
    
    chars = string.ascii_letters + string.digits
    return ''.join(random.choice(chars) for _ in range(length))

def generate_hash(data: str, algorithm: str = 'md5') -> str:
    """
    Gera hash de string.
    
    Args:
        data: Dados a fazer hash
        algorithm: Algoritmo de hash
        
    Returns:
        str: Hash gerado
    """
    hash_obj = hashlib.new(algorithm)
    hash_obj.update(data.encode('utf-8'))
    return hash_obj.hexdigest()

# =============================================================================
# UTILITÁRIOS DE SISTEMA
# =============================================================================

def get_system_info() -> Dict[str, Any]:
    """
    Obtém informações do sistema.
    
    Returns:
        Dict[str, Any]: Informações do sistema
    """
    import platform
    import sys
    
    try:
        import psutil
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        disk = psutil.disk_usage('/')
        
        system_info = {
            'platform': platform.system(),
            'platform_release': platform.release(),
            'platform_version': platform.version(),
            'architecture': platform.machine(),
            'hostname': platform.node(),
            'processor': platform.processor(),
            'python_version': sys.version,
            'cpu_count': psutil.cpu_count(),
            'cpu_percent': cpu_percent,
            'memory_total': memory.total,
            'memory_available': memory.available,
            'memory_percent': memory.percent,
            'disk_total': disk.total,
            'disk_free': disk.free,
            'disk_percent': disk.percent
        }
    except ImportError:
        # Fallback sem psutil
        system_info = {
            'platform': platform.system(),
            'platform_release': platform.release(),
            'architecture': platform.machine(),
            'hostname': platform.node(),
            'python_version': sys.version
        }
    
    return system_info

def is_docker_container() -> bool:
    """Verifica se está executando em container Docker."""
    try:
        with open('/proc/1/cgroup', 'r') as f:
            return 'docker' in f.read()
    except Exception:
        return False

# =============================================================================
# FUNÇÕES DE DEBUG E PROFILING
# =============================================================================

def debug_function_call(func: Callable):
    """
    Decorator para debug de chamadas de função.
    
    Args:
        func: Função a debugar
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger('debug')
        
        # Log de entrada
        logger.debug(f"Chamando {func.__name__} com args={args}, kwargs={kwargs}")
        
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            
            logger.debug(f"{func.__name__} concluída em {duration:.3f}s, resultado: {type(result)}")
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            logger.debug(f"{func.__name__} falhou em {duration:.3f}s com erro: {e}")
            raise
    
    return wrapper

def profile_memory_usage():
    """
    Decorator para profilear uso de memória.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                import psutil
                process = psutil.Process()
                
                # Memória antes
                mem_before = process.memory_info().rss
                
                result = func(*args, **kwargs)
                
                # Memória depois
                mem_after = process.memory_info().rss
                mem_diff = mem_after - mem_before
                
                logger = get_logger('memory_profile')
                logger.info(f"{func.__name__}: Δ memória = {format_bytes(mem_diff)}")
                
                return result
                
            except ImportError:
                # Fallback sem psutil
                return func(*args, **kwargs)
        
        return wrapper
    return decorator

# =============================================================================
# EXPORTAÇÕES
# =============================================================================

__all__ = [
    # Validações
    'validate_email', 'validate_url', 'validate_json', 'is_numeric',
    'validate_confidence_score', 'validate_coordinates',
    
    # Conversões
    'safe_float', 'safe_int', 'format_duration', 'format_timestamp',
    'format_bytes', 'format_percentage', 'slugify',
    
    # Arquivos
    'ensure_directory', 'safe_filename', 'get_file_hash', 'copy_file_safe',
    'read_file_safe', 'write_file_safe',
    
    # JSON/Pickle
    'load_json_safe', 'save_json_safe', 'load_pickle_safe', 'save_pickle_safe',
    
    # Threading
    'ThreadSafeCounter', 'ThreadSafeDict', 'run_in_thread',
    'timeout_function', 'retry_on_exception',
    
    # Matemática
    'clamp', 'normalize', 'lerp', 'moving_average', 'calculate_confidence_interval',
    'euclidean_distance', 'cosine_similarity',
    
    # Cache
    'LRUCache', 'memoize',
    
    # Rede
    'is_port_open', 'get_local_ip', 'parse_url_params',
    
    # Imagem
    'resize_image_safe', 'crop_image_safe',
    
    # Tempo
    'get_current_timestamp', 'timestamp_to_datetime', 'datetime_to_timestamp',
    'format_relative_time',
    
    # Geração
    'generate_uuid', 'generate_short_id', 'generate_hash',
    
    # Sistema
    'get_system_info', 'is_docker_container',
    
    # Debug
    'debug_function_call', 'profile_memory_usage'
]