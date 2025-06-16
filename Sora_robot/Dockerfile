# =========================================================================
# SORA ROBOT - DOCKERFILE
# Containerização completa do sistema de assistente virtual inteligente
# =========================================================================

# =========================================================================
# MULTI-STAGE BUILD - Para otimizar tamanho da imagem final
# =========================================================================

# Estágio 1: Base com dependências do sistema
FROM python:3.10-slim-bullseye as base

# Metadados da imagem
LABEL maintainer="Sora Robot Team <contato@sora-robot.com>"
LABEL version="1.0.0"
LABEL description="Sora Robot - Assistente Virtual Inteligente"
LABEL org.opencontainers.image.source="https://github.com/seu-usuario/sora-robot"
LABEL org.opencontainers.image.documentation="https://github.com/seu-usuario/sora-robot/blob/main/README.md"
LABEL org.opencontainers.image.licenses="MIT"

# Variáveis de ambiente para build
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Atualizar sistema e instalar dependências essenciais
RUN apt-get update && apt-get install -y \
    # Dependências básicas do sistema
    build-essential \
    pkg-config \
    cmake \
    wget \
    curl \
    git \
    # Dependências para OpenCV
    libopencv-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # Dependências para áudio
    libportaudio2 \
    portaudio19-dev \
    libasound2-dev \
    libpulse-dev \
    # Dependências para multimedia
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    # Dependências para dlib
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    # Dependências para face_recognition
    libdlib-dev \
    # Dependências SSL/TLS
    libssl-dev \
    libffi-dev \
    # Limpeza
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# =========================================================================
# Estágio 2: Instalação de dependências Python
# =========================================================================

FROM base as python-deps

# Atualizar pip e ferramentas
RUN pip install --upgrade pip setuptools wheel

# Copiar apenas requirements.txt primeiro (para cache do Docker)
COPY requirements.txt /tmp/requirements.txt

# Instalar dependências Python com otimizações
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && pip install --no-cache-dir gunicorn[gevent] \
    && rm /tmp/requirements.txt

# Baixar modelos necessários (cache layer)
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('vader_lexicon')" \
    && python -c "import spacy; spacy.cli.download('pt_core_news_sm')" \
    && python -c "import torch; torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)" \
    || echo "Alguns downloads de modelos falharam - continuando..."

# =========================================================================
# Estágio 3: Imagem final otimizada
# =========================================================================

FROM python-deps as final

# Criar usuário não-root para segurança
RUN groupadd --gid 1000 sora \
    && useradd --uid 1000 --gid sora --shell /bin/bash --create-home sora

# Definir diretório de trabalho
WORKDIR /app

# Criar diretórios necessários com permissões corretas
RUN mkdir -p /app/logs /app/data /app/temp /app/cache /app/models \
    && chown -R sora:sora /app

# Copiar código fonte
COPY --chown=sora:sora . /app/

# Ajustar permissões
RUN chmod +x /app/main.py \
    && find /app -type f -name "*.py" -exec chmod 644 {} \; \
    && find /app -type d -exec chmod 755 {} \;

# Configurar variáveis de ambiente para runtime
ENV PYTHONPATH=/app
ENV SORA_ENVIRONMENT=production
ENV SORA_LOG_LEVEL=INFO
ENV SORA_HOST=0.0.0.0
ENV SORA_PORT=8000
ENV SORA_WORKERS=1

# Variáveis para OpenCV (evitar warnings)
ENV QT_X11_NO_MITSHM=1
ENV DISPLAY=:99

# Variáveis para PyTorch
ENV TORCH_HOME=/app/cache/torch
ENV TRANSFORMERS_CACHE=/app/cache/transformers

# Variáveis para otimização
ENV OMP_NUM_THREADS=1
ENV NUMEXPR_MAX_THREADS=1

# Trocar para usuário não-root
USER sora

# Verificar instalação (healthcheck build-time)
RUN python -c "import cv2, torch, transformers, fastapi; print('✅ Dependências verificadas')"

# Criar script de entrada personalizado
USER root
RUN cat > /app/docker-entrypoint.sh << 'EOF'
#!/bin/bash
set -e

echo "🤖 Iniciando Sora Robot..."
echo "Ambiente: ${SORA_ENVIRONMENT}"
echo "Host: ${SORA_HOST}:${SORA_PORT}"

# Verificar dependências críticas
echo "Verificando dependências..."
python -c "import cv2, torch, transformers, fastapi" || {
    echo "❌ Erro: Dependências críticas não encontradas"
    exit 1
}

# Verificar variáveis de ambiente críticas
if [ "$SORA_ENVIRONMENT" = "production" ]; then
    echo "Verificando configuração de produção..."
    
    # Avisar sobre chaves de API ausentes (não bloquear)
    [ -z "$OPENAI_API_KEY" ] && echo "⚠️  OPENAI_API_KEY não configurada"
    [ -z "$GOOGLE_AI_API_KEY" ] && echo "⚠️  GOOGLE_AI_API_KEY não configurada"
fi

# Criar diretórios se não existirem
mkdir -p /app/logs /app/data /app/temp /app/cache

# Executar comando passado ou comando padrão
if [ $# -eq 0 ]; then
    echo "Iniciando servidor API..."
    exec python main.py --mode api --host ${SORA_HOST} --port ${SORA_PORT}
else
    echo "Executando comando customizado: $@"
    exec "$@"
fi
EOF

RUN chmod +x /app/docker-entrypoint.sh \
    && chown sora:sora /app/docker-entrypoint.sh

# Voltar para usuário não-root
USER sora

# Expor portas
EXPOSE 8000 8001 8002

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${SORA_PORT}/health || exit 1

# Volumes para persistência de dados
VOLUME ["/app/logs", "/app/data", "/app/cache"]

# Ponto de entrada
ENTRYPOINT ["/app/docker-entrypoint.sh"]

# Comando padrão (pode ser sobrescrito)
CMD []

# =========================================================================
# INSTRUÇÕES DE BUILD E EXECUÇÃO
# =========================================================================

# Build da imagem:
# docker build -t sora-robot:latest .
# docker build -t sora-robot:1.0.0 .

# Build com argumentos:
# docker build --build-arg SORA_VERSION=1.0.0 -t sora-robot:latest .

# Execução básica:
# docker run -p 8000:8000 sora-robot:latest

# Execução com variáveis de ambiente:
# docker run -p 8000:8000 \
#   -e OPENAI_API_KEY=your-key \
#   -e SORA_ENVIRONMENT=production \
#   sora-robot:latest

# Execução com volumes:
# docker run -p 8000:8000 \
#   -v $(pwd)/logs:/app/logs \
#   -v $(pwd)/data:/app/data \
#   sora-robot:latest

# Execução em modo interativo:
# docker run -it --rm sora-robot:latest python main.py --mode interactive

# Execução com comando customizado:
# docker run -it --rm sora-robot:latest bash

# Debug mode:
# docker run -it --rm \
#   -e SORA_ENVIRONMENT=development \
#   -p 8000:8000 \
#   sora-robot:latest python main.py --debug

# =========================================================================
# OTIMIZAÇÕES PARA PRODUÇÃO
# =========================================================================

# Para reduzir tamanho da imagem final:
# 1. Usar imagem Alpine (mais complexo devido a dependências)
# 2. Multi-stage build (já implementado)
# 3. Remover caches de build
# 4. Otimizar layers

# Para melhor performance:
# 1. Usar gunicorn com workers
# 2. Configurar variáveis de ambiente adequadas
# 3. Usar volumes para cache de modelos
# 4. Configurar limits de recursos

# Para segurança:
# 1. Usuário não-root (já implementado)
# 2. Scanning de vulnerabilidades
# 3. Secrets via Docker secrets ou variables
# 4. Read-only filesystem quando possível

# =========================================================================
# TROUBLESHOOTING
# =========================================================================

# Se erro de permissão:
# docker run --user root sora-robot:latest chown -R sora:sora /app

# Se erro de memória:
# docker run --memory=4g sora-robot:latest

# Se erro de GPU:
# docker run --gpus all sora-robot:latest

# Para debug:
# docker run -it --entrypoint bash sora-robot:latest

# Para logs detalhados:
# docker run -e SORA_LOG_LEVEL=DEBUG sora-robot:latest

# =========================================================================
# VARIANTES DA IMAGEM
# =========================================================================

# Versão CPU-only (menor):
# FROM python:3.10-slim as cpu-only
# [instalar apenas dependências CPU]

# Versão GPU (CUDA):
# FROM nvidia/cuda:11.8-runtime-ubuntu20.04 as gpu
# [instalar dependências CUDA]

# Versão minimal (funcionalidades básicas):
# [instalar apenas subset de requirements.txt]

# Versão development (com ferramentas de dev):
# [instalar pytest, black, mypy, etc.]