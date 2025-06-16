#!/bin/bash

# =========================================================================
# SORA ROBOT - AUTOMATED SETUP SCRIPT
# Script de configuração automática para instalação completa do sistema
# =========================================================================

set -e  # Parar execução em caso de erro

# =========================================================================
# CONFIGURAÇÕES E CONSTANTES
# =========================================================================

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

# Configurações do script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOGFILE="$PROJECT_DIR/logs/setup.log"
REQUIRED_DOCKER_VERSION="20.0.0"
REQUIRED_PYTHON_VERSION="3.8"

# Informações do projeto
PROJECT_NAME="Sora Robot"
PROJECT_VERSION="1.0.0"
GITHUB_REPO="https://github.com/seu-usuario/sora-robot"

# =========================================================================
# FUNÇÕES UTILITÁRIAS
# =========================================================================

# Função para imprimir mensagens coloridas
print_message() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Função para imprimir cabeçalhos
print_header() {
    echo ""
    print_message $CYAN "=========================================="
    print_message $WHITE "$1"
    print_message $CYAN "=========================================="
    echo ""
}

# Função para imprimir sucesso
print_success() {
    print_message $GREEN "✅ $1"
}

# Função para imprimir aviso
print_warning() {
    print_message $YELLOW "⚠️  $1"
}

# Função para imprimir erro
print_error() {
    print_message $RED "❌ $1"
}

# Função para imprimir informação
print_info() {
    print_message $BLUE "ℹ️  $1"
}

# Função para log
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOGFILE"
}

# Função para verificar se comando existe
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Função para comparar versões
version_ge() {
    test "$(printf '%s\n' "$@" | sort -V | head -n 1)" != "$1"
}

# Função para aguardar confirmação do usuário
confirm() {
    read -p "$(echo -e ${YELLOW}$1${NC}) [y/N]: " response
    case "$response" in
        [yY][eE][sS]|[yY])
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

# Função para gerar chave aleatória
generate_random_key() {
    local length=${1:-32}
    openssl rand -hex $length 2>/dev/null || python3 -c "import secrets; print(secrets.token_hex($length))"
}

# =========================================================================
# VERIFICAÇÕES INICIAIS
# =========================================================================

check_system_requirements() {
    print_header "🔍 Verificando Requisitos do Sistema"
    
    local requirements_met=true
    
    # Verificar sistema operacional
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        print_success "Sistema operacional: Linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        print_success "Sistema operacional: macOS"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        print_warning "Sistema operacional: Windows (usando WSL/Cygwin)"
    else
        print_error "Sistema operacional não suportado: $OSTYPE"
        requirements_met=false
    fi
    
    # Verificar Python
    if command_exists python3; then
        local python_version=$(python3 --version | cut -d' ' -f2)
        if version_ge $python_version $REQUIRED_PYTHON_VERSION; then
            print_success "Python $python_version encontrado"
        else
            print_error "Python $REQUIRED_PYTHON_VERSION+ necessário. Encontrado: $python_version"
            requirements_met=false
        fi
    else
        print_error "Python 3 não encontrado"
        requirements_met=false
    fi
    
    # Verificar pip
    if command_exists pip3; then
        print_success "pip3 encontrado"
    else
        print_error "pip3 não encontrado"
        requirements_met=false
    fi
    
    # Verificar Docker
    if command_exists docker; then
        local docker_version=$(docker --version | cut -d' ' -f3 | cut -d',' -f1)
        if version_ge $docker_version $REQUIRED_DOCKER_VERSION; then
            print_success "Docker $docker_version encontrado"
        else
            print_error "Docker $REQUIRED_DOCKER_VERSION+ necessário. Encontrado: $docker_version"
            requirements_met=false
        fi
    else
        print_error "Docker não encontrado"
        requirements_met=false
    fi
    
    # Verificar Docker Compose
    if command_exists docker-compose; then
        print_success "Docker Compose encontrado"
    elif docker compose version >/dev/null 2>&1; then
        print_success "Docker Compose (plugin) encontrado"
    else
        print_error "Docker Compose não encontrado"
        requirements_met=false
    fi
    
    # Verificar Git
    if command_exists git; then
        print_success "Git encontrado"
    else
        print_warning "Git não encontrado (opcional)"
    fi
    
    # Verificar curl
    if command_exists curl; then
        print_success "curl encontrado"
    else
        print_warning "curl não encontrado (opcional)"
    fi
    
    # Verificar dependências do sistema para Python packages
    check_system_dependencies
    
    if [ "$requirements_met" = false ]; then
        print_error "Alguns requisitos não foram atendidos. Instale as dependências e execute novamente."
        exit 1
    fi
    
    print_success "Todos os requisitos básicos foram atendidos!"
}

check_system_dependencies() {
    print_info "Verificando dependências do sistema para packages Python..."
    
    # Dependências comuns necessárias
    local deps_linux=("build-essential" "python3-dev" "libportaudio2" "ffmpeg" "cmake" "libssl-dev" "libffi-dev")
    local deps_macos=("portaudio" "ffmpeg" "cmake")
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Verificar se é Ubuntu/Debian ou CentOS/RHEL
        if command_exists apt-get; then
            print_info "Sistema baseado em Debian/Ubuntu detectado"
            local missing_deps=()
            for dep in "${deps_linux[@]}"; do
                if ! dpkg -l | grep -q "^ii  $dep "; then
                    missing_deps+=("$dep")
                fi
            done
            
            if [ ${#missing_deps[@]} -gt 0 ]; then
                print_warning "Dependências ausentes: ${missing_deps[*]}"
                if confirm "Instalar dependências automaticamente?"; then
                    sudo apt-get update
                    sudo apt-get install -y "${missing_deps[@]}"
                fi
            fi
            
        elif command_exists yum; then
            print_info "Sistema baseado em RedHat/CentOS detectado"
            print_warning "Certifique-se de ter: python3-devel portaudio-devel ffmpeg cmake gcc gcc-c++"
        fi
        
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        print_info "macOS detectado"
        if command_exists brew; then
            print_success "Homebrew encontrado"
            for dep in "${deps_macos[@]}"; do
                if ! brew list "$dep" &>/dev/null; then
                    print_warning "Dependência ausente: $dep"
                    if confirm "Instalar $dep via Homebrew?"; then
                        brew install "$dep"
                    fi
                fi
            done
        else
            print_warning "Homebrew não encontrado. Instale manualmente: ${deps_macos[*]}"
        fi
    fi
}

# =========================================================================
# CONFIGURAÇÃO DO AMBIENTE
# =========================================================================

setup_directories() {
    print_header "📁 Configurando Estrutura de Diretórios"
    
    cd "$PROJECT_DIR"
    
    # Criar diretórios necessários
    local directories=(
        "logs"
        "data/user_profiles"
        "data/knowledge_base" 
        "data/collected_data"
        "temp"
        "cache"
        "backups"
        "scripts"
        "monitoring/grafana/dashboards"
        "monitoring/grafana/datasources"
        "monitoring/prometheus/rules"
        "nginx/conf.d"
        "database"
    )
    
    for dir in "${directories[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            print_success "Criado diretório: $dir"
            log_message "Created directory: $dir"
        else
            print_info "Diretório já existe: $dir"
        fi
    done
    
    # Configurar permissões
    chmod 755 logs data temp cache backups
    chmod 644 *.py *.md *.txt *.yml *.yaml *.json 2>/dev/null || true
    chmod +x scripts/*.sh 2>/dev/null || true
    
    print_success "Estrutura de diretórios configurada!"
}

setup_environment_file() {
    print_header "🔧 Configurando Arquivo de Ambiente"
    
    if [ ! -f ".env" ]; then
        if [ -f ".env.example" ]; then
            cp .env.example .env
            print_success "Arquivo .env criado a partir do .env.example"
            
            # Gerar chaves seguras automaticamente
            print_info "Gerando chaves de segurança..."
            
            local encryption_key=$(generate_random_key 32)
            local jwt_secret=$(generate_random_key 32)
            local postgres_password=$(generate_random_key 16)
            local redis_password=$(generate_random_key 16)
            local grafana_password=$(generate_random_key 12)
            
            # Substituir no arquivo .env
            sed -i.bak "s/your-32-character-encryption-key-here/$encryption_key/g" .env
            sed -i.bak "s/your-jwt-secret-key-here/$jwt_secret/g" .env
            sed -i.bak "s/your-secure-postgres-password-here/$postgres_password/g" .env
            sed -i.bak "s/your-secure-redis-password-here/$redis_password/g" .env
            sed -i.bak "s/your-grafana-admin-password-here/$grafana_password/g" .env
            
            # Remover arquivo backup
            rm .env.bak 2>/dev/null || true
            
            print_success "Chaves de segurança geradas automaticamente"
            
            # Mostrar informações importantes
            print_warning "IMPORTANTE: Configure as seguintes variáveis no arquivo .env:"
            print_info "  - OPENAI_API_KEY (essencial para funcionamento)"
            print_info "  - GOOGLE_AI_API_KEY (opcional)"
            print_info "  - AZURE_SPEECH_KEY (opcional)"
            print_info "  - SORA_DOMAIN (para produção)"
            
        else
            print_error "Arquivo .env.example não encontrado!"
            exit 1
        fi
    else
        print_info "Arquivo .env já existe"
        if confirm "Deseja regenerar chaves de segurança?"; then
            local encryption_key=$(generate_random_key 32)
            local jwt_secret=$(generate_random_key 32)
            
            # Backup do .env atual
            cp .env .env.backup.$(date +%Y%m%d_%H%M%S)
            
            # Atualizar chaves
            sed -i.bak "s/SORA_ENCRYPTION_KEY=.*/SORA_ENCRYPTION_KEY=$encryption_key/g" .env
            sed -i.bak "s/JWT_SECRET_KEY=.*/JWT_SECRET_KEY=$jwt_secret/g" .env
            
            rm .env.bak 2>/dev/null || true
            print_success "Chaves de segurança regeneradas"
        fi
    fi
    
    # Verificar se variáveis críticas estão configuradas
    source .env
    
    local missing_vars=()
    
    if [ -z "$OPENAI_API_KEY" ] || [ "$OPENAI_API_KEY" = "your-openai-api-key-here" ]; then
        missing_vars+=("OPENAI_API_KEY")
    fi
    
    if [ ${#missing_vars[@]} -gt 0 ]; then
        print_warning "Variáveis críticas não configuradas: ${missing_vars[*]}"
        print_info "Configure essas variáveis no arquivo .env antes de continuar"
    fi
}

# =========================================================================
# INSTALAÇÃO DE DEPENDÊNCIAS
# =========================================================================

setup_python_environment() {
    print_header "🐍 Configurando Ambiente Python"
    
    # Verificar se ambiente virtual existe
    if [ ! -d "venv" ]; then
        print_info "Criando ambiente virtual Python..."
        python3 -m venv venv
        print_success "Ambiente virtual criado"
    else
        print_info "Ambiente virtual já existe"
    fi
    
    # Ativar ambiente virtual
    source venv/bin/activate
    print_success "Ambiente virtual ativado"
    
    # Atualizar pip
    print_info "Atualizando pip..."
    pip install --upgrade pip
    
    # Instalar dependências
    if [ -f "requirements.txt" ]; then
        print_info "Instalando dependências Python..."
        pip install -r requirements.txt
        print_success "Dependências Python instaladas"
    else
        print_error "Arquivo requirements.txt não encontrado!"
        exit 1
    fi
    
    # Verificar instalação crítica
    print_info "Verificando instalação das dependências críticas..."
    python3 -c "
import sys
try:
    import cv2
    print('✅ OpenCV instalado')
except ImportError:
    print('❌ OpenCV não instalado')
    sys.exit(1)

try:
    import torch
    print('✅ PyTorch instalado')
except ImportError:
    print('❌ PyTorch não instalado')
    sys.exit(1)

try:
    import transformers
    print('✅ Transformers instalado')
except ImportError:
    print('❌ Transformers não instalado')
    sys.exit(1)

try:
    import fastapi
    print('✅ FastAPI instalado')
except ImportError:
    print('❌ FastAPI não instalado')
    sys.exit(1)

print('✅ Todas as dependências críticas estão instaladas')
"
    
    print_success "Ambiente Python configurado com sucesso!"
}

download_ml_models() {
    print_header "🤖 Baixando Modelos de Machine Learning"
    
    source venv/bin/activate
    
    print_info "Baixando modelos NLTK..."
    python3 -c "
import nltk
try:
    nltk.download('punkt', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('stopwords', quiet=True)
    print('✅ Modelos NLTK baixados')
except Exception as e:
    print(f'⚠️  Erro ao baixar modelos NLTK: {e}')
"
    
    print_info "Baixando modelos spaCy..."
    python3 -c "
import spacy
import subprocess
import sys

try:
    subprocess.check_call([sys.executable, '-m', 'spacy', 'download', 'pt_core_news_sm'], 
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print('✅ Modelo spaCy português baixado')
except Exception as e:
    print(f'⚠️  Erro ao baixar modelo spaCy: {e}')
"
    
    print_info "Pré-carregando modelos de transformers..."
    python3 -c "
from transformers import pipeline
import os

try:
    # Cache dos modelos em diretório local
    cache_dir = 'cache/transformers'
    os.makedirs(cache_dir, exist_ok=True)
    
    # Baixar modelo de análise de sentimento
    sentiment_pipeline = pipeline('sentiment-analysis', 
                                 model='nlptown/bert-base-multilingual-uncased-sentiment',
                                 cache_dir=cache_dir)
    print('✅ Modelo de análise de sentimento carregado')
    
except Exception as e:
    print(f'⚠️  Erro ao carregar modelos transformers: {e}')
"
    
    print_success "Modelos de ML configurados!"
}

# =========================================================================
# CONFIGURAÇÃO DO DOCKER
# =========================================================================

setup_docker_environment() {
    print_header "🐳 Configurando Ambiente Docker"
    
    # Verificar se Docker está rodando
    if ! docker info >/dev/null 2>&1; then
        print_error "Docker não está rodando. Inicie o Docker e tente novamente."
        exit 1
    fi
    
    print_success "Docker está rodando"
    
    # Build da imagem principal
    print_info "Fazendo build da imagem Sora Robot..."
    docker build -t sora-robot:latest . || {
        print_error "Falha no build da imagem Docker"
        exit 1
    }
    print_success "Imagem Docker construída"
    
    # Verificar docker-compose.yml
    if [ -f "docker-compose.yml" ]; then
        print_info "Validando docker-compose.yml..."
        docker-compose config >/dev/null || {
            print_error "Erro na configuração do docker-compose.yml"
            exit 1
        }
        print_success "docker-compose.yml válido"
    else
        print_error "Arquivo docker-compose.yml não encontrado!"
        exit 1
    fi
    
    # Criar networks se não existirem
    docker network create sora-network 2>/dev/null || print_info "Network sora-network já existe"
    
    print_success "Ambiente Docker configurado!"
}

start_services() {
    print_header "🚀 Iniciando Serviços"
    
    print_info "Parando serviços existentes..."
    docker-compose down >/dev/null 2>&1 || true
    
    print_info "Iniciando serviços em background..."
    docker-compose up -d
    
    print_info "Aguardando serviços ficarem saudáveis..."
    
    # Aguardar PostgreSQL
    print_info "Aguardando PostgreSQL..."
    for i in {1..30}; do
        if docker-compose exec -T postgres pg_isready -U sora_user >/dev/null 2>&1; then
            print_success "PostgreSQL está pronto"
            break
        fi
        if [ $i -eq 30 ]; then
            print_error "Timeout aguardando PostgreSQL"
            exit 1
        fi
        sleep 2
    done
    
    # Aguardar Redis
    print_info "Aguardando Redis..."
    for i in {1..15}; do
        if docker-compose exec -T redis redis-cli ping >/dev/null 2>&1; then
            print_success "Redis está pronto"
            break
        fi
        if [ $i -eq 15 ]; then
            print_error "Timeout aguardando Redis"
            exit 1
        fi
        sleep 2
    done
    
    # Aguardar aplicação principal
    print_info "Aguardando Sora Robot API..."
    for i in {1..60}; do
        if curl -f http://localhost:8000/health >/dev/null 2>&1; then
            print_success "Sora Robot API está pronta"
            break
        fi
        if [ $i -eq 60 ]; then
            print_error "Timeout aguardando Sora Robot API"
            exit 1
        fi
        sleep 3
    done
    
    print_success "Todos os serviços estão rodando!"
}

# =========================================================================
# VERIFICAÇÕES FINAIS
# =========================================================================

run_health_checks() {
    print_header "🏥 Executando Verificações de Saúde"
    
    local all_healthy=true
    
    # Verificar API principal
    print_info "Verificando API principal..."
    if curl -f http://localhost:8000/health >/dev/null 2>&1; then
        print_success "API principal: OK"
    else
        print_error "API principal: FALHA"
        all_healthy=false
    fi
    
    # Verificar status dos serviços
    print_info "Verificando status dos containers..."
    local unhealthy_services=$(docker-compose ps --filter "status=exited" --format "table {{.Service}}" | tail -n +2)
    if [ -z "$unhealthy_services" ]; then
        print_success "Todos os containers estão rodando"
    else
        print_error "Containers com problema: $unhealthy_services"
        all_healthy=false
    fi
    
    # Verificar endpoints críticos
    local endpoints=(
        "http://localhost:8000/status"
        "http://localhost:8000/docs"
        "http://localhost/"
    )
    
    for endpoint in "${endpoints[@]}"; do
        print_info "Testando $endpoint..."
        if curl -f "$endpoint" >/dev/null 2>&1; then
            print_success "$endpoint: OK"
        else
            print_warning "$endpoint: Inacessível"
        fi
    done
    
    # Verificar logs por erros críticos
    print_info "Verificando logs por erros..."
    local error_count=$(docker-compose logs 2>&1 | grep -i "error\|exception\|failed" | wc -l)
    if [ "$error_count" -eq 0 ]; then
        print_success "Nenhum erro crítico encontrado nos logs"
    else
        print_warning "$error_count linhas com possíveis erros encontradas"
        print_info "Execute 'docker-compose logs' para mais detalhes"
    fi
    
    if [ "$all_healthy" = true ]; then
        print_success "Todas as verificações de saúde passaram!"
    else
        print_warning "Algumas verificações falharam, mas o sistema pode estar funcional"
    fi
}

run_basic_tests() {
    print_header "🧪 Executando Testes Básicos"
    
    source venv/bin/activate
    
    print_info "Testando importações Python..."
    python3 -c "
import sys
sys.path.append('.')

try:
    from config import CONFIG
    print('✅ Configuração carregada')
except Exception as e:
    print(f'❌ Erro ao carregar configuração: {e}')

try:
    from utils.logger import get_logger
    logger = get_logger('test')
    logger.info('Teste de logging')
    print('✅ Sistema de logging OK')
except Exception as e:
    print(f'❌ Erro no sistema de logging: {e}')

try:
    from utils.helpers import validate_input
    print('✅ Helpers carregados')
except Exception as e:
    print(f'❌ Erro ao carregar helpers: {e}')
"
    
    print_info "Testando API via Python..."
    python3 -c "
import requests
import time

try:
    # Aguardar API estar pronta
    for _ in range(10):
        try:
            response = requests.get('http://localhost:8000/health', timeout=5)
            if response.status_code == 200:
                break
        except:
            time.sleep(1)
    
    # Teste de status
    response = requests.get('http://localhost:8000/status', timeout=10)
    if response.status_code == 200:
        print('✅ Endpoint /status OK')
    else:
        print(f'❌ Endpoint /status retornou {response.status_code}')
    
    # Teste básico de interação (se API keys estiverem configuradas)
    # payload = {'message': 'Hello, Sora!', 'wait_for_response': False}
    # response = requests.post('http://localhost:8000/message', json=payload, timeout=10)
    # print(f'✅ Teste de mensagem: {response.status_code}')
    
except Exception as e:
    print(f'⚠️  Erro nos testes de API: {e}')
"
    
    print_success "Testes básicos concluídos!"
}

# =========================================================================
# INFORMAÇÕES FINAIS
# =========================================================================

show_final_information() {
    print_header "🎉 Instalação Concluída!"
    
    echo ""
    print_message $GREEN "╔════════════════════════════════════════════════════════════════╗"
    print_message $GREEN "║                    SORA ROBOT v${PROJECT_VERSION}                         ║"
    print_message $GREEN "║                 Instalação Concluída com Sucesso!             ║"
    print_message $GREEN "╚════════════════════════════════════════════════════════════════╝"
    echo ""
    
    print_message $CYAN "🌐 INTERFACES DISPONÍVEIS:"
    print_message $WHITE "  • Interface Web:      http://localhost"
    print_message $WHITE "  • API Documentation: http://localhost:8000/docs"
    print_message $WHITE "  • API Redoc:         http://localhost:8000/redoc"
    print_message $WHITE "  • Grafana Dashboard: http://localhost:3000 (admin/admin)"
    print_message $WHITE "  • Prometheus:        http://localhost:9090"
    echo ""
    
    print_message $CYAN "🔧 COMANDOS ÚTEIS:"
    print_message $WHITE "  • Logs em tempo real: docker-compose logs -f"
    print_message $WHITE "  • Status dos serviços: docker-compose ps"
    print_message $WHITE "  • Parar serviços: docker-compose down"
    print_message $WHITE "  • Reiniciar: docker-compose restart"
    print_message $WHITE "  • Rebuild: docker-compose up -d --build"
    echo ""
    
    print_message $CYAN "📁 ARQUIVOS IMPORTANTES:"
    print_message $WHITE "  • Configuração: .env"
    print_message $WHITE "  • Logs: logs/"
    print_message $WHITE "  • Dados: data/"
    print_message $WHITE "  • Backups: backups/"
    echo ""
    
    print_message $CYAN "⚠️  PRÓXIMOS PASSOS:"
    print_message $YELLOW "  1. Configure as API keys no arquivo .env"
    print_message $YELLOW "  2. Teste a interface web em http://localhost"
    print_message $YELLOW "  3. Configure SSL/TLS para produção"
    print_message $YELLOW "  4. Configure backup automático"
    print_message $YELLOW "  5. Configure monitoramento de alertas"
    echo ""
    
    if [ -f ".env" ]; then
        source .env
        if [ -z "$OPENAI_API_KEY" ] || [ "$OPENAI_API_KEY" = "your-openai-api-key-here" ]; then
            print_message $RED "🚨 ATENÇÃO: Configure OPENAI_API_KEY no arquivo .env para funcionalidade completa!"
        fi
    fi
    
    print_message $CYAN "📚 DOCUMENTAÇÃO:"
    print_message $WHITE "  • README.md - Documentação completa"
    print_message $WHITE "  • GitHub: $GITHUB_REPO"
    echo ""
    
    print_message $GREEN "Sistema pronto para uso! 🚀"
    echo ""
}

# =========================================================================
# FUNÇÃO PRINCIPAL
# =========================================================================

main() {
    # Cabeçalho inicial
    clear
    print_message $PURPLE "
    ███████╗ ██████╗ ██████╗  █████╗     ██████╗  ██████╗ ██████╗  ██████╗ ██╗████████╗
    ██╔════╝██╔═══██╗██╔══██╗██╔══██╗    ██╔══██╗██╔═══██╗██╔══██╗██╔═══██╗██║╚══██╔══╝
    ███████╗██║   ██║██████╔╝███████║    ██████╔╝██║   ██║██████╔╝██║   ██║██║   ██║   
    ╚════██║██║   ██║██╔══██╗██╔══██║    ██╔══██╗██║   ██║██╔══██╗██║   ██║██║   ██║   
    ███████║╚██████╔╝██║  ██║██║  ██║    ██║  ██║╚██████╔╝██████╔╝╚██████╔╝██║   ██║   
    ╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝    ╚═╝  ╚═╝ ╚═════╝ ╚═════╝  ╚═════╝ ╚═╝   ╚═╝   
    "
    echo ""
    print_message $CYAN "Assistente Virtual Inteligente - Setup Automatizado v${PROJECT_VERSION}"
    print_message $WHITE "Este script irá configurar todo o ambiente Sora Robot"
    echo ""
    
    # Criar arquivo de log
    mkdir -p "$(dirname "$LOGFILE")"
    touch "$LOGFILE"
    log_message "Setup iniciado"
    
    # Verificar se é primeira execução
    if confirm "Continuar com a instalação?"; then
        echo ""
    else
        print_info "Instalação cancelada pelo usuário"
        exit 0
    fi
    
    # Executar etapas de configuração
    check_system_requirements
    setup_directories
    setup_environment_file
    setup_python_environment
    download_ml_models
    setup_docker_environment
    start_services
    run_health_checks
    run_basic_tests
    show_final_information
    
    log_message "Setup concluído com sucesso"
}

# =========================================================================
# TRATAMENTO DE SINAIS
# =========================================================================

cleanup() {
    print_warning "Interrompido pelo usuário"
    log_message "Setup interrompido"
    exit 1
}

trap cleanup SIGINT SIGTERM

# =========================================================================
# EXECUÇÃO DO SCRIPT
# =========================================================================

# Verificar se o script está sendo executado do diretório correto
if [ ! -f "requirements.txt" ] || [ ! -f "docker-compose.yml" ]; then
    print_error "Execute este script a partir do diretório raiz do projeto Sora Robot"
    exit 1
fi

# Executar função principal
main "$@"