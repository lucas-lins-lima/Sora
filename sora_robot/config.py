# sora_robot/config.py
#
# Este arquivo contém as configurações globais para o projeto Sora.
# Aqui são definidos caminhos de arquivos, chaves de API, e outras variáveis
# que controlam o comportamento do robô.

# --- Chaves de API ---
# Armazena as chaves de API necessárias para interagir com serviços externos.
# IMPORTANTE: Substitua os placeholders com suas chaves de API reais.
API_KEYS = {
    "google_speech_to_text": "SUA_CHAVE_API_GOOGLE_SPEECH_TO_TEXT", # Chave para o serviço de transcrição de fala.
    "google_text_to_speech": "SUA_CHAVE_API_GOOGLE_TEXT_TO_SPEECH", # Chave para o serviço de síntese de fala.
    "gemini": "SUA_CHAVE_API_GEMINI", # Chave para a API do modelo Gemini (Large Language Model).
    # Adicione outras chaves de API aqui conforme a necessidade de novos serviços.
}

# --- Caminhos para Diretórios de Dados ---
# Define os caminhos relativos ou absolutos para os arquivos de dados do projeto.
DATA_PATHS = {
    # Caminho para o arquivo JSON de perfis de usuário.
    "user_profiles": "sora_robot/data/user_profiles/profiles.json",
    # Caminho para o arquivo JSON da base de conhecimento do evento.
    "knowledge_base": "sora_robot/data/knowledge_base/event_info.json",
    # Caminho para o arquivo JSONL de logs de interações (JSON Lines).
    "interaction_logs": "sora_robot/data/collected_data/interaction_logs.jsonl",
    # Caminho para o arquivo JSONL de dados formatados para aprendizado de máquina.
    "learning_data": "sora_robot/data/collected_data/learning_data.jsonl",
    # Adicione outros caminhos de dados aqui conforme a necessidade.
}

# --- Configurações de Logging ---
# Define o nome do arquivo de log e o nível mínimo de mensagens a serem registradas.
LOG_FILE = "sora_robot.log" # Nome do arquivo onde os logs serão salvos.
LOG_LEVEL = "INFO" # Nível de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL).
                   # INFO: Mensagens informativas sobre o fluxo normal da aplicação.
                   # DEBUG: Mensagens detalhadas para depuração.

# --- Configurações de Hardware (Exemplo) ---
# Parâmetros de configuração para dispositivos de hardware como câmeras e microfones.
CAMERA_INDEX = 0  # Índice da câmera a ser usada (0 para a câmera padrão do sistema, 1 para a segunda, etc.).
MICROPHONE_DEVICE_INDEX = None # Índice do dispositivo de microfone (None para usar o microfone padrão do sistema).
                               # Pode ser um número inteiro se houver múltiplos microfones.

# --- Configurações da API Backend ---
API_SERVER_HOST = "127.0.0.1" # Endereço IP do servidor da API (geralmente localhost para desenvolvimento).
API_SERVER_PORT = 5000       # Porta onde o servidor da API irá escutar.

# --- Outras Configurações ---
# Caminho base para os arquivos de animação do avatar (vídeos MP4).
# Estes arquivos devem conter as animações pré-renderizadas para as expressões do Sora.
AVATAR_ANIMATIONS_PATH = "sora_robot/assets/avatar_animations/"
# Exemplo de caminho para um modelo 3D do avatar, caso seja usado um motor de renderização.
# AVATAR_MODEL_PATH = "sora_robot/assets/sora_avatar.glb"