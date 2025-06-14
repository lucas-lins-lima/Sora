# Visão Geral do Projeto

O projeto Sora é uma iniciativa de Iniciação Científica focada na construção de um robô interativo e empático. O objetivo principal é criar uma experiência mais humana, engajadora e interativa em grandes eventos, redefinindo a forma como as pessoas se conectam e interagem em ambientes dinâmicos. Sora busca humanizar as interações tecnológicas, superando a impessoalidade comum em grandes aglomerações e oferecendo uma conexão genuína e personalizada que ressoa emocionalmente com os participantes.

O robô Sora será capaz de:
* **Capturar Dados Sensoriais:** Processamento de vídeo (detecção facial e corporal) e áudio (fala).
* **Processar Informações:** Analisar expressões faciais, emoções corporais, converter fala em texto, identificar emoções e intenções na voz e no texto.
* **Gerar Respostas:** Utilizar modelos de linguagem avançados (LLMs) para gerar respostas contextualmente relevantes.
* **Executar Ações:** Sincronizar animações de avatar (expressões faciais, movimentos labiais, gestos corporais) e, futuramente, controlar movimentos físicos do robô, além de síntese de fala.
* **Aprender e Persistir Dados:** Coletar e armazenar logs detalhados de interações para análise e potencial retreinamento de modelos, permitindo que a Sora melhore suas respostas ao longo do tempo.
* **Interface Imersiva:** Oferecer uma interface de usuário (front-end web) para uma experiência de interação mais imersiva, com streaming de vídeo do avatar e chat em tempo real.

# Estrutura de Pastas e Arquivos
A arquitetura do Sora é modular, visando facilitar o desenvolvimento, a manutenção e a escalabilidade. Cada módulo tem uma responsabilidade bem definida, o que permite que atualizações em uma parte do sistema minimizem o impacto em outras.
```
sora_robot/                                 # Diretório raiz do projeto Sora. Contém toda a lógica e dados do robô.
├── main.py                                 # Ponto de entrada principal do robô. Orquestra a execução de todos os módulos.
├── config.py                               # Armazena configurações globais, chaves de API e caminhos de arquivos.
├── utils/                                  # Utilitários e funções de suporte que podem ser usadas em múltiplos módulos.
│   ├── __init__.py                         # Torna 'utils' um pacote Python.
│   └── logger.py                           # Módulo para configuração e gestão centralizada de logs.
│   └── constants.py                        # Definições de constantes (emoções, intenções, limiares, etc.) usadas globalmente.
├── data/                                   # Armazena dados persistentes e coletados pelo robô.
│   ├── user_profiles/                      # Gerencia perfis de usuários com quem o Sora interage.
│   │   ├── __init__.py                     # Torna 'user_profiles' um pacote Python.
│   │   └── profiles.json                   # Arquivo JSON para armazenar dados estruturados de perfis de usuário (ex: histórico de interações, preferências).
│   ├── knowledge_base/                     # Contém a base de conhecimento estática sobre o evento ou outros tópicos.
│   │   ├── __init__.py                     # Torna 'knowledge_base' um pacote Python.
│   │   └── event_info.json                 # Arquivo JSON com informações pré-definidas sobre o evento (agenda, locais, etc.).
│   └── collected_data/                     # Destinado a armazenar dados brutos e processados coletados em tempo real para análise e aprendizado.
│       ├── __init__.py                     # Torna 'collected_data' um pacote Python.
│       ├── interaction_logs.py             # Módulo para registrar logs detalhados de cada interação do Sora.
│       ├── learning_data_manager.py        # Módulo para gerenciar dados formatados especificamente para treinamento de modelos de ML.
│       └── raw_sensor_logs/                # Opcional: Para logs brutos (vídeo/áudio) dos sensores, se a persistência for necessária.
│           └── __init__.py                 # Torna 'raw_sensor_logs' um pacote Python.
├── vision_processing/                      # Módulo dedicado ao processamento de dados visuais (câmera).
│   ├── __init__.py                         # Torna 'vision_processing' um pacote Python.
│   ├── camera_handler.py                   # Gerencia a captura de frames da câmera.
│   ├── facial_recognition.py               # Lida com a detecção e, opcionalmente, o reconhecimento facial.
│   ├── emotion_analysis.py                 # Analisa emoções a partir de dados faciais e corporais.
│   └── body_pose_estimation.py             # Estima a pose corporal (esqueleto) para identificar gestos e postura.
├── audio_processing/                       # Módulo dedicado ao processamento de dados de áudio (microfone).
│   ├── __init__.py                         # Torna 'audio_processing' um pacote Python.
│   ├── microphone_handler.py               # Gerencia a captura de dados de áudio do microfone.
│   ├── speech_recognition.py               # Converte fala em texto (Automatic Speech Recognition - ASR).
│   └── audio_analysis.py                   # Analisa características do áudio para identificar emoções e intenções vocais.
├── nlp/                                    # Módulo para Processamento de Linguagem Natural (NLP).
│   ├── __init__.py                         # Torna 'nlp' um pacote Python.
│   ├── sentiment_analysis.py               # Analisa o sentimento (positivo, negativo, neutro) do texto.
│   ├── intent_recognition.py               # Identifica a intenção principal da fala do usuário.
│   └── dialogue_manager.py                 # Gerencia o fluxo da conversa, contexto e histórico do diálogo.
├── response_generation/                    # Módulo responsável por gerar as respostas e animações do Sora.
│   ├── __init__.py                         # Torna 'response_generation' um pacote Python.
│   ├── llm_integration.py                  # Integração com Large Language Models (LLMs) para geração de texto dinâmico.
│   ├── avatar_animation.py                 # Controlador de alto nível para as animações do avatar 3D (expressões faciais, gestos).
│   └── video_animation_player.py           # Gerencia a reprodução de arquivos de vídeo MP4 para animação fluida do avatar.
├── action_execution/                       # Módulo para a execução de ações físicas e verbais do robô.
│   ├── __init__.py                         # Torna 'action_execution' um pacote Python.
│   ├── speech_synthesis.py                 # Converte texto em fala (Text-to-Speech - TTS).
│   └── movement_control.py                 # Controla os movimentos físicos do robô (motores, navegação).
├── backend_api/                            # NOVO: Módulo para a API RESTful e WebSocket do backend.
│   ├── __init__.py                         # Torna 'backend_api' um pacote Python.
│   └── api_server.py                       # Servidor Flask que expõe endpoints para comunicação com o frontend.
├── frontend/                               # NOVO: Diretório para o código do front-end web (HTML, CSS, JavaScript).
│   ├── index.html                          # Página HTML principal que conterá a interface do usuário.
│   ├── style.css                           # Arquivo CSS para estilização da interface.
│   ├── script.js                           # Arquivo JavaScript para lógica do cliente e comunicação com o backend.
│   └── assets/                             # Pasta para ativos estáticos do frontend (imagens, ícones, etc.).
└── requirements.txt                        # Lista todas as dependências Python do projeto para instalação via pip.
```
# Como Fazer o Projeto Funcionar (Passo a Passo)
Siga estas instruções para configurar e rodar o projeto Sora em seu ambiente local.

**1. Pré-requisitos**
Certifique-se de ter o seguinte instalado em seu sistema:

* **Python 3.8+:** Recomendado Python 3.8 ou versão superior.
* **pip:** O gerenciador de pacotes do Python (geralmente vem com o Python).
* **Git** (Opcional, mas recomendado para controle de versão).
* **mpg123 (para Linux/macOS) ou uma ferramenta similar (para Windows):** Usado para reproduzir arquivos de áudio MP3 gerados pela síntese de fala.
** **Linux (Ubuntu/Debian):** sudo apt-get install mpg123
** **macOS:** brew install mpg123 (se tiver Homebrew)
** **Windows:** Você pode precisar de ffplay (do pacote ffmpeg) ou usar uma biblioteca Python como playsound (que pode exigir mais configuração).

**2. Configuração do Projeto**
* **1.Crie o Diretório do Projeto:**
Crie uma pasta principal para o seu projeto, por exemplo, sora_robot/.
* **2.Crie a Estrutura de Pastas:**
Dentro de sora_robot/, crie manualmente todas as pastas e subpastas conforme a "Estrutura de Pastas e Arquivos" detalhada acima. Lembre-se de criar os arquivos __init__.py vazios em cada subpasta para que o Python os reconheça como pacotes.

Script para criar a estrutura de pastas e arquivos:
```
#!/bin/bash
echo "Iniciando a criação da estrutura de pastas para o projeto Sora..."

# 1. Cria o diretório raiz do projeto
mkdir -p sora_robot
cd sora_robot

# 2. Cria os diretórios principais
mkdir -p utils data vision_processing audio_processing nlp response_generation action_execution backend_api frontend assets

# 3. Cria subdiretórios dentro de 'data'
mkdir -p data/user_profiles
mkdir -p data/knowledge_base
mkdir -p data/collected_data
mkdir -p data/collected_data/raw_sensor_logs

# 4. Cria subdiretórios dentro de 'frontend' e 'assets'
mkdir -p frontend/assets
mkdir -p assets/avatar_animations # Onde os vídeos de animação MP4 serão armazenados

# 5. Cria os arquivos __init__.py para cada pacote Python
echo "Criando arquivos __init__.py..."
touch utils/__init__.py
touch data/__init__.py
touch data/user_profiles/__init__.py
touch data/knowledge_base/__init__.py
touch data/collected_data/__init__.py
touch data/collected_data/raw_sensor_logs/__init__.py
touch vision_processing/__init__.py
touch audio_processing/__init__.py
touch nlp/__init__.py
touch response_generation/__init__.py
touch action_execution/__init__.py
touch backend_api/__init__.py

echo "Estrutura de pastas e arquivos __init__.py criados com sucesso!"
echo "Agora você pode prosseguir com a criação dos arquivos .py principais e de dados."
```
* **3.Crie os Arquivos Python e de Configuração:**
Crie cada arquivo .py e .txt nas suas respectivas pastas e cole o código fornecido nos artefatos anteriores (main.py, config.py, logger.py, etc.).

* **4.Crie os Arquivos do Frontend:**
Crie index.html, style.css e script.js na pasta frontend/ e cole o código fornecido nos artefatos.

* **5.Configure as Chaves de API (config.py):**
Abra o arquivo sora_robot/config.py e substitua os placeholders ("SUA_CHAVE_API_...") pelas suas chaves de API reais para os serviços do Google (Speech-to-Text, Text-to-Speech) e Gemini.
```
# sora_robot/config.py
API_KEYS = {
    "google_speech_to_text": "SUA_CHAVE_API_GOOGLE_SPEECH_TO_TEXT_AQUI",
    "google_text_to_speech": "SUA_CHAVE_API_GOOGLE_TEXT_TO_SPEECH_AQUI",
    "gemini": "SUA_CHAVE_API_GEMINI_AQUI",
}
```
Você também pode ajustar CAMERA_INDEX, MICROPHONE_DEVICE_INDEX, API_SERVER_HOST, API_SERVER_PORT e LOG_LEVEL se necessário.

* **6.Crie Arquivos de Dados Iniciais:**
Crie os seguintes arquivos vazios ou com conteúdo inicial básico para o sistema de dados:

- sora_robot/data/user_profiles/profiles.json
```
  {}
```
- sora_robot/data/knowledge_base/event_info.json
```
  {
  "agenda": "A agenda completa está disponível no site do evento.",
  "localizacao_estandes": "Os estandes principais estão no Pavilhão Azul.",
  "contato_suporte": "Para suporte, procure a equipe com crachás azuis.",
  "redes_sociais": "Use #SoraEvent nas redes sociais para compartilhar sua experiência!"
}
```
- sora_robot/data/collected_data/interaction_logs.jsonl (arquivo vazio, será preenchido automaticamente)

- sora_robot/data/collected_data/learning_data.jsonl (arquivo vazio, será preenchido automaticamente)

* **7.Crie Arquivos de Animação do Avatar:**
Na pasta sora_robot/assets/avatar_animations/, você precisará colocar seus arquivos de vídeo MP4 que representam as animações do avatar. Certifique-se de que os nomes dos arquivos correspondem aos mapeamentos definidos em response_generation/avatar_animation.py. Por exemplo, se animation_map["neutral"] aponta para neutral.mp4, certifique-se de que sora_robot/assets/avatar_animations/neutral.mp4 exista.

Exemplos de nomes de arquivos MP4 esperados:
```
neutral.mp4
speaking_loop.mp4
happy.mp4
sad.mp4
angry.mp4
surprised.mp4
fear.mp4
disgust.mp4
agitated_gesture.mp4
attentive_pose.mp4
wave_gesture.mp4
```
**3. Instalação das Dependências**
Abra seu terminal, navegue até o diretório raiz do projeto (sora_robot/) e execute o seguinte comando para instalar todas as bibliotecas Python necessárias:
```
pip install -r requirements.txt
```
**4. Executando o Projeto**
Após a configuração, você pode iniciar o robô Sora e o servidor da API.

- 1.Inicie o Backend:
No terminal, a partir do diretório sora_robot/, execute:
```
python main.py
```
- 2.Acesse o Frontend:
Abra seu navegador web (Google Chrome, Firefox, etc.) e navegue para o seguinte endereço:
```
http://127.0.0.1:5000/
```
A interface do Sora deverá carregar, exibir o avatar e permitir que você interaja via chat.

# Mapa de Conexões de Arquivos
Entender como os arquivos do projeto Sora se conectam é crucial para depuração e atualização. O main.py atua como o orquestrador central, instanciando e chamando os métodos dos outros módulos. As setas (→) indicam a direção da dependência ou do fluxo de informações.
```
+-------------------+                                                                +------------------+
|   frontend/       |                                                                |   backend_api/   |
|                   |                                                                |   api_server.py  |
| - index.html      |<---------- HTTPS/WebSocket ----------------------------------- |                  |
| - style.css       |                                                                |------------------+
| - script.js       |                                                                         | (global vars)
|                   |                                                                         |
+-------------------+                                                                         v
                                                                                           +-------------+
                                                                                           |   main.py   |
                                                                                           |  (Orquestra)|
                                                                                           +-------------+
                                                                                                  |
                                                      +---------------------------------------------+---------------------------------------------+
                                                      |                                           |                                               |
                                                      v                                           v                                               v
+-----------------------+           +-----------------------------+           +-----------------------+           +-----------------------+           +-----------------------+
|  utils/               |           |  data/                      |           |  vision_processing/   |           |  audio_processing/      |           |  nlp/                   |
|                       |           |                             |           |                       |           |                         |           |                         |
| - logger.py   <-------+-----------+ - user_profiles/            |           | - camera_handler.py   |<----------+ - microphone_handler.py |           | - sentiment_analysis.py |
| - constants.py<-------------------+ - knowledge_base/           |           | - facial_recognition.py --+       |                         |           | - intent_recognition.py |
+-----------------------+           | - collected_data/           |           | - body_pose_estimation.py--+      | - speech_recognition.py |<----------+                         |
                                    |   - interaction_logs.py     |           |                       |           | - audio_analysis.py     |           |                         |
                                    |   - learning_data_manager.py|      <-------+ emotion_analysis.py <--+-----------+-----------------------+         |+-----------------------+|
                                    +-----------------------------+                                       |                                             |
                                               ^                                                     |                                                  |
                                               | (dialogue_manager logs)                             |                                                  v
                                               |                                                     |                                   +-------------------+
                                               |                                                     |                                   |  dialogue_manager.py|
                                               |                                                     |                                   +-------------------+
                                               |                                                     |                                             |
                                               +-----------------------------------------------------+---------------------------------------------+
                                                                                                     | (Context, Estado do Diálogo)
                                                                                                     v
                                                                                           +-----------------------+
                                                                                           |  response_generation/ |
                                                                                           |                       |
                                                                                           | - llm_integration.py  |<----------------------------+
                                                                                           | - avatar_animation.py |<----------------------------+
                                                                                           |                       |
                                                                                           +-----------------------+
                                                                                                     |
                                                      +----------------------------------------------+---------------------------------------------+
                                                      |                                              |
                                                      v                                              v
                                            +-----------------------+                     +---------------------------+
                                            |  action_execution/    |                     | video_animation_player.py |
                                            |                       |                     |                           |
                                            | - speech_synthesis.py |<--------------------+ - (frames de vídeo)       |
                                            | - movement_control.py |                     +---------------------------+
                                            +-----------------------+
```
