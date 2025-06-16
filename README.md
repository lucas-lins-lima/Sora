# Visão Geral do Projeto

O projeto Sora é uma iniciativa de Iniciação Científica focada na construção de um robô interativo e empático. O objetivo principal é criar uma experiência mais humana, engajadora e interativa em grandes eventos, redefinindo a forma como as pessoas se conectam e interagem em ambientes dinâmicos. Sora busca humanizar as interações tecnológicas, superando a impessoalidade comum em grandes aglomerações e oferecendo uma conexão genuína e personalizada que ressoa emocionalmente com os participantes.

O robô Sora será capaz de:
* **Capturar Dados Sensoriais:** Processamento de vídeo (detecção facial e corporal) e áudio (fala).
* **Processar Informações:** Analisar expressões faciais, emoções corporais, converter fala em texto, identificar emoções e intenções na voz e no texto.
* **Gerar Respostas:** Utilizar modelos de linguagem avançados (LLMs) para gerar respostas contextualmente relevantes.
* **Executar Ações:** Sincronizar animações de avatar (expressões faciais, movimentos labiais, gestos corporais) e, futuramente, controlar movimentos físicos do robô, além de síntese de fala.
* **Aprender e Persistir Dados:** Coletar e armazenar logs detalhados de interações para análise e potencial retreinamento de modelos, permitindo que a Sora melhore suas respostas ao longo do tempo.
* **Interface Imersiva:** Oferecer uma interface de usuário (front-end web) para uma experiência de interação mais imersiva, com streaming de vídeo do avatar e chat em tempo real.

# 🗂️ Estrutura de Pastas e Arquivos
A arquitetura do Sora é modular, visando facilitar o desenvolvimento, a manutenção e a escalabilidade. Cada módulo tem uma responsabilidade bem definida, o que permite que atualizações em uma parte do sistema minimizem o impacto em outras.

**📁 RAIZ DO PROJETO**
```
├── .env                     # 🔧 Variáveis de ambiente (senhas, API keys, configurações)
├── config.py               # ⚙️ Configurações centralizadas do sistema
├── docker-compose.yml      # 🐳 Orquestração de múltiplos containers (app, banco, redis, nginx)
├── Dockerfile             # 📦 Receita para criar imagem Docker do Sora Robot
├── main.py                # 🎮 PONTO DE ENTRADA PRINCIPAL - inicia todo o sistema
└── requirements.txt       # 📚 Lista de todas as dependências Python necessárias
```
**🤖 action_execution/ - Execução de Ações Físicas**
```
├── movement_control.py    # 🦾 [FUTURO] Controle de movimentos robóticos/servos
├── speech_synthesis.py    # 🗣️ Síntese de voz (TTS) - converte texto em fala
└── __init__.py           # 📦 Marca como módulo Python
```
**🌐 api/ - Interface de Comunicação**
```
├── api_interface.py       # 🔌 API REST + WebSocket - ponte entre frontend e backend
└── __init__.py           # 📦 Marca como módulo Python
```
**🎤 audio_processing/ - Processamento de Áudio**
```
├── audio_analysis.py      # 🎵 Análise prosódica e emocional do áudio
├── microphone_handler.py  # 🎙️ Captura de áudio do microfone
├── speech_recognition.py  # 🗣️➡️📝 Conversão de fala em texto (STT)
└── __init__.py           # 📦 Marca como módulo Python
```
**🧠 Core/ - Núcleo do Sistema**
```
├── main_controller.py     # 🎮 Controlador principal - orquestra todos os módulos
├── system_integration.py # 🔗 Integração entre todos os componentes do sistema
└── __init__.py           # 📦 Marca como módulo Python
```
**📊 data/ - Gerenciamento de Dados**
```
├── collected_data/        # 📈 Dados coletados das interações
│   ├── interaction_logs.py     # 📝 Sistema de logging de interações
│   ├── learning_data_manager.py # 🎓 Gerenciador de dados para aprendizagem
│   ├── __init__.py             # 📦 Marca como módulo Python
│   └── raw_sensor_logs/        # 🔍 Logs brutos de sensores
│       └── __init__.py         # 📦 Marca como módulo Python
│
├── knowledge_base/        # 📚 Base de conhecimento
│   ├── event_info.json         # 📋 [FUTURO] Informações sobre eventos
│   └── __init__.py             # 📦 Marca como módulo Python
│
└── user_profiles/         # 👥 Perfis de usuários
    ├── profiles.json           # 👤 Dados dos perfis de usuários
    └── __init__.py             # 📦 Marca como módulo Python
```
**🌍 frontend/ - Interface Web**
```
├── index.html            # 📄 Página web principal - interface do usuário
├── script.js            # ⚡ JavaScript - lógica de interação e WebSocket
├── style.css            # 🎨 Estilos visuais - design responsivo e moderno
└── assets/              # 🖼️ Recursos estáticos (imagens, ícones, etc.)
```
**📊 monitoring/ - Observabilidade**
```
├── prometheus.yml        # 📈 Configuração de coleta de métricas
└── grafana/             # 📊 Visualização de dados
    ├── sora-dashboard.json    # 📋 Dashboard principal com métricas de IA
    └── dashboards/            # 📊 Diretório para dashboards adicionais
```
**🌐 nginx/ - Proxy Reverso**
```
└── nginx.conf           # ⚖️ Configuração do load balancer e proxy
```
**🧠 nlp/ - Processamento de Linguagem Natural**
```
├── dialogue_manager.py   # 💬 Gerenciamento de contexto conversacional
├── intent_recognition.py # 🎯 Reconhecimento de intenções do usuário
├── sentiment_analysis.py # 💭 Análise de sentimento das mensagens
└── __init__.py          # 📦 Marca como módulo Python
```
**🤖 response_generation/ - Geração de Respostas**
```
├── avatar_animation.py        # 🎭 Animações faciais e corporais do avatar
├── llm_integration.py         # 🧠 Integração com LLMs (GPT, Claude, Gemini)
├── video_animation_player.py  # 🎬 Player de animações de vídeo
└── __init__.py               # 📦 Marca como módulo Python
```
**🛠️ scripts/ - Automação**
```
└── setup.sh             # 🚀 Script de instalação automática completa
```
**🔧 utils/ - Utilitários**
```
├── constants.py         # 📊 Constantes globais do sistema
├── helpers.py          # 🛠️ Funções auxiliares reutilizáveis
├── logger.py           # 📝 Sistema de logging estruturado
└── __init__.py         # 📦 Marca como módulo Python
```
**👁️ vision_processing/ - Visão Computacional**
```
├── body_pose_estimation.py # 🏃 Estimativa de pose corporal
├── camera_handler.py       # 📷 Captura de frames da câmera
├── emotion_analysis.py     # 😊 Detecção de emoções faciais
├── facial_recognition.py   # 👤 Reconhecimento e detecção facial
└── __init__.py            # 📦 Marca como módulo Python
```
