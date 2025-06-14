# sora_robot/backend_api/api_server.py
#
# Este módulo define e inicia o servidor da API Flask para o robô Sora.
# Ele expõe endpoints RESTful e WebSocket para que o front-end possa interagir
# com a lógica de backend do Sora em tempo real.

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS # Para permitir requisições de origens diferentes (do frontend).
from flask_socketio import SocketIO, emit # Para comunicação WebSocket.
import threading
import time
import base64
import cv2
import queue # Para acessar a fila de frames do main.py.

from utils.logger import setup_logger
from config import API_SERVER_HOST, API_SERVER_PORT # Importa configurações do servidor.

# Configura o logger para este módulo.
logger = setup_logger(__name__)

# Instância do Flask e SocketIO.
app = Flask(__name__, static_folder='../../frontend', static_url_path='/') # Define a pasta 'frontend' como pasta estática.
# Permite CORS para todas as origens (para desenvolvimento, pode ser restrito em produção).
CORS(app)
# Inicializa o SocketIO com a aplicação Flask.
socketio = SocketIO(app, cors_allowed_origins="*") # Permite CORS para WebSockets também.

# Variáveis para armazenar referências à instância do SoraRobot e à fila de frames.
# Essas variáveis serão preenchidas pela função start_api_server().
sora_robot_instance = None
frame_queue_ref = None

def start_api_server(sora_instance_param, frame_queue_param):
    """
    Função para iniciar o servidor Flask.
    Recebe a instância do SoraRobot e a fila de frames para permitir a comunicação.

    Args:
        sora_instance_param (object): A instância principal do SoraRobot.
        frame_queue_param (queue.Queue): A fila global de frames de vídeo.
    """
    global sora_robot_instance, frame_queue_ref
    sora_robot_instance = sora_instance_param
    frame_queue_ref = frame_queue_param
    logger.info(f"Servidor API Flask inicializando em http://{API_SERVER_HOST}:{API_SERVER_PORT}")
    # Usa socketio.run() que integra o Flask e o SocketIO.
    # Permite que o servidor escute em todas as interfaces ('0.0.0.0') ou no host especificado.
    socketio.run(app, host=API_SERVER_HOST, port=API_SERVER_PORT, debug=False, allow_unsafe_werkzeug=True)
    # debug=False para produção. allow_unsafe_werkzeug=True pode ser necessário em alguns ambientes.

# --- Endpoints RESTful ---

@app.route('/')
def serve_index():
    """
    Serve o arquivo index.html do diretório frontend.
    """
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static_files(path):
    """
    Serve outros arquivos estáticos (CSS, JS, assets) do diretório frontend.
    """
    return send_from_directory(app.static_folder, path)


@app.route('/api/chat', methods=['POST'])
def chat_endpoint():
    """
    Endpoint para o frontend enviar a entrada de texto do usuário e receber a resposta do Sora.
    """
    if not sora_robot_instance:
        logger.error("Instância do SoraRobot não disponível para o endpoint /api/chat.")
        return jsonify({"response": "Erro interno: Robô não inicializado."}), 500

    data = request.json
    user_input = data.get('message')
    user_id = data.get('userId', 'frontend_user') # ID do usuário, pode vir do frontend ou ser padrão.

    if not user_input:
        return jsonify({"response": "Mensagem vazia"}), 400

    logger.info(f"Recebida mensagem do frontend para o usuário {user_id}: '{user_input}'")

    # Em um ambiente real, você faria uma chamada assíncrona para o loop do Sora.
    # Para simplicidade no esqueleto, vamos usar os objetos globais diretamente.
    # O ideal seria usar uma fila de mensagens para comunicar com a thread principal do Sora.
    response_text = ""
    try:
        # Define o user_id no dialogue_manager para registrar a interação corretamente.
        sora_robot_instance.dialogue_manager.current_user_id = user_id

        # Simula o processamento do Sora para uma resposta de chat.
        # Estas chamadas (NLP, LLM) seriam idealmente executadas no loop principal do Sora,
        # mas aqui as chamamos diretamente para demonstrar a interação da API.
        # Note que a chamada ao LLMIntegration é assíncrona, então precisamos de asyncio.
        # Como estamos em uma thread Flask síncrona, usamos loop.run_until_complete.
        # Importante: Tenha cuidado ao chamar funções assíncronas de threads síncronas.
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Simula a transcrição de áudio e a análise para alimentar o diálogo.
        # Em um cenário real, o frontend enviaria o áudio ou o texto transcrito.
        # Aqui, usamos o user_input como texto já transcrito.
        audio_emotion, audio_intention = sora_robot_instance.audio_analysis.analyze(b"dummy_audio_data") # Dados dummy
        text_sentiment = sora_robot_instance.sentiment_analysis.analyze(user_input)
        text_intent = sora_robot_instance.intent_recognition.recognize(user_input)

        context = sora_robot_instance.dialogue_manager.get_current_context()
        response_text = loop.run_until_complete(sora_robot_instance.llm_integration.generate_response(user_input, context))
        loop.close()

        # Atualiza o estado do diálogo e loga a interação.
        sora_robot_instance.dialogue_manager.update_dialogue_state(
            user_input=user_input,
            sora_response=response_text,
            facial_emotion=sora_robot_instance.dialogue_manager.current_emotion, # Usa a emoção que Sora percebeu
            body_emotion=None, # Para simplicidade, não repassa aqui da API por enquanto
            audio_emotion=audio_emotion,
            audio_intention=audio_intention,
            text_sentiment=text_sentiment,
            text_intent=text_intent
        )

        # Dispara a animação labial e a síntese de fala (TTS) para a resposta.
        # Estas também seriam idealmente executadas na thread principal do Sora.
        sora_robot_instance.avatar_animation.synchronize_mouth_with_speech(response_text)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(sora_robot_instance.speech_synthesis.speak(response_text))
        loop.close()

    except Exception as e:
        logger.error(f"Erro ao processar mensagem do frontend: {e}", exc_info=True)
        response_text = "Desculpe, tive um problema ao processar sua solicitação."

    return jsonify({"response": response_text})

# --- WebSocket Endpoints ---

@socketio.on('connect')
def test_connect():
    """
    Evento disparado quando um cliente WebSocket se conecta.
    """
    logger.info("Cliente WebSocket conectado!")
    emit('status', {'data': 'Conectado ao Sora API WebSocket!'})
    # Inicia o streaming de frames para o cliente recém-conectado.
    threading.Thread(target=send_frames_to_frontend).start()


@socketio.on('disconnect')
def test_disconnect():
    """
    Evento disparado quando um cliente WebSocket se desconecta.
    """
    logger.info("Cliente WebSocket desconectado.")

def send_frames_to_frontend():
    """
    Função que envia frames de vídeo para o frontend via WebSocket.
    Lê frames da fila global e os emite como eventos 'video_frame'.
    """
    logger.info("Iniciando streaming de frames via WebSocket...")
    while True:
        if not sora_robot_instance or not sora_robot_instance.video_animation_player:
            logger.warning("Player de vídeo do SoraRobot não disponível para streaming. Aguardando...")
            time.sleep(1)
            continue

        try:
            # Tenta pegar um frame da fila global (não bloqueia).
            frame_data = frame_queue_ref.get(timeout=1) # Espera 1 segundo por um frame.
            # Emite o frame codificado em base64 como um evento WebSocket.
            socketio.emit('video_frame', {'image': frame_data})
            #logger.debug("Frame de vídeo emitido via WebSocket.")
        except queue.Empty:
            # logger.debug("Fila de frames vazia, aguardando próximo frame...")
            pass # Sem frames disponíveis na fila.
        except Exception as e:
            logger.error(f"Erro ao enviar frame via WebSocket: {e}")
            break # Quebra o loop em caso de erro.
        time.sleep(0.033) # Aproximadamente 30 FPS (1 segundo / 30 frames). Ajuste conforme o FPS do seu vídeo.