# sora_robot/main.py
#
# Este é o ponto de entrada principal para o robô Sora.
# Ele orquestra o fluxo de informações, integrando todos os módulos
# de percepção, processamento, geração de resposta e execução de ações.
# Agora, também gerencia a execução do servidor API para o frontend.

import time
import asyncio # Necessário para executar funções assíncronas.
import threading # Para executar o servidor Flask em uma thread separada.
import queue # Para comunicação segura entre threads (frames de vídeo).
import base64 # Para codificar frames de vídeo para transmissão web.
import cv2 # Para codificação de imagem JPEG.

from config import API_KEYS, DATA_PATHS, AVATAR_ANIMATIONS_PATH, API_SERVER_HOST, API_SERVER_PORT
from utils.logger import setup_logger
from data.user_profiles.profiles import UserProfileManager
from data.knowledge_base.event_info import EventInfoManager
from data.collected_data.interaction_logs import InteractionLogger
from data.collected_data.learning_data_manager import LearningDataManager
from vision_processing.camera_handler import CameraHandler
from vision_processing.facial_recognition import FacialRecognition
from vision_processing.emotion_analysis import EmotionAnalysis
from vision_processing.body_pose_estimation import BodyPoseEstimation
from audio_processing.microphone_handler import MicrophoneHandler
from audio_processing.speech_recognition import SpeechRecognition
from audio_processing.audio_analysis import AudioAnalysis
from nlp.sentiment_analysis import SentimentAnalysis
from nlp.intent_recognition import IntentRecognition
from nlp.dialogue_manager import DialogueManager
from response_generation.llm_integration import LLMIntegration
from response_generation.avatar_animation import AvatarAnimation
from response_generation.video_animation_player import VideoAnimationPlayer
from action_execution.speech_synthesis import SpeechSynthesis
from action_execution.movement_control import MovementControl
from backend_api.api_server import start_api_server # Importa a função para iniciar o servidor API.
from flask_socketio import SocketIO # Para emitir eventos WebSocket para o frontend.

# Configura o logger global para este arquivo e módulos relacionados.
logger = setup_logger(__name__)

# Fila global para frames de vídeo (comunicação entre a thread principal e a thread da API).
# Isso permite que a thread principal coloque frames aqui e a thread da API os retire para enviar ao frontend.
# maxsize=1 para garantir que sempre peguemos o frame mais recente e não acumulemos atrasos.
global_frame_queue = queue.Queue(maxsize=1)

# Variáveis globais para compartilhar instâncias de objetos entre threads.
# Usadas principalmente pela API para interagir com a lógica do Sora.
sora_instance = None
sora_llm_integration = None
sora_dialogue_manager = None
sora_audio_analysis = None
sora_speech_recognition = None
sora_sentiment_analysis = None
sora_intent_recognition = None
sora_speech_synthesis = None
sora_avatar_animation = None


class SoraRobot:
    """
    Classe principal que encapsula a lógica de operação do robô Sora.
    Gerencia a inicialização e o ciclo de vida de todos os componentes do robô.
    """
    def __init__(self, frame_queue: queue.Queue, socketio_instance: SocketIO):
        """
        Construtor da classe SoraRobot.
        Inicializa todas as dependências e módulos necessários para o funcionamento do robô.

        Args:
            frame_queue (queue.Queue): Fila para comunicação de frames de vídeo com o servidor API.
            socketio_instance (SocketIO): Instância do SocketIO do servidor API para emitir eventos.
        """
        logger.info("Inicializando o robô Sora...")
        self.frame_queue = frame_queue
        self.socketio = socketio_instance

        # --- Inicialização dos Gerenciadores de Dados ---
        self.user_profile_manager = UserProfileManager(DATA_PATHS['user_profiles'])
        self.event_info_manager = EventInfoManager(DATA_PATHS['knowledge_base'])
        self.interaction_logger = InteractionLogger(DATA_PATHS['interaction_logs'])
        self.learning_data_manager = LearningDataManager(DATA_PATHS['learning_data'])


        # --- Inicialização dos Módulos de Entrada (Sensores) ---
        self.camera_handler = CameraHandler()
        self.microphone_handler = MicrophoneHandler()

        # --- Inicialização dos Módulos de Processamento de Percepção ---
        self.facial_recognition = FacialRecognition()
        self.emotion_analysis = EmotionAnalysis()
        self.body_pose_estimation = BodyPoseEstimation()
        self.speech_recognition = SpeechRecognition(api_key=API_KEYS['google_speech_to_text'])
        self.audio_analysis = AudioAnalysis()
        self.sentiment_analysis = SentimentAnalysis()
        self.intent_recognition = IntentRecognition()
        self.dialogue_manager = DialogueManager(user_profile_manager=self.user_profile_manager,
                                                event_info_manager=self.event_info_manager,
                                                interaction_logger=self.interaction_logger)

        # --- Inicialização dos Módulos de Saída (Ações) ---
        self.llm_integration = LLMIntegration(api_key=API_KEYS['gemini'])
        self.video_animation_player = VideoAnimationPlayer()
        self.avatar_animation = AvatarAnimation(self.video_animation_player, AVATAR_ANIMATIONS_PATH)
        self.speech_synthesis = SpeechSynthesis(api_key=API_KEYS['google_text_to_speech'])
        self.movement_control = MovementControl()

        # Define as instâncias globais para acesso pela API.
        global sora_instance, sora_llm_integration, sora_dialogue_manager, sora_audio_analysis, \
            sora_speech_recognition, sora_sentiment_analysis, sora_intent_recognition, sora_speech_synthesis, \
            sora_avatar_animation
        sora_instance = self # A própria instância do robô.
        sora_llm_integration = self.llm_integration
        sora_dialogue_manager = self.dialogue_manager
        sora_audio_analysis = self.audio_analysis
        sora_speech_recognition = self.speech_recognition
        sora_sentiment_analysis = self.sentiment_analysis
        sora_intent_recognition = self.intent_recognition
        sora_speech_synthesis = self.speech_synthesis
        sora_avatar_animation = self.avatar_animation


        logger.info("Robô Sora inicializado com sucesso!")

    async def run(self):
        """
        Inicia o ciclo de operação principal do robô Sora.
        Este loop contínuo captura dados, os processa, gera respostas e executa ações.
        """
        logger.info("Iniciando o ciclo de operação do Sora...")
        try:
            # Garante que a animação inicial (neutra) esteja sendo reproduzida ao iniciar.
            self.avatar_animation.set_neutral_state()

            while True:
                # --- 1. Captura de Dados Sensoriais ---
                video_frame = self.camera_handler.capture_frame()
                audio_data = self.microphone_handler.capture_audio()

                # Variáveis para armazenar os resultados de percepção para a interação atual.
                facial_emotion = None
                body_emotion = None
                audio_emotion = None
                audio_intention = None
                text_sentiment = None
                text_intent = None
                transcribed_text = ""
                response_text = ""

                # --- 2. Processamento de Imagens ---
                if video_frame is not None:
                    # Detecção facial e análise de emoções faciais.
                    faces = self.facial_recognition.detect_faces(video_frame)
                    if faces:
                        facial_emotion = self.emotion_analysis.analyze_facial_emotion(faces[0])
                        logger.info(f"Emoção facial detectada: {facial_emotion}")
                        self.avatar_animation.update_facial_expression(facial_emotion)
                    else:
                        logger.info("Nenhuma face detectada.")
                        self.avatar_animation.set_neutral_state()

                    # Estimativa de pose corporal e análise de emoções corporais.
                    body_landmarks = self.body_pose_estimation.estimate_pose(video_frame)
                    if body_landmarks is not None:
                        body_emotion = self.emotion_analysis.analyze_body_emotion(body_landmarks)
                        logger.info(f"Emoção corporal detectada: {body_emotion}")
                        self.avatar_animation.update_body_gesture(body_emotion)
                    else:
                        logger.info("Nenhum corpo detectado ou landmarks insuficientes.")


                    # Combina as emoções (facial e corporal) para uma emoção geral do usuário.
                    combined_emotion = self.emotion_analysis.combine_emotions(facial_emotion, body_emotion)
                    if combined_emotion:
                        logger.info(f"Emoção combinada: {combined_emotion}")
                        self.dialogue_manager.update_current_emotion(combined_emotion)

                    # Obtém o frame atual da animação do avatar.
                    # Este frame será enviado para o frontend via WebSocket.
                    avatar_frame = self.video_animation_player.get_current_frame()
                    if avatar_frame is not None:
                        # Codifica o frame para JPEG e depois para base64 para transmissão via WebSocket.
                        # `cv2.imencode` é mais eficiente que salvar para arquivo e ler de volta.
                        ret, buffer = cv2.imencode('.jpg', avatar_frame)
                        if ret:
                            frame_base64 = base64.b64encode(buffer).decode('utf-8')
                            # Coloca o frame na fila para ser consumido pela thread da API.
                            try:
                                self.frame_queue.put_nowait(frame_base64)
                            except queue.Full:
                                # Se a fila estiver cheia, descarta o frame mais antigo para pegar o mais recente.
                                self.frame_queue.get_nowait()
                                self.frame_queue.put_nowait(frame_base64)

                # --- 3. Análise de Áudio (se áudio direto do microfone for usado aqui, ou virá do frontend) ---
                # A lógica de processamento de áudio direto do microfone pode ser movida para
                # o frontend (captura de áudio no navegador e envio via WebSocket) ou
                # continuar aqui se o Sora tiver um microfone físico dedicado.
                # Por simplicidade, vamos manter a lógica aqui por enquanto,
                # mas note que para o frontend interagir, o áudio do usuário virá do navegador.
                if audio_data is not None:
                    transcribed_text = await self.speech_recognition.transcribe(audio_data)
                    if transcribed_text:
                        logger.info(f"Texto transcrito: '{transcribed_text}'")
                        audio_emotion, audio_intention = self.audio_analysis.analyze(audio_data)
                        logger.info(f"Emoção de áudio: {audio_emotion}, Intenção de áudio: {audio_intention}")

                        text_sentiment = self.sentiment_analysis.analyze(transcribed_text)
                        text_intent = self.intent_recognition.recognize(transcribed_text)
                        logger.info(f"Sentimento: {text_sentiment}, Intenção: {text_intent}")

                        context = self.dialogue_manager.get_current_context()
                        response_text = await self.llm_integration.generate_response(transcribed_text, context)
                        logger.info(f"Resposta gerada: '{response_text}'")

                        self.avatar_animation.synchronize_mouth_with_speech(response_text)

                        self.dialogue_manager.update_dialogue_state(
                            user_input=transcribed_text,
                            sora_response=response_text,
                            facial_emotion=facial_emotion,
                            body_emotion=body_emotion,
                            audio_emotion=audio_emotion,
                            audio_intention=audio_intention,
                            text_sentiment=text_sentiment,
                            text_intent=text_intent
                        )

                        # 5. Execução de Ação
                        await self.speech_synthesis.speak(response_text)
                        if text_intent == "navegar":
                            self.movement_control.move_forward()

                # Pequena pausa para evitar sobrecarga de CPU e permitir um loop de processamento fluido.
                await asyncio.sleep(0.05)

        except KeyboardInterrupt:
            logger.info("Desligando o Sora...")
        except Exception as e:
            logger.exception(f"Ocorreu um erro fatal no Sora: {e}")
        finally:
            self.camera_handler.release()
            self.microphone_handler.release()
            self.video_animation_player.release()
            logger.info("Sora desligado.")


# Função para iniciar o servidor Flask em uma thread separada.
# Isso permite que o loop principal do robô (asyncio) continue executando
# enquanto o servidor web espera por requisições.
def start_flask_thread(sora_instance_ref, frame_queue_ref):
    """
    Inicia o servidor Flask em uma nova thread.

    Args:
        sora_instance_ref (SoraRobot): Referência à instância principal do SoraRobot.
        frame_queue_ref (queue.Queue): Referência à fila de frames para streaming.
    """
    logger.info("Iniciando a thread do servidor Flask...")
    # Passa a fila de frames para a função de inicialização da API.
    # A instância do socketio é criada dentro de api_server.py.
    start_api_server(sora_instance_ref, frame_queue_ref)

if __name__ == "__main__":
    # Inicializa o SocketIO que será usado pelo servidor Flask e pela instância do SoraRobot.
    # Permite a comunicação WebSocket entre o backend e o frontend.
    # Note: socketio.run() é quem inicia o servidor web. Aqui, criamos a instância para passar ao Sora.
    socketio_app = SocketIO(cors_allowed_origins="*") # Permite CORS de qualquer origem para facilitar o desenvolvimento.

    # Cria a instância do SoraRobot, passando a fila de frames e a instância do SocketIO.
    sora = SoraRobot(global_frame_queue, socketio_app)

    # Inicia o servidor Flask em uma thread separada.
    # Isso é feito para que o loop assíncrono do Sora (sora.run()) possa continuar a execução
    # na thread principal, enquanto o servidor Flask lida com as requisições web.
    api_thread = threading.Thread(target=start_flask_thread, args=(sora, global_frame_queue))
    api_thread.daemon = True # Torna a thread um daemon para que ela encerre automaticamente com o programa principal.
    api_thread.start()
    logger.info("Thread da API iniciada.")

    # Executa o loop principal assíncrono do Sora.
    asyncio.run(sora.run())

    logger.info("Programa principal do Sora finalizado.")