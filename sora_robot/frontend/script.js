// sora_robot/frontend/script.js
//
// Este script JavaScript controla a interface do usuário do front-end do Sora.
// Ele lida com a interação do usuário, comunicação com o backend via WebSocket e HTTP,
// e a exibição das animações do avatar e mensagens de chat.

document.addEventListener('DOMContentLoaded', () => {
    // Referências aos elementos HTML da interface.
    const avatarImage = document.getElementById('avatarImage');
    const chatBox = document.getElementById('chatBox');
    const userInput = document.getElementById('userInput');
    const sendButton = document.getElementById('sendButton');
    const statusMessage = document.getElementById('statusMessage');

    // Configurações da API. Devem corresponder às configurações do backend (config.py).
    const API_HOST = '127.0.0.1';
    const API_PORT = 5000;
    const SOCKET_IO_URL = `http://${API_HOST}:${API_PORT}`;
    const CHAT_API_URL = `http://${API_HOST}:${API_PORT}/api/chat`;

    let userId = 'frontend_user_' + Math.random().toString(36).substring(2, 9); // Gera um ID de usuário simples.

    // --- Inicialização do Socket.IO para comunicação em tempo real (frames de vídeo) ---
    const socket = io(SOCKET_IO_URL, {
        reconnection: true,
        reconnectionAttempts: 5,
        reconnectionDelay: 1000,
        transports: ['websocket']
    });

    // Evento de conexão bem-sucedida com o WebSocket.
    socket.on('connect', () => {
        console.log('Conectado ao WebSocket do Sora!');
        statusMessage.textContent = 'Conectado. Digite sua mensagem!';
        userInput.disabled = false;
        sendButton.disabled = false;
        addMessageToChat('Sora', 'Olá! Como posso te ajudar hoje?', 'sora-message');
    });

    // Evento de desconexão do WebSocket.
    socket.on('disconnect', () => {
        console.log('Desconectado do WebSocket do Sora.');
        statusMessage.textContent = 'Desconectado. Tentando reconectar...';
        userInput.disabled = true;
        sendButton.disabled = true;
    });

    // Evento de erro do WebSocket.
    socket.on('connect_error', (error) => {
        console.error('Erro de conexão WebSocket:', error);
        statusMessage.textContent = `Erro de conexão: ${error.message}. Tentando reconectar...`;
        userInput.disabled = true;
        sendButton.disabled = true;
    });

    // Evento 'status' enviado pelo backend.
    socket.on('status', (data) => {
        console.log('Status do Backend:', data.data);
    });

    // Evento 'video_frame' para receber os frames de vídeo do avatar.
    socket.on('video_frame', (data) => {
        if (data.image) {
            // Atualiza a imagem do avatar com o frame base64 recebido.
            avatarImage.src = 'data:image/jpeg;base64,' + data.image;
        }
    });

    // --- Funções de Manipulação da Interface ---

    // Adiciona uma mensagem ao chat box.
    function addMessageToChat(sender, message, type) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('chat-message', type);
        messageElement.textContent = `${sender}: ${message}`;
        chatBox.appendChild(messageElement);
        // Garante que o scroll esteja sempre na parte inferior para ver a última mensagem.
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    // Envia a mensagem do usuário para o backend.
    async function sendMessage() {
        const message = userInput.value.trim();
        if (message === '') {
            return; // Não envia mensagens vazias.
        }

        // Adiciona a mensagem do usuário ao chat.
        addMessageToChat('Você', message, 'user-message');
        userInput.value = ''; // Limpa o input.
        userInput.disabled = true; // Desabilita input e botão enquanto espera resposta.
        sendButton.disabled = true;
        statusMessage.textContent = 'Sora está pensando...';

        try {
            // Faz uma requisição POST para a API de chat do backend.
            const response = await fetch(CHAT_API_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: message, userId: userId }), // Envia a mensagem e o ID do usuário.
            });

            const data = await response.json();
            if (response.ok) {
                // Adiciona a resposta do Sora ao chat.
                addMessageToChat('Sora', data.response, 'sora-message');
            } else {
                console.error('Erro na resposta da API:', data.response || 'Erro desconhecido');
                addMessageToChat('Sora', 'Desculpe, houve um erro ao processar sua mensagem.', 'sora-message');
            }
        } catch (error) {
            console.error('Erro ao enviar mensagem para o backend:', error);
            addMessageToChat('Sora', 'Não foi possível conectar ao servidor. Verifique sua conexão.', 'sora-message');
        } finally {
            userInput.disabled = false; // Reabilita input e botão.
            sendButton.disabled = false;
            statusMessage.textContent = 'Pronta para sua próxima pergunta!';
            userInput.focus(); // Coloca o foco de volta no input.
        }
    }

    // --- Escutadores de Eventos ---

    // Envia mensagem ao clicar no botão.
    sendButton.addEventListener('click', sendMessage);

    // Envia mensagem ao pressionar "Enter" no input.
    userInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            sendMessage();
        }
    });

    // Inicia o estado da UI como desabilitada até a conexão WebSocket.
    userInput.disabled = true;
    sendButton.disabled = true;
    statusMessage.textContent = 'Tentando conectar ao servidor Sora...';
});