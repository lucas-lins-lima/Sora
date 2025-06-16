/**
 * SORA ROBOT - FRONTEND JAVASCRIPT
 * Sistema de interface interativa para comunicação com o backend Sora Robot
 * Conecta com a API REST e WebSocket para funcionalidade completa
 */

// =========================================================================
// CONFIGURAÇÕES E CONSTANTES
// =========================================================================

const CONFIG = {
    API_BASE_URL: window.location.origin,
    WS_URL: `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws`,
    UPDATE_INTERVAL: 1000, // 1 segundo
    CHART_UPDATE_INTERVAL: 5000, // 5 segundos
    MAX_CHAT_MESSAGES: 100,
    MAX_LOG_LINES: 500,
    RECONNECT_INTERVAL: 3000, // 3 segundos
    REQUEST_TIMEOUT: 30000 // 30 segundos
};

const ENDPOINTS = {
    STATUS: '/status',
    INITIALIZE: '/initialize',
    START: '/start',
    STOP: '/stop',
    MESSAGE: '/message',
    CONFIG: '/config',
    HISTORY: '/history',
    METRICS: '/metrics',
    HEALTH: '/health'
};

// =========================================================================
// ESTADO GLOBAL DA APLICAÇÃO
// =========================================================================

class AppState {
    constructor() {
        this.systemStatus = 'unknown';
        this.isConnected = false;
        this.isRecording = false;
        this.currentConfig = {};
        this.metrics = {};
        this.chatMessages = [];
        this.logs = [];
        this.charts = {};
        this.websocket = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        
        // Bind methods
        this.updateStatus = this.updateStatus.bind(this);
        this.addChatMessage = this.addChatMessage.bind(this);
        this.addLogEntry = this.addLogEntry.bind(this);
    }
    
    updateStatus(status) {
        this.systemStatus = status;
        this.updateStatusDisplay();
    }
    
    updateStatusDisplay() {
        const statusElement = document.getElementById('systemStatus');
        const statusText = document.getElementById('statusText');
        const avatarStatus = document.getElementById('avatarStatus');
        
        if (statusElement && statusText) {
            statusElement.className = 'badge';
            
            switch (this.systemStatus) {
                case 'active':
                    statusElement.classList.add('status-active');
                    statusText.textContent = 'Sistema Ativo';
                    if (avatarStatus) {
                        avatarStatus.innerHTML = '<i class="fas fa-circle text-success"></i>';
                    }
                    break;
                case 'inactive':
                case 'ready':
                    statusElement.classList.add('status-inactive');
                    statusText.textContent = 'Sistema Inativo';
                    if (avatarStatus) {
                        avatarStatus.innerHTML = '<i class="fas fa-circle text-warning"></i>';
                    }
                    break;
                case 'processing':
                    statusElement.classList.add('status-loading');
                    statusText.textContent = 'Processando...';
                    if (avatarStatus) {
                        avatarStatus.innerHTML = '<i class="fas fa-circle text-info"></i>';
                    }
                    break;
                default:
                    statusElement.classList.add('status-inactive');
                    statusText.textContent = 'Carregando...';
                    if (avatarStatus) {
                        avatarStatus.innerHTML = '<i class="fas fa-circle text-secondary"></i>';
                    }
            }
        }
    }
    
    addChatMessage(message, isUser = false, metadata = {}) {
        const chatMessage = {
            id: generateUUID(),
            text: message,
            isUser: isUser,
            timestamp: new Date(),
            metadata: metadata
        };
        
        this.chatMessages.push(chatMessage);
        
        // Limita número de mensagens
        if (this.chatMessages.length > CONFIG.MAX_CHAT_MESSAGES) {
            this.chatMessages = this.chatMessages.slice(-CONFIG.MAX_CHAT_MESSAGES);
        }
        
        this.renderChatMessage(chatMessage);
    }
    
    renderChatMessage(message) {
        const chatMessages = document.getElementById('chatMessages');
        if (!chatMessages) return;
        
        const messageElement = document.createElement('div');
        messageElement.className = `message ${message.isUser ? 'user-message' : 'bot-message'}`;
        messageElement.innerHTML = `
            <div class="message-avatar">
                <i class="fas ${message.isUser ? 'fa-user' : 'fa-robot'}"></i>
            </div>
            <div class="message-content">
                <div class="message-bubble">
                    <p class="mb-0">${escapeHtml(message.text)}</p>
                </div>
                <div class="message-meta">
                    <span class="timestamp">${formatTime(message.timestamp)}</span>
                    ${message.metadata.confidence ? 
                        `<span class="confidence badge bg-success ms-2">${Math.round(message.metadata.confidence * 100)}%</span>` : 
                        ''
                    }
                </div>
            </div>
        `;
        
        chatMessages.appendChild(messageElement);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    addLogEntry(level, message, timestamp = new Date()) {
        const logEntry = {
            id: generateUUID(),
            level: level.toLowerCase(),
            message: message,
            timestamp: timestamp
        };
        
        this.logs.push(logEntry);
        
        // Limita número de logs
        if (this.logs.length > CONFIG.MAX_LOG_LINES) {
            this.logs = this.logs.slice(-CONFIG.MAX_LOG_LINES);
        }
        
        this.updateLogsDisplay();
        this.updateQuickLogs();
    }
    
    updateLogsDisplay() {
        const logsContainer = document.getElementById('logsContainer');
        if (!logsContainer) return;
        
        const filteredLogs = this.getFilteredLogs();
        logsContainer.innerHTML = filteredLogs.map(log => `
            <div class="log-line level-${log.level}">
                <span class="log-timestamp">[${formatTime(log.timestamp)}]</span>
                <span class="log-level">[${log.level.toUpperCase()}]</span>
                <span class="log-message">${escapeHtml(log.message)}</span>
            </div>
        `).join('');
        
        logsContainer.scrollTop = logsContainer.scrollHeight;
    }
    
    updateQuickLogs() {
        const quickLogs = document.getElementById('quickLogs');
        if (!quickLogs) return;
        
        const recentLogs = this.logs.slice(-5);
        quickLogs.innerHTML = recentLogs.map(log => `
            <div class="log-entry text-muted small log-${log.level}">
                ${formatTime(log.timestamp, 'short')} - ${escapeHtml(log.message)}
            </div>
        `).join('');
    }
    
    getFilteredLogs() {
        const activeFilter = document.querySelector('input[name="logLevel"]:checked')?.id;
        
        if (!activeFilter || activeFilter === 'logAll') {
            return this.logs;
        }
        
        const levelMap = {
            'logInfo': 'info',
            'logWarning': 'warning',
            'logError': 'error'
        };
        
        const targetLevel = levelMap[activeFilter];
        return this.logs.filter(log => log.level === targetLevel);
    }
}

// =========================================================================
// GERENCIADOR DE API
// =========================================================================

class APIManager {
    constructor() {
        this.baseURL = CONFIG.API_BASE_URL;
        this.headers = {
            'Content-Type': 'application/json'
        };
    }
    
    async request(endpoint, options = {}) {
        const url = `${this.baseURL}${endpoint}`;
        const config = {
            headers: this.headers,
            timeout: CONFIG.REQUEST_TIMEOUT,
            ...options
        };
        
        try {
            showLoading(true);
            const response = await fetch(url, config);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            return data;
            
        } catch (error) {
            console.error(`API Error [${endpoint}]:`, error);
            showError(`Erro na comunicação: ${error.message}`);
            throw error;
        } finally {
            showLoading(false);
        }
    }
    
    async get(endpoint) {
        return this.request(endpoint, { method: 'GET' });
    }
    
    async post(endpoint, data) {
        return this.request(endpoint, {
            method: 'POST',
            body: JSON.stringify(data)
        });
    }
    
    async put(endpoint, data) {
        return this.request(endpoint, {
            method: 'PUT',
            body: JSON.stringify(data)
        });
    }
    
    async delete(endpoint) {
        return this.request(endpoint, { method: 'DELETE' });
    }
    
    // Métodos específicos da API Sora
    async getStatus() {
        return this.get(ENDPOINTS.STATUS);
    }
    
    async initializeSystem(config = {}) {
        return this.post(ENDPOINTS.INITIALIZE, config);
    }
    
    async startSystem() {
        return this.post(ENDPOINTS.START);
    }
    
    async stopSystem() {
        return this.post(ENDPOINTS.STOP);
    }
    
    async sendMessage(message, waitForResponse = true, timeout = null) {
        return this.post(ENDPOINTS.MESSAGE, {
            message: message,
            wait_for_response: waitForResponse,
            timeout: timeout
        });
    }
    
    async updateConfig(configUpdates) {
        return this.put(ENDPOINTS.CONFIG, configUpdates);
    }
    
    async getConfig() {
        return this.get(ENDPOINTS.CONFIG);
    }
    
    async getHistory(limit = 10) {
        return this.get(`${ENDPOINTS.HISTORY}?limit=${limit}`);
    }
    
    async clearHistory() {
        return this.delete(ENDPOINTS.HISTORY);
    }
    
    async getMetrics() {
        return this.get(ENDPOINTS.METRICS);
    }
    
    async getHealth() {
        return this.get(ENDPOINTS.HEALTH);
    }
}

// =========================================================================
// GERENCIADOR DE WEBSOCKET
// =========================================================================

class WebSocketManager {
    constructor(appState) {
        this.appState = appState;
        this.ws = null;
        this.clientId = generateUUID();
        this.isConnecting = false;
        this.reconnectTimer = null;
        this.heartbeatTimer = null;
    }
    
    connect() {
        if (this.isConnecting || (this.ws && this.ws.readyState === WebSocket.OPEN)) {
            return;
        }
        
        this.isConnecting = true;
        const wsUrl = `${CONFIG.WS_URL}/${this.clientId}`;
        
        try {
            this.ws = new WebSocket(wsUrl);
            this.setupEventListeners();
            
            this.appState.addLogEntry('info', 'Conectando WebSocket...');
            
        } catch (error) {
            console.error('Erro ao conectar WebSocket:', error);
            this.appState.addLogEntry('error', `Erro WebSocket: ${error.message}`);
            this.scheduleReconnect();
        }
    }
    
    setupEventListeners() {
        this.ws.onopen = (event) => {
            console.log('WebSocket conectado');
            this.appState.isConnected = true;
            this.appState.reconnectAttempts = 0;
            this.isConnecting = false;
            
            this.appState.addLogEntry('info', 'WebSocket conectado com sucesso');
            
            // Inicia heartbeat
            this.startHeartbeat();
            
            // Solicita status inicial
            this.send({
                type: 'status'
            });
        };
        
        this.ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.handleMessage(data);
            } catch (error) {
                console.error('Erro ao processar mensagem WebSocket:', error);
            }
        };
        
        this.ws.onclose = (event) => {
            console.log('WebSocket desconectado:', event.code, event.reason);
            this.appState.isConnected = false;
            this.isConnecting = false;
            
            this.stopHeartbeat();
            
            if (event.code !== 1000) { // Não foi fechamento normal
                this.appState.addLogEntry('warning', 'Conexão WebSocket perdida');
                this.scheduleReconnect();
            }
        };
        
        this.ws.onerror = (error) => {
            console.error('Erro WebSocket:', error);
            this.appState.addLogEntry('error', 'Erro na conexão WebSocket');
            this.isConnecting = false;
        };
    }
    
    handleMessage(data) {
        switch (data.type) {
            case 'connection':
                console.log('Mensagem de conexão:', data.message);
                break;
                
            case 'response':
                if (data.data && data.data.text) {
                    this.appState.addChatMessage(data.data.text, false, {
                        confidence: data.data.confidence,
                        emotion: data.data.emotion_detected,
                        intent: data.data.intent_detected
                    });
                }
                break;
                
            case 'system_response':
                if (data.data && data.data.text) {
                    this.appState.addChatMessage(data.data.text, false, data.data);
                }
                break;
                
            case 'state_change':
                if (data.data && data.data.state) {
                    this.appState.updateStatus(data.data.state);
                    this.appState.addLogEntry('info', `Estado alterado para: ${data.data.state}`);
                }
                break;
                
            case 'error':
                this.appState.addLogEntry('error', data.message || 'Erro do sistema');
                showError(data.message || 'Erro do sistema');
                break;
                
            case 'pong':
                // Resposta do heartbeat
                break;
                
            default:
                console.log('Mensagem WebSocket não reconhecida:', data);
        }
    }
    
    send(data) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(data));
            return true;
        }
        return false;
    }
    
    sendMessage(message) {
        return this.send({
            type: 'message',
            text: message,
            timestamp: new Date().toISOString()
        });
    }
    
    startHeartbeat() {
        this.heartbeatTimer = setInterval(() => {
            this.send({ type: 'ping' });
        }, 30000); // 30 segundos
    }
    
    stopHeartbeat() {
        if (this.heartbeatTimer) {
            clearInterval(this.heartbeatTimer);
            this.heartbeatTimer = null;
        }
    }
    
    scheduleReconnect() {
        if (this.appState.reconnectAttempts >= this.appState.maxReconnectAttempts) {
            this.appState.addLogEntry('error', 'Máximo de tentativas de reconexão atingido');
            return;
        }
        
        this.appState.reconnectAttempts++;
        
        this.reconnectTimer = setTimeout(() => {
            this.appState.addLogEntry('info', `Tentativa de reconexão ${this.appState.reconnectAttempts}/${this.appState.maxReconnectAttempts}`);
            this.connect();
        }, CONFIG.RECONNECT_INTERVAL);
    }
    
    disconnect() {
        this.stopHeartbeat();
        
        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
        }
        
        if (this.ws) {
            this.ws.close(1000, 'Desconexão solicitada');
            this.ws = null;
        }
    }
}

// =========================================================================
// GERENCIADOR DE GRÁFICOS
// =========================================================================

class ChartManager {
    constructor() {
        this.charts = {};
        this.chartData = {
            responseTime: [],
            emotions: {},
            intents: {},
            sentiments: {}
        };
    }
    
    initializeCharts() {
        this.initResponseTimeChart();
        this.initEmotionsChart();
        this.initIntentsChart();
        this.initSentimentsChart();
    }
    
    initResponseTimeChart() {
        const ctx = document.getElementById('responseTimeChart');
        if (!ctx) return;
        
        this.charts.responseTime = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Tempo de Resposta (s)',
                    data: [],
                    borderColor: '#2563eb',
                    backgroundColor: 'rgba(37, 99, 235, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: {
                            color: '#e2e8f0'
                        }
                    },
                    x: {
                        grid: {
                            color: '#e2e8f0'
                        }
                    }
                }
            }
        });
    }
    
    initEmotionsChart() {
        const ctx = document.getElementById('emotionsChart');
        if (!ctx) return;
        
        this.charts.emotions = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: [],
                datasets: [{
                    data: [],
                    backgroundColor: [
                        '#10b981', '#f59e0b', '#ef4444', 
                        '#06b6d4', '#8b5cf6', '#f97316'
                    ]
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
    }
    
    initIntentsChart() {
        const ctx = document.getElementById('intentsChart');
        if (!ctx) return;
        
        this.charts.intents = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: 'Frequência',
                    data: [],
                    backgroundColor: '#2563eb'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    }
    
    initSentimentsChart() {
        const ctx = document.getElementById('sentimentsChart');
        if (!ctx) return;
        
        this.charts.sentiments = new Chart(ctx, {
            type: 'pie',
            data: {
                labels: ['Positivo', 'Neutro', 'Negativo'],
                datasets: [{
                    data: [0, 0, 0],
                    backgroundColor: ['#10b981', '#64748b', '#ef4444']
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
    }
    
    updateResponseTimeChart(responseTime) {
        const chart = this.charts.responseTime;
        if (!chart) return;
        
        const now = new Date();
        const timeLabel = now.toLocaleTimeString();
        
        chart.data.labels.push(timeLabel);
        chart.data.datasets[0].data.push(responseTime);
        
        // Mantém apenas últimos 20 pontos
        if (chart.data.labels.length > 20) {
            chart.data.labels.shift();
            chart.data.datasets[0].data.shift();
        }
        
        chart.update('none');
    }
    
    updateEmotionsChart(emotion) {
        const chart = this.charts.emotions;
        if (!chart || !emotion) return;
        
        // Atualiza dados de emoções
        if (!this.chartData.emotions[emotion]) {
            this.chartData.emotions[emotion] = 0;
        }
        this.chartData.emotions[emotion]++;
        
        // Atualiza gráfico
        const emotions = Object.keys(this.chartData.emotions);
        const counts = Object.values(this.chartData.emotions);
        
        chart.data.labels = emotions;
        chart.data.datasets[0].data = counts;
        chart.update();
    }
    
    updateChartsFromMetrics(metrics) {
        // Atualiza gráficos baseado nas métricas do sistema
        if (metrics.system_metrics && metrics.system_metrics.average_response_time) {
            this.updateResponseTimeChart(metrics.system_metrics.average_response_time);
        }
    }
}

// =========================================================================
// GERENCIADOR DE ÁUDIO
// =========================================================================

class AudioManager {
    constructor(appState) {
        this.appState = appState;
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.isRecording = false;
        this.stream = null;
    }
    
    async initializeAudio() {
        try {
            this.stream = await navigator.mediaDevices.getUserMedia({ 
                audio: true,
                video: false
            });
            return true;
        } catch (error) {
            console.error('Erro ao acessar microfone:', error);
            showError('Não foi possível acessar o microfone');
            return false;
        }
    }
    
    async startRecording() {
        if (this.isRecording) return false;
        
        if (!this.stream && !(await this.initializeAudio())) {
            return false;
        }
        
        try {
            this.audioChunks = [];
            this.mediaRecorder = new MediaRecorder(this.stream);
            
            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    this.audioChunks.push(event.data);
                }
            };
            
            this.mediaRecorder.onstop = () => {
                const audioBlob = new Blob(this.audioChunks, { type: 'audio/wav' });
                this.processAudioBlob(audioBlob);
            };
            
            this.mediaRecorder.start();
            this.isRecording = true;
            this.appState.isRecording = true;
            
            this.updateRecordingUI(true);
            this.appState.addLogEntry('info', 'Gravação de áudio iniciada');
            
            return true;
            
        } catch (error) {
            console.error('Erro ao iniciar gravação:', error);
            showError('Erro ao iniciar gravação de áudio');
            return false;
        }
    }
    
    stopRecording() {
        if (!this.isRecording || !this.mediaRecorder) return;
        
        this.mediaRecorder.stop();
        this.isRecording = false;
        this.appState.isRecording = false;
        
        this.updateRecordingUI(false);
        this.appState.addLogEntry('info', 'Gravação de áudio finalizada');
    }
    
    updateRecordingUI(isRecording) {
        const voiceBtn = document.getElementById('voiceInputBtn');
        const recordingIndicator = document.getElementById('recordingIndicator');
        
        if (voiceBtn) {
            if (isRecording) {
                voiceBtn.classList.add('recording');
                voiceBtn.innerHTML = '<i class="fas fa-stop"></i>';
                voiceBtn.title = 'Parar gravação';
            } else {
                voiceBtn.classList.remove('recording');
                voiceBtn.innerHTML = '<i class="fas fa-microphone"></i>';
                voiceBtn.title = 'Iniciar gravação';
            }
        }
        
        if (recordingIndicator) {
            recordingIndicator.style.display = isRecording ? 'block' : 'none';
        }
    }
    
    async processAudioBlob(audioBlob) {
        try {
            // Converte blob para base64
            const arrayBuffer = await audioBlob.arrayBuffer();
            const base64Audio = btoa(String.fromCharCode(...new Uint8Array(arrayBuffer)));
            
            // Em uma implementação real, enviaria para o backend
            // Por enquanto, simula transcrição
            this.simulateTranscription("Áudio gravado (transcrição simulada)");
            
        } catch (error) {
            console.error('Erro ao processar áudio:', error);
            showError('Erro ao processar áudio gravado');
        }
    }
    
    simulateTranscription(text) {
        const messageInput = document.getElementById('messageInput');
        if (messageInput) {
            messageInput.value = text;
            messageInput.focus();
        }
    }
}

// =========================================================================
// INSTÂNCIAS GLOBAIS
// =========================================================================

let appState;
let apiManager;
let wsManager;
let chartManager;
let audioManager;

// =========================================================================
// INICIALIZAÇÃO DA APLICAÇÃO
// =========================================================================

document.addEventListener('DOMContentLoaded', function() {
    initializeApplication();
});

async function initializeApplication() {
    try {
        // Inicializa estado da aplicação
        appState = new AppState();
        apiManager = new APIManager();
        wsManager = new WebSocketManager(appState);
        chartManager = new ChartManager();
        audioManager = new AudioManager(appState);
        
        // Log inicial
        appState.addLogEntry('info', 'Aplicação frontend inicializada');
        
        // Configura event listeners
        setupEventListeners();
        
        // Inicializa gráficos
        chartManager.initializeCharts();
        
        // Conecta WebSocket
        wsManager.connect();
        
        // Carrega status inicial
        await loadInitialData();
        
        // Inicia loops de atualização
        startUpdateLoops();
        
        appState.addLogEntry('info', 'Sistema frontend pronto');
        
    } catch (error) {
        console.error('Erro na inicialização:', error);
        showError('Erro ao inicializar aplicação');
    }
}

function setupEventListeners() {
    // Controles do sistema
    document.getElementById('startSystemBtn')?.addEventListener('click', startSystem);
    document.getElementById('pauseSystemBtn')?.addEventListener('click', pauseSystem);
    document.getElementById('stopSystemBtn')?.addEventListener('click', stopSystem);
    
    // Chat
    document.getElementById('chatForm')?.addEventListener('submit', handleChatSubmit);
    document.getElementById('voiceInputBtn')?.addEventListener('click', toggleVoiceInput);
    document.getElementById('clearChatBtn')?.addEventListener('click', clearChat);
    document.getElementById('exportChatBtn')?.addEventListener('click', exportChat);
    
    // Configurações
    document.getElementById('personalitySelect')?.addEventListener('change', updatePersonality);
    document.getElementById('languageSelect')?.addEventListener('change', updateLanguage);
    document.getElementById('voiceEnabled')?.addEventListener('change', updateVoiceEnabled);
    document.getElementById('visionEnabled')?.addEventListener('change', updateVisionEnabled);
    document.getElementById('animationEnabled')?.addEventListener('change', updateAnimationEnabled);
    
    // Histórico
    document.getElementById('clearHistoryBtn')?.addEventListener('click', clearHistory);
    
    // Logs
    document.querySelectorAll('input[name="logLevel"]').forEach(radio => {
        radio.addEventListener('change', () => appState.updateLogsDisplay());
    });
    document.getElementById('pauseLogsBtn')?.addEventListener('click', toggleLogsPause);
    document.getElementById('downloadLogsBtn')?.addEventListener('click', downloadLogs);
    
    // Tabs
    document.querySelectorAll('[data-bs-toggle="tab"]').forEach(tab => {
        tab.addEventListener('shown.bs.tab', handleTabChange);
    });
    
    // Keyboard shortcuts
    document.addEventListener('keydown', handleKeyboardShortcuts);
    
    // Window events
    window.addEventListener('beforeunload', handleBeforeUnload);
    window.addEventListener('online', () => {
        appState.addLogEntry('info', 'Conexão com internet restaurada');
        wsManager.connect();
    });
    window.addEventListener('offline', () => {
        appState.addLogEntry('warning', 'Conexão com internet perdida');
    });
}

// =========================================================================
// HANDLERS DE EVENTOS
// =========================================================================

async function startSystem() {
    try {
        const response = await apiManager.startSystem();
        if (response.success) {
            showSuccess('Sistema iniciado com sucesso');
            appState.updateStatus('active');
        }
    } catch (error) {
        console.error('Erro ao iniciar sistema:', error);
    }
}

async function pauseSystem() {
    try {
        // API pause não existe ainda, simula
        showSuccess('Sistema pausado');
        appState.updateStatus('paused');
    } catch (error) {
        console.error('Erro ao pausar sistema:', error);
    }
}

async function stopSystem() {
    try {
        const response = await apiManager.stopSystem();
        if (response.success) {
            showSuccess('Sistema parado');
            appState.updateStatus('inactive');
        }
    } catch (error) {
        console.error('Erro ao parar sistema:', error);
    }
}

async function handleChatSubmit(event) {
    event.preventDefault();
    
    const messageInput = document.getElementById('messageInput');
    const message = messageInput.value.trim();
    
    if (!message) return;
    
    // Adiciona mensagem do usuário ao chat
    appState.addChatMessage(message, true);
    
    // Limpa input
    messageInput.value = '';
    
    try {
        // Envia via WebSocket se conectado, senão via API
        if (wsManager.isConnected) {
            wsManager.sendMessage(message);
        } else {
            const response = await apiManager.sendMessage(message);
            if (response.success && response.data) {
                appState.addChatMessage(response.data.text, false, {
                    confidence: response.data.confidence,
                    emotion: response.data.emotion_detected,
                    intent: response.data.intent_detected
                });
            }
        }
        
    } catch (error) {
        console.error('Erro ao enviar mensagem:', error);
        appState.addChatMessage('Erro ao processar sua mensagem. Tente novamente.', false);
    }
}

async function toggleVoiceInput() {
    if (audioManager.isRecording) {
        audioManager.stopRecording();
    } else {
        await audioManager.startRecording();
    }
}

function clearChat() {
    const chatMessages = document.getElementById('chatMessages');
    if (chatMessages) {
        // Mantém apenas mensagem de boas-vindas
        const welcomeMessage = chatMessages.querySelector('.bot-message');
        chatMessages.innerHTML = '';
        if (welcomeMessage) {
            chatMessages.appendChild(welcomeMessage);
        }
    }
    
    appState.chatMessages = [];
    showSuccess('Chat limpo');
}

function exportChat() {
    const chatData = {
        timestamp: new Date().toISOString(),
        messages: appState.chatMessages
    };
    
    const blob = new Blob([JSON.stringify(chatData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = `sora_chat_${formatDateForFilename(new Date())}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    showSuccess('Chat exportado');
}

async function updatePersonality(event) {
    const personality = event.target.value;
    try {
        await apiManager.updateConfig({ personality });
        document.getElementById('avatarMood').textContent = `Modo: ${capitalizeFirst(personality)}`;
        showSuccess('Personalidade atualizada');
    } catch (error) {
        console.error('Erro ao atualizar personalidade:', error);
    }
}

async function updateLanguage(event) {
    const language = event.target.value;
    try {
        await apiManager.updateConfig({ language });
        showSuccess('Idioma atualizado');
    } catch (error) {
        console.error('Erro ao atualizar idioma:', error);
    }
}

async function updateVoiceEnabled(event) {
    const enabled = event.target.checked;
    try {
        await apiManager.updateConfig({ voice_enabled: enabled });
        showSuccess(`Síntese de voz ${enabled ? 'habilitada' : 'desabilitada'}`);
    } catch (error) {
        console.error('Erro ao atualizar configuração de voz:', error);
    }
}

async function updateVisionEnabled(event) {
    const enabled = event.target.checked;
    try {
        await apiManager.updateConfig({ vision_enabled: enabled });
        showSuccess(`Processamento de visão ${enabled ? 'habilitado' : 'desabilitado'}`);
    } catch (error) {
        console.error('Erro ao atualizar configuração de visão:', error);
    }
}

async function updateAnimationEnabled(event) {
    const enabled = event.target.checked;
    try {
        await apiManager.updateConfig({ animation_enabled: enabled });
        showSuccess(`Animações ${enabled ? 'habilitadas' : 'desabilitadas'}`);
    } catch (error) {
        console.error('Erro ao atualizar configuração de animação:', error);
    }
}

async function clearHistory() {
    try {
        await apiManager.clearHistory();
        updateHistoryDisplay([]);
        showSuccess('Histórico limpo');
    } catch (error) {
        console.error('Erro ao limpar histórico:', error);
    }
}

function toggleLogsPause() {
    const btn = document.getElementById('pauseLogsBtn');
    // Implementar pausa de logs
    showSuccess('Funcionalidade em desenvolvimento');
}

function downloadLogs() {
    const logs = appState.logs.map(log => 
        `[${formatTime(log.timestamp)}] [${log.level.toUpperCase()}] ${log.message}`
    ).join('\n');
    
    const blob = new Blob([logs], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = `sora_logs_${formatDateForFilename(new Date())}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    showSuccess('Logs baixados');
}

function handleTabChange(event) {
    const tabId = event.target.getAttribute('data-bs-target');
    
    switch (tabId) {
        case '#history':
            loadHistory();
            break;
        case '#analytics':
            updateAnalytics();
            break;
        case '#logs':
            appState.updateLogsDisplay();
            break;
    }
}

function handleKeyboardShortcuts(event) {
    // Ctrl/Cmd + Enter para enviar mensagem
    if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
        const chatForm = document.getElementById('chatForm');
        if (chatForm) {
            chatForm.dispatchEvent(new Event('submit'));
        }
    }
    
    // Esc para parar gravação
    if (event.key === 'Escape' && audioManager.isRecording) {
        audioManager.stopRecording();
    }
}

function handleBeforeUnload() {
    // Desconecta WebSocket ao sair
    if (wsManager) {
        wsManager.disconnect();
    }
}

// =========================================================================
// CARREGAMENTO DE DADOS
// =========================================================================

async function loadInitialData() {
    try {
        // Carrega status do sistema
        const status = await apiManager.getStatus();
        if (status.success) {
            appState.updateStatus(status.data.controller_state);
            updateSystemInfo(status.data);
        }
        
        // Carrega configuração atual
        const config = await apiManager.getConfig();
        if (config.success) {
            updateConfigDisplay(config.data);
        }
        
        // Carrega métricas
        await loadMetrics();
        
        // Carrega histórico
        await loadHistory();
        
    } catch (error) {
        console.error('Erro ao carregar dados iniciais:', error);
        appState.addLogEntry('error', 'Erro ao carregar dados iniciais');
    }
}

async function loadMetrics() {
    try {
        const metrics = await apiManager.getMetrics();
        if (metrics.success) {
            updateMetricsDisplay(metrics.data);
            chartManager.updateChartsFromMetrics(metrics.data);
        }
    } catch (error) {
        console.error('Erro ao carregar métricas:', error);
    }
}

async function loadHistory() {
    try {
        const history = await apiManager.getHistory(20);
        if (history.success) {
            updateHistoryDisplay(history.data.history);
        }
    } catch (error) {
        console.error('Erro ao carregar histórico:', error);
    }
}

// =========================================================================
// ATUALIZAÇÃO DE INTERFACE
// =========================================================================

function updateSystemInfo(statusData) {
    // Atualiza informações dos componentes
    const components = ['vision', 'audio', 'nlp', 'animation'];
    
    components.forEach(component => {
        const statusElement = document.getElementById(`${component}Status`);
        const detailsElement = document.getElementById(`${component}Details`);
        
        if (statusElement) {
            const isEnabled = statusData.components_enabled?.[component];
            statusElement.textContent = isEnabled ? 'Ativo' : 'Inativo';
            statusElement.className = isEnabled ? 'stat-value text-success' : 'stat-value text-muted';
        }
    });
}

function updateConfigDisplay(configData) {
    // Atualiza displays de configuração
    const personalitySelect = document.getElementById('personalitySelect');
    if (personalitySelect && configData.personality) {
        personalitySelect.value = configData.personality;
    }
    
    const languageSelect = document.getElementById('languageSelect');
    if (languageSelect && configData.language) {
        languageSelect.value = configData.language;
    }
    
    const voiceEnabled = document.getElementById('voiceEnabled');
    if (voiceEnabled) {
        voiceEnabled.checked = configData.voice_enabled ?? true;
    }
    
    const visionEnabled = document.getElementById('visionEnabled');
    if (visionEnabled) {
        visionEnabled.checked = configData.vision_enabled ?? true;
    }
    
    const animationEnabled = document.getElementById('animationEnabled');
    if (animationEnabled) {
        animationEnabled.checked = configData.animation_enabled ?? true;
    }
    
    // Atualiza mood do avatar
    const avatarMood = document.getElementById('avatarMood');
    if (avatarMood && configData.personality) {
        avatarMood.textContent = `Modo: ${capitalizeFirst(configData.personality)}`;
    }
}

function updateMetricsDisplay(metricsData) {
    // Atualiza métricas na sidebar
    const totalInteractions = document.getElementById('totalInteractions');
    if (totalInteractions) {
        totalInteractions.textContent = metricsData.controller_metrics?.total_interactions || 0;
    }
    
    const avgResponseTime = document.getElementById('avgResponseTime');
    if (avgResponseTime) {
        const time = metricsData.controller_metrics?.average_response_time || 0;
        avgResponseTime.textContent = `${time.toFixed(1)}s`;
    }
    
    const systemUptime = document.getElementById('systemUptime');
    if (systemUptime) {
        const uptime = metricsData.uptime || 0;
        systemUptime.textContent = formatDuration(uptime);
    }
    
    // Atualiza progress bars de recursos
    const cpuProgress = document.getElementById('cpuProgress');
    if (cpuProgress && metricsData.system_metrics?.resource_usage?.cpu_percent) {
        const cpu = metricsData.system_metrics.resource_usage.cpu_percent;
        cpuProgress.style.width = `${cpu}%`;
        cpuProgress.setAttribute('aria-valuenow', cpu);
    }
    
    const memoryProgress = document.getElementById('memoryProgress');
    if (memoryProgress && metricsData.system_metrics?.resource_usage?.memory_percent) {
        const memory = metricsData.system_metrics.resource_usage.memory_percent;
        memoryProgress.style.width = `${memory}%`;
        memoryProgress.setAttribute('aria-valuenow', memory);
    }
}

function updateHistoryDisplay(historyData) {
    const historyList = document.getElementById('historyList');
    if (!historyList) return;
    
    if (!historyData || historyData.length === 0) {
        historyList.innerHTML = '<p class="text-muted text-center">Nenhum histórico disponível</p>';
        return;
    }
    
    historyList.innerHTML = historyData.map(item => `
        <div class="history-item">
            <div class="history-meta">
                ${formatTime(new Date(item.timestamp * 1000))} - Qualidade: ${Math.round((item.quality || 0) * 100)}%
            </div>
            <div class="history-preview">
                <strong>Usuário:</strong> ${escapeHtml(item.text_input || 'N/A')}<br>
                <strong>Sora:</strong> ${escapeHtml(item.response_text || 'N/A')}
            </div>
        </div>
    `).join('');
}

function updateAnalytics() {
    // Atualiza gráficos de analytics
    const performanceMetrics = document.getElementById('performanceMetrics');
    if (performanceMetrics) {
        performanceMetrics.innerHTML = `
            <div class="performance-item">
                <span>Interações Totais:</span>
                <span>${appState.metrics.total_interactions || 0}</span>
            </div>
            <div class="performance-item">
                <span>Taxa de Sucesso:</span>
                <span>${Math.round((appState.metrics.success_rate || 0) * 100)}%</span>
            </div>
            <div class="performance-item">
                <span>Satisfação Média:</span>
                <span>${Math.round((appState.metrics.user_satisfaction || 0) * 100)}%</span>
            </div>
        `;
    }
}

// =========================================================================
// LOOPS DE ATUALIZAÇÃO
// =========================================================================

function startUpdateLoops() {
    // Loop principal de atualização
    setInterval(async () => {
        try {
            await loadMetrics();
        } catch (error) {
            // Silencioso - evita spam de erros
        }
    }, CONFIG.UPDATE_INTERVAL);
    
    // Loop de atualização de gráficos
    setInterval(() => {
        // Atualiza gráficos se necessário
    }, CONFIG.CHART_UPDATE_INTERVAL);
}

// =========================================================================
// FUNÇÕES UTILITÁRIAS
// =========================================================================

function showLoading(show = true) {
    const loadingOverlay = document.getElementById('loadingOverlay');
    if (loadingOverlay) {
        loadingOverlay.classList.toggle('show', show);
    }
}

function showError(message) {
    const toast = document.getElementById('errorToast');
    const toastBody = document.getElementById('errorToastBody');
    
    if (toast && toastBody) {
        toastBody.textContent = message;
        const bsToast = new bootstrap.Toast(toast);
        bsToast.show();
    }
    
    console.error('Error:', message);
}

function showSuccess(message) {
    const toast = document.getElementById('successToast');
    const toastBody = document.getElementById('successToastBody');
    
    if (toast && toastBody) {
        toastBody.textContent = message;
        const bsToast = new bootstrap.Toast(toast);
        bsToast.show();
    }
}

function generateUUID() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
        const r = Math.random() * 16 | 0;
        const v = c == 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatTime(date, format = 'full') {
    if (!(date instanceof Date)) {
        date = new Date(date);
    }
    
    switch (format) {
        case 'short':
            return date.toLocaleTimeString('pt-BR', { 
                hour: '2-digit', 
                minute: '2-digit' 
            });
        case 'full':
        default:
            return date.toLocaleTimeString('pt-BR');
    }
}

function formatDuration(seconds) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    
    return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}

function formatDateForFilename(date) {
    return date.toISOString().replace(/[:.]/g, '-').slice(0, 19);
}

function capitalizeFirst(str) {
    return str.charAt(0).toUpperCase() + str.slice(1);
}

// =========================================================================
// EXPORT PARA DEBUGGING
// =========================================================================

// Expõe objetos globais para debugging no console
window.soraDebug = {
    appState,
    apiManager,
    wsManager,
    chartManager,
    audioManager,
    CONFIG
};

console.log('🤖 Sora Robot Frontend carregado com sucesso!');
console.log('Para debug, use: window.soraDebug');