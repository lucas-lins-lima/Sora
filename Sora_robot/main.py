#!/usr/bin/env python3
# sora_robot/main.py

"""
Sora Robot - Assistente Virtual Inteligente
Ponto de entrada principal do sistema

Uso:
    python main.py                          # Modo interativo padrão
    python main.py --mode api               # Modo servidor API
    python main.py --mode demo              # Modo demonstração
    python main.py --personality friendly   # Com personalidade específica
    python main.py --config config.json    # Com arquivo de configuração
"""

import sys
import os
import argparse
import json
import time
import signal
import asyncio
from pathlib import Path
from typing import Dict, Optional, Any
import threading

# Adiciona o diretório raiz ao Python path
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

# Imports principais
try:
    from utils.logger import get_logger, setup_logging
    from core.main_controller import MainController, SoraConfig, OperationMode
    from api.api_interface import SoraAPIServer
    import config
except ImportError as e:
    print(f"Erro ao importar módulos: {e}")
    print("Certifique-se de que todas as dependências estão instaladas.")
    sys.exit(1)

class SoraLauncher:
    """
    Lançador principal do sistema Sora.
    Gerencia inicialização, configuração e execução em diferentes modos.
    """
    
    def __init__(self):
        """Inicializa o lançador do Sora."""
        # Configura logging antes de tudo
        setup_logging()
        self.logger = get_logger(__name__)
        
        # Estado do sistema
        self.sora_controller: Optional[MainController] = None
        self.api_server: Optional[SoraAPIServer] = None
        self.is_running = False
        
        # Configurações
        self.mode = OperationMode.INTERACTIVE
        self.sora_config = SoraConfig()
        self.api_config = {}
        
        # Handlers para shutdown graceful
        self._setup_signal_handlers()
        
        self.logger.info("SoraLauncher inicializado")
    
    def _setup_signal_handlers(self):
        """Configura handlers para shutdown graceful."""
        def signal_handler(signum, frame):
            self.logger.info(f"Sinal recebido: {signum}. Iniciando shutdown graceful...")
            self.shutdown()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler)  # Termination
    
    def parse_arguments(self) -> argparse.Namespace:
        """Parse argumentos da linha de comando."""
        parser = argparse.ArgumentParser(
            description="Sora Robot - Assistente Virtual Inteligente",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Exemplos de uso:
  python main.py                                    # Modo interativo
  python main.py --mode api --port 8080           # Servidor API na porta 8080
  python main.py --mode demo --personality casual  # Demo com personalidade casual
  python main.py --config my_config.json          # Com configuração customizada
  python main.py --no-voice --no-vision           # Apenas texto
            """
        )
        
        # Modo de operação
        parser.add_argument(
            "--mode", "-m",
            choices=["interactive", "api", "demo", "single", "batch"],
            default="interactive",
            help="Modo de operação do sistema"
        )
        
        # Configurações de personalidade
        parser.add_argument(
            "--personality", "-p",
            choices=["friendly", "professional", "casual", "empathetic", "energetic"],
            default="friendly",
            help="Personalidade do Sora"
        )
        
        parser.add_argument(
            "--language", "-l",
            choices=["pt-BR", "en-US", "es-ES"],
            default="pt-BR",
            help="Idioma do sistema"
        )
        
        # Componentes
        parser.add_argument(
            "--no-voice",
            action="store_true",
            help="Desabilita síntese de voz"
        )
        
        parser.add_argument(
            "--no-vision",
            action="store_true",
            help="Desabilita processamento de visão"
        )
        
        parser.add_argument(
            "--no-animation",
            action="store_true",
            help="Desabilita animações do avatar"
        )
        
        # Configurações de API
        parser.add_argument(
            "--host",
            default="0.0.0.0",
            help="Host para servidor API (padrão: 0.0.0.0)"
        )
        
        parser.add_argument(
            "--port",
            type=int,
            default=8000,
            help="Porta para servidor API (padrão: 8000)"
        )
        
        parser.add_argument(
            "--api-key",
            help="Chave da API para autenticação"
        )
        
        # Configurações de qualidade/performance
        parser.add_argument(
            "--quality",
            choices=["low", "medium", "high"],
            default="high",
            help="Nível de qualidade do processamento"
        )
        
        parser.add_argument(
            "--debug",
            action="store_true",
            help="Habilita modo debug"
        )
        
        # Arquivo de configuração
        parser.add_argument(
            "--config", "-c",
            help="Caminho para arquivo de configuração JSON"
        )
        
        # Opções de execução
        parser.add_argument(
            "--auto-start",
            action="store_true",
            help="Inicia sistema automaticamente"
        )
        
        parser.add_argument(
            "--check-dependencies",
            action="store_true",
            help="Verifica dependências e sai"
        )
        
        parser.add_argument(
            "--version", "-v",
            action="store_true",
            help="Mostra versão e sai"
        )
        
        return parser.parse_args()
    
    def load_config_file(self, config_path: str) -> Dict:
        """Carrega configuração de arquivo JSON."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            self.logger.info(f"Configuração carregada de: {config_path}")
            return config_data
        
        except FileNotFoundError:
            self.logger.error(f"Arquivo de configuração não encontrado: {config_path}")
            return {}
        except json.JSONDecodeError as e:
            self.logger.error(f"Erro ao decodificar JSON: {e}")
            return {}
        except Exception as e:
            self.logger.error(f"Erro ao carregar configuração: {e}")
            return {}
    
    def apply_cli_config(self, args: argparse.Namespace):
        """Aplica configurações da linha de comando."""
        try:
            # Configuração do Sora
            self.sora_config.personality = args.personality
            self.sora_config.language = args.language
            self.sora_config.voice_enabled = not args.no_voice
            self.sora_config.vision_enabled = not args.no_vision
            self.sora_config.animation_enabled = not args.no_animation
            self.sora_config.debug_mode = args.debug
            
            # Modo de operação
            mode_map = {
                "interactive": OperationMode.INTERACTIVE,
                "api": OperationMode.API_MODE,
                "demo": OperationMode.DEMO_MODE,
                "single": OperationMode.SINGLE_SHOT,
                "batch": OperationMode.BATCH
            }
            self.sora_config.operation_mode = mode_map[args.mode]
            self.mode = mode_map[args.mode]
            
            # Qualidade
            quality_map = {
                "low": "low",
                "medium": "medium", 
                "high": "high"
            }
            self.sora_config.processing_quality = quality_map[args.quality]
            self.sora_config.speech_quality = quality_map[args.quality]
            
            # Configurações da API
            self.api_config = {
                'host': args.host,
                'port': args.port,
                'api_key_required': args.api_key is not None,
                'api_key': args.api_key or 'sora-default-key'
            }
            
            self.logger.info(f"Configuração aplicada: modo={args.mode}, personalidade={args.personality}")
            
        except Exception as e:
            self.logger.error(f"Erro ao aplicar configuração CLI: {e}")
    
    def apply_file_config(self, config_data: Dict):
        """Aplica configurações do arquivo."""
        try:
            # Configurações do Sora
            sora_settings = config_data.get('sora', {})
            for key, value in sora_settings.items():
                if hasattr(self.sora_config, key):
                    setattr(self.sora_config, key, value)
            
            # Configurações da API
            api_settings = config_data.get('api', {})
            self.api_config.update(api_settings)
            
            # Configurações de logging
            logging_settings = config_data.get('logging', {})
            if logging_settings:
                # Reconfigura logging se necessário
                pass
            
            self.logger.info("Configuração do arquivo aplicada")
            
        except Exception as e:
            self.logger.error(f"Erro ao aplicar configuração do arquivo: {e}")
    
    def check_dependencies(self) -> bool:
        """Verifica se todas as dependências estão disponíveis."""
        self.logger.info("Verificando dependências do sistema...")
        
        dependencies = {
            "OpenCV": True,
            "NumPy": True,
            "Threads": True,
            "Audio": True,
            "Vision": True,
            "NLP": True,
            "API": True
        }
        
        try:
            # Verifica OpenCV
            import cv2
            dependencies["OpenCV"] = True
            self.logger.info("✓ OpenCV disponível")
        except ImportError:
            dependencies["OpenCV"] = False
            self.logger.warning("✗ OpenCV não encontrado")
        
        try:
            # Verifica NumPy
            import numpy as np
            dependencies["NumPy"] = True
            self.logger.info("✓ NumPy disponível")
        except ImportError:
            dependencies["NumPy"] = False
            self.logger.error("✗ NumPy não encontrado (OBRIGATÓRIO)")
        
        try:
            # Verifica componentes de áudio
            from audio_processing.microphone_handler import MicrophoneHandler
            dependencies["Audio"] = True
            self.logger.info("✓ Componentes de áudio disponíveis")
        except Exception as e:
            dependencies["Audio"] = False
            self.logger.warning(f"✗ Problemas com áudio: {e}")
        
        try:
            # Verifica componentes de visão
            from vision_processing.camera_handler import CameraHandler
            dependencies["Vision"] = True
            self.logger.info("✓ Componentes de visão disponíveis")
        except Exception as e:
            dependencies["Vision"] = False
            self.logger.warning(f"✗ Problemas com visão: {e}")
        
        try:
            # Verifica componentes de NLP
            from nlp.dialogue_manager import DialogueManager
            dependencies["NLP"] = True
            self.logger.info("✓ Componentes de NLP disponíveis")
        except Exception as e:
            dependencies["NLP"] = False
            self.logger.error(f"✗ Problemas com NLP: {e}")
        
        try:
            # Verifica API
            from api.api_interface import SoraAPI
            dependencies["API"] = True
            self.logger.info("✓ Componentes de API disponíveis")
        except Exception as e:
            dependencies["API"] = False
            self.logger.warning(f"✗ Problemas com API: {e}")
        
        # Verifica dependências críticas
        critical_deps = ["NumPy", "NLP"]
        missing_critical = [dep for dep in critical_deps if not dependencies[dep]]
        
        if missing_critical:
            self.logger.error(f"Dependências críticas ausentes: {missing_critical}")
            return False
        
        # Ajusta configuração baseado nas dependências disponíveis
        if not dependencies["Audio"]:
            self.sora_config.voice_enabled = False
            self.logger.warning("Áudio desabilitado devido a dependências ausentes")
        
        if not dependencies["Vision"]:
            self.sora_config.vision_enabled = False
            self.logger.warning("Visão desabilitada devido a dependências ausentes")
        
        self.logger.info("Verificação de dependências concluída")
        return True
    
    def initialize_system(self) -> bool:
        """Inicializa o sistema Sora."""
        try:
            self.logger.info("Inicializando sistema Sora...")
            
            # Cria controlador principal
            self.sora_controller = MainController(self.sora_config)
            
            # Inicializa o sistema
            if not self.sora_controller.initialize():
                self.logger.error("Falha na inicialização do sistema")
                return False
            
            self.logger.info("Sistema Sora inicializado com sucesso")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro na inicialização: {e}")
            return False
    
    def start_interactive_mode(self):
        """Inicia modo interativo via console."""
        try:
            self.logger.info("Iniciando modo interativo...")
            
            if not self.sora_controller.start():
                self.logger.error("Falha ao iniciar sistema")
                return
            
            self.is_running = True
            
            print("\n" + "="*60)
            print("🤖 SORA ROBOT - ASSISTENTE VIRTUAL INTELIGENTE")
            print("="*60)
            print(f"Personalidade: {self.sora_config.personality}")
            print(f"Idioma: {self.sora_config.language}")
            print(f"Voz: {'✓' if self.sora_config.voice_enabled else '✗'}")
            print(f"Visão: {'✓' if self.sora_config.vision_enabled else '✗'}")
            print("="*60)
            print("\nDigite 'sair' para encerrar, 'help' para ajuda")
            print("Fale naturalmente ou digite sua mensagem:\n")
            
            while self.is_running:
                try:
                    # Lê input do usuário
                    user_input = input("Você: ").strip()
                    
                    if not user_input:
                        continue
                    
                    # Comandos especiais
                    if user_input.lower() in ['sair', 'exit', 'quit']:
                        break
                    elif user_input.lower() == 'help':
                        self._show_help()
                        continue
                    elif user_input.lower() == 'status':
                        self._show_status()
                        continue
                    elif user_input.lower() == 'config':
                        self._show_config()
                        continue
                    
                    # Processa mensagem
                    print("Sora: Processando...", end="", flush=True)
                    
                    response = self.sora_controller.send_message(user_input)
                    
                    print("\r" + " "*20 + "\r", end="")  # Limpa "Processando..."
                    
                    if response and response.success:
                        print(f"Sora: {response.text}")
                        
                        # Mostra informações adicionais em modo debug
                        if self.sora_config.debug_mode:
                            print(f"[DEBUG] Emoção: {response.emotion_detected}, "
                                  f"Intenção: {response.intent_detected}, "
                                  f"Confiança: {response.confidence:.2f}, "
                                  f"Tempo: {response.processing_time:.2f}s")
                    else:
                        error_msg = response.error_message if response else "Erro desconhecido"
                        print(f"Sora: Desculpe, houve um problema: {error_msg}")
                    
                    print()  # Linha vazia
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    self.logger.error(f"Erro no modo interativo: {e}")
                    print(f"Erro: {e}")
            
            print("\nEncerrando modo interativo...")
            
        except Exception as e:
            self.logger.error(f"Erro no modo interativo: {e}")
        finally:
            self.shutdown()
    
    def start_api_mode(self):
        """Inicia modo servidor API."""
        try:
            self.logger.info("Iniciando modo servidor API...")
            
            # Cria servidor API
            self.api_server = SoraAPIServer(self.sora_config, self.api_config)
            
            print("\n" + "="*60)
            print("🌐 SORA API SERVER")
            print("="*60)
            print(f"Host: {self.api_config['host']}")
            print(f"Porta: {self.api_config['port']}")
            print(f"Documentação: http://{self.api_config['host']}:{self.api_config['port']}/docs")
            print("="*60)
            
            self.is_running = True
            
            # Inicia servidor (bloqueante)
            self.api_server.start(auto_initialize_sora=True)
            
        except Exception as e:
            self.logger.error(f"Erro no modo API: {e}")
        finally:
            self.shutdown()
    
    def start_demo_mode(self):
        """Inicia modo demonstração."""
        try:
            self.logger.info("Iniciando modo demonstração...")
            
            if not self.sora_controller.start():
                self.logger.error("Falha ao iniciar sistema")
                return
            
            self.is_running = True
            
            print("\n" + "="*60)
            print("🎭 SORA ROBOT - MODO DEMONSTRAÇÃO")
            print("="*60)
            
            # Sequência de demonstração
            demo_messages = [
                "Olá! Eu sou a Sora, sua assistente virtual.",
                "Posso ajudar com informações sobre eventos, responder perguntas e conversar.",
                "Tenho processamento de voz, visão e posso expressar emoções.",
                "Como posso ajudá-lo hoje?"
            ]
            
            for i, message in enumerate(demo_messages, 1):
                print(f"\n[Demo {i}/{len(demo_messages)}]")
                print(f"Sora: {message}")
                
                if self.sora_config.voice_enabled:
                    print("(🔊 Reproduzindo áudio...)")
                    # Aqui reproduziria o áudio
                
                time.sleep(3)  # Pausa entre mensagens
            
            print("\n" + "="*60)
            print("Demonstração concluída!")
            print("Para interação completa, use: python main.py --mode interactive")
            print("="*60)
            
        except Exception as e:
            self.logger.error(f"Erro no modo demo: {e}")
        finally:
            self.shutdown()
    
    def _show_help(self):
        """Mostra ajuda no modo interativo."""
        print("\n" + "="*40)
        print("COMANDOS DISPONÍVEIS:")
        print("="*40)
        print("help     - Mostra esta ajuda")
        print("status   - Mostra status do sistema")
        print("config   - Mostra configuração atual")
        print("sair     - Encerra o programa")
        print("="*40)
        print("Ou simplesmente converse naturalmente!")
        print("="*40 + "\n")
    
    def _show_status(self):
        """Mostra status do sistema."""
        if self.sora_controller:
            status = self.sora_controller.get_status()
            print("\n" + "="*40)
            print("STATUS DO SISTEMA:")
            print("="*40)
            print(f"Estado: {status.get('controller_state', 'desconhecido')}")
            print(f"Ativo: {'Sim' if status.get('is_active', False) else 'Não'}")
            print(f"Modo: {status.get('operation_mode', 'desconhecido')}")
            
            # Componentes
            config = status.get('configuration', {})
            print(f"Voz: {'✓' if config.get('voice_enabled', False) else '✗'}")
            print(f"Visão: {'✓' if config.get('vision_enabled', False) else '✗'}")
            print(f"Animação: {'✓' if config.get('animation_enabled', False) else '✗'}")
            
            # Estatísticas
            stats = status.get('usage_stats', {})
            print(f"Interações: {stats.get('total_interactions', 0)}")
            print(f"Tempo médio: {stats.get('average_response_time', 0):.2f}s")
            print("="*40 + "\n")
    
    def _show_config(self):
        """Mostra configuração atual."""
        print("\n" + "="*40)
        print("CONFIGURAÇÃO ATUAL:")
        print("="*40)
        print(f"Nome: {self.sora_config.name}")
        print(f"Idioma: {self.sora_config.language}")
        print(f"Personalidade: {self.sora_config.personality}")
        print(f"Modo: {self.sora_config.operation_mode.value}")
        print(f"Voz: {'Habilitada' if self.sora_config.voice_enabled else 'Desabilitada'}")
        print(f"Visão: {'Habilitada' if self.sora_config.vision_enabled else 'Desabilitada'}")
        print(f"Animação: {'Habilitada' if self.sora_config.animation_enabled else 'Desabilitada'}")
        print(f"Debug: {'Sim' if self.sora_config.debug_mode else 'Não'}")
        print("="*40 + "\n")
    
    def shutdown(self):
        """Encerra sistema graciosamente."""
        try:
            self.logger.info("Iniciando shutdown do sistema...")
            self.is_running = False
            
            if self.sora_controller:
                self.sora_controller.stop()
                self.logger.info("Controlador Sora parado")
            
            if self.api_server:
                self.api_server.stop()
                self.logger.info("Servidor API parado")
            
            self.logger.info("Shutdown concluído")
            
        except Exception as e:
            self.logger.error(f"Erro durante shutdown: {e}")
    
    def run(self):
        """Executa o sistema baseado na configuração."""
        try:
            if self.mode == OperationMode.INTERACTIVE:
                self.start_interactive_mode()
            elif self.mode == OperationMode.API_MODE:
                self.start_api_mode()
            elif self.mode == OperationMode.DEMO_MODE:
                self.start_demo_mode()
            else:
                self.logger.error(f"Modo não implementado: {self.mode}")
                
        except KeyboardInterrupt:
            self.logger.info("Interrompido pelo usuário")
        except Exception as e:
            self.logger.error(f"Erro na execução: {e}")
        finally:
            self.shutdown()


def show_version():
    """Mostra informações da versão."""
    version_info = {
        "version": "1.0.0",
        "build": "2024.12.1",
        "author": "Equipe Sora",
        "description": "Assistente Virtual Inteligente",
        "components": [
            "Processamento de Visão",
            "Reconhecimento de Fala",
            "Análise de Sentimento",
            "Gerenciamento de Diálogo",
            "Síntese de Fala",
            "Animação de Avatar",
            "API REST"
        ]
    }
    
    print("\n" + "="*50)
    print("🤖 SORA ROBOT")
    print("="*50)
    for key, value in version_info.items():
        if isinstance(value, list):
            print(f"{key.title()}:")
            for item in value:
                print(f"  • {item}")
        else:
            print(f"{key.title()}: {value}")
    print("="*50 + "\n")


def main():
    """Função principal."""
    try:
        # Cria launcher
        launcher = SoraLauncher()
        
        # Parse argumentos
        args = launcher.parse_arguments()
        
        # Mostra versão se solicitado
        if args.version:
            show_version()
            return
        
        # Verifica dependências se solicitado
        if args.check_dependencies:
            success = launcher.check_dependencies()
            sys.exit(0 if success else 1)
        
        # Carrega configuração do arquivo se especificada
        if args.config:
            config_data = launcher.load_config_file(args.config)
            launcher.apply_file_config(config_data)
        
        # Aplica configuração da linha de comando
        launcher.apply_cli_config(args)
        
        # Verifica dependências
        if not launcher.check_dependencies():
            print("Dependências críticas ausentes. Use --check-dependencies para detalhes.")
            sys.exit(1)
        
        # Inicializa sistema
        if not launcher.initialize_system():
            print("Falha na inicialização do sistema.")
            sys.exit(1)
        
        # Executa sistema
        launcher.run()
        
    except KeyboardInterrupt:
        print("\nPrograma interrompido pelo usuário.")
    except Exception as e:
        print(f"Erro fatal: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()