#!/usr/bin/env python3
# ------------------------------------------------------------------------------
# Copyright (c) 2026 Nguyen Huu Duc (DUCNGUYEN-creator)
# Project: Draco AI V15 Ultra
#
# This file is part of Draco AI.
# Licensed under the MIT License. See LICENSE file in the project root.
# ------------------------------------------------------------------------------
"""
DRACO V15 ULTRA - FINAL PRODUCTION EDITION
Main entry point với lazy loading hoàn chỉnh
"""
import os
import sys
import time
import signal
from pathlib import Path

# Setup paths
sys.path.insert(0, str(Path(__file__).parent))


class DracoUltra:
    """Main Draco class với lazy loading hoàn chỉnh"""

    def __init__(self):
        self.version = "V15.3.0-Ultra-Final"
        self.start_time = time.time()

        # Parse arguments
        self.args = self._parse_args()

        # Components
        self.config = None
        self.ai_core = None
        self.vision = None
        self.automation = None
        self.search = None
        self.voice = None
        self.gui = None

        # State
        self.running = False
        self.initialized = False

        # Signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        print(f"\n🚀 DRACO {self.version} - Lazy Loading Edition")
        print("=" * 50)

    def _parse_args(self):
        """Parse command line arguments"""
        args = {
            "safe": "--safe" in sys.argv,
            "cli": "--cli" in sys.argv,
            "no-voice": "--no-voice" in sys.argv,
            "no-automation": "--no-automation" in sys.argv,
            "no-vision": "--no-vision" in sys.argv,
            "help": "--help" in sys.argv
        }

        if args["help"]:
            self._show_help()
            sys.exit(0)

        return args

    def _show_help(self):
        """Show help"""
        print("""
DRACO V15 ULTRA - AI Assistant

Usage: python main_final.py [options]

Options:
  --safe           Safe mode (no automation)
  --cli            CLI mode only
  --no-voice       Disable voice activation
  --no-vision      Disable vision system
  --no-automation  Disable automation
  --help           Show this help

Examples:
  python main_final.py              # Normal mode
  python main_final.py --safe       # Safe mode
  python main_final.py --cli        # CLI mode
        """)

    def initialize(self):
        """Khởi tạo Draco với lazy loading"""
        try:
            print("\n🔄 Initializing Draco...")

            # Load configuration
            from config import DracoConfig
            self.config = DracoConfig()

            print(f"📁 Storage: {self.config.storage_path}")
            print(f"🔒 Safe Mode: {self.args['safe']}")

            # Initialize AI Core
            print("\n🧠 Initializing AI Core...")
            from ai_core_fixed import DracoAICore
            self.ai_core = DracoAICore(self.config)

            # Initialize Vision System
            if not self.args["no-vision"] and self.config.get("ai.enable_vision", True):
                print("👁️ Initializing Vision System...")
                from vision_system import DracoVision
                self.vision = DracoVision(self.config)

            # Initialize Automation
            if not self.args["no-automation"] and not self.args["safe"]:
                print("🤖 Initializing Automation System...")
                from automation_system import DracoAutomation
                self.automation = DracoAutomation(self.config)

            # Initialize Search Agent
            if self.config.get("ai.enable_search", True):
                print("🔍 Initializing Search Agent...")
                from search_agent import DracoSearchAgent
                self.search = DracoSearchAgent(self.config)

            # Initialize Voice Activation
            if not self.args["no-voice"] and self.config.get("ai.enable_voice", True):
                print("🎤 Initializing Voice Activation...")
                from voice_activation_fixed import get_voice_activation
                self.voice = get_voice_activation(self.config)

                if self.voice:
                    def on_voice_command(text):
                        print(f"\n🔊 Voice command: {text}")
                        self._handle_voice_command(text)

                    self.voice.on_activation = on_voice_command
                    self.voice.start()

            self.initialized = True

            print("\n" + "=" * 50)
            print("✅ DRACO INITIALIZED SUCCESSFULLY!")
            print("=" * 50)

            # Show system info
            self._show_system_info()

            return True

        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _show_system_info(self):
        """Hiển thị system info"""
        print("\n📊 SYSTEM INFORMATION:")
        print(f"  Version: {self.version}")
        print(f"  Safe Mode: {'Yes' if self.args['safe'] else 'No'}")
        print(f"  CLI Mode: {'Yes' if self.args['cli'] else 'No'}")

        if self.ai_core:
            status = self.ai_core.get_status()
            print(f"  AI Core: Ready (lazy loaded)")

        if self.vision:
            status = self.vision.get_status()
            print(f"  Vision: {'Ready' if status['initialized'] else 'Disabled'}")

        if self.automation:
            status = self.automation.get_status()
            print(f"  Automation: {'Ready' if status['initialized'] else 'Disabled'}")

        if self.voice:
            status = self.voice.get_status()
            print(f"  Voice: {'Active' if status['listening'] else 'Inactive'}")

        if self.search:
            print(f"  Search: Ready")

        print("\n💡 TIPS:")
        print("  • Models load on-demand to save RAM")
        print("  • Unused components auto-unload after 60s")
        print("  • Type 'help' for available commands")
        print()

    def _handle_voice_command(self, text):
        """Xử lý lệnh giọng nói"""
        if not self.ai_core:
            return

        try:
            # Extract command
            keyword = self.config.get("voice.wake_word", "hey draco")
            if keyword in text.lower():
                command = text.lower().split(keyword, 1)[1].strip()
            else:
                command = text

            print(f"🎤 Processing: {command}")

            # Process with AI
            response = self.ai_core.process_query(command)

            if response.get("success"):
                print(f"🤖 Draco: {response['response']}")

                # Execute actions based on response
                self._execute_voice_action(command, response['response'])

        except Exception as e:
            print(f"Voice command error: {e}")

    def _execute_voice_action(self, command: str, response: str):
        """Thực thi action từ voice command"""
        command_lower = command.lower()

        try:
            if "screenshot" in command_lower or "capture" in command_lower:
                if self.vision:
                    screenshot = self.vision.capture_screen()
                    if screenshot:
                        print("📸 Screenshot captured")

            elif "search" in command_lower:
                query = command_lower.replace("search", "").replace("for", "").strip()
                if self.search and query:
                    result = self.search.search(query)
                    if result.get("success"):
                        print(f"🔍 Found {len(result['results'])} results")

            elif "weather" in command_lower:
                if self.search:
                    result = self.search.search_weather()
                    if result.get("success"):
                        print(f"🌤️ Weather: {result['weather_info']}")

            elif "click" in command_lower and self.automation:
                # Simple click at center
                self.automation.execute_command("click", {"x": 500, "y": 500})
                print("🖱️ Clicked at center")

        except Exception as e:
            print(f"Action execution error: {e}")

    def _signal_handler(self, signum, frame):
        """Xử lý signals"""
        print(f"\n⚠️ Signal {signum} received, shutting down...")
        self.shutdown()

    def run_cli(self):
        """Chạy CLI mode"""
        print("\n💻 DRACO CLI MODE")
        print("Type 'help' for commands, 'quit' to exit\n")

        self.running = True

        while self.running:
            try:
                command = input("Draco> ").strip()

                if not command:
                    continue

                command_lower = command.lower()

                if command_lower in ["quit", "exit", "q"]:
                    self.shutdown()
                    break

                elif command_lower == "help":
                    self._show_cli_help()

                elif command_lower == "status":
                    self._show_status()

                elif command_lower.startswith("ask "):
                    question = command[4:].strip()
                    if self.ai_core:
                        response = self.ai_core.process_query(question)
                        if response.get("success"):
                            print(f"\n🤖 Draco: {response['response']}")
                        else:
                            print(f"\n❌ Error: {response.get('response', 'Unknown')}")
                    else:
                        print("❌ AI Core not available")

                elif command_lower.startswith("search "):
                    query = command[7:].strip()
                    if self.search:
                        result = self.search.search(query)
                        if result.get("success"):
                            print(f"\n🔍 Found {len(result['results'])} results:")
                            for i, r in enumerate(result['results'][:3], 1):
                                print(f"  {i}. {r['title']}")
                                print(f"     {r['snippet'][:80]}...")
                                print(f"     {r['url']}")
                                print()
                        else:
                            print(f"\n❌ Search failed: {result.get('error', 'Unknown')}")
                    else:
                        print("❌ Search not available")

                elif command_lower == "screenshot":
                    if self.vision:
                        screenshot = self.vision.capture_screen()
                        if screenshot:
                            print("📸 Screenshot captured")
                        else:
                            print("❌ Failed to capture screenshot")
                    else:
                        print("❌ Vision system not available")

                elif command_lower == "weather":
                    if self.search:
                        result = self.search.search_weather()
                        if result.get("success"):
                            print(f"\n🌤️ Weather: {result['weather_info']}")
                        else:
                            print(f"\n❌ Weather check failed")
                    else:
                        print("❌ Search not available")

                elif command_lower == "memory":
                    self._show_memory_status()

                elif command_lower == "voice":
                    if self.voice:
                        if self.voice.listening:
                            self.voice.stop()
                            print("🎤 Voice stopped")
                        else:
                            self.voice.start()
                            print("🎤 Voice started")
                    else:
                        print("❌ Voice not available")

                else:
                    print("❌ Unknown command. Type 'help' for available commands.")

            except KeyboardInterrupt:
                print("\n⚠️ Interrupted")
                continue

            except EOFError:
                print("\n👋 Goodbye!")
                self.shutdown()
                break

            except Exception as e:
                print(f"❌ Error: {e}")

    def _show_cli_help(self):
        """Hiển thị CLI help"""
        print("""
Available Commands:
  help                    - Show this help
  status                  - Show system status
  memory                  - Show memory usage
  ask <question>          - Ask Draco AI
  search <query>          - Web search
  screenshot              - Capture screen
  weather                 - Check weather
  voice                   - Toggle voice activation
  quit/exit/q             - Exit Draco
        """)

    def _show_status(self):
        """Hiển thị status"""
        uptime = time.time() - self.start_time
        hours, rem = divmod(uptime, 3600)
        minutes, seconds = divmod(rem, 60)

        print(f"\n📊 DRACO STATUS:")
        print(f"  Version: {self.version}")
        print(f"  Uptime: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        print(f"  Mode: {'CLI' if self.args['cli'] else 'GUI'}")
        print(f"  Safe Mode: {'Yes' if self.args['safe'] else 'No'}")

        if self.ai_core:
            status = self.ai_core.get_status()
            print(f"  🤖 AI Core: Ready (Session: {status['session_id']})")
            print(f"     Requests: {status['request_count']}")

        if self.vision:
            status = self.vision.get_status()
            print(f"  👁️ Vision: {'Ready' if status['initialized'] else 'Disabled'}")

        if self.automation:
            status = self.automation.get_status()
            print(f"  🤖 Automation: {'Ready' if status['initialized'] else 'Disabled'}")

        if self.voice:
            status = self.voice.get_status()
            print(f"  🎤 Voice: {'Active' if status['listening'] else 'Inactive'}")

        if self.search:
            print(f"  🔍 Search: Ready")

        print()

    def _show_memory_status(self):
        """Hiển thị memory status"""
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / (1024 * 1024)

        print(f"\n💾 MEMORY USAGE: {memory_mb:.1f} MB")

        # Show lazy loader status
        from lazy_loader import get_lazy_loader
        loader = get_lazy_loader()
        status = loader.get_status()

        print("  Loaded Components:")
        for name, comp in status.items():
            state = comp['state']
            idle = comp['idle_seconds']
            if state == 'loaded':
                print(f"    • {name}: LOADED (idle {idle:.1f}s)")
            elif state == 'loading':
                print(f"    • {name}: LOADING...")
            else:
                print(f"    • {name}: NOT LOADED")

        print()

    def run_gui(self):
        """Chạy GUI mode"""
        try:
            import customtkinter as ctk
            from gui import DracoGUI

            print("🖥️ Starting GUI...")

            self.gui = DracoGUI(
                ai_core=self.ai_core,
                vision=self.vision,
                automation=self.automation,
                search=self.search,
                config=self.config
            )

            self.running = True
            self.gui.run()

        except ImportError as e:
            print(f"⚠️ GUI not available: {e}")
            print("Falling back to CLI mode...")
            self.run_cli()
        except Exception as e:
            print(f"❌ GUI failed: {e}")
            self.run_cli()

    def shutdown(self):
        """Shutdown Draco"""
        print("\n🔴 Shutting down Draco...")
        self.running = False

        # Cleanup components
        if self.voice:
            self.voice.cleanup()

        if self.ai_core:
            self.ai_core.cleanup()

        if self.vision:
            self.vision.cleanup()

        if self.automation:
            self.automation.cleanup()

        if self.search:
            self.search.cleanup()

        # Cleanup lazy loader
        from lazy_loader import get_lazy_loader
        loader = get_lazy_loader()
        loader.cleanup()

        uptime = time.time() - self.start_time
        print(f"👋 Draco stopped. Uptime: {uptime:.1f}s")

        sys.exit(0)


def main():
    """Main entry point"""
    # Create Draco instance
    draco = DracoUltra()

    # Initialize
    if not draco.initialize():
        print("❌ Failed to initialize Draco")
        return 1

    # Run in CLI or GUI mode
    if draco.args["cli"]:
        draco.run_cli()
    else:
        draco.run_gui()

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)