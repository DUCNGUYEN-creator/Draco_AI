#!/usr/bin/env python3
# ------------------------------------------------------------------------------
# Copyright (c) 2026 Nguyen Huu Duc (DUCNGUYEN-creator)
# Project: Draco AI V15 Ultra
#
# This file is part of Draco AI.
# Licensed under the MIT License. See LICENSE file in the project root.
# ------------------------------------------------------------------------------
"""
DRACO GUI - Simple GUI interface
"""
import customtkinter as ctk
import threading
import time


class DracoGUI:
    """Simple GUI for Draco AI"""

    def __init__(self, ai_core, vision, automation, search, config):
        self.ai_core = ai_core
        self.vision = vision
        self.automation = automation
        self.search = search
        self.config = config

        # Window
        self.window = None
        self.chat_text = None
        self.input_entry = None

        # State
        self.running = True

    def run(self):
        """Run the GUI"""
        # Setup appearance
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Create window
        self.window = ctk.CTk()
        self.window.title(f"DRACO AI - {self.config.version}")
        self.window.geometry("1000x700")

        # Configure grid
        self.window.grid_rowconfigure(0, weight=1)
        self.window.grid_columnconfigure(0, weight=1)

        # Main frame
        main_frame = ctk.CTkFrame(self.window)
        main_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)

        # Chat display
        self.chat_text = ctk.CTkTextbox(main_frame, wrap="word")
        self.chat_text.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="nsew")
        self._add_chat_message("system", "🤖 Draco AI Assistant Ready!")
        self._add_chat_message("system", "Type your message below and press Enter.")

        # Input frame
        input_frame = ctk.CTkFrame(main_frame)
        input_frame.grid(row=1, column=0, padx=10, pady=(5, 10), sticky="ew")
        input_frame.grid_columnconfigure(0, weight=1)

        # Input entry
        self.input_entry = ctk.CTkEntry(input_frame, placeholder_text="Type your message here...")
        self.input_entry.grid(row=0, column=0, padx=(0, 10), pady=10, sticky="ew")
        self.input_entry.bind("<Return>", lambda e: self._process_input())

        # Send button
        send_button = ctk.CTkButton(input_frame, text="Send", command=self._process_input)
        send_button.grid(row=0, column=1, pady=10)

        # Status bar
        status_frame = ctk.CTkFrame(self.window)
        status_frame.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="ew")

        status_label = ctk.CTkLabel(
            status_frame,
            text="Ready | Voice: Active | Memory: Optimized"
        )
        status_label.pack(side="left", padx=10, pady=5)

        # Handle window close
        self.window.protocol("WM_DELETE_WINDOW", self._on_closing)

        # Start GUI
        self.window.mainloop()

    def _add_chat_message(self, sender: str, message: str):
        """Add message to chat"""
        self.chat_text.configure(state="normal")

        if sender == "user":
            self.chat_text.insert("end", f"\n👤 You: {message}\n")
        elif sender == "ai":
            self.chat_text.insert("end", f"\n🤖 Draco: {message}\n")
        else:
            self.chat_text.insert("end", f"\n⚡ {message}\n")

        self.chat_text.see("end")
        self.chat_text.configure(state="disabled")

    def _process_input(self):
        """Process user input"""
        message = self.input_entry.get().strip()
        if not message:
            return

        # Clear input
        self.input_entry.delete(0, "end")

        # Add user message
        self._add_chat_message("user", message)

        # Process in background
        thread = threading.Thread(target=self._process_message, args=(message,))
        thread.daemon = True
        thread.start()

    def _process_message(self, message: str):
        """Process message in background"""
        try:
            # Handle commands
            if message.lower() == "help":
                help_text = """
Available commands:
• ask [question] - Ask Draco AI
• search [query] - Web search
• screenshot - Capture screen
• weather - Check weather
• status - System status
• memory - Memory usage
                """
                self._add_chat_message("ai", help_text)
                return

            elif message.lower() == "status":
                import psutil
                memory = psutil.virtual_memory()
                status_text = f"System Status:\n"
                status_text += f"• Memory: {memory.percent}% used\n"
                status_text += f"• CPU: {psutil.cpu_percent()}% used\n"

                if self.ai_core:
                    ai_status = self.ai_core.get_status()
                    status_text += f"• AI Core: Ready\n"
                    status_text += f"• Requests: {ai_status.get('request_count', 0)}\n"

                self._add_chat_message("ai", status_text)
                return

            elif message.lower() == "memory":
                import psutil
                process = psutil.Process()
                memory_mb = process.memory_info().rss / (1024 * 1024)

                from lazy_loader import get_lazy_loader
                loader = get_lazy_loader()
                status = loader.get_status()

                memory_text = f"Memory Usage: {memory_mb:.1f} MB\n"
                memory_text += "Loaded components:\n"

                for name, comp in status.items():
                    if comp['state'] == 'loaded':
                        idle = comp['idle_seconds']
                        memory_text += f"• {name}: {comp['memory_mb']}MB (idle {idle:.1f}s)\n"

                self._add_chat_message("ai", memory_text)
                return

            elif message.lower().startswith("search "):
                query = message[7:].strip()
                if self.search:
                    self._add_chat_message("system", f"Searching for: {query}")
                    result = self.search.search(query)

                    if result.get("success"):
                        response = f"Found {len(result['results'])} results:\n"
                        for i, r in enumerate(result['results'][:2], 1):
                            response += f"{i}. {r['title']}\n"
                            response += f"   {r['snippet'][:60]}...\n"
                        self._add_chat_message("ai", response)
                    else:
                        self._add_chat_message("ai", f"Search failed: {result.get('error', 'Unknown')}")
                else:
                    self._add_chat_message("ai", "Search not available")
                return

            elif message.lower() == "screenshot":
                if self.vision:
                    self._add_chat_message("system", "Capturing screenshot...")
                    screenshot = self.vision.capture_screen()
                    if screenshot:
                        self._add_chat_message("ai", "Screenshot captured successfully!")
                    else:
                        self._add_chat_message("ai", "Failed to capture screenshot")
                else:
                    self._add_chat_message("ai", "Vision system not available")
                return

            elif message.lower() == "weather":
                if self.search:
                    self._add_chat_message("system", "Checking weather...")
                    result = self.search.search_weather()
                    if result.get("success"):
                        self._add_chat_message("ai", f"Weather: {result['weather_info']}")
                    else:
                        self._add_chat_message("ai", "Weather check failed")
                else:
                    self._add_chat_message("ai", "Search not available")
                return

            # Use AI Core for other messages
            if self.ai_core:
                self._add_chat_message("system", "Thinking...")
                response = self.ai_core.process_query(message)

                if response.get("success"):
                    self._add_chat_message("ai", response['response'])
                else:
                    self._add_chat_message("ai", f"Error: {response.get('response', 'Unknown error')}")
            else:
                self._add_chat_message("ai", "AI Core not available")

        except Exception as e:
            self._add_chat_message("ai", f"Error: {str(e)}")

    def _on_closing(self):
        """Handle window closing"""
        self.running = False
        if self.window:
            self.window.quit()
            self.window.destroy()

    def cleanup(self):
        """Cleanup GUI"""
        self._on_closing()