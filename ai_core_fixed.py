#!/usr/bin/env python3
# ------------------------------------------------------------------------------
# Copyright (c) 2026 Nguyen Huu Duc (DUCNGUYEN-creator)
# Project: Draco AI V15 Ultra
#
# This file is part of Draco AI.
# Licensed under the MIT License. See LICENSE file in the project root.
# ------------------------------------------------------------------------------
"""
DRACO AI CORE - Complete với lazy loading và memory management
"""
import os
import time
import threading
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent))
from lazy_loader import get_lazy_loader


class DracoAICore:
    """AI Core với lazy loading thông minh"""

    def __init__(self, config):
        self.config = config
        self.initialized = False

        # Models sẽ được lazy load
        self.chat_model = None
        self.model_loaded = False

        # State management
        self.state_lock = threading.RLock()
        self.current_session = f"session_{int(time.time())}"
        self.request_count = 0

        # Lazy loader
        self.lazy_loader = get_lazy_loader()

        # Register components
        self._register_components()

        # Initialize
        self.initialize()

    def _register_components(self):
        """Đăng ký components cho lazy loading"""

        def load_chat_model():
            """Load chat model on-demand"""
            try:
                from llama_cpp import Llama

                model_name = self.config.get("ai.core_model", "gemma-2-2b-it-Q4_K_M")
                model_path = self._find_model_file(model_name)

                if not model_path:
                    raise FileNotFoundError(f"Model not found: {model_name}")

                # Verify integrity
                if not self._verify_model_integrity(model_path, model_name):
                    print("⚠️ Model integrity check failed, loading anyway")

                # Load với config
                n_ctx = self.config.get("ai.context_size", 4096)

                model = Llama(
                    model_path=str(model_path),
                    n_ctx=n_ctx,
                    n_gpu_layers=0,  # CPU only for safety
                    verbose=False
                )

                print(f"✅ AI Model loaded: {model_name}")
                return model

            except ImportError as e:
                print(f"❌ Llama-cpp-python not available: {e}")
                raise
            except Exception as e:
                print(f"❌ Failed to load AI model: {e}")
                raise

        def unload_chat_model(model):
            """Unload chat model"""
            if model:
                try:
                    del model
                    print("✅ AI Model unloaded")
                except:
                    pass

        # Đăng ký với lazy loader
        self.lazy_loader.register_component(
            name="chat_model",
            loader_func=load_chat_model,
            unloader_func=unload_chat_model,
            estimated_memory_mb=1600  # ~1.6GB
        )

    def initialize(self):
        """Khởi tạo AI Core (chỉ setup, không load model)"""
        print("🧠 Initializing AI Core (Lazy Loading)...")

        # Tạo thư mục models nếu chưa có
        model_dir = self.config.get_storage_paths()["models"]
        model_dir.mkdir(parents=True, exist_ok=True)

        self.initialized = True
        print("✅ AI Core initialized (model will load on-demand)")
        return True

    def _find_model_file(self, model_name: str) -> Optional[Path]:
        """Tìm model file"""
        model_dir = self.config.get_storage_paths()["models"]
        extensions = ['.gguf', '.bin', '.pt', '.safetensors']

        for ext in extensions:
            model_path = model_dir / f"{model_name}{ext}"
            if model_path.exists():
                return model_path

        # Check without extension
        model_path = model_dir / model_name
        if model_path.exists():
            return model_path

        print(f"⚠️ Model not found: {model_name}")
        print(f"   Expected in: {model_dir}")
        return None

    def _verify_model_integrity(self, model_path: Path, model_name: str) -> bool:
        """Verify model integrity"""
        try:
            hashes = self.config.get("ai.model_hashes", {})
            expected_hash = hashes.get(model_name)

            if not expected_hash or expected_hash.startswith("to_be_filled"):
                print("⚠️ No hash available, skipping verification")
                return True

            # Calculate hash
            sha256 = hashlib.sha256()
            with open(model_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256.update(chunk)

            actual_hash = sha256.hexdigest()

            if expected_hash == actual_hash:
                print("✅ Model integrity verified")
                return True
            else:
                print(f"⚠️ Model hash mismatch!")
                print(f"   Expected: {expected_hash[:16]}...")
                print(f"   Got: {actual_hash[:16]}...")
                return False

        except Exception as e:
            print(f"⚠️ Integrity check failed: {e}")
            return True

    def process_query(self, query: str, context: Dict = None) -> Dict[str, Any]:
        """Xử lý query với lazy loading"""
        start_time = time.time()
        self.request_count += 1
        request_id = f"{self.current_session}_{self.request_count}"

        print(f"📝 Processing query {request_id}: {query[:50]}...")

        try:
            # Load model on-demand
            model = self.lazy_loader.get_component("chat_model")

            # Prepare prompt
            prompt = self._prepare_prompt(query, context)

            # Generate response
            response = model(
                prompt,
                max_tokens=self.config.get("ai.max_tokens", 1024),
                temperature=self.config.get("ai.temperature", 0.7),
                top_p=self.config.get("ai.top_p", 0.95),
                echo=False
            )

            response_text = response["choices"][0]["text"].strip()
            processing_time = time.time() - start_time

            # Schedule unload sau 60s idle
            self.lazy_loader.schedule_unload("chat_model", timeout=60)

            # Check dangerous commands
            danger_check = self.config.check_dangerous_command(response_text)

            result = {
                "success": True,
                "response": response_text,
                "processing_time": processing_time,
                "error": False,
                "model": self.config.get("ai.core_model"),
                "request_id": request_id,
                "session_id": self.current_session,
                "danger_check": danger_check
            }

            if danger_check["is_dangerous"]:
                result["warning"] = f"⚠️ Contains dangerous keywords: {danger_check['keywords_found']}"

            print(f"✅ Request {request_id} completed in {processing_time:.2f}s")
            return result

        except Exception as e:
            return {
                "success": False,
                "response": f"AI processing error: {str(e)}",
                "processing_time": time.time() - start_time,
                "error": True,
                "error_code": "PROCESSING_ERROR",
                "request_id": request_id
            }

    def _prepare_prompt(self, query: str, context: Dict = None) -> str:
        """Chuẩn bị prompt"""
        base_prompt = f"""You are Draco, a helpful AI assistant.

User: {query}

Draco: """

        if context:
            context_str = "\n".join([f"{k}: {v}" for k, v in context.items()])
            base_prompt = f"Context:\n{context_str}\n\n{base_prompt}"

        return base_prompt

    def get_status(self) -> Dict[str, Any]:
        """Lấy trạng thái AI Core"""
        loader_status = self.lazy_loader.get_status()

        return {
            "initialized": self.initialized,
            "session_id": self.current_session,
            "request_count": self.request_count,
            "loader_status": loader_status.get("chat_model", {})
        }

    def unload_model(self):
        """Unload model ngay lập tức"""
        self.lazy_loader.unload_component("chat_model")

    def cleanup(self):
        """Dọn dẹp AI Core"""
        self.unload_model()
        print("AI Core cleaned up")