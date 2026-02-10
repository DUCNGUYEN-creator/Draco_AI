#!/usr/bin/env python3
"""
DRACO VISION SYSTEM - Complete với lazy loading
"""
from pathlib import Path
import os
import time
import threading
from typing import Dict, Any, Optional
from PIL import Image
import sys

sys.path.insert(0, str(Path(__file__).parent))
from lazy_loader import get_lazy_loader


class DracoVision:
    """Vision system với lazy loading cho từng model"""

    def __init__(self, config):
        self.config = config
        self.initialized = False
        self.safe_mode = config.get("system.safe_mode", False)

        # Lazy loader
        self.lazy_loader = get_lazy_loader()

        # Screen capture
        self.screen_capturer = None
        self.capture_enabled = False

        # OCR path
        self.tesseract_path = None

        # Initialize basic components
        self.initialize()

    def initialize(self):
        """Khởi tạo vision system (chỉ setup cơ bản)"""
        if self.initialized:
            return True

        try:
            print("👁️ Initializing Vision System (Lazy Loading)...")

            # Tìm Tesseract path
            self._find_tesseract()

            # Register components
            self._register_components()

            # Setup screen capture nếu có quyền
            self._setup_screen_capture()

            self.initialized = True
            print("✅ Vision System initialized (models will load on-demand)")
            return True

        except Exception as e:
            print(f"❌ Vision initialization failed: {e}")
            return False

    def _find_tesseract(self):
        """Tìm Tesseract OCR"""
        # Check portable path
        bin_path = self.config.get_storage_paths()["bin"] / "tesseract" / "tesseract.exe"
        if bin_path.exists():
            self.tesseract_path = str(bin_path)
            return

        # Check system paths
        import pytesseract
        try:
            pytesseract.get_tesseract_version()
            self.tesseract_path = "system"
        except:
            self.tesseract_path = None
            print("⚠️ Tesseract OCR not found")

    def _register_components(self):
        """Đăng ký components cho lazy loading"""

        def load_ocr_engine():
            """Load OCR engine on-demand"""
            try:
                import pytesseract

                if self.tesseract_path and self.tesseract_path != "system":
                    pytesseract.pytesseract.tesseract_cmd = self.tesseract_path

                print("✅ OCR Engine loaded")
                return pytesseract

            except ImportError:
                print("❌ pytesseract not installed")
                raise

        def load_vision_model():
            """Load vision AI model on-demand"""
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                import torch

                model_id = "vikhyatk/moondream2"

                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    torch_dtype=torch.float32,
                    device_map="auto"
                )

                tokenizer = AutoTokenizer.from_pretrained(
                    model_id,
                    trust_remote_code=True
                )

                print("✅ Vision AI Model loaded")
                return {"model": model, "tokenizer": tokenizer}

            except ImportError:
                print("❌ transformers/torch not installed")
                raise

        def load_object_detector():
            """Load object detector on-demand"""
            try:
                import cv2

                # Check for YOLO files
                model_dir = self.config.get_storage_paths()["models"]
                cfg_file = model_dir / "yolov4-tiny.cfg"
                weights_file = model_dir / "yolov4-tiny.weights"

                if not cfg_file.exists() or not weights_file.exists():
                    print("⚠️ YOLO model files not found")
                    return None

                net = cv2.dnn.readNet(str(weights_file), str(cfg_file))
                net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

                print("✅ Object Detector loaded")
                return net

            except ImportError:
                print("❌ OpenCV not installed")
                return None

        def unload_vision_model(model_dict):
            """Unload vision model"""
            if model_dict and "model" in model_dict:
                try:
                    del model_dict["model"]
                    del model_dict["tokenizer"]
                except:
                    pass

        # Đăng ký components
        self.lazy_loader.register_component(
            name="ocr_engine",
            loader_func=load_ocr_engine,
            estimated_memory_mb=50
        )

        self.lazy_loader.register_component(
            name="vision_model",
            loader_func=load_vision_model,
            unloader_func=unload_vision_model,
            estimated_memory_mb=1400  # ~1.4GB
        )

        self.lazy_loader.register_component(
            name="object_detector",
            loader_func=load_object_detector,
            estimated_memory_mb=50
        )

    def _setup_screen_capture(self):
        """Setup screen capture"""
        try:
            import mss
            self.screen_capturer = mss.mss()
            self.capture_enabled = True
            print("✅ Screen capture ready")
        except ImportError:
            print("⚠️ Screen capture not available (mss not installed)")

    def analyze_image(self, image_path: str = None, image: Image = None,
                      question: str = "What's in this image?") -> Dict[str, Any]:
        """Phân tích ảnh với lazy loading"""
        start_time = time.time()

        # Load image
        if image_path and os.path.exists(image_path):
            img = Image.open(image_path)
        elif image:
            img = image
        else:
            return {"error": "No image provided"}

        result = {
            "image_size": f"{img.width}x{img.height}",
            "timestamp": time.time()
        }

        try:
            # OCR (luôn thử)
            try:
                ocr = self.lazy_loader.get_component("ocr_engine")
                if ocr:
                    # Convert to grayscale
                    gray_img = img.convert("L")
                    text = ocr.image_to_string(gray_img)
                    result["text"] = text.strip()

                    # Schedule unload
                    self.lazy_loader.schedule_unload("ocr_engine", timeout=30)
            except:
                result["text"] = ""

            # Vision AI (nếu được yêu cầu)
            if "analyze" in question.lower() or "describe" in question.lower():
                try:
                    vision_model = self.lazy_loader.get_component("vision_model")
                    if vision_model:
                        # Simple analysis
                        enc_image = vision_model["model"].encode_image(img)
                        answer = vision_model["model"].answer_question(
                            enc_image,
                            question,
                            vision_model["tokenizer"]
                        )
                        result["vision_analysis"] = answer

                        # Schedule unload
                        self.lazy_loader.schedule_unload("vision_model", timeout=60)
                except:
                    result["vision_analysis"] = "Vision analysis failed"

            # Object detection (nếu được yêu cầu)
            if "object" in question.lower() or "detect" in question.lower():
                try:
                    detector = self.lazy_loader.get_component("object_detector")
                    if detector:
                        # Simple object count
                        import cv2
                        import numpy as np

                        open_cv_image = np.array(img)
                        open_cv_image = open_cv_image[:, :, ::-1].copy()

                        blob = cv2.dnn.blobFromImage(
                            open_cv_image, 1 / 255, (416, 416),
                            swapRB=True, crop=False
                        )

                        detector.setInput(blob)
                        outputs = detector.forward(detector.getUnconnectedOutLayersNames())

                        objects = []
                        for output in outputs:
                            for detection in output:
                                scores = detection[5:]
                                confidence = scores[np.argmax(scores)]
                                if confidence > 0.5:
                                    objects.append(float(confidence))

                        result["objects_detected"] = len(objects)

                        # Schedule unload
                        self.lazy_loader.schedule_unload("object_detector", timeout=30)
                except:
                    result["objects_detected"] = 0

            result["processing_time"] = time.time() - start_time
            result["success"] = True

            return result

        except Exception as e:
            return {
                "error": str(e),
                "processing_time": time.time() - start_time,
                "success": False
            }

    def capture_screen(self, region=None):
        """Chụp màn hình"""
        if not self.capture_enabled or not self.screen_capturer:
            return None

        try:
            if region:
                screenshot = self.screen_capturer.grab(region)
            else:
                screenshot = self.screen_capturer.grab(self.screen_capturer.monitors[1])

            img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
            return img

        except Exception as e:
            print(f"Screen capture error: {e}")
            return None

    def get_status(self) -> Dict[str, Any]:
        """Lấy trạng thái vision system"""
        loader_status = self.lazy_loader.get_status()

        return {
            "initialized": self.initialized,
            "screen_capture_enabled": self.capture_enabled,
            "tesseract_available": self.tesseract_path is not None,
            "loader_status": loader_status
        }

    def unload_all(self):
        """Unload tất cả models"""
        self.lazy_loader.unload_all()

    def cleanup(self):
        """Dọn dẹp vision system"""
        self.unload_all()

        if self.screen_capturer:
            try:
                self.screen_capturer.close()
            except:
                pass

        print("Vision System cleaned up")