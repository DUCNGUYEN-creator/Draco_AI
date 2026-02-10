import hashlib
import json
import os
from pathlib import Path


def calculate_sha256(file_path):
    """Tính dấu vân tay SHA-256 cho file nặng (3GB - 6GB+)"""
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            # Đọc từng khối 4096 bytes để bảo vệ RAM
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except Exception as e:
        print(f"❌ Lỗi khi đọc file {file_path.name}: {e}")
        return None


def update_hashes():
    # 1. Đường dẫn thư mục models (Đức nhớ để weights vào đây nhé)
    # Theo config của bạn: C:\Users\<Name>\DracoAI_Data\models
    model_dir = Path.home() / "DracoAI_Data" / "models"
    json_path = Path(__file__).parent / "model_hashes.json"

    if not model_dir.exists():
        print(f"⚠️ Thư mục models không tồn tại tại: {model_dir}")
        return

    # 2. Quét các file model thực tế
    found_hashes = {}
    extensions = ['.gguf', '.weights', '.bin', '.pth']

    print(f"🔍 Đang quét 'linh hồn' của Draco tại: {model_dir}...")

    for file in model_dir.glob("*"):
        if file.suffix.lower() in extensions:
            print(f"⏳ Đang lấy hash cho {file.name} (vui lòng đợi)...")
            h = calculate_sha256(file)
            if h:
                found_hashes[file.stem] = h  # Lưu theo tên file (không đuôi)

    # 3. Ghi vào file JSON để ai_core_fixed.py sử dụng
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(found_hashes, f, indent=4)

    print(f"✅ Đã cập nhật {len(found_hashes)} model vào {json_path.name}!")


if __name__ == "__main__":
    update_hashes()