"""
تنظیمات بهینه‌سازی برای سیستم‌های با حافظه محدود (16GB RAM)
"""

import os
import torch

# تنظیمات کلی سیستم
SYSTEM_CONFIG = {
    "max_ram_usage_gb": 16,
    "target_ram_usage_gb": 12,  # 75% از کل RAM
    "enable_memory_optimization": True,
    "use_cpu_for_heavy_models": True,
    "batch_size_reduction": True
}

# تنظیمات مدل‌های YOLO
YOLO_CONFIG = {
    "model_size": "nano",  # nano, small, medium, large, xlarge
    "model_path": "yolov8n.pt",  # کوچک‌ترین مدل
    "confidence_threshold": 0.3,  # افزایش آستانه برای کاهش تشخیص‌ها
    "iou_threshold": 0.5,
    "max_detections": 50,  # کاهش تعداد تشخیص‌ها
    "image_size": 640,
    "device": "cpu" if not torch.cuda.is_available() else "cuda"
}

# تنظیمات مدل‌های SAM
SAM_CONFIG = {
    "use_small_model": True,  # استفاده از مدل کوچک‌تر
    "model_type": "vit_b",  # vit_b کوچک‌تر از vit_h
    "checkpoint_path": "sam2.1_hiera_small.pt",  # مدل کوچک‌تر
    "points_per_side": 8,  # کاهش نقاط (پیش‌فرض: 16)
    "pred_iou_thresh": 0.88,  # افزایش آستانه
    "stability_score_thresh": 0.95,
    "min_mask_region_area": 200,  # افزایش حداقل اندازه
    "device": "cpu" if not torch.cuda.is_available() else "cuda"
}

# تنظیمات مدل‌های LLM
LLM_CONFIG = {
    "preferred_providers": [
        "G4FProvider",  # مدل‌های رایگان و کوچک
        "OpenRouterProvider",  # مدل‌های کوچک و رایگان
        "AnthropicProvider"  # فقط مدل‌های کوچک
    ],
    "small_models": [
        "google/gemini-2.0-flash-001",
        "claude-3.5-haiku",
        "mistralai/mistral-small-3.1-24b-instruct:free"
    ],
    "max_tokens": 2048,  # کاهش تعداد توکن‌ها
    "temperature": 0.7
}

# تنظیمات RAG
RAG_CONFIG = {
    "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",  # مدل کوچک
    "chunk_size": 500,  # کاهش اندازه chunk
    "chunk_overlap": 100,
    "max_documents": 1000,  # محدود کردن تعداد اسناد
    "use_faiss_cpu": True  # استفاده از FAISS CPU
}

# تنظیمات پردازش تصویر
IMAGE_CONFIG = {
    "max_image_size": 1024,  # کاهش اندازه تصاویر
    "compression_quality": 85,  # فشرده‌سازی تصاویر
    "use_pillow_optimization": True,
    "cache_images": False  # عدم کش کردن تصاویر
}

# تنظیمات حافظه
MEMORY_CONFIG = {
    "enable_garbage_collection": True,
    "clear_cache_interval": 10,  # پاک کردن کش هر 10 درخواست
    "max_cache_size_mb": 500,
    "use_memory_mapping": False
}

def get_optimized_settings():
    """دریافت تنظیمات بهینه بر اساس سیستم"""
    return {
        "system": SYSTEM_CONFIG,
        "yolo": YOLO_CONFIG,
        "sam": SAM_CONFIG,
        "llm": LLM_CONFIG,
        "rag": RAG_CONFIG,
        "image": IMAGE_CONFIG,
        "memory": MEMORY_CONFIG
    }

def is_low_memory_system():
    """بررسی اینکه آیا سیستم حافظه محدودی دارد"""
    import psutil
    total_ram = psutil.virtual_memory().total / (1024**3)  # GB
    return total_ram <= 16

def get_recommended_model_size():
    """دریافت اندازه مدل توصیه شده بر اساس سیستم"""
    if is_low_memory_system():
        return "nano"  # کوچک‌ترین مدل
    else:
        return "small"  # مدل کوچک 