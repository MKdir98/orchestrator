import json
import os
import hashlib
from typing import Dict, Optional, List
import cv2
import numpy as np


class ElementMemoryService:
    """
    سرویس مدیریت حافظه المان‌ها برای تمام کاربرها
    """
    
    def __init__(self):
        self.memory_file_path = "orchestrator_data/element_memory.json"
        self.ensure_memory_file_exists()
    
    def ensure_memory_file_exists(self):
        """اطمینان از وجود فایل حافظه المان‌ها"""
        os.makedirs(os.path.dirname(self.memory_file_path), exist_ok=True)
        if not os.path.exists(self.memory_file_path):
            self._save_memory({})
    
    def _load_memory(self) -> Dict:
        """بارگذاری حافظه المان‌ها از فایل"""
        try:
            with open(self.memory_file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    
    def _save_memory(self, memory_data: Dict):
        """ذخیره حافظه المان‌ها در فایل"""
        with open(self.memory_file_path, 'w', encoding='utf-8') as f:
            json.dump(memory_data, f, indent=2, ensure_ascii=False)
    
    def generate_element_signature(self, image_crop: np.ndarray, coordinates: List[int]) -> str:
        """
        تولید امضای منحصر به فرد برای المان بر اساس ویژگی‌های بصری
        """
        try:
            # تبدیل به سایز ثابت برای مقایسه
            resized = cv2.resize(image_crop, (64, 64))
            
            # محاسبه histogram
            hist = cv2.calcHist([resized], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist_normalized = cv2.normalize(hist, hist).flatten()
            
            # اضافه کردن ابعاد و نسبت طول به عرض
            height, width = image_crop.shape[:2]
            aspect_ratio = width / height if height > 0 else 1.0
            size_info = f"{width}x{height}_{aspect_ratio:.2f}"
            
            # ترکیب اطلاعات برای تولید hash
            signature_data = f"{hist_normalized.tobytes().hex()[:32]}_{size_info}"
            return hashlib.md5(signature_data.encode()).hexdigest()
            
        except Exception as e:
            # fallback: استفاده از مختصات
            coords_str = f"{coordinates[0]}_{coordinates[1]}_{coordinates[2]}_{coordinates[3]}"
            return hashlib.md5(coords_str.encode()).hexdigest()
    
    def check_element_in_memory(self, signature: str) -> Optional[Dict]:
        """
        بررسی وجود المان در حافظه بر اساس امضا
        """
        memory = self._load_memory()
        return memory.get(signature)
    
    def save_element_to_memory(self, signature: str, element_info: Dict):
        """
        ذخیره اطلاعات المان در حافظه
        """
        memory = self._load_memory()
        
        # اضافه کردن timestamp
        import time
        element_info['discovered_at'] = time.time()
        element_info['discovery_count'] = memory.get(signature, {}).get('discovery_count', 0) + 1
        
        memory[signature] = element_info
        self._save_memory(memory)
    
    def find_similar_elements(self, signature: str, threshold: float = 0.8) -> List[Dict]:
        """
        یافتن المان‌های مشابه در حافظه
        """
        memory = self._load_memory()
        similar_elements = []
        
        for stored_signature, element_info in memory.items():
            # محاسبه شباهت ساده (می‌توان بهبود داد)
            if self._calculate_signature_similarity(signature, stored_signature) > threshold:
                similar_elements.append({
                    'signature': stored_signature,
                    'element_info': element_info,
                    'similarity': self._calculate_signature_similarity(signature, stored_signature)
                })
        
        return sorted(similar_elements, key=lambda x: x['similarity'], reverse=True)
    
    def _calculate_signature_similarity(self, sig1: str, sig2: str) -> float:
        """
        محاسبه شباهت بین دو امضا
        """
        if sig1 == sig2:
            return 1.0
        
        # مقایسه ساده بر اساس کاراکترهای مشترک
        common_chars = sum(1 for a, b in zip(sig1, sig2) if a == b)
        return common_chars / max(len(sig1), len(sig2))
    
    def get_memory_stats(self) -> Dict:
        """
        آمار حافظه المان‌ها
        """
        memory = self._load_memory()
        return {
            'total_elements': len(memory),
            'memory_file_size': os.path.getsize(self.memory_file_path) if os.path.exists(self.memory_file_path) else 0,
            'most_discovered_elements': sorted(
                [(k, v.get('discovery_count', 0), v.get('element_type', 'unknown')) 
                 for k, v in memory.items()], 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
        }
    
    def clean_old_elements(self, max_age_days: int = 30):
        """
        پاک‌سازی المان‌های قدیمی
        """
        import time
        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 60 * 60
        
        memory = self._load_memory()
        cleaned_memory = {}
        
        for signature, element_info in memory.items():
            discovered_at = element_info.get('discovered_at', 0)
            if current_time - discovered_at < max_age_seconds:
                cleaned_memory[signature] = element_info
        
        self._save_memory(cleaned_memory)
        return len(memory) - len(cleaned_memory)  # تعداد المان‌های حذف شده


# نمونه singleton
_element_memory_service = None

def get_element_memory_service() -> ElementMemoryService:
    global _element_memory_service
    if _element_memory_service is None:
        _element_memory_service = ElementMemoryService()
    return _element_memory_service 