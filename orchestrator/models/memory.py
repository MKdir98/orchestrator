from datetime import datetime, timedelta
from typing import List, Dict, Any
import json
import os

class Memory:
    def __init__(self, user_id: int, max_short_term_items: int = 10, short_term_duration: timedelta = timedelta(hours=1)):
        self.user_id = user_id
        self.short_term_memory: List[Dict[str, Any]] = []
        self.long_term_memory: List[Dict[str, Any]] = []
        self.task_relationships: List[Dict[str, Any]] = []
        self.max_short_term_items = max_short_term_items
        self.short_term_duration = short_term_duration
        self.memory_filepath = self._get_memory_filepath()
        self._load_memory()

    def _get_memory_filepath(self) -> str:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        services_dir = os.path.join(base_dir, "services")
        data_dir = os.path.join(services_dir, "data")
        user_dir = os.path.join(data_dir, str(self.user_id))
        os.makedirs(user_dir, exist_ok=True)
        return os.path.join(user_dir, "memory.json")

    def _load_memory(self):
        if os.path.exists(self.memory_filepath):
            try:
                with open(self.memory_filepath, 'r') as f:
                    data = json.load(f)
                    self.short_term_memory = data.get('short_term_memory', [])
                    self.long_term_memory = data.get('long_term_memory', [])
                    self.task_relationships = data.get('task_relationships', [])
            except Exception as e:
                print(f"Error loading memory: {e}")

    def add_to_short_term(self, experience: Dict[str, Any]):
        experience['timestamp'] = datetime.now().isoformat()
        self.short_term_memory.append(experience)
        self._cleanup_short_term()
        self._save_memory()

    def add_task_relationship(self, source_id, target_id, rel_type):
        self.task_relationships.append({
            'source': source_id,
            'target': target_id,
            'type': rel_type,
            'timestamp': datetime.now().isoformat()
        })
        self._save_memory()

    def get_recent_experiences(self, count: int = None) -> List[Dict[str, Any]]:
        if count is None:
            count = self.max_short_term_items
        return self.short_term_memory[-count:]

    def can_task_proceed(self, task_id):
        """بررسی امکان اجرای تسک بر اساس روابط والد-فرزندی"""
        for rel in self.task_relationships:
            if rel['target'] == task_id and rel['type'] == 'subtask':
                parent_task = self.db.query(Task).get(rel['source'])
                if parent_task and parent_task.status != TaskStatus.FINISH:
                    return False
        return True

    def _cleanup_short_term(self):
        current_time = datetime.now()
        self.short_term_memory = [
            exp for exp in self.short_term_memory
            if (current_time - datetime.fromisoformat(exp['timestamp'])).total_seconds() <= self.short_term_duration.total_seconds()
        ][-self.max_short_term_items:]

    def _save_memory(self):
        try:
            with open(self.memory_filepath, 'w') as f:
                json.dump({
                    'short_term_memory': self.short_term_memory,
                    'long_term_memory': self.long_term_memory,
                    'task_relationships': self.task_relationships
                }, f)
        except Exception as e:
            print(f"Error saving memory: {e}")

    def save_to_file(self):
        self._save_memory()

    def check_task_status(self, task_description, task_id):
        """بررسی خودکار وضعیت تسک بر اساس تجربیات قبلی"""
        # جستجو در حافظه بلند مدت برای یافتن راه حل‌های مشابه
        similar_solutions = self.get_similar_solutions(task_description)
        
        if similar_solutions:
            # بررسی زمان آخرین راه حل موفق
            latest_solution = max(similar_solutions, key=lambda x: x.get('timestamp', datetime.min))
            solution_time = latest_solution.get('timestamp', datetime.min)
            
            # اگر راه حل کمتر از 5 دقیقه پیش موفق بوده
            if (datetime.now() - solution_time).total_seconds() < 300:
                return {
                    'status': 'auto_complete',
                    'reason': f'Task automatically completed based on recent successful solution: {latest_solution.get("solution", "")}',
                    'solution': latest_solution.get('solution', '')
                }
        
        return None

    def add_task_solution(self, task_description, solution, success=True):
        """ذخیره راه حل برای یک تسک"""
        solution_data = {
            'task_description': task_description,
            'solution': solution,
            'success': success,
            'timestamp': datetime.now()
        }
        
        # اضافه کردن به حافظه بلند مدت
        self.long_term_memory.append(solution_data)
        
        # ذخیره در فایل
        self._save_memory()

    def get_similar_solutions(self, task_description: str) -> List[Dict[str, Any]]:
        """جستجوی راه حل‌های مشابه در حافظه بلند مدت"""
        similar_solutions = []
        
        # جستجو در حافظه بلند مدت
        for solution in self.long_term_memory:
            # محاسبه شباهت بین توضیحات تسک
            similarity = self._calculate_similarity(task_description, solution.get('task_description', ''))
            
            # اگر شباهت بیشتر از 70 درصد باشد
            if similarity > 0.7:
                similar_solutions.append(solution)
        
        return similar_solutions

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """محاسبه شباهت بین دو متن با استفاده از الگوریتم ساده"""
        # تبدیل متن‌ها به حروف کوچک
        text1 = text1.lower()
        text2 = text2.lower()
        
        # تقسیم متن‌ها به کلمات
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        # محاسبه تعداد کلمات مشترک
        common_words = words1.intersection(words2)
        
        # محاسبه تعداد کل کلمات منحصر به فرد
        total_unique_words = len(words1.union(words2))
        
        # محاسبه نسبت شباهت
        if total_unique_words == 0:
            return 0.0
            
        return len(common_words) / total_unique_words