import json
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional
from difflib import SequenceMatcher


class SharedLearningService:
    """
    سرویس مدیریت یادگیری مشترک برای ذخیره و بازیابی اشتباهات
    این سرویس برای همه کاربران مشترک است
    """
    
    def __init__(self):
        self.mistakes_file_path = "orchestrator_data/shared_mistakes.json"
        self.ensure_mistakes_file_exists()
    
    def ensure_mistakes_file_exists(self):
        """اطمینان از وجود فایل اشتباهات"""
        os.makedirs(os.path.dirname(self.mistakes_file_path), exist_ok=True)
        if not os.path.exists(self.mistakes_file_path):
            self._save_mistakes({"mistakes": [], "version": "1.0"})
    
    def _load_mistakes(self) -> Dict:
        """بارگذاری اشتباهات از فایل"""
        try:
            with open(self.mistakes_file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading mistakes: {e}")
            return {"mistakes": [], "version": "1.0"}
    
    def _save_mistakes(self, mistakes_data: Dict):
        """ذخیره اشتباهات در فایل"""
        try:
            with open(self.mistakes_file_path, 'w', encoding='utf-8') as f:
                json.dump(mistakes_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving mistakes: {e}")
    
    def save_mistake(self, 
                    action_context: Dict,
                    expected_outcome: str,
                    actual_outcome: str,
                    problem_description: str,
                    suggested_solution: str = "") -> str:
        """
        ذخیره یک اشتباه جدید یا به‌روزرسانی اشتباه مشابه موجود
        
        Args:
            action_context: اطلاعات action انجام شده (type, coordinates, element_name, window_title, etc.)
            expected_outcome: نتیجه مورد انتظار
            actual_outcome: نتیجه واقعی که رخ داد
            problem_description: توضیح مشکل
            suggested_solution: راه‌حل پیشنهادی برای تصحیح
            
        Returns:
            mistake_id: شناسه اشتباه ذخیره شده
        """
        mistakes_data = self._load_mistakes()
        mistakes = mistakes_data.get("mistakes", [])
        
        # جستجوی اشتباه مشابه موجود
        similar_mistake = self._find_similar_existing_mistake(
            mistakes, 
            action_context, 
            expected_outcome,
            actual_outcome
        )
        
        if similar_mistake:
            # به‌روزرسانی اشتباه موجود
            similar_mistake["occurrences"] += 1
            similar_mistake["last_seen"] = datetime.now().isoformat()
            
            # اگر solution جدید بهتر است، به‌روزرسانی کن
            if suggested_solution and suggested_solution not in similar_mistake.get("solutions", []):
                if "solutions" not in similar_mistake:
                    similar_mistake["solutions"] = []
                similar_mistake["solutions"].append(suggested_solution)
            
            mistake_id = similar_mistake["id"]
            print(f"Updated existing mistake: {mistake_id} (occurrences: {similar_mistake['occurrences']})")
        else:
            # ایجاد اشتباه جدید
            mistake_id = str(uuid.uuid4())[:8]
            new_mistake = {
                "id": mistake_id,
                "pattern": self._generate_pattern_name(action_context, actual_outcome),
                "action_context": action_context,
                "expected_outcome": expected_outcome,
                "actual_outcome": actual_outcome,
                "problem": problem_description,
                "solutions": [suggested_solution] if suggested_solution else [],
                "occurrences": 1,
                "first_seen": datetime.now().isoformat(),
                "last_seen": datetime.now().isoformat()
            }
            mistakes.append(new_mistake)
            print(f"Created new mistake: {mistake_id}")
        
        # ذخیره به فایل
        mistakes_data["mistakes"] = mistakes
        self._save_mistakes(mistakes_data)
        
        return mistake_id
    
    def _find_similar_existing_mistake(self, 
                                      mistakes: List[Dict],
                                      action_context: Dict,
                                      expected_outcome: str,
                                      actual_outcome: str) -> Optional[Dict]:
        """
        جستجوی اشتباه مشابه در لیست موجود
        """
        for mistake in mistakes:
            similarity_score = self._calculate_similarity(
                mistake["action_context"],
                mistake["expected_outcome"],
                mistake["actual_outcome"],
                action_context,
                expected_outcome,
                actual_outcome
            )
            
            # اگر شباهت بالای 80٪ بود، اشتباه مشابه است
            if similarity_score > 0.8:
                return mistake
        
        return None
    
    def _calculate_similarity(self,
                            context1: Dict,
                            expected1: str,
                            actual1: str,
                            context2: Dict,
                            expected2: str,
                            actual2: str) -> float:
        """
        محاسبه میزان شباهت بین دو اشتباه
        """
        # مقایسه action type
        action_match = context1.get("action_type") == context2.get("action_type")
        
        # مقایسه window titles
        window_similarity = SequenceMatcher(
            None,
            context1.get("window_title_after", ""),
            context2.get("window_title_after", "")
        ).ratio()
        
        # مقایسه actual outcomes
        outcome_similarity = SequenceMatcher(
            None,
            actual1.lower(),
            actual2.lower()
        ).ratio()
        
        # میانگین وزن‌دار
        similarity = (
            (0.3 * (1.0 if action_match else 0.0)) +
            (0.4 * window_similarity) +
            (0.3 * outcome_similarity)
        )
        
        return similarity
    
    def _generate_pattern_name(self, action_context: Dict, actual_outcome: str) -> str:
        """
        ایجاد نام pattern برای اشتباه
        """
        action_type = action_context.get("action_type", "unknown")
        window_after = action_context.get("window_title_after", "")
        
        # استخراج کلمات کلیدی
        keywords = []
        if "password" in window_after.lower() or "password" in actual_outcome.lower():
            keywords.append("password_manager")
        if "popup" in actual_outcome.lower() or "dialog" in actual_outcome.lower():
            keywords.append("popup")
        if "tab" in actual_outcome.lower() or "window" in actual_outcome.lower():
            keywords.append("unexpected_window")
        
        pattern_name = f"{action_type}_{('_'.join(keywords) if keywords else 'unexpected_outcome')}"
        return pattern_name
    
    def get_relevant_mistakes(self, task_context: str, top_k: int = 5) -> List[Dict]:
        """
        دریافت اشتباهات مرتبط با task فعلی
        
        Args:
            task_context: متن task یا context فعلی
            top_k: تعداد اشتباهات برگشتی
            
        Returns:
            لیست اشتباهات مرتبط مرتب شده بر اساس relevance
        """
        mistakes_data = self._load_mistakes()
        mistakes = mistakes_data.get("mistakes", [])
        
        if not mistakes:
            return []
        
        # محاسبه relevance score برای هر اشتباه
        scored_mistakes = []
        task_lower = task_context.lower()
        
        for mistake in mistakes:
            score = 0.0
            
            # امتیاز بر اساس occurrences (اشتباهات رایج‌تر مهم‌تر هستند)
            score += min(mistake["occurrences"] * 0.1, 1.0)
            
            # امتیاز بر اساس تطابق کلمات کلیدی
            pattern = mistake.get("pattern", "").lower()
            problem = mistake.get("problem", "").lower()
            
            # بررسی کلمات کلیدی مشترک
            for keyword in ["login", "password", "firefox", "browser", "click", "tab", "window"]:
                if keyword in task_lower and (keyword in pattern or keyword in problem):
                    score += 0.3
            
            scored_mistakes.append((score, mistake))
        
        # مرتب‌سازی بر اساس score
        scored_mistakes.sort(key=lambda x: x[0], reverse=True)
        
        # برگرداندن top_k اشتباه
        return [mistake for score, mistake in scored_mistakes[:top_k] if score > 0]
    
    def format_mistakes_for_prompt(self, mistakes: List[Dict]) -> str:
        """
        فرمت کردن اشتباهات برای نمایش در prompt
        """
        if not mistakes:
            return "No similar mistakes found in learning history."
        
        formatted = "=== LEARNED MISTAKES (Avoid These) ===\n\n"
        
        for i, mistake in enumerate(mistakes, 1):
            formatted += f"{i}. Pattern: {mistake.get('pattern', 'Unknown')}\n"
            formatted += f"   Problem: {mistake.get('problem', 'Unknown')}\n"
            formatted += f"   Expected: {mistake.get('expected_outcome', 'N/A')}\n"
            formatted += f"   Actually Happened: {mistake.get('actual_outcome', 'N/A')}\n"
            
            solutions = mistake.get("solutions", [])
            if solutions:
                formatted += f"   Solutions:\n"
                for sol in solutions:
                    if sol:  # فقط solution های غیرخالی
                        formatted += f"      - {sol}\n"
            
            formatted += f"   (Occurred {mistake.get('occurrences', 1)} times)\n\n"
        
        return formatted
    
    def get_all_mistakes(self) -> List[Dict]:
        """دریافت تمام اشتباهات ذخیره شده"""
        mistakes_data = self._load_mistakes()
        return mistakes_data.get("mistakes", [])
    
    def clear_all_mistakes(self):
        """پاک کردن تمام اشتباهات (برای testing)"""
        self._save_mistakes({"mistakes": [], "version": "1.0"})
        print("All mistakes cleared")
    
    def get_mistake_by_id(self, mistake_id: str) -> Optional[Dict]:
        """دریافت یک اشتباه خاص با ID"""
        mistakes = self.get_all_mistakes()
        for mistake in mistakes:
            if mistake.get("id") == mistake_id:
                return mistake
        return None
    
    def get_statistics(self) -> Dict:
        """دریافت آمار کلی از اشتباهات"""
        mistakes = self.get_all_mistakes()
        
        if not mistakes:
            return {
                "total_mistakes": 0,
                "total_occurrences": 0,
                "most_common_pattern": None
            }
        
        total_occurrences = sum(m.get("occurrences", 1) for m in mistakes)
        
        # یافتن رایج‌ترین pattern
        pattern_counts = {}
        for m in mistakes:
            pattern = m.get("pattern", "unknown")
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + m.get("occurrences", 1)
        
        most_common = max(pattern_counts.items(), key=lambda x: x[1]) if pattern_counts else (None, 0)
        
        return {
            "total_mistakes": len(mistakes),
            "total_occurrences": total_occurrences,
            "most_common_pattern": most_common[0],
            "most_common_count": most_common[1],
            "patterns": pattern_counts
        }

