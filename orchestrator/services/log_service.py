import os
import json
from datetime import datetime
import urllib.parse

from orchestrator.models import ResourceNotFoundException, Log
from orchestrator.services.data_service import DataService
from orchestrator.models.base import get_db
from orchestrator.models.task import Task
from orchestrator.models.exceptions import (
    AccessDeniedException,
)


class BaseLogService:
    def _save_image(self, image_data, log_id: int, timestamp):

        # اگر image_data یک مسیر فایل باشد
        if isinstance(image_data, str) and os.path.isfile(image_data):
            ext = os.path.splitext(image_data)[1]
            image_path = os.path.join(DataService().get_logs_image_path(), f"{log_id}_{timestamp}{ext}")
            with open(image_data, "rb") as src, open(image_path, "wb") as dst:
                dst.write(src.read())
            return image_path

        # اگر image_data یک شیء Image از PIL باشد
        elif hasattr(image_data, 'save'):
            image_path = os.path.join(DataService().get_logs_image_path(), f"{log_id}_{timestamp}.png")
            image_data.save(image_path)
            return image_path

        # اگر image_data باینری باشد
        elif isinstance(image_data, bytes):
            image_path = os.path.join(DataService().get_logs_image_path(), f"{log_id}_{timestamp}.png")
            with open(image_path, "wb") as f:
                f.write(image_data)
            return image_path

        return None

    def create_log(self, log_id: int, content, log_type="text", image_data=None):
        """ایجاد لاگ جدید - این متد در کلاس‌های فرزند پیاده‌سازی می‌شود"""
        raise NotImplementedError("Subclasses must implement this method")

    def append_log(self, log_id: int, content, log_type="text", image_data=None):
        """اضافه کردن به یک لاگ موجود - این متد در کلاس‌های فرزند پیاده‌سازی می‌شود"""
        raise NotImplementedError("Subclasses must implement this method")

    def get_logs(self, log_id):
        """دریافت لاگ‌ها - این متد در کلاس‌های فرزند پیاده‌سازی می‌شود"""
        raise NotImplementedError("Subclasses must implement this method")

    def _convert_image_path_to_url(self, image_path):
        """تبدیل مسیر فایل تصویر به URL"""
        if not image_path:
            return None

        # مسیر فایل را به URL تبدیل می‌کنیم - این مسیر در نهایت باید به API مناسب تغییر کند
        file_name = os.path.basename(image_path)
        encoded_file_name = urllib.parse.quote(file_name)
        return f"/api/logs/images/{encoded_file_name}"

    def create_step_log(self, task_id: int, step_name: str, content, log_type="text", image_data=None):
        step_content = f"step:{step_name}|{content}"
        return self.create_log(task_id, step_content, log_type, image_data)

    def append_step_log(self, task_id: int, step_name: str, content, log_type="text", image_data=None):
        step_content = f"step:{step_name}|{content}"
        return self.append_log(task_id, step_content, log_type, image_data)


class LocalLogService(BaseLogService):
    """پیاده‌سازی سرویس لاگ مبتنی بر فایل محلی"""

    def create_log(self, log_id: int, content, log_type="text", image_data=None):
        timestamp = datetime.now().isoformat()

        log_entry = {
            "id": log_id,
            "type": log_type,
            "timestamp": timestamp,
            "content": content,
            "image_path": None
        }

        if image_data:
            image_path = self._save_image(image_data, log_id, timestamp)
            log_entry["image_path"] = image_path

        log_file = DataService().get_log_path(log_id)
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump([log_entry], f, indent=2, ensure_ascii=False)

        return log_id

    def append_log(self, log_id: int, content, log_type="text", image_data=None):
        log_file = DataService().get_log_path(log_id)

        if not os.path.exists(log_file):
            return self.create_log(log_id, content, log_type, image_data)

        timestamp = datetime.now().isoformat()
        new_entry = {
            "type": log_type,
            "timestamp": timestamp,
            "content": content,
            "image_path": None
        }

        if image_data:
            image_path = self._save_image(image_data, log_id, timestamp)
            new_entry["image_path"] = image_path

        with open(log_file, "r+", encoding="utf-8") as f:
            logs = json.load(f)
            logs.append(new_entry)
            f.seek(0)
            json.dump(logs, f, indent=2, ensure_ascii=False)

        return log_id

    def get_logs(self, log_id):
        log_file = DataService().get_log_path(log_id)
        if not os.path.exists(log_file):
            return None

        with open(log_file, "r", encoding="utf-8") as f:
            return json.load(f)


class DBLogService(BaseLogService):
    """پیاده‌سازی سرویس لاگ مبتنی بر پایگاه داده"""

    def create_log(self, log_id: int, content, log_type="text", image_data=None):
        timestamp = datetime.now()
        image_path = None

        if image_data:
            image_path = self._save_image(image_data, log_id, timestamp.isoformat())

        with get_db() as db:
            log_entry = Log(
                log_id=log_id,
                type=log_type,
                timestamp=timestamp,
                content=content,
                image_path=image_path
            )
            db.add(log_entry)
            db.commit()

        return log_id

    def append_log(self, step_id: int, task_id: int, content, log_type="text", image_data=None):
        timestamp = datetime.now()
        image_path = None

        if image_data:
            image_path = self._save_image(image_data, step_id, timestamp.isoformat())

        with get_db() as db:
            log_entry = Log(
                step_id=step_id,
                task_id=task_id,
                type=log_type,
                timestamp=timestamp,
                content=content,
                image_path=image_path
            )
            db.add(log_entry)
            db.commit()

        return step_id

    def get_logs(self, log_id):
        with get_db() as db:
            logs = db.query(Log).filter(Log.log_id == log_id).order_by(Log.timestamp).all()
            if not logs:
                return None
            return [log.to_dict() for log in logs]

    def get_logs_by_task_step(self, task_id, current_user, include_images=True, page=1, per_page=50, log_types=None):

        with get_db() as db:
            task = db.query(Task).filter(Task.id == task_id).first()
            if not task:
                raise ResourceNotFoundException(Task, task_id)
            if task.user.group.system_user_id != current_user.id:
                raise AccessDeniedException()
            
            # شروع کوئری با فیلتر task_id
            query = db.query(Log).filter(Log.task_id == task_id)
            
            # فیلتر بر اساس نوع لاگ‌ها
            if log_types and len(log_types) > 0:
                query = query.filter(Log.type.in_(log_types))
            
            # ترتیب نزولی (جدیدترین بالا) و pagination
            query = query.order_by(Log.timestamp.desc())
            
            # محاسبه total count قبل از pagination
            total_count = query.count()
            
            # اعمال pagination
            offset = (page - 1) * per_page
            logs = query.offset(offset).limit(per_page).all()
            
            if not logs and page == 1:
                return {
                    "step_logs": {},
                    "total_count": 0,
                    "page": page,
                    "per_page": per_page,
                    "total_pages": 0
                }

            step_logs = {}

            for log in logs:
                log_dict = log.to_dict()

                if include_images and log_dict["image_path"]:
                    log_dict["image_url"] = self._convert_image_path_to_url(log_dict["image_path"])

                content = log_dict["content"]
                step_name = "default"

                if "|" in content:
                    step_info, actual_content = content.split("|", 1)
                    if step_info.startswith("step:"):
                        step_name = step_info[5:]
                        log_dict["content"] = actual_content

                # اضافه کردن لاگ به استپ مربوطه
                if step_name not in step_logs:
                    step_logs[step_name] = []

                step_logs[step_name].append(log_dict)

            # محاسبه تعداد صفحات
            total_pages = (total_count + per_page - 1) // per_page
            
            return {
                "step_logs": step_logs,
                "total_count": total_count,
                "page": page,
                "per_page": per_page,
                "total_pages": total_pages
            }


logService = DBLogService()
