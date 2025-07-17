from sqlalchemy import Column, Integer, String, Text, DateTime
from .base import Base
from datetime import datetime


class Log(Base):
    __tablename__ = "step_logs"

    id = Column(Integer, primary_key=True, index=True)
    step_id = Column(Integer, index=True)
    task_id = Column(Integer, index=True)
    type = Column(String(50), default="text")
    timestamp = Column(DateTime, default=datetime.now)
    content = Column(Text)
    image_path = Column(String(255), nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "step_id": self.step_id,
            "task_id": self.task_id,
            "type": self.type,
            "timestamp": self.timestamp.isoformat(),
            "content": self.content,
            "image_path": self.image_path
        } 