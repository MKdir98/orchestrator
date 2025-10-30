from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Boolean
from sqlalchemy.orm import relationship
from .base import Base
from datetime import datetime


class Message(Base):
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True, index=True)
    rocketchat_message_id = Column(String, unique=True, index=True)  # ID پیغام در RocketChat
    user_id = Column(Integer, ForeignKey("users.id"))  # Agent که پیغام برای اون هست
    sender_username = Column(String)  # نام کاربری فرستنده (مدیر)
    content = Column(String)  # متن پیغام
    is_read = Column(Boolean, default=False)  # آیا agent خونده؟
    is_processed = Column(Boolean, default=False)  # آیا پردازش شده؟
    created_at = Column(DateTime, default=datetime.utcnow)
    rocketchat_timestamp = Column(DateTime)  # زمان پیغام در RocketChat
    
    user = relationship("User", backref="messages")
    
    def __repr__(self):
        return f"<Message(id={self.id}, user_id={self.user_id}, sender={self.sender_username}, is_read={self.is_read})>"
    
    def summary(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "sender_username": self.sender_username,
            "content": self.content,
            "is_read": self.is_read,
            "is_processed": self.is_processed,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "rocketchat_timestamp": self.rocketchat_timestamp.isoformat() if self.rocketchat_timestamp else None
        }

