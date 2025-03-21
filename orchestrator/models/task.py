from sqlalchemy import Column, Integer, String, ForeignKey, Enum, DateTime
from sqlalchemy.orm import relationship
from .base import Base
from enum import Enum as PyEnum
from datetime import datetime
import json


# تعریف Enum برای وضعیت‌های تسک
class TaskStatus(PyEnum):
    NEW = "new"
    IN_PROGRESS = "in_progress"
    BLOCK_BY_OTHER_TASK = "block_by_other_task"
    WAIT_FOR_APPROVE = "wait_for_approve"
    FINISH = "finish"


# مدل Task
class Task(Base):
    __tablename__ = "tasks"

    id = Column(Integer, primary_key=True, index=True)
    description = Column(String)
    status = Column(Enum(TaskStatus), default=TaskStatus.NEW)
    user_id = Column(Integer, ForeignKey("users.id"))
    parent_task_id = Column(Integer, ForeignKey("tasks.id"), nullable=True)

    # Relationship with user
    user = relationship("User", back_populates="tasks")

    # Relationship for parent and child tasks
    parent_task = relationship("Task", remote_side=[id], back_populates="child_tasks")
    child_tasks = relationship("Task", back_populates="parent_task")

    # Relationship with messages (یک تسک می‌تواند چندین پیام داشته باشد)
    task_messages = relationship("TaskMessage", back_populates="task", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Task(id={self.id}, description={self.description}, status={self.status}, user_id={self.user_id}, parent_task_id={self.parent_task_id})>"

    def messages(self):
        # print(self.task_messages)
        # for test in self.task_messages:
        #     print(test.content)
        return [json.loads(task_message.content) for task_message in self.task_messages]


def Message(content, role="assistant"):
    return {"role": role, "content": content}

# مدل Message
class TaskMessage(Base):
    __tablename__ = "task_messages"

    id = Column(Integer, primary_key=True, index=True)
    content = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)  # زمان ایجاد پیام
    task_id = Column(Integer, ForeignKey("tasks.id"))  # کلید خارجی به تسک مربوطه

    task = relationship("Task", back_populates="task_messages")
