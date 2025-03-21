from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship
from .base import Base, SessionLocal


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    parent_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    group_id = Column(Integer, ForeignKey("groups.id"))
    vnc_port = Column(Integer, unique=True)
    novnc_port = Column(Integer, unique=True)
    description = Column(String, nullable=True)  # اضافه کردن فیلد description

    # Relationships
    group = relationship("Group", back_populates="users")
    tasks = relationship("Task", back_populates="user")

    parent = relationship("User", remote_side=[id], back_populates="children")
    children = relationship("User", back_populates="parent")

    def __repr__(self):
        return f"<User(id={self.id}, name={self.name}, group_id={self.group_id}, vnc_port={self.vnc_port}, description={self.description})>"

    def save(self):
        db_session = SessionLocal()
        db_session.add(self)
        db_session.commit()
        db_session.close()
