from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship
from .base import Base


class Group(Base):
    __tablename__ = "groups"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True, nullable=False)
    root_user = Column(String, nullable=False)
    system_user_id = Column(Integer, ForeignKey("system_users.id"), nullable=False)

    users = relationship("User", back_populates="group")
    system_user = relationship("SystemUser", back_populates="groups")

    def __repr__(self):
        return f"<Group(id={self.id}, name={self.name}, root_user={self.root_user})>"

    def summary(self):
        return {
            'id': self.id,
            'name': self.name,
            'users': [user.summary() for user in self.users]
        }
