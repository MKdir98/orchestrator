from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship
from .base import Base


class Group(Base):
    __tablename__ = "groups"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    root_user = Column(String)

    # Relationship with users
    users = relationship("User", back_populates="group")

    def __repr__(self):
        return f"<Group(id={self.id}, name={self.name}, root_user={self.root_user})>"
