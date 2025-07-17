from sqlalchemy import Column, Integer, String, Boolean
from sqlalchemy.orm import relationship
from werkzeug.security import generate_password_hash, check_password_hash

from .base import Base


class SystemUser(Base):
    __tablename__ = "system_users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, nullable=False)
    hash_password = Column(String, nullable=False)
    is_admin = Column(Boolean, default=False)

    groups = relationship("Group", back_populates="system_user")

    def __repr__(self):
        return f"<SystemUser(id={self.id}, email={self.email})>"

    def set_password(self, password):
        self.hash_password = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.hash_password, password)