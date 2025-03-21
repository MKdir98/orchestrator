from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager

# Create the SQLAlchemy engine and session
DATABASE_URL = "sqlite:///orchestrator.db"
engine = create_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


# Create tables if they don't exist
def init_db():
    Base.metadata.create_all(bind=engine)


# Initialize the database when this module is imported
# init_db()
@contextmanager
def get_db():
    db_session = SessionLocal()
    try:
        yield db_session
    finally:
        db_session.close()
