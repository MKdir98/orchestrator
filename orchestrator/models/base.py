from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager

from orchestrator.services.data_service import DataService

# Create the SQLAlchemy engine and session
DATABASE_URL = "sqlite:///" + DataService().get_orchestrator_path() + "orchestrator.db"
engine = create_engine(DATABASE_URL, echo=False)
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


def clear_database():
    try:
        Base.metadata.drop_all(bind=engine)
    except Exception:
        pass
