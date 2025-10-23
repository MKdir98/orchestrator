from orchestrator.models import SystemUser, AccessDeniedException, ResourceNotFoundException
from orchestrator.models.base import SessionLocal
from sqlalchemy.orm import Session
from orchestrator.services.container_service import ContainerService
from orchestrator.models.user import User
from orchestrator.services.group_service import GroupService
from orchestrator.services.websocket_service import websocket_manager


def get_user(db: Session, user_id: int):
    return db.query(User).filter(User.id == user_id).first()


def create_user(
        db: Session, name: str, parent_user_id: int | None, group_id: int | None, description: str
):
    if group_id is None:
        parent_user = db.query(User).filter(User.id == parent_user_id).first()
        if not parent_user:
            raise ValueError("Parent user does not exist.")
        group_id = parent_user.group_id
    user = User(
        name=name, parent_user_id=parent_user_id, group_id=group_id, description=description
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    GroupService().update_group_for_websockets(user.group)
    return user


def check_and_create_containers():
    """Check and create containers for all users."""
    container_service = ContainerService()
    db = SessionLocal()
    users = db.query(User).all()

    for user in users:
        if not container_service.find_container_by_user(user):
            print(f"Container for user '{user.name}' does not exist. Creating one...")
            container = container_service.create_container(user, db)
            if container:
                print(f"Container created successfully for user '{user.name}'.")
            else:
                print(f"Failed to create container for user '{user.name}'.")


def get_users_by_group(db: Session, group_id: int):
    return db.query(User).filter(User.group_id == group_id).all()


class UserService:
    def patch_user(self, db, data: dict, user_id: int, system_user: SystemUser):
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise ResourceNotFoundException(User, user_id)

        if user.group.system_user_id != system_user.id:
            raise AccessDeniedException()

        if 'continue_automatically' in data.keys():
            user.continue_automatically = data['continue_automatically']
            db.add(user)
            db.commit()
        return user

    def get_user(self, db, user_id: int, system_user: SystemUser):
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise ResourceNotFoundException(User, user_id)

        if user.group.system_user_id != system_user.id:
            raise AccessDeniedException()
        user = db.query(User).filter(User.id == user_id).first()
        return user


user_service = UserService()
