from sqlalchemy.orm import Session

from orchestrator.models import SystemUser
from orchestrator.models.WebSocketType import WebSocketType
from orchestrator.models.group import Group
from orchestrator.services.websocket_service import websocket_manager


def get_group(db: Session, group_id: int):
    return db.query(Group).filter(Group.id == group_id).first()


def get_groups(db: Session, system_user: SystemUser):
    return db.query(Group).filter(Group.system_user_id == system_user.id).all()


class GroupService:
    def update_group_for_websockets(self, group: Group):
        websocket_manager.send_to_user_with_format(group.system_user_id, WebSocketType.GROUP_UPDATE, group.summary())

    def create_group(self, db: Session, name: str, root_user: str, description: str, system_user: SystemUser):
        group = Group(name=name, root_user=root_user, system_user_id=system_user.id)
        db.add(group)
        db.commit()
        db.refresh(group)

        from orchestrator.services.user_service import create_user
        create_user(db, root_user, None, group.id, description)
        return group


group_service = GroupService()
