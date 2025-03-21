from sqlalchemy.orm import Session
from models.group import Group
from models.user import User


def create_group(db: Session, name: str, root_user: str, description: str):
    # Create the group
    group = Group(name=name, root_user=root_user)
    db.add(group)
    db.commit()
    db.refresh(group)

    # Create the root user
    root_user_obj = User(name=root_user, group_id=group.id, description=description)
    db.add(root_user_obj)
    db.commit()
    db.refresh(root_user_obj)

    return group


def get_group(db: Session, group_id: int):
    return db.query(Group).filter(Group.id == group_id).first()


def get_groups(db: Session):
    return db.query(Group).all()
