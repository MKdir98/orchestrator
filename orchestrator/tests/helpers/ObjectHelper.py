from orchestrator.models.base import SessionLocal
from orchestrator.models.group import Group


def group_helper(name, root_user):
    db = SessionLocal()
    group = Group(name=name, root_user=root_user)
    db.add(group)
    db.commit()
    db.close()
