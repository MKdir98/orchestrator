import unittest

from orchestrator.models.base import SessionLocal
from orchestrator.services.processor_service import ProcessorService
from orchestrator.services.task_service import TaskService


class ProcessorServiceTestCase(unittest.TestCase):
    def test_something(self):
        user_id = 1
        # user = db.query(User).get(user_id)
        # subordinates = db.query(User).filter(User.parent_user_id == user_id).all()
        db = SessionLocal()
        tasks = TaskService.get_tasks_by_user(db, user_id)
        service = ProcessorService(db, tasks[0])
        service.process_task()
        # prompt = build_command_prompt(user, subordinates, tasks)
        # response = PromptService.send_prompt_to_model(prompt, '/home/mehdi/all/repositories/github.com/orchestrator/orchestrator/services/data/1/screenshot.png')
        db.commit()


if __name__ == '__main__':
    unittest.main()
