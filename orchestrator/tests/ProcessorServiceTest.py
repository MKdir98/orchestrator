import unittest

from orchestrator.models.base import SessionLocal
from orchestrator.services.command_service import CommandService
from orchestrator.services.processor_service import ProcessorService
from orchestrator.services.task_service import TaskService


class ProcessorServiceTestCase(unittest.TestCase):
    def test_something(self):
        user_id = 1
        # user = db.query(User).get(user_id)
        # subordinates = db.query(User).filter(User.parent_user_id == user_id).all()
        db = SessionLocal()
        service = ProcessorService(db, user_id)
        while True:
            service.process_next_task()
        # prompt = build_command_prompt(user, subordinates, tasks)
        # response = PromptService.send_prompt_to_model(prompt, '/home/mehdi/all/repositories/github.com/orchestrator/orchestrator/services/data/1/screenshot.png')
        # db.commit()

    def test_click_test(self):
        # CommandService.typing( "test", 1)
        CommandService.right_click(143, 396, 1)
        # CommandService.click(32, 64, 1)

    def test_send_send_keys_test(self):
        # CommandService.typing( "test", 1)
        CommandService.send_key('Enter key', 1)
        # CommandService.click(32, 64, 1)

if __name__ == '__main__':
    unittest.main()
