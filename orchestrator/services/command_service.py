from typing import Dict
from orchestrator.models.base import SessionLocal
from orchestrator.models.task import Task
from orchestrator.models.user import User
from orchestrator.services.container_service import ContainerService
from sqlalchemy.orm import Session
import asyncio, asyncvnc
from PIL import Image

from orchestrator.services.data_service import DataService


class CommandService:
    """
    Service class for executing commands related to users and tasks.
    """

    @staticmethod
    def execute_command(db: Session, command: Dict, current_user_id: int):
        """
        Execute a command based on its type.

        Args:
            db (Session): Database session.
            command (Dict): Command to execute.
            current_user_id (int): ID of the current user executing the command.

        Raises:
            ValueError: If the command type is missing or unknown.
            Exception: If the command execution fails.
        """
        command_type = command.get("type")
        if not command_type:
            raise ValueError("Command type is missing.")

        # Map command types to their respective methods
        command_handlers = {
            "create_task": CommandService.create_task,
            "create_child": CommandService.create_child,
            "type": CommandService.typing,
            "click": CommandService.click,
        }

        handler = command_handlers.get(command_type)
        if not handler:
            raise ValueError(f"Unknown command type: {command_type}")

        try:
            handler(db, command, current_user_id)
        except Exception as e:
            db.rollback()  # Rollback in case of error
            raise Exception(f"Failed to execute command: {str(e)}")

    @staticmethod
    def create_task(db: Session, command: Dict, current_user_id: int):
        """
        Create a new task for a subordinate user.

        Args:
            db (Session): Database session.
            command (Dict): Command containing task details.
            current_user_id (int): ID of the current user executing the command.

        Raises:
            ValueError: If required fields are missing or the user is not a subordinate.
        """
        parameters = command.get("parameters")
        if not parameters or len(parameters) < 2:
            raise ValueError("Task description or user_id is missing.")

        task_description = parameters[0]
        user_id = parameters[1]

        # Check if the user_id is a subordinate of the current user
        subordinate = db.query(User).filter(
            User.id == user_id,
            User.parent_user_id == current_user_id
        ).first()
        if not subordinate:
            raise ValueError(f"User with ID {user_id} is not a subordinate of the current user.")

        new_task = Task(description=task_description, user_id=user_id)
        db.add(new_task)
        db.commit()

    @staticmethod
    def create_child(db: Session, command: Dict, current_user_id: int):
        """
        Create a new subordinate user for the current user.

        Args:
            db (Session): Database session.
            command (Dict): Command containing subordinate details.
            current_user_id (int): ID of the current user executing the command.

        Raises:
            ValueError: If required fields are missing.
        """
        parameters = command.get("parameters")
        if not parameters or len(parameters) < 2:
            raise ValueError("Name or description is missing.")

        name = parameters[0]
        description = parameters[1]

        new_subordinate = User(name=name, description=description, parent_user_id=current_user_id)
        db.add(new_subordinate)
        db.commit()

    @staticmethod
    def typing(text, current_user_id: int):
        """
        Simulate typing text via VNC (synchronous version).

        Args:
            db (Session): Database session.
            command (Dict): Command containing text to type.
            current_user_id (int): ID of the current user executing the command.

        Raises:
            ValueError: If required fields are missing.
        """
        db = SessionLocal()

        user = db.query(User).filter(User.id == current_user_id).first()
        if not user:
            raise ValueError(f"User with ID {current_user_id} not found.")

        async def async_typing():
            async with asyncvnc.connect('127.0.0.1', user.vnc_port) as client:
                client.keyboard.write(text)

        asyncio.run(async_typing())

    @staticmethod
    def click(x, y, current_user_id: int):
        db = SessionLocal()

        user = db.query(User).filter(User.id == current_user_id).first()
        if not user:
            raise ValueError(f"User with ID {current_user_id} not found.")

        async def async_click():
            async with asyncvnc.connect('127.0.0.1', user.vnc_port) as client:
                client.mouse.move(int(x), int(y))
                client.mouse.click()

        asyncio.run(async_click())

    @staticmethod
    def double_click(x, y, current_user_id: int):
        db = SessionLocal()

        user = db.query(User).filter(User.id == current_user_id).first()
        if not user:
            raise ValueError(f"User with ID {current_user_id} not found.")

        async def async_click():
            async with asyncvnc.connect('127.0.0.1', user.vnc_port) as client:
                client.mouse.move(int(x), int(y))
                client.mouse.click()
                client.mouse.click()

        asyncio.run(async_click())

    @staticmethod
    def right_click(x, y, current_user_id: int):
        db = SessionLocal()

        user = db.query(User).filter(User.id == current_user_id).first()
        if not user:
            raise ValueError(f"User with ID {current_user_id} not found.")

        async def async_click():
            async with asyncvnc.connect('127.0.0.1', user.vnc_port) as client:
                client.mouse.move(int(x), int(y))
                client.mouse.right_click()

        asyncio.run(async_click())

    @staticmethod
    def move_mouse(x, y, current_user_id: int):
        db = SessionLocal()

        user = db.query(User).filter(User.id == current_user_id).first()
        if not user:
            raise ValueError(f"User with ID {current_user_id} not found.")

        async def async_click():
            async with asyncvnc.connect('127.0.0.1', user.vnc_port) as client:
                client.mouse.move(int(x), int(y))

        asyncio.run(async_click())

    @staticmethod
    def send_key(name, current_user_id: int):
        db = SessionLocal()

        user = db.query(User).filter(User.id == current_user_id).first()
        if not user:
            raise ValueError(f"User with ID {current_user_id} not found.")

        async def async_click():
            async with asyncvnc.connect('127.0.0.1', user.vnc_port) as client:
                client.keyboard.press(name)

        asyncio.run(async_click())

    @staticmethod
    def screenshot(current_user_id: int):
        """
        Take a screenshot via VNC (synchronous version).

        Args:
            current_user_id (int): ID of the current user executing the command.

        Returns:
            bytes: The screenshot image data.
        """
        db = SessionLocal()
        user = db.query(User).filter(User.id == current_user_id).first()
        if not user:
            raise ValueError(f"User with ID {current_user_id} not found.")

        async def async_screenshot():
            async with asyncvnc.connect('127.0.0.1', user.vnc_port) as client:
                pixels = await client.screenshot()
                image = Image.fromarray(pixels)
                image.save(DataService().get_user_path_screenshot(user.id))
                return pixels

        return asyncio.run(async_screenshot())

    @staticmethod
    def run_command_via_container(command, current_user_id: int):
        """
        Simulate running a command via container service.
        """
        command_to_execute = f"python3 /root/Desktop/orchestrator/app.py run '{command}'"
        container_service = ContainerService()
        output = container_service.exec_command_in_container(current_user_id, command_to_execute)
        print(output)

    @staticmethod
    def run_background_command_via_container(command, current_user_id: int):
        """
        Simulate running a background command via container service.
        """
        command_to_execute = f"python3 /root/Desktop/orchestrator/app.py runbg '{command}'"
        container_service = ContainerService()
        output = container_service.exec_command_in_container(current_user_id, command_to_execute)
        print(output)
