from typing import Dict, List, Tuple, Union
from orchestrator.models.task import Task
from orchestrator.models.user import User
from orchestrator.services.processor_service import ProcessorService
from sqlalchemy.orm import Session
import json
import os
import re

def extract_json_blocks(text: str) -> List[Union[dict, list]]:
    """
    Extract JSON blocks from a text string.

    Args:
        text (str): The input text containing JSON blocks.

    Returns:
        List[Union[dict, list]]: A list of extracted JSON blocks.
    """
    json_blocks = []
    pattern = r'(?s)(\[.*?\]|\{.*?\})'  # Regex to match JSON-like blocks
    for match in re.finditer(pattern, text):
        try:
            json_data = json.loads(match.group(1))
            json_blocks.append(json_data)
        except json.JSONDecodeError:
            continue  # Skip invalid JSON blocks
    return json_blocks


def merge_json_blocks(json_blocks: List[Union[dict, list]]) -> List[dict]:
    """
    Merge JSON blocks into a single list of commands.

    Args:
        json_blocks (List[Union[dict, list]]): A list of JSON blocks.

    Returns:
        List[dict]: A merged list of commands.
    """
    merged_commands = []
    for block in json_blocks:
        if isinstance(block, list):
            merged_commands.extend(block)
        elif isinstance(block, dict):
            merged_commands.append(block)
    return merged_commands


def build_command_prompt(user: User, subordinates: List[User], tasks: List[Task]) -> str:
    """
    Build a prompt for the model based on user, subordinates, and tasks.

    Args:
        user (User): The user object.
        subordinates (List[User]): A list of subordinate users.
        tasks (List[Task]): A list of tasks.

    Returns:
        str: The generated prompt.
    """
    subordinates_list = "\n".join(
        [f"• {sub.name} ({sub.description})" for sub in subordinates]
    ) if subordinates else "No subordinates available"

    tasks_list = "\n".join(
        [f"• Task {task.id}: {task.description} [{task.status}]" for task in tasks]
    )

    return f"""
Your Role:
You are {user.name} with description: {user.description}
OS: Ubuntu 20.04 LXDE

Screen Dimensions:
The screen resolution is 1536x500 pixels. Coordinates are based on this resolution.

Available Resources:
Subordinates:
{subordinates_list}

Current Tasks:
{tasks_list}

Allowed Commands:
Use ONLY these command formats:

{{
  "type": "create_task",
  "task_id": "related_id",  // Must be an integer
  "parameters": [
    "task_description",  // String
    user_id              // Integer
  ]
}}

{{
  "type": "type",
  "task_id": "related_id",  // Must be an integer
  "parameters": [
    "text_to_type"  // String
  ]
}}

{{
  "type": "click",
  "task_id": "related_id",  // Must be an integer
  "parameters": [
    X,  // Integer (X coordinate, 0 <= X < 1536)
    Y   // Integer (Y coordinate, 0 <= Y < 500)
  ]
}}

{{
  "type": "create_child",
  "task_id": "related_id",  // Must be an integer
  "parameters": [
    "name",  // String
    "description"  // String
  ]
}}

Important Rules:
1. If the task is simple (e.g., installing software) and the user can handle it directly, use "type" commands instead of creating a new task.
2. Always generate valid JSON output.
3. Do not include explanations or extra text in the output.
4. Before typing commands, ensure the terminal is open. Use "click" to open the terminal if necessary.
5. The "task_id" must always be an integer or a string that can be converted to an integer.
6. To open the start menu, click at the bottom-left corner of the screen (coordinates: [10, 490]).
7. After opening the start menu, search for the terminal and click on it to open it.
    """


class TaskService:
    @staticmethod
    def create_task(db: Session, description: str, user_id: int) -> Task:
        """
        Create a new task for a user.

        Args:
            db (Session): Database session.
            description (str): Description of the task.
            user_id (int): ID of the user.

        Returns:
            Task: The created task.

        Raises:
            Exception: If task creation fails.
        """
        try:
            task = Task(description=description, user_id=user_id)
            db.add(task)
            db.commit()
            db.refresh(task)
            return task
        except Exception as e:
            db.rollback()
            raise Exception(f"Failed to create task: {str(e)}")

    @staticmethod
    def get_tasks_by_user(db: Session, user_id: int) -> List[Task]:
        """
        Fetch all tasks for a user.

        Args:
            db (Session): Database session.
            user_id (int): ID of the user.

        Returns:
            List[Task]: A list of tasks.

        Raises:
            Exception: If fetching tasks fails.
        """
        try:
            return db.query(Task).filter(Task.user_id == user_id).all()
        except Exception as e:
            raise Exception(f"Failed to fetch tasks for user {user_id}: {str(e)}")

    @staticmethod
    def process_tasks(db: Session, user_id: int) -> Tuple[str, str]:
        """
        Process tasks for a user, generate a prompt, send it to the model, and execute the returned commands.

        Args:
            db (Session): Database session.
            user_id (int): ID of the user.

        Returns:
            Tuple[str, Dict]: The prompt and the model's output.

        Raises:
            Exception: If processing tasks fails.
        """
        try:
            # Fetch user information
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                raise Exception(f"User with ID {user_id} not found.")

            # Fetch subordinates and tasks
            # subordinates = db.query(User).filter(User.parent_user_id == user_id).all()
            tasks = TaskService.get_tasks_by_user(db, user_id)
            for task in tasks:
                service = ProcessorService()
                service.process_task(task, db)
            db.commit()
        except Exception as e:
            db.rollback()  # Rollback in case of error
            raise Exception(f"Failed to process tasks for user {user_id}: {str(e)}")

    def new_process_task():
        tools = {
            "stop": {
                "description": "Indicate that the task has been completed.",
                "params": {},
            }
        }

