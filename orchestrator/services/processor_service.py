import os
from sqlalchemy import or_
from PIL import Image
import json
from datetime import datetime
import time
from typing import Optional
import traceback

from orchestrator.models.task import TaskStatus, Task, TaskMessage
from orchestrator.services.command_service import CommandService
from orchestrator.services.config_service import grounding_model, vision_model, action_model
from orchestrator.services.grounding_service import draw_big_dot
from orchestrator.models.memory import Memory

TYPING_DELAY_MS = 12
TYPING_GROUP_SIZE = 5
MODEL_WIDTH = 1420.0
MODEL_HEIGHT = 650.0
OS_WIDTH = 1920.0
OS_HEIGHT = 915.0

tools = {
    "complete_task": {
        "description": "Mark the current task as completed",
        "params": {
            "description": "Reason for completion"
        }
    },
    "stop": {"description": "Signal task completion", "params": {}},
    "wait": {
        "description": "Pause execution",
        "params": {"seconds": "Wait duration", "description": "Reason"}
    },
    "click": {
        "description": "Click UI element",
        "params": {"x": "X", "y": "Y", "description": "Reason"}
    },
    "double_click": {
        "description": "Double-click UI element",
        "params": {"x": "X", "y": "Y", "description": "Reason"}
    },
    "right_click": {
        "description": "Right-click UI element",
        "params": {"x": "X", "y": "Y", "description": "Reason"}
    },
    "type_text": {
        "description": "Input text",
        "params": {"text": "Text to type", "description": "Reason"}
    },
    "send_key": {
        "description": "Send keyboard key. The main keys are: Enter Escape Shift Alt Ctrl Tab",
        "params": {"name": "Key name", "description": "Reason"}
    },
    "create_task": {
        "description": "Create new task",
        "params": {
            "task": "Task description",
            "priority": "Priority level",
            "description": "Reason"
        }
    }
}


class ProcessorService:
    def __init__(self, db, user_id):
        self.db = db
        self.user_id = user_id
        self.memory = Memory(user_id=user_id)
        self.current_task: Optional[Task] = None
        self.latest_screenshot: Optional[str] = None

    def screenshot(self):
        CommandService.screenshot(self.user_id)
        with open(self.screenshot_path, "rb") as f:
            return f.read()

    @property
    def screenshot_path(self):
        return os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data",
            str(self.user_id),
            'screenshot.png'
        )

    @property
    def dot_screenshot_path(self):
        return os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data",
            str(self.user_id),
            'screenshot_dot.png'
        )

    def save_image(self, image):
        os.makedirs(os.path.dirname(self.dot_screenshot_path), exist_ok=True)
        if isinstance(image, Image.Image):
            image.save(self.dot_screenshot_path)
        else:
            with open(self.dot_screenshot_path, "wb") as f:
                f.write(image)

    def find_x_y(self, query):
        self.screenshot()
        position = grounding_model.call(query, self.screenshot_path)
        dot_image = draw_big_dot(Image.open(self.screenshot_path), position)
        self.save_image(dot_image)
        return position

    def process_next_task(self):

        # اگر تسک paused نبود، تسک‌های جدید یا در حال انجام را بررسی می‌کنیم
        self.current_task = self.db.query(Task).filter(
            Task.user_id == self.user_id,
            or_(
                Task.status == TaskStatus.NEW,
                Task.status == TaskStatus.IN_PROGRESS,
                Task.status == TaskStatus.PAUSED,
            )
        ).order_by(Task.priority.desc()).first()

        if not self.current_task:
            return False

        # بررسی خودکار وضعیت تسک
        auto_status = self.memory.check_task_status(self.current_task.description, self.current_task.id)
        if auto_status and auto_status['status'] == 'auto_complete':
            # اگر تسک به صورت خودکار حل شده
            self.current_task.status = TaskStatus.FINISH
            self.current_task.task_messages.append(
                TaskMessage(content=json.dumps({
                    "role": "system",
                    "content": f"Task automatically completed: {auto_status['reason']}"
                }))
            )
            self.db.commit()
            return True

        self.current_task.status = TaskStatus.IN_PROGRESS
        self.db.commit()
        self.process_task()
        return True

    def process_task(self):
        if not self.current_task:
            return

        # تبدیل محتوای task_messages به دیکشنری
        initial_message = {
            "role": "system",
            "content": f"""OBJECTIVE: {self.current_task.description}
                PRIORITY: {self.current_task.priority}

                IMPORTANT INSTRUCTIONS:
                1. Each action can have pre-actions and post-actions that must be executed before and after the main action
                2. Pre-actions are required steps that must be completed before executing the main action
                3. Post-actions are cleanup or follow-up steps that must be executed after the main action
                4. When planning an action, consider:
                   - What needs to be prepared before the action (pre-actions)
                   - What needs to be cleaned up after the action (post-actions)
                   - The sequence of all required steps
                5. Always execute pre-actions before the main action
                6. Always execute post-actions after the main action
                7. If any step fails, handle the error appropriately

                Example action structure:
                {{
                    "action": "main_action",
                    "pre_actions": [
                        {{"action": "prepare_step1", "thought": "Why this preparation is needed"}},
                        {{"action": "prepare_step2", "thought": "Why this preparation is needed"}}
                    ],
                    "post_actions": [
                        {{"action": "cleanup_step1", "thought": "Why this cleanup is needed"}},
                        {{"action": "cleanup_step2", "thought": "Why this cleanup is needed"}}
                    ],
                    "thought": "Why this main action is needed"
                }}

                Remember: Always consider the complete sequence of actions, including necessary preparations and cleanup steps.
                """
        }

        init_msg = {
            "role": "system",
            "content": "TASK EXECUTION STARTED\nFocus on proper action sequencing"
        }
        self.current_task.task_messages.append(
            TaskMessage(content=json.dumps(init_msg))
        )

        for _ in range(20):
            if not self.process_task_step():
                break

        self.memory.save_to_file()

    def execute_action(self, action_call):
        action_name = action_call["name"]
        params = action_call.get("parameters", {})

        try:
            result = getattr(self, action_name)(**params)

            self.current_task.task_messages.append(
                TaskMessage(content=json.dumps({
                    "role": "action",
                    "content": f"{action_name}: {result}"
                }))
            )

            return "stop" if action_name == "stop" else "continue"
        except Exception as e:
            error_msg = f"Action failed: {action_name} - {str(e)}"
            self.current_task.task_messages.append(
                TaskMessage(content=json.dumps({
                    "role": "error",
                    "content": error_msg
                }))
            )
            return "continue"

    def stop(self):
        if self.current_task:
            # ذخیره راه حل ناموفق در حافظه
            self.memory.add_task_solution(
                self.current_task.description,
                "Task stopped due to errors or issues",
                success=False
            )
        return "Task completed"

    def wait(self, seconds=5, description=""):
        time.sleep(int(seconds))
        return f"Waited {seconds} seconds"

    def click(self, x, y, description):
        return CommandService.click(float(x) * (OS_WIDTH / MODEL_WIDTH), float(y) * (OS_HEIGHT / MODEL_HEIGHT), self.user_id)

    def complete_task(self, description):
        if self.current_task:
            self.current_task.status = TaskStatus.FINISH
            # ذخیره راه حل موفق در حافظه
            self.memory.add_task_solution(
                self.current_task.description,
                description,
                success=True
            )
            self.db.commit()
            return f"Task {self.current_task.id} marked as completed: {description}"
        return "No active task to complete"

    def double_click(self, x, y, description):
        return CommandService.double_click(float(x) * (OS_WIDTH / MODEL_WIDTH), float(y) * (OS_HEIGHT / MODEL_HEIGHT), self.user_id)

    def right_click(self, x, y, description):
        return CommandService.right_click(float(x) * (OS_WIDTH / MODEL_WIDTH), float(y) * (OS_HEIGHT / MODEL_HEIGHT), self.user_id)

    def type_text(self, text, description):
        CommandService.typing(text, self.user_id)
        return f"Typed: {text[:50]}..."

    def send_key(self, name, description):
        return CommandService.send_key(name, self.user_id)

    def append_screenshot(self):
        vision_prompt = f"""Analyze this screenshot and respond in this EXACT format:

        Current Objective: {self.current_task.description}
        Priority: {self.current_task.priority}

        Screen Analysis:
        - [Windows]: [[Describe first visible window and its state and it's coordinates], [Describe next visible window and its state and it's coordinates], ...]
        - [Task UI Elements]: [[Describe first visible element relevant to the task and its state and it's coordinates], [Describe next visible element relevant to the task and its state and it's coordinates], ...]
        - [Other UI Elements]: [[Describe first visible element and its state and it's coordinates], [Describe next visible element and its state and it's coordinates], ...]
        - [Other Observations]: [Any other important details]

        Task Status: [complete/not complete]
        Reason: [Explain why task is complete or not]

        Next Actions (list ALL required actions in order):

        1. Action:
        - Type: [click/double_click/right_click/type_text/send_key/wait/create_task/stop]
        - Target: [Specific element or location]
        - Detail: [Detail of type in text]
        - Reason: [Why this action is needed]
        - Pre-Actions: [List any preparation steps]
        - Post-Actions: [List any follow-up steps]
        - Thought: [Thought of this action]

        2. Action:
        - Type: [Next action type]
        - Target: [Specific element or location]
        - ...

        Now analyze the screenshot and provide your response in the requested format.
        Rules to follow:
        1. ALWAYS maintain this exact format
        2. List ALL required actions in proper sequence
        3. Include ALL necessary pre/post actions
        4. For terminal interactions:
           - Ensure terminal is focused first
           - Include appropriate wait times
        5. For system commands:
           - Check for errors in response
        6. Be specific about targets and reasons
        7. For open apps and files you should use double_click instead of click
        8. For errors and something like them you can use create_task to create new task with higher priority.
        9. If a task with higher priority is created via create_task:
            - Immediately terminate the current task’s action list.
            - Only output the new task (no other actions or tasks).
            - Do NOT add subsequent actions to the original task.

        Current task objective: {self.current_task.description}
        Priority: {self.current_task.priority}
        """
        screenshot_data = self.screenshot()
        response = vision_model.call(
            messages=[
                {
                    "role": "user",
                    "content": [
                        screenshot_data,
                        vision_prompt,
                    ]
                }
            ]
        )
        return response

    def process_task_step(self):
        try:
            screenshot_thought = self.append_screenshot()
            print(f"screenshot: {screenshot_thought}")

            # ذخیره تجربه در حافظه کوتاه مدت
            self.memory.add_to_short_term({
                'type': 'screenshot_analysis',
                'thought': screenshot_thought,
                'timestamp': datetime.now()
            })

            action_prompt = f"""Based on the screenshot analysis and task objective, determine the next action to take.
                    Each action can have pre-actions and post-actions that must be executed before and after the main action.
                    
                    Screenshot analysis:
                    {screenshot_thought}

                    Format your response as a JSON object with this structure:
                    {{
                        "action": "main_action_name",
                        "pre_actions": [
                            {{
                                "action": "prepare_step1",
                                "detail": "Why this preparation is needed"
                            }}
                        ],
                        "post_actions": [
                            {{
                                "action": "cleanup_step1",
                                "detail": "Why this cleanup is needed"
                            }}
                        ],
                        "detail": "The detail of the action in text format"
                    }}

                    Available actions:
                    - click: Click at specific coordinates
                    - double_click: Double click at specific coordinates
                    - type_text: Type text at current position
                    - send_key: Send a key press
                    - wait: Wait for specified seconds
                    - create_task: Create new task (will pause current task if higher priority)
                    
                    Rules to follow:
                    1. If a task with a higher priority number (higher urgency) is created via create_task:
                        - Immediately terminate the current task’s action list.
                        - Only output the new task (no other actions or tasks).
                        - Do NOT add subsequent actions to the original task.
                    2. Pre-actions are CRITICAL – They determine if the main action can execute:
                        - Strictly enforce pre-actions from screen_shot analysis.
                    3. Window Focus Requirement:
                        - If pre-actions specify focusing on a particular window:
                            - Physically click on the **window itself** (not its icon or other UI elements).
                            - Example: For "LX Terminal," click the terminal window, not the LX Terminal icon.
                            - Verify focus by checking window attributes (title, active state) before proceeding.
                    4. In send_key for enter something you should use `Return` word
                    5. If you want apps opened in the taskbar you should say it's name that you hve in screenshot analysis
                    6. For ensure the apps is focused you need to click on them before the main action
                    

                    Current task objective: {self.current_task.description}
                    Priority: {self.current_task.priority}

                    Determine the next action to take in the specified JSON format."""

            response = action_model.call(
                [
                    {"role": "system", "content": action_prompt},
                    {"role": "user", "content": "Determine next action using available tools."}
                ],
                tools
            )
            print(f'ACTION MODEL RESPONSE: {response}')

            should_continue = True
            for tool_call in response[1]:
                try:
                    action = tool_call['name']
                    args = tool_call['parameters']
                    print(f'action: {action} args: {args}')

                    # پردازش pre-actions اگر وجود دارد
                    if 'pre_actions' in args:
                        for pre_action in args['pre_actions']:
                            pre_action_name = pre_action.get('action')
                            pre_action_args = {k: v for k, v in pre_action.items()
                                               if k != 'action' and k != 'thought'}
                            if hasattr(self, pre_action_name):
                                getattr(self, pre_action_name)(**pre_action_args)

                    # اجرای اقدام اصلی
                    if hasattr(self, action):
                        result = getattr(self, action)(**args)

                        # ذخیره نتیجه اقدام
                        self.memory.add_to_short_term({
                            'type': 'action',
                            'action': action,
                            'args': args,
                            'result': result,
                            'timestamp': datetime.now()
                        })

                        # اگر action ایجاد تسک بود و تسک جدید اولویت بالاتری داشت، باید ادامه ندهیم
                        if action == "create_task" and "switched to higher priority task" in result:
                            should_continue = False

                    # پردازش post-actions اگر وجود دارد
                    if 'post_actions' in args:
                        for post_action in args['post_actions']:
                            post_action_name = post_action.get('action')
                            post_action_args = {k: v for k, v in post_action.items()
                                                if k != 'action' and k != 'thought'}
                            if hasattr(self, post_action_name):
                                getattr(self, post_action_name)(**post_action_args)

                    # ذخیره پیام تسک
                    self.current_task.task_messages.append(
                        TaskMessage(content=json.dumps({
                            "role": "assistant",
                            "content": json.dumps(tool_call),
                        }))
                    )

                    self.db.commit()

                    if not should_continue:
                        return False

                except Exception as e:
                    traceback.print_exc()
                    print(f"Error processing tool call {action}: {str(e)}")
                    continue

            return should_continue
        except Exception as e:
            traceback.print_exc()
            print(f"Error in process_task_step: {str(e)}")
            return False

    def execute_tool_call(self, tool_call):
        try:
            tool_name = tool_call["name"]
            tool_params = tool_call.get("parameters", {})

            # لاگ قبل از اجرای ابزار
            print(f"\n[DEBUG] Executing {tool_name} with params: {tool_params}")

            result = getattr(self, tool_name)(**tool_params)

            # لاگ پس از اجرای ابزار
            print(f"[DEBUG] Tool {tool_name} executed. Result: {result}")

            self.current_task.task_messages.append(
                TaskMessage(content=json.dumps({
                    "role": "tool",
                    "name": tool_name,
                    "content": result
                }))
            )

            return "stop" if tool_name == "stop" else "continue"

        except Exception as e:
            error_msg = f"Tool {tool_name} failed: {str(e)}"
            print(f"\n[ERROR] {error_msg}")
            self.current_task.task_messages.append(
                TaskMessage(content=json.dumps({
                    "role": "error",
                    "content": error_msg
                }))
            )
            return "continue"

    def _format_recent_experiences(self):
        """فرمت‌بندی تجربیات اخیر برای نمایش در پرامپت"""
        recent_experiences = []
        for exp in self.memory.short_term_memory[-5:]:  # فقط 5 تجربه اخیر
            exp_type = exp.get('type', 'unknown')
            thought = exp.get('thought', exp.get('result', 'No details available'))
            recent_experiences.append(f"- {exp_type}: {thought}")
        return "\n".join(recent_experiences)

    def create_task(self, task, description, priority=None):
        priority = priority or (self.current_task.priority - 1 if self.current_task else 0)
        priority = int(priority)
        if self.current_task and priority > self.current_task.priority:
            # ذخیره وضعیت فعلی برای بازگشت بعدی
            self.memory.add_to_short_term({
                'type': 'task_interrupted',
                'task_id': self.current_task.id,
                'reason': f'Interrupted by higher priority task: {task}',
                'timestamp': datetime.now()
            })

            # متوقف کردن تسک فعلی
            self.current_task.status = TaskStatus.PAUSED
            self.db.commit()

            # ایجاد تسک جدید
            new_task = Task(
                description=task,
                user_id=self.current_task.user_id,
                priority=priority,
                parent_task_id=self.current_task.id,
                status=TaskStatus.IN_PROGRESS
            )
            self.db.add(new_task)
            self.db.commit()

            # تنظیم تسک جدید به عنوان تسک جاری
            self.current_task = new_task
            return f"Created and switched to higher priority task {new_task.id}"

        new_task = Task(
            description=task,
            user_id=self.current_task.user_id,
            priority=priority,
            parent_task_id=self.current_task.id,
            status=TaskStatus.NEW
        )
        self.db.add(new_task)
        self.db.commit()
        return f"Created task {new_task.id} (current task continues)"
