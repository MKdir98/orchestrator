import os
import tempfile

from sqlalchemy import or_
from copy import deepcopy
from PIL import Image
import json
from datetime import datetime
import time
from typing import Optional
import traceback

from orchestrator.models.task import TaskStatus, Task, TaskMessage
from orchestrator.services.command_service import CommandService
from orchestrator.services.config_service import grounding_model, vision_model, action_model, position_model
from orchestrator.services.data_service import DataService
from orchestrator.models.memory import Memory
from orchestrator.services.rag_service import RAGSystem
from orchestrator.models.user import User

TYPING_DELAY_MS = 12
TYPING_GROUP_SIZE = 5
# MODEL_WIDTH = 1420.0
# MODEL_HEIGHT = 650.0
OS_WIDTH = 1920.0
OS_HEIGHT = 407.0

tools = {
    "complete_task": {
        "description": "Mark the current task as completed",
        "params": {
            "description": "Reason for completion",
            "last_action_result": "What was the result of the last action",
            "image_width": "The screenshot width size",
            "image_height": "The screenshot height size",
        }
    },
    # "stop": {"description": "Signal task completion", "params": {}},
    "wait": {
        "description": "Pause execution",
        "params": {
            "seconds": "Wait duration", "description": "Reason",
        }
    },
    "single_click": {
        "description": "Click UI element one time",
        "params": {
            "x": "X", "y": "Y",
        }
    },
    "double_click": {
        "description": "Double-click UI element",
        "params": {
            "x": "X", "y": "Y",
        }
    },
    "move_mouse": {
        "description": "Move mouse",
        "params": {
            "x": "X", "y": "Y",
        }
    },
    "right_click": {
        "description": "Right-click UI element",
        "params": {
            "x": "X", "y": "Y",
        }
    },
    "type_text": {
        "description": "Input text",
        "params": {
            "text": "Text to type",
        }
    },
    "send_key": {
        "description": "Send keyboard key. The main keys are: Enter Escape Shift Alt Ctrl Tab",
        "params": {
            "name": "Key name",
        }
    },
    "create_task": {
        "description": "Create new task",
        "params": {
            "task": "Task description",
            "priority": "Priority level",
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
        self.model_width = None
        self.model_height = None
        self.task_rag = RAGSystem()

    def screenshot(self):
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

    # def find_x_y(self, query):
    #     self.screenshot()
    #     position = grounding_model.call(query, self.screenshot_path)
    #     dot_image = draw_big_dot(Image.open(self.screenshot_path), position)
    #     self.save_image(dot_image)
    #     return position

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
            # self.current_task.task_messages.append(
            #     TaskMessage(content=json.dumps({
            #         "role": "system",
            #         "content": f"Task automatically completed: {auto_status['reason']}"
            #     }))
            # )
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
        # self.current_task.task_messages.append(
        #     TaskMessage(content=json.dumps(init_msg))
        # )

        for _ in range(20):
            if not self.process_task_step():
                break

        self.memory.save_to_file()

    def execute_action(self, action_call):
        action_name = action_call["name"]
        params = action_call.get("parameters", {})

        try:
            result = getattr(self, action_name)(**params)

            # self.current_task.task_messages.append(
            #     TaskMessage(content=json.dumps({
            #         "role": "action",
            #         "content": f"{action_name}: {result}"
            #     }))
            # )

            return "stop" if action_name == "stop" else "continue"
        except Exception as e:
            # error_msg = f"Action failed: {action_name} - {str(e)}"
            # self.current_task.task_messages.append(
            #     TaskMessage(content=json.dumps({
            #         "role": "error",
            #         "content": error_msg
            #     }))
            # )
            return "continue"

    # def stop(self, description):
    #     if self.current_task:
    #         # ذخیره راه حل ناموفق در حافظه
    #         self.memory.add_task_solution(
    #             self.current_task.description,
    #             "Task stopped due to errors or issues",
    #             success=False
    #         )
    #     return "Task completed"

    def wait(self, seconds=5, description=""):
        time.sleep(int(seconds))
        return f"Waited {seconds} seconds"

    def single_click(self, x, y, description):
        return CommandService.click(float(x) * (OS_WIDTH / self.model_width),
                                    float(y) * (OS_HEIGHT / self.model_height),
                                    self.user_id)

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
        width = float(x) * (OS_WIDTH / self.model_width)
        height = float(y) * (OS_HEIGHT / self.model_height)
        print(width, height)
        return CommandService.double_click(width, height, self.user_id)

    def move_mouse(self, x, y, description):
        width = float(x) * (OS_WIDTH / self.model_width)
        height = float(y) * (OS_HEIGHT / self.model_height)
        return CommandService.move_mouse(width, height, self.user_id)

    def right_click(self, x, y, description):
        return CommandService.right_click(float(x) * (OS_WIDTH / self.model_width),
                                          float(y) * (OS_HEIGHT / self.model_height),
                                          self.user_id)

    def type_text(self, text, description):
        CommandService.typing(text, self.user_id)
        return f"Typed: {text[:50]}..."

    def send_key(self, name, description):
        return CommandService.send_key(name, self.user_id)

    def append_screenshot(self):
        CommandService.screenshot(self.user_id)
        enhanced_prompt = self.task_rag.query_context(
            self.user_id, f"Memory data: {self.current_task.description}"
        )
        # recent_actions = "\n".join(k for k in self._get_recent_actions(7))
        coordinates = position_model.call('''give all the windows and icons and buttons and picture size coordinates and it's type of that you can see EXACTLY in this FORMAT 
            {"desktop": [Xmin, Ymin, Xmax, Ymax], "terminal_window": [Xmin, Ymin, Xmax, Ymax], "terminal_icon": [Xmin, Ymin, Xmax, Ymax], ...}
            DO NOT ADD OTHER WORDS JUST FOLLOW THE FORMAT 
            Add the type of that thing in the end of the name of that thing like example
            ''', DataService().get_user_path_screenshot(self.user_id))
        user = self.db.query(User).filter(User.id == self.user_id).first()
        username = user.name.replace(' ', '_').lower()
        vision_prompt = f"""Analyze this screenshot and respond in this EXACT format:

        Current Objective: {self.current_task.description}
        Priority: {self.current_task.priority}

        Screen Analysis:
        - [Windows]: [[Describe first visible window and its exact state and it's coordinates], [Describe next visible window and its state and it's coordinates], ...]
        - [Task UI Elements]: [[Describe first visible element relevant to the task and its state and it's coordinates], [Describe next visible element relevant to the task and its state and it's coordinates], ...]
        - [Other UI Elements]: [[Describe first visible element and its state and it's coordinates], [Describe next visible element and its state and it's coordinates], ...]
        - [Result of last action]: [What was the result of the last action base of picture and recent actions part and that was succeed or do you see problems? And if you see problem why this happened and what we can do for this to rollback or handle this situation]
        - [width and height size of the picture]: [The last x and y of whole picture]

        Task Status: [complete/not complete]
        Reason: [Explain why task is complete or not]
        Something should store in memory: [The things you learned from the last actions and it's needed to save in memory]

        Next Actions (list ALL required actions in order):

        1. Action:
        - Type: [single_click/double_click/right_click/type_text/send_key/wait/create_task/stop/move_mouse]
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
        7. For open apps and files you MUST use double_click instead of single_click
        8. For errors and something like them you can use create_task to create new task with higher priority.
        9. If a task with higher priority is created via create_task:
            - Immediately terminate the current task’s action list.
            - Only output the new task (no other actions or tasks).
            - Do NOT add subsequent actions to the original task.
        10. When describing coordinates or locations of icons, windows, or any rectangular area, you must provide all four corner points (top-left, top-right, bottom-right, bottom-left) to precisely define the boundaries of the object. The format should be: [(x1,y1), (x2,y2), (x3,y3), (x4,y4)] where these points represent the four vertices of the rectangle in clockwise or counter-clockwise order.
        11. Please do not forget details in screen analysis like press enter or click
        12. Sometimes os got shut down and everything on it got disappeared. So maybe the windows got closed. Consider screenshot detail in .
        13. Sometimes you see command for enter in command line you should report all of it in the detail of windows
        14. Actions must be explicit and deterministic—no conditionals (e.g., 'if', 'or') or ambiguity. Define all possible paths as separate, concrete steps (e.g., 'double_click icon_X' OR 'run command_Y'), but never combine them conditionally.
        15. If you want to open an app you can use application menu or desktop or terminal to open it.
        16. The current task state is defined by the provided Recent Actions; before executing any UI‑ or system‑altering Action, you must review the Recent Actions to understand the context, perform a quick Screen Analysis or CLI check to confirm the actual system/UI state, and then select the next Action that aligns with both the Recent Actions history and the verified state.
        17. For downloading big files first check this address in your firefox `ftp://ftp-server:20` username: myuser password: mypassword
        18. A messenger in the http://rocketchat:3000 is up. And you should check it times to times. Your lead will give you messages in there. And sometimes for some interaction with others you will need it. Your username is {username} and password is `mypassword`

        Current task objective: {self.current_task.description}
        Priority: {self.current_task.priority}
        Coordinates of the picture(You should use this to generate the response that I want): {coordinates}
        
        """
        # Recent actions: {recent_actions}
        response = vision_model.call(
            enhanced_prompt + vision_prompt, DataService().get_user_path_screenshot(self.user_id))
        print(f'screenshot response: {response}')
        return response

    def process_task_step(self):
        try:
            screenshot_thought = self.append_screenshot()
            print(f"screenshot: {screenshot_thought}")

            self.memory.add_to_short_term({
                'type': 'screenshot_analysis',
                'thought': screenshot_thought,
                'timestamp': datetime.now()
            })

            # task_context = self.task_rag.query_context(
            #     f"Current task: {self.current_task.description}"
            # )
            # enhanced_prompt = f"""
            #             TASK CONTEXT:
            #             {task_context}
            #
            #             CURRENT OBJECTIVE:
            #             {self.current_task.description}
            #             """
            action_prompt = f"""Based on the recent actions and screenshot analysis and task objective, determine the next action to take.

                    Screenshot analysis:
                    {screenshot_thought}

                    Rules to follow:
                    1. If a task with a higher priority number (higher urgency) is created via create_task:
                        - Immediately terminate the current task’s action list.
                        - Only output the new task (no other actions or tasks).
                        - Do NOT add subsequent actions to the original task.
                    2. Pre-actions are CRITICAL – They determine if the main action can execute. But before that check recent actions maybe it happened in the past
                    3. Window Focus Requirement:
                        - If pre-actions specify focusing on a particular window:
                            - Physically click on the **window itself** (not its icon or other UI elements).
                            - Example: For "LX Terminal," click the terminal window, not the LX Terminal icon.
                            - Verify focus by checking window attributes (title, active state) before proceeding.
                    4. In send_key for enter something you should use `Return` word
                    5. If you want apps opened in the taskbar you should say it's name that you hve in screenshot analysis
                    6. For ensure the apps is focused you need to click on them before the main action. But before that check recent actions maybe it happened in the past
                    7. For each action of screenshot thought you should give me one action
                    8. For click or open something click on center of it
                    9. Before executing any Pre-Action (e.g., clicking to Focus), always check if the same action exists in Recent Actions. If present, do NOT repeat it."
                    10. Try to use EXACTLY the action that I told you in the promt. double_click => double_click
                    11. For more accuracy you MUST just answer one action.(just for pre-focus on windows you can use one more action before the real action)

                    
                    Current task objective: {self.current_task.description}
                    Priority: {self.current_task.priority}
                    Determine next action using available tools.
"""
            # Recent actions (in desc format): {self._get_recent_actions(7)}

            for x, y in tools.items():
                y['params']["last_action_result"] = "What was the result of the last action"
                y['params']["image_width"] = "The screenshot width size"
                y['params']["image_height"] = "The screenshot height size"
                y['params']["description"] = "Reason"
                y['params']["memory_data"] = "The things need to save in memory"
            response = action_model.call(
                action_prompt,
                DataService().get_user_path_screenshot(self.user_id),
                tools
            )
            print(f'ACTION MODEL RESPONSE: {response}')

            should_continue = True
            for tool_call in response[1]:
                try:
                    action = tool_call['name']
                    args = deepcopy(tool_call['parameters'])
                    memory_data = args['memory_data']
                    self.task_rag.add_text(self.user_id,
                                           "event: " + memory_data + " timestamp: " + datetime.now().isoformat())
                    print(f'action: {action} args: {args}')

                    if hasattr(self, action):
                        if args['image_width'] and args['image_height']:
                            self.model_height = float(args['image_height'])
                            self.model_width = float(args['image_width'])
                        args.pop('image_height')
                        args.pop('image_width')
                        args.pop('last_action_result')
                        args.pop('memory_data')
                        result = getattr(self, action)(**args)

                        self.task_rag.add_text(
                            self.user_id, "action: " + action + " args:" + " ".join(
                                f'{k}: {v}' for k, v in args.items()) + " timestamp: " + datetime.now().isoformat()
                        )
                        # ذخیره نتیجه اقدام
                        # self.memory.add_to_short_term({
                        #     'type': 'action',
                        #     'action': action,
                        #     'args': args,
                        #     'result': result,
                        #     'timestamp': datetime.now()
                        # })

                        # اگر action ایجاد تسک بود و تسک جدید اولویت بالاتری داشت، باید ادامه ندهیم
                        if action == "create_task" and "switched to higher priority task" in result:
                            should_continue = False

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

        except Exception as e:
            error_msg = f"Error in process_task_step: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            self.current_task.task_messages.append(
                TaskMessage(content=json.dumps({
                    "role": "system_error",
                    "content": error_msg,
                    "timestamp": datetime.now().isoformat()
                }))
            )
            self.db.commit()
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

    def _get_last_predictions(self):
        """خواندن ۳ پیش‌بینی آخر از task_messages"""
        predictions = []
        for msg in reversed(self.current_task.task_messages):
            try:
                data = json.loads(msg.content)
                if isinstance(data, dict) and data.get("role") == "prediction":
                    if "predictions" in data:
                        predictions.extend(data["predictions"])
                    elif "content" in data and isinstance(data["content"], dict) and "predictions" in data["content"]:
                        predictions.extend(data["content"]["predictions"])

                    if len(predictions) >= 3:
                        break
            except json.JSONDecodeError:
                continue

        # اطمینان از ساختار صحیح پیش‌بینی‌ها
        validated_predictions = []
        for pred in predictions[-3:]:
            if isinstance(pred, dict) and "main_action" in pred:
                validated_predictions.append({
                    "main_action": pred["main_action"],
                    "pre_actions": pred.get("pre_actions", []),
                    "post_actions": pred.get("post_actions", []),
                    "reason": pred.get("reason", "")
                })

        return validated_predictions

    def _get_last_actions(self, count=7):
        """خواندن آخرین اکشن‌ها از تاریخچه"""
        actions = []
        # پیمایش معکوس پیام‌ها
        for msg in reversed(self.current_task.task_messages):
            try:
                data = json.loads(msg.content)
                if data.get("role") == "action":
                    actions.append(data)
                    # توقف وقتی به تعداد مورد نیاز رسیدیم
                    if len(actions) >= count:
                        break
            except:
                continue
        return list(reversed(actions[-count:]))  # حفظ ترتیب زمانی

    def _prepare_action_context(self):
        """تهیه زمینه از task_messages برای مدل"""
        recent_actions = self._get_recent_actions(7)

        return {
            "recent_actions": [{
                "name": a.get("name"),
                "params": a.get("params", {}),
                "timestamp": a.get("timestamp")
            } for a in recent_actions],
        }

    def _get_recent_actions(self, count=7):
        actions = []
        for msg in reversed(self.current_task.task_messages):
            try:
                data = json.loads(msg.content)
                if isinstance(data, dict):
                    copy_data = deepcopy(json.loads(data['content']))
                    sentence = copy_data['name']
                    copy_data['parameters'].pop('image_height')
                    copy_data['parameters'].pop('image_width')
                    if 'save_in_memory' in copy_data['parameters'].keys():
                        copy_data['parameters'].pop('save_in_memory')
                    sentence = sentence + " " + " ".join(f"{k}:{v}" for k, v in copy_data['parameters'].items())
                    actions.append(sentence + ' time: ' + str(msg.created_at))
                    if len(actions) >= count:
                        break
            except json.JSONDecodeError:
                continue

        return list(reversed(actions[-count:]))

    def _execute_action_sequence(self, action_chain):
        """اجرای کامل یک دنباله اکشن با pre و post actions"""
        try:
            # اجرای pre-actions
            for pre_action in action_chain.get("pre_actions", []):
                self._log_action(
                    pre_action["name"],
                    pre_action.get("params", {}),
                    "pre_action",
                    action_chain.get("reason", "")
                )
                result = self._execute_single_action(pre_action)
                if result == "high_priority_task":
                    return result

            # اجرای اکشن اصلی
            main_action = action_chain["main_action"]
            self._log_action(
                main_action["name"],
                main_action.get("params", {}),
                "main_action",
                action_chain.get("reason", "")
            )
            result = self._execute_single_action(main_action)
            if result == "high_priority_task":
                return result

            # اجرای post-actions
            for post_action in action_chain.get("post_actions", []):
                self._log_action(
                    post_action["name"],
                    post_action.get("params", {}),
                    "post_action",
                    action_chain.get("reason", "")
                )
                result = self._execute_single_action(post_action)
                if result == "high_priority_task":
                    return result

            return "continue"

        except Exception as e:
            error_msg = f"Error executing action sequence: {str(e)}"
            print(error_msg)
            # self.current_task.task_messages.append(
            #     TaskMessage(content=json.dumps({
            #         "role": "action_error",
            #         "content": error_msg,
            #         "timestamp": datetime.now().isoformat()
            #     })))
            # self.db.commit()
            return "error"

    def _log_action(self, action_name, params, action_type, reason=""):
        action_msg = {
            "role": "action",
            "type": action_type,
            "name": action_name,
            "params": params,
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        }
        self.current_task.task_messages.append(TaskMessage(content=json.dumps(action_msg)))
        self.db.commit()

        def _execute_single_action(self, action):
            """اجرای یک اکشن منفرد"""
            try:
                action_name = action["name"]
                action_params = action.get("params", {})

                # بررسی وجود متد اجرایی
                if not hasattr(self, action_name):
                    raise ValueError(f"Unknown action: {action_name}")

                # اجرای اکشن
                result = getattr(self, action_name)(**action_params)

                # بررسی تغییر اولویت
                if action_name == "create_task":
                    new_priority = action_params.get("priority", 0)
                    if isinstance(new_priority, str):
                        new_priority = int(new_priority)
                    if new_priority > self.current_task.priority:
                        return "high_priority_task"

                return "continue"

            except Exception as e:
                error_msg = f"Failed to execute {action_name}: {str(e)}"
                print(error_msg)
                self.current_task.task_messages.append(
                    TaskMessage(content=json.dumps({
                        "role": "action_error",
                        "action": action_name,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    })))
                self.db.commit()
                return "error"
