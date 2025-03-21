import os
import tempfile
from PIL import Image
import json
import base64  # اضافه کردن کتابخانه‌ی base64

from orchestrator.models import user
from orchestrator.models.task import Message, TaskStatus, Task, TaskMessage
from orchestrator.services.command_service import CommandService
from orchestrator.services.config_service import grounding_model, vision_model, action_model
from orchestrator.services.grounding_service import draw_big_dot

TYPING_DELAY_MS = 12
TYPING_GROUP_SIZE = 50

tools = {
    "stop": {
        "description": "Indicate that the task has been completed.",
        "params": {},
    }
}


class ProcessorService:

    def __init__(self, db, task):
        self.db = db
        self.task = task
        # self.image_counter = 0  # Current screenshot number
        # self.tmp_dir = tempfile.mkdtemp()  # Folder to store screenshots

        print("The agent will use the following actions:")
        for action, details in tools.items():
            param_str = ", ".join(details.get("params").keys())
            print(f"- {action}({param_str})")

    def call_function(self, name, arguments):
        func_impl = getattr(self, name.lower()) if name.lower() in tools else None
        if func_impl:
            try:
                result = func_impl(**arguments) if arguments else func_impl()
                return result
            except Exception as e:
                return f"Error executing function: {str(e)}"
        else:
            return "Function not implemented."

    def tool(description, params):
        def decorator(func):
            tools[func.__name__] = {"description": description, "params": params}
            return func

        return decorator

    def save_image(self, image, prefix="image"):
        self.image_counter += 1
        filename = f"{prefix}_{self.image_counter}.png"
        filepath = os.path.join(self.tmp_dir, filename)
        if isinstance(image, Image.Image):
            image.save(filepath)
        else:
            with open(filepath, "wb") as f:
                f.write(image)
        return filepath

    def screenshot(self):
        CommandService.screenshot(self.task.user_id)
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(BASE_DIR, "data", str(self.task.user_id), 'screenshot.png')
        self.latest_screenshot = filename
        with open(filename, "rb") as image_file:
            return image_file.read()

    @tool(
        description="Run a shell command and return the result.",
        params={"command": "Shell command to run synchronously"},
    )
    def run_command(self, command):
        result = CommandService.run_command_via_container(command, self.task.user_id)
        return result

    @tool(
        description="Run a shell command in the background.",
        params={"command": "Shell command to run asynchronously"},
    )
    def run_background_command(self, command):
        result = CommandService.run_background_command_via_container(command, self.task.user_id)
        return result

    @tool(
        description="Send a key or combination of keys to the system.",
        params={"name": "Key or combination (e.g. 'Return', 'Ctl-C')"},
    )
    def send_key(self, name):
        result = CommandService.send_key(name, self.task.user_id)
        return result

    @tool(
        description="Type a specified text into the system.",
        params={"text": "Text to type"},
    )
    def type_text(self, text):
        CommandService.typing(text, self.task.user_id)
        return "The text has been typed."

    def find_x_y(self, query):
        """Base method for all click operations"""
        self.screenshot()
        position = grounding_model.call(query, self.latest_screenshot)
        dot_image = draw_big_dot(Image.open(self.latest_screenshot), position)
        filepath = self.save_image(dot_image, "location")
        return position

    @tool(
        description="Click on a specified UI element.",
        params={"query": "Item or UI element on the screen to click"},
    )
    def click(self, query):
        print('clicking')
        print(query)
        x, y = self.find_x_y(query)
        result = CommandService.click(x, y, self.task.user_id)
        return result

    @tool(
        description="Double click on a specified UI element.",
        params={"query": "Item or UI element on the screen to double click"},
    )
    def double_click(self, query):
        x, y = self.find_x_y(query)
        result = CommandService.double_click(x, y, self.task.user_id)
        return result

    @tool(
        description="Right click on a specified UI element.",
        params={"query": "Item or UI element on the screen to right click"},
    )
    def right_click(self, query):
        x, y = self.find_x_y(query)
        result = CommandService.right_click(x, y, self.task.user_id)
        return result


    def append_screenshot(self):
        screenshot_bytes = self.screenshot()
        screenshot_message_for_model = {
            "role": "user",
            "content": [
                screenshot_bytes,  # ارسال به صورت باینری
                "This image shows the current display of the computer. Please respond in the following format:\n"
                "The objective is: [put the objective here]\n"
                "On the screen, I see: [an extensive list of everything that might be relevant to the objective including windows, icons, menus, apps, and UI elements]\n"
                "This means the objective is: [complete|not complete]\n\n"
                "(Only continue if the objective is not complete.)\n"
                "The next step is to [click|type|run the shell command] [put the next single step here] in order to [put what you expect to happen here]."
            ]
        }
        # print(messages)
        # ارسال پیام به مدل
        model_response = vision_model.call(self.task.messages() + [screenshot_message_for_model])
        
        # تبدیل تصویر به Base64 برای ذخیره‌سازی در دیتابیس
        # screenshot_base64 = base64.b64encode(screenshot_bytes).decode('utf-8')
        
        # ایجاد پیام برای ذخیره‌سازی در دیتابیس
        screenshot_message_for_db = TaskMessage(
            content=json.dumps({
                "role": "user",
                "content": 
                    """This image shows the current display of the computer. Please respond in the following format:
                    The objective is: [put the objective here]
                    On the screen, I see: [an extensive list of everything that might be relevant to the objective including windows, icons, menus, apps, and UI elements]
                    This means the objective is: [complete|not complete]
                    (Only continue if the objective is not complete.)
                    The next step is to [click|type|run the shell command] [put the next single step here] in order to [put what you expect to happen here]."""
                
            })
        )
        
        # اضافه کردن پیام به task_messages
        # self.task.task_messages.append(screenshot_message_for_db)
        
        return model_response

    def process_task(self):
        # تبدیل محتوای task_messages به دیکشنری

        initial_message = {
            "role": "system",
            "content": f"OBJECTIVE: {self.task.description}"
        }
        # اضافه کردن پیام اولیه (هدف تسک)
        self.task.task_messages.append(TaskMessage(content=json.dumps(initial_message)))
        
        should_continue = True
        while should_continue:
            # اضافه کردن اسکرین‌شات و دریافت پاسخ از مدل
            screenshot_thought = self.append_screenshot()
            self.task.task_messages.append(TaskMessage(content=json.dumps({
                "role": "user",
                "content": f"THOUGHT: {screenshot_thought}"
            })))
            
            # فراخوانی مدل برای دریافت محتوا و فراخوانی ابزارها
            content, tool_calls = action_model.call( 
                    [{"role": "system", "content": "You are an AI assistant with computer use abilities."}] +
                    self.task.messages() +
                    [{"role": "assistant", "content": "I will now use tool calls to take these actions, or use the stop command if the objective is complete."}],
                tools,
            )
            print(content, tool_calls)
            
            if content:
                # اضافه کردن پاسخ مدل به پیام‌ها
                thought_message = TaskMessage(
                    content=json.dumps({
                        "role": "assistant",
                        "content": f"THOUGHT: {content}"
                    })
                )
                self.task.task_messages.append(thought_message)
                
            should_continue = False
            for tool_call in tool_calls:
                name, parameters = tool_call.get("name"), tool_call.get("parameters")
                should_continue = name != "stop"
                if not should_continue:
                    break
                
                # اضافه کردن فراخوانی ابزار به پیام‌ها
                # tool_call_message = TaskMessage(
                #     content=json.dumps(tool_call)
                # )
                # self.task.task_messages.append(tool_call_message)
                
                # اجرای ابزار و دریافت نتیجه
                result = self.call_function(name, parameters)
                
                # اضافه کردن نتیجه به پیام‌ها
                observation_message = TaskMessage(
                    content=json.dumps({
                        "role": "assistant",
                        "content": f"OBSERVATION: {result}"
                    })
                )
                self.task.task_messages.append(observation_message)
        
        # ذخیره‌سازی تسک در دیتابیس
        self.db.add(self.task)
        self.db.commit()
