import json
import unittest

from g4f.Provider import Blackbox, LMArenaProvider
from gradio_client import Client, handle_file
import g4f
from PIL import Image, ImageDraw, ImageFont
import json
from orchestrator.models.base import SessionLocal
from orchestrator.services.command_service import CommandService
from orchestrator.services.data_service import DataService
from orchestrator.services.llm_provider import G4FProvider
from orchestrator.services.processor_service import ProcessorService
import os

from orchestrator.services.qwen_service import QwenProvider, HF_TOKEN
from orchestrator.services.rag_service import RAGSystem

OSATLAS_HUGGINGFACE_SOURCE = "maxiw/os-atlas"
OSATLAS_HUGGINGFACE_MODEL = "os-copilot/os-atlas-base-7b"
OSATLAS_HUGGINGFACE_API = "/run_example"


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
        # db.commit()

    def test_click_test(self):
        # CommandService.typing( "test", 1)
        CommandService.right_click(
            47.0,
            211.80555555555554
            , 1)
        # CommandService.click(32, 64, 1)

    def test_send_send_keys_test(self):
        # CommandService.typing( "test", 1)
        CommandService.send_key('Enter key', 1)
        # CommandService.click(32, 64, 1)

    # def test_huggingface(self):
    #     HF_TOKEN = os.getenv("HF_TOKEN")
    #     client = Client(OSATLAS_HUGGINGFACE_SOURCE, hf_token=HF_TOKEN)
    #     result = client.predict(
    #         image=handle_file(image_data),
    #         text_input="",
    #         model_id=OSATLAS_HUGGINGFACE_MODEL,
    #         api_name=OSATLAS_HUGGINGFACE_API,
    #     )
    def test_g4f(self):
        testMessage = '''
        
                    TASK CONTEXT:
                    Context 1:
[TASK 1]
            Description: Install telegram desktop on your system
            Status: TaskStatus.IN_PROGRESS
            Priority: 0
            Created at: None
            
                    [MESSAGE 1]
                    Type: assistant
                    Timestamp: 2025-04-14 08:11:08.201798

Context 2:
Content: {"type": "function", "name": "double_click", "parameters": {"x": "57", "image_width": "1700", "description": "To open the terminal and start the installation process of Telegram desktop.", "image_height": "500", "y": "237", "last_action_result": "The terminal is focused, but it is not currently open on the screen. The previous action was to type the installation command, but the terminal is not visible, indicating that it might not have been opened yet."}}

Context 3:
Timestamp: 2025-04-16 10:41:44.933286
                    Content: {"type": "function", "name": "double_click", "parameters": {"x": "60", "image_width": "1420", "description": "To open the terminal and start the installation process of Telegram desktop.", "last_action_result": "The terminal is focused, but it is not currently open on the screen. The Display Settings window is open, which is unrelated to the current task.", "image_height": "670", "y": "238"}}

                    CURRENT OBJECTIVE:
                    Install telegram desktop on your system
                    Analyze this screenshot and respond in this EXACT format:

        Current Objective: Install telegram desktop on your system
        Priority: 0

        Screen Analysis:
        - [Windows]: [[Describe first visible window and its exact state and it's coordinates], [Describe next visible window and its state and it's coordinates], ...]
        - [Task UI Elements]: [[Describe first visible element relevant to the task and its state and it's coordinates], [Describe next visible element relevant to the task and its state and it's coordinates], ...]
        - [Other UI Elements]: [[Describe first visible element and its state and it's coordinates], [Describe next visible element and its state and it's coordinates], ...]
        - [Result of last action]: [What was the result of the last action base of picture and recent actions part]
        - [width and height size]: [The last x and y of the window]

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
        10. When describing coordinates or locations of icons, windows, or any rectangular area, you must provide all four corner points (top-left, top-right, bottom-right, bottom-left) to precisely define the boundaries of the object. The format should be: [(x1,y1), (x2,y2), (x3,y3), (x4,y4)] where these points represent the four vertices of the rectangle in clockwise or counter-clockwise order.
        11. Please do not forget details in screen analysis like press enter or click
        12. Sometimes os got shut down and everything on it got disappeared. So maybe the windows got closed. Consider screenshot detail in .
        13. Sometimes you see command for enter in command line you should report all of it in the detail of windows

        Current task objective: Install telegram desktop on your system
        Priority: 0
        Recent actions: [{'role': 'assistant', 'content': '{"type": "function", "name": "type_text", "parameters": {"description": "To update the package list and install Telegram desktop.", "image_height": "700", "text": "sudo apt update && sudo apt install telegram-desktop", "image_width": "1400", "last_action_result": "The last action was to add a GPG key for the Google Chrome repository, which was successfully executed. This action is unrelated to the current task of installing Telegram desktop."}}'}, {'role': 'assistant', 'content': '{"type": "function", "name": "click", "parameters": {"last_action_result": "The last action was to add the GPG key for the Google Chrome repository and update the package list. The terminal is ready to execute the command to install Telegram Desktop.", "image_height": "495", "y": "325", "image_width": "958", "description": "Ensure the terminal window is focused", "x": "714"}}'}, {'role': 'assistant', 'content': '{"type": "function", "name": "send_key", "parameters": {"description": "Press Enter to execute the command `sudo apt update && sudo apt install telegram-desktop`.", "name": "Return", "image_width": "1295", "image_height": "680", "last_action_result": "The user is now ready to update the package list and install Telegram Desktop."}}'}, {'role': 'assistant', 'content': '{"type": "function", "name": "wait", "parameters": {"description": "Wait for the terminal to complete the connection and installation process", "image_height": "496", "image_width": "958", "seconds": "30", "last_action_result": "The last action was to add the GPG key and start the installation of Telegram Desktop. The terminal is currently connecting to the Ubuntu mirrors to fetch the package."}}'}, {'role': 'assistant', 'content': '{"type": "function", "name": "click", "parameters": {"y": "325", "image_height": "500", "description": "Focus on LXTerminal window", "last_action_result": "None", "x": "713", "image_width": "1000"}}'}, {'role': 'assistant', 'content': '{"type": "function", "name": "type_text", "parameters": {"description": "Type \'Y\' and press Enter", "image_height": "1080", "text": "Y", "image_width": "1920", "last_action_result": "The last action was likely the execution of a command to install Telegram Desktop, which has prompted the user to confirm the installation by typing \'Y\' or \'n\'."}}'}, {'role': 'assistant', 'content': '{"type": "function", "name": "type_text", "parameters": {"text": "y", "image_width": "955", "last_action_result": "None", "description": "The terminal is awaiting user confirmation to proceed with the installation. Typing \\"y\\" will confirm the action and allow the installation to continue.", "image_height": "495"}}'}]
        '''
        response = G4FProvider('deepseek-v3').call({'role': 'user', 'content': testMessage})
        print(response)

    def test_action_model_with_g4f(self):

        tool_calls = {
            "click": {
                "description": "Click UI element",
                "params": {
                    "x": "X", "y": "Y",
                    "last_action_result": "What was the result of the last action",
                    "image_width": "The screenshot width size",
                    "image_height": "The screenshot height size",
                    "description": "Reason"}
            },
            "double_click": {
                "description": "Double-click UI element",
                "params": {
                    "x": "X", "y": "Y",
                    "last_action_result": "What was the result of the last action",
                    "image_width": "The screenshot width size",
                    "image_height": "The screenshot height size",
                    "description": "Reason"}
            }
        }
        messages = [{
            'role': 'user',
            'content': 'randomly click and double_click'
        }]
        G4FProvider('').call(messages, tool_calls)

    def test_another_action_model(self):
        messages = json.loads("""
[{"role": "user", "content": "                    TASK CONTEXT:                    Context 1:[TASK 1]            Description: Install telegram desktop on your system            Status: TaskStatus.IN_PROGRESS            Priority: 0            Created at: None                                [MESSAGE 1]                    Type: assistant                    Timestamp: 2025-04-14 08:11:08.201798Context 2:Content: {\"type\": \"function\", \"name\": \"double_click\", \"parameters\": {\"x\": \"57\", \"image_width\": \"1700\", \"description\": \"To open the terminal and start the installation process of Telegram desktop.\", \"image_height\": \"500\", \"y\": \"237\", \"last_action_result\": \"The terminal is focused, but it is not currently open on the screen. The previous action was to type the installation command, but the terminal is not visible, indicating that it might not have been opened yet.\"}}Context 3:Timestamp: 2025-04-16 10:41:44.933286                    Content: {\"type\": \"function\", \"name\": \"double_click\", \"parameters\": {\"x\": \"60\", \"image_width\": \"1420\", \"description\": \"To open the terminal and start the installation process of Telegram desktop.\", \"last_action_result\": \"The terminal is focused, but it is not currently open on the screen. The Display Settings window is open, which is unrelated to the current task.\", \"image_height\": \"670\", \"y\": \"238\"}}                    CURRENT OBJECTIVE:                    Install telegram desktop on your system                    Analyze this screenshot and respond in this EXACT format:        Current Objective: Install telegram desktop on your system        Priority: 0        Screen Analysis:        - [Windows]: [[Describe first visible window and its exact state and it's coordinates], [Describe next visible window and its state and it's coordinates], ...]        - [Task UI Elements]: [[Describe first visible element relevant to the task and its state and it's coordinates], [Describe next visible element relevant to the task and its state and it's coordinates], ...]        - [Other UI Elements]: [[Describe first visible element and its state and it's coordinates], [Describe next visible element and its state and it's coordinates], ...]        - [Result of last action]: [What was the result of the last action base of picture and recent actions part]        - [width and height size]: [The last x and y of the window]        Task Status: [complete/not complete]        Reason: [Explain why task is complete or not]        Next Actions (list ALL required actions in order):        1. Action:        - Type: [click/double_click/right_click/type_text/send_key/wait/create_task/stop]        - Target: [Specific element or location]        - Detail: [Detail of type in text]        - Reason: [Why this action is needed]        - Pre-Actions: [List any preparation steps]        - Post-Actions: [List any follow-up steps]        - Thought: [Thought of this action]        2. Action:        - Type: [Next action type]        - Target: [Specific element or location]        - ...        Now analyze the screenshot and provide your response in the requested format.        Rules to follow:        1. ALWAYS maintain this exact format        2. List ALL required actions in proper sequence        3. Include ALL necessary pre/post actions        4. For terminal interactions:           - Ensure terminal is focused first           - Include appropriate wait times        5. For system commands:           - Check for errors in response        6. Be specific about targets and reasons        7. For open apps and files you should use double_click instead of click        8. For errors and something like them you can use create_task to create new task with higher priority.        9. If a task with higher priority is created via create_task:            - Immediately terminate the current task\u2019s action list.            - Only output the new task (no other actions or tasks).            - Do NOT add subsequent actions to the original task.        10. When describing coordinates or locations of icons, windows, or any rectangular area, you must provide all four corner points (top-left, top-right, bottom-right, bottom-left) to precisely define the boundaries of the object. The format should be: [(x1,y1), (x2,y2), (x3,y3), (x4,y4)] where these points represent the four vertices of the rectangle in clockwise or counter-clockwise order.        11. Please do not forget details in screen analysis like press enter or click        12. Sometimes os got shut down and everything on it got disappeared. So maybe the windows got closed. Consider screenshot detail in .        13. Sometimes you see command for enter in command line you should report all of it in the detail of windows        Current task objective: Install telegram desktop on your system        Priority: 0        Recent actions: [{'role': 'assistant', 'content': '{\"type\": \"function\", \"name\": \"type_text\", \"parameters\": {\"description\": \"To update the package list and install Telegram desktop.\", \"image_height\": \"700\", \"text\": \"sudo apt update && sudo apt install telegram-desktop\", \"image_width\": \"1400\", \"last_action_result\": \"The last action was to add a GPG key for the Google Chrome repository, which was successfully executed. This action is unrelated to the current task of installing Telegram desktop.\"}}'}, {'role': 'assistant', 'content': '{\"type\": \"function\", \"name\": \"click\", \"parameters\": {\"last_action_result\": \"The last action was to add the GPG key for the Google Chrome repository and update the package list. The terminal is ready to execute the command to install Telegram Desktop.\", \"image_height\": \"495\", \"y\": \"325\", \"image_width\": \"958\", \"description\": \"Ensure the terminal window is focused\", \"x\": \"714\"}}'}, {'role': 'assistant', 'content': '{\"type\": \"function\", \"name\": \"send_key\", \"parameters\": {\"description\": \"Press Enter to execute the command `sudo apt update && sudo apt install telegram-desktop`.\", \"name\": \"Return\", \"image_width\": \"1295\", \"image_height\": \"680\", \"last_action_result\": \"The user is now ready to update the package list and install Telegram Desktop.\"}}'}, {'role': 'assistant', 'content': '{\"type\": \"function\", \"name\": \"wait\", \"parameters\": {\"description\": \"Wait for the terminal to complete the connection and installation process\", \"image_height\": \"496\", \"image_width\": \"958\", \"seconds\": \"30\", \"last_action_result\": \"The last action was to add the GPG key and start the installation of Telegram Desktop. The terminal is currently connecting to the Ubuntu mirrors to fetch the package.\"}}'}, {'role': 'assistant', 'content': '{\"type\": \"function\", \"name\": \"click\", \"parameters\": {\"y\": \"325\", \"image_height\": \"500\", \"description\": \"Focus on LXTerminal window\", \"last_action_result\": \"None\", \"x\": \"713\", \"image_width\": \"1000\"}}'}, {'role': 'assistant', 'content': '{\"type\": \"function\", \"name\": \"type_text\", \"parameters\": {\"description\": \"Type \\'Y\\' and press Enter\", \"image_height\": \"1080\", \"text\": \"Y\", \"image_width\": \"1920\", \"last_action_result\": \"The last action was likely the execution of a command to install Telegram Desktop, which has prompted the user to confirm the installation by typing \\'Y\\' or \\'n\\'.\"}}'}, {'role': 'assistant', 'content': '{\"type\": \"function\", \"name\": \"type_text\", \"parameters\": {\"text\": \"y\", \"image_width\": \"955\", \"last_action_result\": \"None\", \"description\": \"The terminal is awaiting user confirmation to proceed with the installation. Typing \\\\\"y\\\\\" will confirm the action and allow the installation to continue.\", \"image_height\": \"495\"}}'}]                "}]
        """)
        images = [[open(
            DataService().get_user_path_screenshot(1),
            "rb"), "screenshot.png"]]
        response = g4f.Client(Blackbox).chat.completions.create(messages, 'blackboxai', images=images).choices[
            0].message.content

    def test_qwen(self):
        provider = QwenProvider()
        data = provider.call(
            'What is size of this image? And give all the windows and icons and buttons coordinates of that you can see',
            DataService().get_user_path_screenshot(1))
        print(data)

    def test_g4f1(self):
        provider = G4FProvider('')
        jsonResult = provider.call(
            '''give all the windows and icons and buttons and picture size coordinates of that you can see EXACTLY in this FORMAT 
            {"desktop": [Xmin, Ymin, Xmax, Ymax], "app_name_or_icon_name_or_button_name": [Xmin, Ymin, Xmax, Ymax], ...}
            DO NOT ADD OTHER WORDS JUST FOLLOW THE FORMAT 
            ''', None, LMArenaProvider, 'o3-2025-04-16')
        picture_file_path = DataService().get_user_path_screenshot(1)
        if jsonResult[0:3] == '```':
            self.drawPictureLines(picture_file_path, json.loads(jsonResult[7:-3]))
        else:
            self.drawPictureLines(picture_file_path, json.loads(jsonResult))

    def test_qwen2(self):
        HF_TOKEN = os.getenv("HF_TOKEN")
        client = Client("Qwen/Qwen2.5-VL-72B-Instruct", hf_token=HF_TOKEN)
        history = []
        messages = 'describe the picture'
        history = client.predict(
            history=history,
            text=messages,
            api_name="/add_text"
        )

        history = client.predict(
            history=history,
            file=handle_file(
                DataService().get_user_path_screenshot(1)),
            api_name="/add_file"
        )

        # Step 3: Get final prediction
        response = client.predict(
            _chatbot=history,
            api_name="/predict"
        )

    def drawPictureLines(self, picture_file_path, jsonResult):
        try:
            # Load the image
            image = Image.open(picture_file_path)

            # Create a draw object
            draw = ImageDraw.Draw(image)

            # Define drawing parameters
            rect_color = (255, 0, 0)  # Red color
            text_color = (255, 0, 0)  # Red color
            line_width = 2
            font = ImageFont.load_default()  # Use default font

            # Iterate over each item in jsonResult
            for item, coordinates in jsonResult.items():
                x_min, y_min, x_max, y_max = coordinates

                # Draw rectangle
                draw.rectangle(
                    [x_min, y_min, x_max, y_max],
                    outline=rect_color,
                    width=line_width
                )

                # Draw label text
                text_position = (x_min, y_min - 10)
                draw.text(
                    text_position,
                    item,
                    fill=text_color,
                    font=font
                )

            # Save the modified image
            output_path = picture_file_path.replace('.png', '_annotated.png')
            image.save(output_path)
            print(f"Annotated image saved to: {output_path}")

        except Exception as e:
            print(f"Error processing image: {str(e)}")

    def test_rag(self):
        rag1 = RAGSystem()
        rag = RAGSystem()
        rag1.add_text('The test is important')
        result = rag.query_context('give all the things that you know?')
        print(result)

    def test_lm_arena(self):
        images = [[open(
            DataService().get_user_path_screenshot(1),
            "rb"), "screenshot.png"]]
        response = g4f.Client(LMArenaProvider).chat.completions.create(
            '''give all the windows and icons and buttons and picture size coordinates and it's type of that you can see EXACTLY in this FORMAT 
            {"desktop": [Xmin, Ymin, Xmax, Ymax], "terminal_window": [Xmin, Ymin, Xmax, Ymax], "terminal_icon": [Xmin, Ymin, Xmax, Ymax], ...}
            DO NOT ADD OTHER WORDS JUST FOLLOW THE FORMAT 
            Add the type of that thing in the end of the name of that thing like example
            ''', '', images=images).choices[0].message.content  ## That was not great

    def test_screenshot_path(self):
        service = DataService()
        path = service.get_user_path_screenshot(1)
        self.assertEqual('/tmp/orchestrator/1/screenshot.png', path)

    if __name__ == '__main__':
        unittest.main()
