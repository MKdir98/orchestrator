import os
import time
import cv2

from dotenv import load_dotenv
from flask_jwt_extended import verify_jwt_in_request, get_jwt_identity

from orchestrator.services.desktop_element_detector import DesktopElementDetector

load_dotenv('../../.env-testing', override=True)
os.getenv('MODE')

from orchestrator.models.har import NewHarProvider
import shutil
import unittest
from unittest.mock import MagicMock

import docker
from g4f.Provider import Blackbox, HarProvider, PollinationsAI
from gradio_client import Client, handle_file
import g4f
from PIL import Image, ImageDraw, ImageFont
import json

from orchestrator.models import SystemUser
from orchestrator.models.base import SessionLocal, init_db
from orchestrator.models.task import Task, TaskStatus
from orchestrator.models.user import User
from orchestrator.services.chat_service import ChatService
from orchestrator.services.command_service import CommandService
from orchestrator.services.config_service import position_model, vision_model
from orchestrator.services.container_service import ContainerService
from orchestrator.services.data_service import DataService
from orchestrator.services.llm_provider import G4FProvider, HuggingFaceProvider
from orchestrator.services.divar3.solution import DivarContest
from orchestrator.services.processor_service import ProcessorService
from orchestrator.app import app, socketio

from orchestrator.services.qwen_service import QwenProvider, HF_TOKEN
from orchestrator.services.rag_service import RAGSystem
from orchestrator.models.base import clear_database, engine
from orchestrator.services.user_service import check_and_create_containers

OSATLAS_HUGGINGFACE_SOURCE = "maxiw/os-atlas"
OSATLAS_HUGGINGFACE_MODEL = "os-copilot/os-atlas-base-7b"
OSATLAS_HUGGINGFACE_API = "/run_example"


class ProcessorServiceTestCase(unittest.TestCase):
    db = SessionLocal()
    valid_user_system_token = ''

    def remove_orchestrator_containers(self):
        client = docker.from_env()
        all_containers = client.containers.list(all=True)
        orchestrator_containers = [
            container for container in all_containers
            if container.name.startswith('orchestrator_container_')
        ]
        for container in orchestrator_containers:
            try:
                container.remove(force=True)
            except Exception as e:
                print(f"Error removing container {container.name}: {str(e)}")

    def remove_users_in_rocket_chat(self):
        chatService = ChatService()
        for user in chatService.users():
            chatService.user_delete(user['id'])

    def remove_orchestrator_data_folder(self):
        shutil.rmtree(DataService().get_orchestrator_path())

    def call_post(self, path: str, payload: dict, token: str = None):
        if token is None:
            token = self.valid_user_system_token
        return self.client.post(path, headers={'Authorization': f'Bearer {token}'},
                                json=payload).get_json()

    def call_get(self, path: str, token: str = None):
        if token is None:
            token = self.valid_user_system_token
        return self.client.get(path, headers={'Authorization': f'Bearer {token}'}).get_json()

    def setUp(self):
        super().setUp()
        self.remove_orchestrator_data_folder()
        self.remove_orchestrator_containers()
        self.remove_users_in_rocket_chat()
        clear_database()
        DataService().get_orchestrator_path()
        init_db()
        self.valid_user_system_token = self.create_system_user_and_get_token()
        self.another_valid_user_system_token = self.create_system_user_and_get_token('karam2@karm.krm')

    def create_system_user_and_get_token(self, email='karam@karam.krm', password="testpass"):
        payload = {
            "email": email,
            "password": password
        }
        self.client.post("/signup", json=payload)
        payload = {
            "email": email,
            "password": password
        }
        token_ = self.client.post("/login", json=payload).get_json()['access_token']
        return token_

    def tearDown(self):
        super().tearDown()
        self.remove_orchestrator_containers()
        self.remove_users_in_rocket_chat()
        self.remove_orchestrator_data_folder()
        clear_database()

    client = app.test_client()

    def test_signup(self):
        payload = {
            "email": "test@testi.tst",
            "password": "testpass"
        }
        response = self.client.post("/signup", json=payload)
        assert response.status_code == 200
        users = self.db.query(SystemUser).all()
        self.assertEqual(3, len(users))
        self.assertEqual('test@testi.tst', users[2].email)
        self.assertEqual('testing', os.getenv('MODE'))

    def test_login(self):
        payload = {
            "email": "test@testi.tst",
            "password": "testpass"
        }
        self.client.post("/signup", json=payload)

        payload = {
            "email": "test@testi.tst",
            "password": "testpass"
        }
        response = self.client.post("/login", json=payload)
        assert response.status_code == 200
        self.assertIn('access_token', response.get_json().keys())

    def test_create_and_get_groups(self):
        self.create_group()
        response = self.call_get('/api/groups')
        self.assertEqual(1, len(response))
        self.assertEqual('KARAM group', response[0]['name'])
        self.assertEqual('Mahdi Karami', response[0]['root_user'])

    def test_create_and_get_groups_with_wrong_auth(self):
        response = self.client.post('/api/groups', json={
            'name': 'KARAM group',
            'root_user': 'Mahdi Karami',
            'description': 'The CEO of the KARAM group'
        }, headers={'Authorization': 'Bearer wrong_token'})
        self.assertEqual(422, response.status_code)

    def test_create_and_get_groups_with_other_auth(self):
        self.call_post('/api/groups', {
            'name': 'KARAM group',
            'root_user': 'Mahdi Karami',
            'description': 'The CEO of the KARAM group'
        })
        response = self.call_get('/api/groups', self.another_valid_user_system_token)
        self.assertEqual(0, len(response))

    def create_group(self):
        return self.call_post('/api/groups', {
            'name': 'KARAM group',
            'root_user': 'Mahdi Karami',
            'description': 'The CEO of the KARAM group'
        })

    def test_create_container_for_user_and_initial_task(self):
        socket = socketio.test_client(app, headers={'Authorization': f'Bearer {self.valid_user_system_token}'})
        self.create_group()
        user = self.db.query(User).first()
        check_and_create_containers()
        check_and_create_containers()
        containerService = ContainerService()
        container = containerService.find_container_by_user(user)
        chatService = ChatService()
        chatService.check_and_create_users_in_chat()
        chat_users = chatService.users()
        tasks = self.db.query(Task).all()
        self.assertEqual(1, len(chat_users))
        self.assertEqual(1, len(tasks))
        self.assertEqual('Mahdi Karami', chat_users[0]['name'])
        self.assertEqual('mahdi_karami', chat_users[0]['username'])
        self.assertEqual(TaskStatus.NEW, tasks[0].status)
        self.assertEqual('orchestrator_container_1', container.name)
        messages = socket.get_received()
        message1 = json.loads(messages[0]['args'])
        self.assertEqual('group_update', message1['type'])
        self.assertEqual('KARAM group', message1['data']['name'])
        self.assertIs(None, message1['data']['users'][0]['vnc_port'])
        message2 = json.loads(messages[1]['args'])
        self.assertEqual('group_update', message2['type'])
        self.assertEqual('KARAM group', message2['data']['name'])
        self.assertIsNot(None, message2['data']['users'][0]['vnc_port'])

    def position_call_side_effect(self, *args, **kwargs):
        message, image = args
        self.assertEqual(message, open('texts/position_request.txt').read())
        self.assertEqual(image, DataService().get_user_path_screenshot(1))
        return open('texts/position_response.txt').read()

    def vision_call_side_effect(self, *args, **kwargs):
        message, image = args
        self.assertEqual(message, open('texts/vision_request.txt').read())
        self.assertEqual(image, DataService().get_user_path_screenshot(1))
        return open('texts/vision_response.txt').read()

    def test_do_initial_task(self):
        self.create_group()
        check_and_create_containers()
        ChatService().check_and_create_users_in_chat()
        # position_model.call = MagicMock(side_effect=self.position_call_side_effect)
        # vision_model.call = MagicMock(side_effect=self.vision_call_side_effect)

        db = SessionLocal()
        service = ProcessorService(1, db.query(SystemUser).first())
        service.process_next_task()
        self.assertEqual('Tasks processed successfully', response['message'])

    def test_local_screen_data(self):
        self.create_group()
        check_and_create_containers()
        ChatService().check_and_create_users_in_chat()
        # position_model.call = MagicMock(side_effect=self.position_call_side_effect)
        # vision_model.call = MagicMock(side_effect=self.vision_call_side_effect)

        db = SessionLocal()
        service = ProcessorService(1, db.query(SystemUser).first())
        CommandService.screenshot(1)
        data = json.loads(service.getCoordinatesInLocal())
        data_dict = {str(i): lst for i, lst in enumerate(data)}
        screenshot_path = DataService().get_user_path_screenshot(1)
        data_dict['desktop'] = data_dict['0']
        service.draw_rectangles_on_image(screenshot_path, data_dict, screenshot_path.replace('.png', '_local.png'))

        # نمایش تصویر خروجی
        output_image = Image.open(screenshot_path.replace('.png', '_local.png'))
        output_image.show()

    def test_login_fail(self):
        payload = {
            "email": "test@testi.tst",
            "password": "testpass"
        }
        self.client.post("/signup", json=payload)

        payload = {
            "email": "test@testi.tst",
            "password": "failpass"
        }
        response = self.client.post("/login", json=payload)
        assert response.status_code == 401
        self.assertEqual(response.get_json().get('msg'), 'incorrect username or password')

    def t1est_something(self):
        user_id = 1
        # user = db.query(User).get(user_id)
        # subordinates = db.query(User).filter(User.parent_user_id == user_id).all()
        db = SessionLocal()
        service = ProcessorService(db, user_id)
        while True:
            service.process_next_task()
        # prompt = build_command_prompt(user, subordinates, tasks)
        # db.commit()

    def te1st_click_test(self):
        # CommandService.typing( "test", 1)
        CommandService.right_click(
            47.0,
            211.80555555555554
            , 1)
        # CommandService.click(32, 64, 1)

    def tes1t_send_send_keys_test(self):
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
    #     self.valid_user_system_token)
    def tes1t_g4f(self):
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
            - Immediately terminate the current task's action list.
            - Only output the new task (no other actions or tasks).
            - Do NOT add subsequent actions to the original task.
        10. When describing coordinates or locations of icons, windows, or any rectangular area, you must provide all four corner points (top-left, top-right, bottom-right, bottom-left) to precisely define the boundaries of the object. The format should be: [(x1,y1), (x2,y2), (x3,y3), (x4,y4)] where these points represent the four vertices of the rectangle in clockwise or counter-clockwise order.
        11. Please do not forget details in screen analysis like press enter or click
        12. Sometimes os got shut down and everything on it got disappeared. So maybe the windows got closed. Consider screenshot detail in .
        13. Sometimes you see command for enter in command line you should report all of it in the detail of windows

        Current task objective: Install telegram desktop on your system
        Priority: 0
        Recent actions: [{'role': 'assistant', 'content': '{"type": "function", "name": "type_text", "parameters": {"description": "To update the package list and install Telegram desktop.", "image_height": "700", "text": "sudo apt update && sudo apt install telegram-desktop", "image_width": "1400", "last_action_result": "The last action was to add a GPG key for the Google Chrome repository, which was successfully executed. This action is unrelated to the current task of installing Telegram desktop."}}'}, {'role': 'assistant', 'content': '{"type": "function", "name": "click", "parameters": {"last_action_result": "The last action was to add the GPG key for the Google Chrome repository and update the package list. The terminal is ready to execute the command to install Telegram Desktop.", "image_height": "495", "y": "325", "image_width": "958", "description": "Ensure the terminal window is focused", "x": "714"}}'}, {'role': 'assistant', 'content': '{"type": "function", "name": "send_key", "parameters": {"description": "Press Enter to execute the command `sudo apt update && sudo apt install telegram-desktop`.", "name": "Return", "image_width": "1295", "image_height": "680", "last_action_result": "The user is now ready to update the package list and install Telegram Desktop."}}'}, {'role': 'assistant', 'content': '{"type": "function", "name": "wait", "parameters": {"description": "Wait for the terminal to complete the connection and installation process", "image_height": "496", "image_width": "958", "seconds": "30", "last_action_result": "The last action was to add the GPG key and start the installation of Telegram Desktop. The terminal is currently connecting to the Ubuntu mirrors to fetch the package."}}'}, {'role': 'assistant', 'content': '{"type": "function", "name": "click", "parameters": {"y": "325", "image_height": "500", "description": "Focus on LXTerminal window", "last_action_result": "None", "x": "713", "image_width": "1000"}}'}, {'role': 'assistant', 'content': '{"type": "function", "name": "type_text", "parameters": {"description": "Type 'Y' and press Enter", "image_height": "1080", "text": "Y", "image_width": "1920", "last_action_result": "The last action was likely the execution of a command to install Telegram Desktop, which has prompted the user to confirm the installation by typing 'Y' or 'n'."}}'}, {'role': 'assistant', 'content': '{"type": "function", "name": "type_text", "parameters": {"text": "y", "image_width": "955", "last_action_result": "None", "description": "The terminal is awaiting user confirmation to proceed with the installation. Typing \"y\" will confirm the action and allow the installation to continue.", "image_height": "495"}}'}]
        '''
        response = G4FProvider('deepseek-v3').call({'role': 'user', 'content': testMessage})
        print(response)

    def tes1t_action_model_with_g4f(self):

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

    def tes1t_another_action_model(self):
        messages = json.loads("""
[{"role": "user", "content": "                    TASK CONTEXT:                    Context 1:[TASK 1]            Description: Install telegram desktop on your system            Status: TaskStatus.IN_PROGRESS            Priority: 0            Created at: None                                [MESSAGE 1]                    Type: assistant                    Timestamp: 2025-04-14 08:11:08.201798Context 2:Content: {\"type\": \"function\", \"name\": \"double_click\", \"parameters\": {\"x\": \"57\", \"image_width\": \"1700\", \"description\": \"To open the terminal and start the installation process of Telegram desktop.\", \"image_height\": \"500\", \"y\": \"237\", \"last_action_result\": \"The terminal is focused, but it is not currently open on the screen. The previous action was to type the installation command, but the terminal is not visible, indicating that it might not have been opened yet.\"}}Context 3:Timestamp: 2025-04-16 10:41:44.933286                    Content: {\"type\": \"function\", \"name\": \"double_click\", \"parameters\": {\"x\": \"60\", \"image_width\": \"1420\", \"description\": \"To open the terminal and start the installation process of Telegram desktop.\", \"last_action_result\": \"The terminal is focused, but it is not currently open on the screen. The Display Settings window is open, which is unrelated to the current task.\", \"image_height\": \"670\", \"y\": \"238\"}}                    CURRENT OBJECTIVE:                    Install telegram desktop on your system                    Analyze this screenshot and respond in this EXACT format:        Current Objective: Install telegram desktop on your system        Priority: 0        Screen Analysis:        - [Windows]: [[Describe first visible window and its exact state and it's coordinates], [Describe next visible window and its state and it's coordinates], ...]        - [Task UI Elements]: [[Describe first visible element relevant to the task and its state and it's coordinates], [Describe next visible element relevant to the task and its state and it's coordinates], ...]        - [Other UI Elements]: [[Describe first visible element and its state and it's coordinates], [Describe next visible element and its state and it's coordinates], ...]        - [Result of last action]: [What was the result of the last action base of picture and recent actions part]        - [width and height size]: [The last x and y of the window]        Task Status: [complete/not complete]        Reason: [Explain why task is complete or not]        Next Actions (list ALL required actions in order):        1. Action:        - Type: [click/double_click/right_click/type_text/send_key/wait/create_task/stop]        - Target: [Specific element or location]        - Detail: [Detail of type in text]        - Reason: [Why this action is needed]        - Pre-Actions: [List any preparation steps]        - Post-Actions: [List any follow-up steps]        - Thought: [Thought of this action]        2. Action:        - Type: [Next action type]        - Target: [Specific element or location]        - ...        Now analyze the screenshot and provide your response in the requested format.        Rules to follow:        1. ALWAYS maintain this exact format        2. List ALL required actions in proper sequence        3. Include ALL necessary pre/post actions        4. For terminal interactions:           - Ensure terminal is focused first           - Include appropriate wait times        5. For system commands:           - Check for errors in response        6. Be specific about targets and reasons        7. For open apps and files you should use double_click instead of click        8. For errors and something like them you can use create_task to create new task with higher priority.        9. If a task with higher priority is created via create_task:            - Immediately terminate the current task's action list.            - Only output the new task (no other actions or tasks).            - Do NOT add subsequent actions to the original task.        10. When describing coordinates or locations of icons, windows, or any rectangular area, you must provide all four corner points (top-left, top-right, bottom-right, bottom-left) to precisely define the boundaries of the object. The format should be: [(x1,y1), (x2,y2), (x3,y3), (x4,y4)] where these points represent the four vertices of the rectangle in clockwise or counter-clockwise order.        11. Please do not forget details in screen analysis like press enter or click        12. Sometimes os got shut down and everything on it got disappeared. So maybe the windows got closed. Consider screenshot detail in .        13. Sometimes you see command for enter in command line you should report all of it in the detail of windows        Current task objective: Install telegram desktop on your system        Priority: 0        Recent actions: [{'role': 'assistant', 'content': '{\"type\": \"function\", \"name\": \"type_text\", \"parameters\": {\"description\": \"To update the package list and install Telegram desktop.\", \"image_height\": \"700\", \"text\": \"sudo apt update && sudo apt install telegram-desktop\", \"image_width\": \"1400\", \"last_action_result\": \"The last action was to add a GPG key for the Google Chrome repository, which was successfully executed. This action is unrelated to the current task of installing Telegram desktop.\"}}'}, {'role': 'assistant', 'content': '{\"type\": \"function\", \"name\": \"click\", \"parameters\": {\"last_action_result\": \"The last action was to add the GPG key for the Google Chrome repository and update the package list. The terminal is ready to execute the command to install Telegram Desktop.\", \"image_height\": \"495\", \"y\": \"325\", \"image_width\": \"958\", \"description\": \"Ensure the terminal window is focused\", \"x\": \"714\"}}'}, {'role': 'assistant', 'content': '{\"type\": \"function\", \"name\": \"send_key\", \"parameters\": {\"description\": \"Press Enter to execute the command `sudo apt update && sudo apt install telegram-desktop`.\", \"name\": \"Return\", \"image_width\": \"1295\", \"image_height\": \"680\", \"last_action_result\": \"The user is now ready to update the package list and install Telegram Desktop.\"}}'}, {'role': 'assistant', 'content': '{\"type\": \"function\", \"name\": \"wait\", \"parameters\": {\"description\": \"Wait for the terminal to complete the connection and installation process\", \"image_height\": \"496\", \"image_width\": \"958\", \"seconds\": \"30\", \"last_action_result\": \"The last action was to add the GPG key and start the installation of Telegram Desktop. The terminal is currently connecting to the Ubuntu mirrors to fetch the package.\"}}'}, {'role': 'assistant', 'content': '{\"type\": \"function\", \"name\": \"click\", \"parameters\": {\"y\": \"325\", \"image_height\": \"500\", \"description\": \"Focus on LXTerminal window\", \"last_action_result\": \"None\", \"x\": \"713\", \"image_width\": \"1000\"}}'}, {'role': 'assistant', 'content': '{\"type\": \"function\", \"name\": \"type_text\", \"parameters\": {\"description\": \"Type \\'Y\\' and press Enter\", \"image_height\": \"1080\", \"text\": \"Y\", \"image_width\": \"1920\", \"last_action_result\": \"The last action was likely the execution of a command to install Telegram Desktop, which has prompted the user to confirm the installation by typing \\'Y\\' or \\'n\\'.\"}}'}, {'role': 'assistant', 'content': '{\"type\": \"function\", \"name\": \"type_text\", \"parameters\": {\"text\": \"y\", \"image_width\": \"955\", \"last_action_result\": \"None\", \"description\": \"The terminal is awaiting user confirmation to proceed with the installation. Typing \\\\\"y\\\\\" will confirm the action and allow the installation to continue.\", \"image_height\": \"495\"}}'}]                "}]
        """)
        images = [[open(
            DataService().get_user_path_screenshot(1),
            "rb"), "screenshot.png"]]
        response = g4f.Client(Blackbox).chat.completions.create(messages, 'blackboxai', images=images).choices[
            0].message.content

    def t1est_qwen(self):
        provider = QwenProvider()
        data = provider.call(
            'What is size of this image? And give all the windows and icons and buttons coordinates of that you can see',
            DataService().get_user_path_screenshot(1))
        print(data)

    def te1st_g4f1(self):
        provider = HuggingFaceProvider('')
        jsonResult = provider.call(
            '''give ALL (maximum 50, at least 20) the windows and icons and buttons and textFields and links and ALL gui coordinates detail components and picture size coordinates and it's type of that you can see EXACTLY in this FORMAT 
            {"desktop": [Xmin, Ymin, Xmax, Ymax], "terminal_window": [Xmin, Ymin, Xmax, Ymax], "terminal_icon": [Xmin, Ymin, Xmax, Ymax], ...}
            DO NOT ADD OTHER WORDS JUST FOLLOW THE FORMAT 
            This is so IMPORTANT and CRITICAL to tell ALL coordinates gui details. DO NOT MISS any of them.
            Add the type of that thing in the end of the key of that thing like example like 'username_textfield'
            ''', DataService().get_user_path_screenshot(1))
        picture_file_path = DataService().get_user_path_screenshot(1)
        if jsonResult[0:3] == '```':
            self.drawPictureLines(picture_file_path, json.loads(jsonResult[7:-3]))
        else:
            self.drawPictureLines(picture_file_path, json.loads(jsonResult))

    def t1est_qwen2(self):
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

    def drawPictureLines(self, picture_file_path, jsonResult, model_name, x_resize=1932, y_resize=924):
        try:
            # Load the image
            image = Image.open(picture_file_path)

            # Resize the image to 1612x768
            new_size = (x_resize, y_resize)
            image = image.resize(new_size)

            # Create a draw object
            draw = ImageDraw.Draw(image)

            # Define drawing parameters
            rect_color = (255, 0, 0)  # Red color
            text_color = (255, 0, 0)  # Red color
            line_width = 2
            font = ImageFont.load_default()  # Use default font

            # Iterate over each item in jsonResult
            for item, coordinates in jsonResult.items():
                try:
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
                except Exception as e:
                    pass

            # Save the modified image
            output_path = picture_file_path.replace('.png', model_name + '.png')
            image.save(output_path)
            print(f"Annotated image saved to: {output_path}")

        except Exception as e:
            print(f"Error processing image: {str(e)}")

    def t1est_rag(self):
        rag1 = RAGSystem()
        rag = RAGSystem()
        rag1.add_text('The test is important')
        result = rag.query_context('give all the things that you know?')
        print(result)

    def test_lm_arena(self):
        model_name = 'qwen2.5-vl-72b-instruct'
        images = [[open(
            "/home/mehdi/all/repositories/github.com/orchestrator/orchestrator_data/local/2/screenshot.png",
            "rb"), "screenshot.png"]]
        response = g4f.Client(NewHarProvider).chat.completions.create(
            '''You are an expert UI analyzer. Your ONLY job is to output a single, valid JSON object that lists the exact bounding box coordinates of EVERY visible UI element on the screen. 

**STRICT RULES:**
- Output MUST be a single JSON object, not repeated, not nested, not wrapped in any text or explanation.
- The JSON must start with '{' and end with '}' and appear only ONCE.
- If you output more than one JSON object, your answer is INVALID.
- If you add any text, explanation, or comment, your answer is INVALID.
- If you miss any visible element (icon, window, toolbar, statusbar, etc.), your answer is INVALID.
- If you use relative coordinates, percentages, or non-integer values, your answer is INVALID.
- If you use the same coordinates for different elements, your answer is INVALID.
- If you use the wrong element type or name, your answer is INVALID.

**Naming convention:**
- Use descriptive names and always add the type at the end (e.g., _icon, _window, _toolbar, _statusbar).
- Example: "firefox_icon", "lxterminal_icon", "taskbar_toolbar", "time_statusbar", "firefox_window"

**Coordinates:**
- Format: [Xmin, Ymin, Xmax, Ymax] (top-left and bottom-right pixel, absolute integer values)
- Example: "firefox_icon": [20, 80, 60, 120]

**POSITIVE EXAMPLE (valid):**
{
    "desktop": [0, 0, 1365, 767],
    "firefox_icon": [20, 80, 60, 120],
    "lxterminal_icon": [20, 140, 60, 180],
    "orchestra_icon": [20, 20, 60, 60],
    "taskbar_toolbar": [0, 740, 1365, 767],
    "time_statusbar": [950, 740, 1000, 767]
}

**NEGATIVE EXAMPLES (invalid):**
- More than one JSON object
- Any explanation or comment
- Missing any visible element
- Wrong or duplicate coordinates
- Wrong format

**REMEMBER:**  
Output ONLY a single, valid JSON object with ALL visible UI elements and their exact bounding box coordinates. NOTHING ELSE.
            ''', model_name, images=images).choices[0].message.content  ## That was not great
        self.drawPictureLines(DataService().get_user_path_screenshot(1), response, model_name)
        print(response)

    def tes1t_screenshot_path(self):
        service = DataService()
        path = service.get_user_path_screenshot(1)
        self.assertEqual('/tmp/orchestrator/1/screenshot.png', path)

    def test_desktop_element_detector(self):
        detector = DesktopElementDetector()
        image_path = '/home/mehdi/all/repositories/github.com/orchestrator/orchestrator_data/local/2/screenshot.png'
        result = detector.detect_elements(image_path, conf_threshold=0.005)
        print(result)
        # حالا annotate کنیم:
        from PIL import ImageDraw, ImageFont
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)
        for name, coords in result.items():
            x1, y1, x2, y2 = coords
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            draw.text((x1, y1 - 10), name, fill="red")
        img.save(image_path.replace('.png', '_detected.png'))

    def tes1t_create_container(self):
        client = docker.from_env()
        container = client.containers.run(
            image="karam_orchestrator:latest",
            command="sleep infinity",
            detach=True,
            network="orchestrator_default",
            name=f"orchestrator_container_3",
            ports={
                "80/tcp": 12345,
                "5900/tcp": 54321
            },
            # environment={
            #     "RESOLUTION": "1920x1080"
            # }
        )
        print("test")

    def test_lm_arena2(self):

        images = [[open(
            "/home/mehdi/all/repositories/github.com/orchestrator/orchestrator_data/local/2/screenshot.png",
            "rb"), "screenshot.png"]]
        positionTextRequest = '''
Analyze the screenshot with extreme precision and attention to detail. Focus on identifying ALL UI elements and their exact coordinates.

        CRITICAL FIRST STEP - SCREEN DIMENSIONS:
        The FIRST and MOST IMPORTANT element to detect is the desktop with EXACT screen dimensions.
        You MUST start with "desktop" as the first key in your response.
        The desktop coordinates MUST represent the ENTIRE screen dimensions.
        Example: If screen is 1920x1080, desktop should be: "desktop": [0, 0, 1920, 1080]

        Requirements for coordinate detection:
        1. ALWAYS start with desktop dimensions as the first element
        2. Measure coordinates with pixel-level accuracy
        3. Include ALL visible UI elements
        4. Consider element hierarchy and relationships
        5. Account for element borders and padding
        6. Verify element visibility and accessibility

        Provide coordinates in this EXACT format (IMPORTANT: Each element must be a direct key-value pair, not nested in arrays):
        {
            "desktop": [Xmin, Ymin, Xmax, Ymax], // MUST be first and MUST represent entire screen
            "a_window": [Xmin, Ymin, Xmax, Ymax], // IF EXIST
            "b_icon": [Xmin, Ymin, Xmax, Ymax], // IF EXIST
            "c_button": [Xmin, Ymin, Xmax, Ymax], // IF EXIST
            "d_textfield": [Xmin, Ymin, Xmax, Ymax], // IF EXIST
            "login_button": [Xmin, Ymin, Xmax, Ymax], // IF EXIST
        }

        Element naming rules:
        1. Use descriptive names for elements
        2. Add type suffix to each element name
        3. Use underscores to separate words
        4. DO NOT nest elements in arrays
        5. Each element must be a direct key in the JSON object

        Element type suffixes (add to element name):
        - _window: For application windows
        - _button: For clickable buttons
        - _textfield: For input fields
        - _icon: For application icons
        - _link: For clickable links
        - _menu: For menu items
        - _label: For text labels
        - _checkbox: For checkboxes
        - _dropdown: For dropdown menus
        - _scrollbar: For scrollbars
        - _tab: For tab elements
        - _toolbar: For toolbar elements

        CRITICAL RULES:
        1. ALWAYS start with desktop dimensions as the first element
        2. Desktop MUST represent the entire screen dimensions
        3. DO NOT MISS any visible UI elements
        4. Ensure coordinates are pixel-perfect
        5. Include element type in the key name
        6. NO additional text or comments
        7. Verify element boundaries are complete
        8. Consider overlapping elements
        9. Account for system UI elements
        10. DO NOT use nested arrays or objects
        11. Each element must be a direct key in the JSON object
        12. Use consistent naming convention
        '''
        images = [[open(
            "/home/mehdi/all/repositories/github.com/orchestrator/orchestrator_data/local/2/screenshot.png",
            "rb"), "screenshot.png"]]
        response = g4f.Client(HarProvider).chat.completions.create(
            positionTextRequest, 'qwen2.5-vl-72b-instruct', images=images).choices[
            0].message.content  ## That was not grea
        # t
        self.drawPictureLines(DataService().get_user_path_screenshot(1), response, model_name)
        print(response)

    def test_process_position_request(self):
        self.create_group()
        user = self.db.query(User).first()
        check_and_create_containers()
        time.sleep(3)
        CommandService.screenshot(user.id)
        service = ProcessorService(user.id, self.db.query(SystemUser).first())
        screenshot_path = DataService().get_user_path_screenshot(user.id)
        output_path = screenshot_path.replace('.png', '_processed.png')

        result = service.process_position_request(user.id, screenshot_path, output_path)

        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)
        self.assertIn('desktop', result)

        # بررسی وجود فایل خروجی
        self.assertTrue(os.path.exists(output_path))

        # بررسی ابعاد دسکتاپ
        desktop_coords = result['desktop']
        self.assertEqual(len(desktop_coords), 4)
        self.assertTrue(all(isinstance(x, (int, float)) for x in desktop_coords))

        # بررسی وجود حداقل یک المان تعاملی
        interactive_elements = [k for k in result.keys() if
                                any(k.endswith(suffix) for suffix in ['_button', '_textfield', '_icon', '_link'])]
        self.assertTrue(len(interactive_elements) > 0)

        # بررسی وجود موقعیت موس
        self.assertIn('mouse_cursor', result)

        # بررسی فرمت مختصات موس
        mouse_coords = result['mouse_cursor']
        self.assertEqual(len(mouse_coords), 4)
        self.assertTrue(all(isinstance(x, (int, float)) for x in mouse_coords))

    def test_position(self):
        positionTextRequest = '''Analyze the screenshot with extreme precision and attention to detail. You will be given a list of detected UI elements with their coordinates from OpenCV. Your task is to:

        1. First, analyze and map these detected elements to meaningful UI components
        2. Then, identify ALL visible elements in order of importance:
           - First: Interactive elements (buttons, textfields, etc.)
           - Second: Application windows
           - Third: Desktop icons
           - Fourth: Other visible UI elements
        3. Finally, provide a clean, organized output of these elements

        CRITICAL FIRST STEP - SCREEN DIMENSIONS:
        The FIRST and MOST IMPORTANT element to detect is the desktop with EXACT screen dimensions.
        You MUST start with "desktop" as the first key in your response.
        The desktop coordinates MUST represent the ENTIRE screen dimensions.
        Example: If screen is 1920x1080, desktop should be: "desktop": [0, 0, 1920, 1080]

        ELEMENT MAPPING RULES:
        1. Analyze the provided OpenCV detected elements
        2. Group related elements into meaningful UI components
        3. Report ALL visible elements in this order:
           a. Interactive elements:
              - Buttons
              - Text fields
              - Links
              - Menus
              - Checkboxes
              - Dropdowns
           b. Application windows:
              - Main windows
              - Dialog boxes
              - Popup windows
           c. Desktop icons:
              - Application icons
              - File icons
              - Folder icons
           d. Other UI elements:
              - Labels
              - Images
              - Toolbars
              - Status bars
        4. Only ignore elements that are:
           - Completely hidden
           - Too small to be meaningful (less than 10x10 pixels)
           - Duplicate or overlapping (keep the most important one)
        5. For mouse cursor:
           - Report as "mouse_cursor" with exact coordinates
           - Use a small bounding box (e.g., 20x20 pixels)
           - Place it at the end of the list
           - DO NOT confuse it with other UI elements

        Requirements for coordinate detection:
        1. ALWAYS start with desktop dimensions as the first element
        2. Use the most accurate coordinates from the OpenCV detection
        3. Ensure coordinates are pixel-perfect
        4. Consider element hierarchy and relationships
        5. Account for element borders and padding
        6. Verify element visibility and accessibility
        7. For mouse cursor, use exact center point

        Provide coordinates in this EXACT format (IMPORTANT: Each element must be a direct key-value pair, not nested in arrays):
        {
            "desktop": [Xmin, Ymin, Xmax, Ymax], // MUST be first and MUST represent entire screen
            "interactive_element_1": [Xmin, Ymin, Xmax, Ymax], // Interactive elements first
            "interactive_element_2": [Xmin, Ymin, Xmax, Ymax],
            "main_window": [Xmin, Ymin, Xmax, Ymax], // Then windows
            "desktop_icon_1": [Xmin, Ymin, Xmax, Ymax], // Then desktop icons
            "desktop_icon_2": [Xmin, Ymin, Xmax, Ymax],
            "other_element_1": [Xmin, Ymin, Xmax, Ymax], // Then other elements
            "other_element_2": [Xmin, Ymin, Xmax, Ymax],
            "mouse_cursor": [Xmin, Ymin, Xmax, Ymax] // Mouse cursor last
        }

        Element naming rules:
        1. Use descriptive names that reflect the element's purpose
        2. Add type suffix to each element name
        3. Use underscores to separate words
        4. DO NOT nest elements in arrays
        5. Each element must be a direct key in the JSON object
        6. For desktop icons, use "desktop_icon_X" format
        7. For windows, use the window name with "_window" suffix
        8. For mouse cursor, always use "mouse_cursor"

        Element type suffixes (add to element name):
        - _window: For application windows
        - _button: For clickable buttons
        - _textfield: For input fields
        - _icon: For application icons
        - _link: For clickable links
        - _menu: For menu items
        - _label: For text labels
        - _checkbox: For checkboxes
        - _dropdown: For dropdown menus
        - _scrollbar: For scrollbars
        - _tab: For tab elements
        - _toolbar: For toolbar elements
        - _cursor: For mouse cursor

        CRITICAL RULES:
        1. ALWAYS start with desktop dimensions as the first element
        2. Desktop MUST represent the entire screen dimensions
        3. Report ALL visible elements
        4. Prioritize interactive elements
        5. Ensure coordinates are pixel-perfect
        6. Include element type in the key name
        7. NO additional text or comments
        8. Verify element boundaries are complete
        9. Consider overlapping elements
        10. Account for system UI elements
        11. DO NOT use nested arrays or objects
        12. Each element must be a direct key in the JSON object
        13. Use consistent naming convention
        14. DO NOT include elements from memory or previous screenshots
        15. Report elements in order of importance
        16. Include ALL desktop icons
        17. Remove only completely hidden or too small elements
        18. ALWAYS report mouse cursor position
        19. DO NOT confuse mouse cursor with other elements
        20. Verify element types before reporting
        21. Just use numbers for coordinates. DO NOT write them with X or Y
        22. JUST write coordinates that you can see
'''
        model_name = 'qwen2.5-vl-72b-instruct'
        images = [[open(
            "/home/mehdi/all/repositories/github.com/orchestrator/orchestrator_data/local/2/screenshot.png",
            "rb"), "screenshot.png"]]

        g4f.Client(PollinationsAI).chat.completions.create(positionTextRequest, model_name, images=images)

    def draw_scaled_rectangles_on_image(self, image_path, bounding_box_data, output_path="output_image_scaled.png"):
        """
        این تابع یک تصویر را بارگذاری کرده، مختصات را بر اساس ابعاد 'desktop'
        مقیاس‌بندی می‌کند و مستطیل‌هایی بر اساس مختصات مقیاس‌بندی شده روی آن رسم می‌کند.

        :param image_path: مسیر فایل تصویر ورودی.
        :param bounding_box_data: دیکشنری حاوی نام ناحیه و مختصات [x_min, y_min, x_max, y_max].
                                  باید شامل کلید 'desktop' با ابعاد مرجع باشد.
        :param output_path: مسیر ذخیره تصویر خروجی با مستطیل‌های رسم شده.
        """
        try:
            # باز کردن تصویر
            img = Image.open(image_path)
            actual_width, actual_height = img.size
            draw = ImageDraw.Draw(img)

            # استخراج ابعاد مرجع از کلید 'desktop'
            if "desktop" not in bounding_box_data:
                print("خطا: کلید 'desktop' با ابعاد مرجع در داده‌های JSON یافت نشد.")
                return

            desktop_coords = bounding_box_data["desktop"]
            if len(desktop_coords) != 4:
                print(f"خطا: مختصات 'desktop' نامعتبر است: {desktop_coords}")
                return

            ref_x_min, ref_y_min, ref_x_max, ref_y_max = desktop_coords
            ref_width = ref_x_max - ref_x_min
            ref_height = ref_y_max - ref_y_min

            if ref_width == 0 or ref_height == 0:
                print("خطا: عرض یا ارتفاع مرجع 'desktop' صفر است. امکان مقیاس‌بندی وجود ندارد.")
                return

            # محاسبه ضرایب مقیاس
            scale_x = actual_width / ref_width
            scale_y = actual_height / ref_height

            print(f"ابعاد عکس واقعی: عرض={actual_width}, ارتفاع={actual_height}")
            print(f"ابعاد مرجع (desktop): عرض={ref_width}, ارتفاع={ref_height}")
            print(f"ضریب مقیاس: scale_x={scale_x}, scale_y={scale_y}")

            # تکرار روی داده‌های مختصات
            for label, coords in bounding_box_data.items():
                if label == "desktop":  # نیازی به رسم خود 'desktop' نیست یا می‌توان آن را متفاوت رسم کرد
                    # می‌توانید مستطیل مربوط به دسکتاپ را هم رسم کنید اگر مایلید
                    # draw.rectangle([(0,0), (actual_width-1, actual_height-1)], outline="blue", width=1)
                    continue

                if len(coords) == 4:
                    x_min, y_min, x_max, y_max = coords

                    # مقیاس‌بندی مختصات
                    # مختصات در JSON نسبت به گوشه (ref_x_min, ref_y_min) دسکتاپ مرجع هستند
                    # ابتدا آنها را نسبت به (0,0) دسکتاپ مرجع نرمالایز می‌کنیم
                    # سپس مقیاس‌بندی کرده و به مختصات واقعی تصویر منتقل می‌کنیم

                    scaled_x_min = (x_min - ref_x_min) * scale_x
                    scaled_y_min = (y_min - ref_y_min) * scale_y
                    scaled_x_max = (x_max - ref_x_min) * scale_x
                    scaled_y_max = (y_max - ref_y_min) * scale_y

                    # رسم مستطیل قرمز با ضخامت 2 پیکسل
                    draw.rectangle(
                        [(scaled_x_min, scaled_y_min), (scaled_x_max, scaled_y_max)],
                        outline="red",
                        width=2
                    )
                    # می‌توانید نام هر ناحیه را هم کنار مستطیل بنویسید (اختیاری)
                    # draw.text((scaled_x_min, scaled_y_min - 10), label, fill="red")
                else:
                    print(f"مختصات نامعتبر برای '{label}': {coords}")

            # نمایش تصویر
            img.show()
            # یا ذخیره تصویر
            img.save(output_path)
            print(f"تصویر با موفقیت در مسیر '{output_path}' ذخیره شد.")

        except FileNotFoundError:
            print(f"خطا: فایل تصویر در مسیر '{image_path}' پیدا نشد.")
        except Exception as e:
            print(f"خطایی رخ داد: {e}")

    def test_gemn_result(self):
        a = '''{
  "desktop": [0, 0, 1366, 768],
  "Firefox_back_button": [511, 59, 530, 80],
  "Firefox_forward_button": [536, 59, 555, 80],
  "Firefox_reload_button": [560, 59, 579, 80],
  "Firefox_address_bar_textfield": [586, 58, 1156, 81],
  "Firefox_menu_button": [1332, 59, 1355, 80],
  "Firefox_new_tab_button": [718, 33, 740, 54],
  "Google Search_bar_tab_close_button": [698, 37, 711, 50],
  "Google Search_bar_tab": [534, 33, 714, 55],
  "Firefox_minimize_button": [1277, 33, 1300, 51],
  "Firefox_maximize_button": [1303, 33, 1326, 51],
  "Firefox_close_window_button": [1330, 33, 1355, 51],
  "Google Search_input_textfield": [553, 132, 826, 167],
  "Google Search_action_button": [829, 133, 858, 166],
  "Google_voice_search_button": [862, 134, 887, 165],
  "Google_clear_search_button": [890, 134, 913, 165],
  "Google_settings_button": [1207, 90, 1229, 110],
  "Google_apps_button": [1240, 90, 1264, 110],
  "Google_signin_button": [1277, 88, 1338, 113],
  "Google_all_link": [554, 187, 588, 212],
  "Google_images_link": [603, 187, 662, 212],
  "Google_videos_link": [678, 187, 734, 212],
  "Google_shopping_link": [749, 187, 820, 212],
  "Google_news_link": [835, 187, 886, 212],
  "Google_books_link": [900, 187, 955, 212],
  "Google_more_link": [970, 187, 1020, 212],
  "Speedtest_by_Ookla_link": [552, 260, 710, 280],
  "Cambridge_Dictionary_link": [552, 342, 860, 363],
  "Dictionary_link": [552, 430, 675, 450],
  "Test1_definition_link": [552, 517, 598, 538],
  "Firefox_vertical_scrollbar_thumb_button": [1349, 87, 1362, 186],
  "File_Manager_back_button": [131, 457, 153, 479],
  "File_Manager_forward_button": [157, 457, 179, 479],
  "File_Manager_up_button": [183, 457, 205, 479],
  "File_Manager_refresh_button": [209, 457, 231, 479],
  "File_Manager_home_button": [235, 457, 257, 479],
  "File_Manager_location_bar_textfield": [262, 457, 662, 480],
  "File_Manager_search_button": [667, 457, 690, 480],
  "File_Manager_file_menu": [132, 435, 160, 452],
  "File_Manager_edit_menu": [165, 435, 198, 452],
  "File_Manager_view_menu": [203, 435, 238, 452],
  "File_Manager_bookmarks_menu": [243, 435, 305, 452],
  "File_Manager_go_menu": [310, 435, 337, 452],
  "File_Manager_tools_menu": [342, 435, 380, 452],
  "File_Manager_help_menu": [385, 435, 420, 452],
  "File_Manager_places_home_folder_link": [136, 516, 230, 532],
  "File_Manager_places_desktop_link": [136, 535, 230, 551],
  "File_Manager_places_applications_link": [136, 554, 230, 570],
  "File_Manager_data_folder_button": [249, 499, 296, 545],
  "File_Manager_app_py_file_button": [249, 550, 308, 596],
  "File_Manager_minimize_button": [622, 435, 642, 452],
  "File_Manager_maximize_button": [647, 435, 667, 452],
  "File_Manager_close_window_button": [673, 435, 693, 452],
  "Terminal_minimize_button": [622, 34, 642, 50],
  "Terminal_maximize_button": [647, 34, 667, 50],
  "Terminal_close_window_button": [673, 34, 693, 50],
  "Terminal_content_area_textfield": [71, 54, 697, 423],
  "Start_menu_button": [0, 748, 25, 767],
  "Taskbar_Firefox_icon_button": [32, 748, 57, 767],
  "Taskbar_File_Manager_icon_button": [61, 748, 86, 767],
  "Taskbar_LXTerminal_icon_button": [90, 748, 115, 767],
  "Taskbar_Google Search_tab_button": [119, 748, 287, 767],
  "Desktop_pager_1_button": [926, 748, 940, 767],
  "Desktop_pager_2_button": [941, 748, 955, 767],
  "Network_status_icon_button": [1255, 748, 1275, 767],
  "Volume_icon_button": [1279, 748, 1299, 767],
  "Time_date_button": [1328, 748, 1365, 767],
  "Firefox_main_window": [503, 29, 1365, 747],
  "File_Manager_main_window": [126, 432, 702, 747],
  "Terminal_main_window": [67, 30, 702, 427],
  "desktop_icon_1": [8, 75, 58, 126],
  "desktop_icon_2": [8, 136, 58, 187],
  "desktop_icon_3": [8, 197, 58, 248],
  "Firefox_window_title_label": [512, 33, 770, 51],
  "File_Manager_window_title_label": [133, 435, 230, 451],
  "Terminal_window_title_label": [73, 34, 260, 49],
  "Firefox_active_tab_label": [538, 36, 693, 51],
  "Google_logo_image": [620, 80, 724, 115],
  "Speedtest_by_Ookla_url_label": [553, 281, 660, 294],
  "Speedtest_by_Ookla_description_label": [553, 298, 1040, 332],
  "Cambridge_Dictionary_url_label": [553, 364, 750, 377],
  "Cambridge_Dictionary_description_label": [553, 380, 1040, 410],
  "Dictionary_url_label": [553, 451, 690, 464],
  "Dictionary_description_label": [553, 467, 1040, 497],
  "Test1_word_type_label": [553, 539, 585, 552],
  "Test1_definition_text_label": [553, 556, 1040, 600],
  "File_manager_status_bar_text_label": [130, 729, 485, 744],
  "Firefox_vertical_scrollbar": [1348, 86, 1363, 745],
  "Taskbar_panel": [0, 747, 1366, 768],
  "Firefox_tab_bar_toolbar": [503, 29, 1365, 55],
  "Firefox_navigation_toolbar": [503, 56, 1365, 85],
  "File_Manager_menu_toolbar": [126, 432, 702, 454],
  "File_Manager_navigation_toolbar": [126, 455, 702, 482],
  "File_Manager_sidebar_panel": [126, 483, 240, 725],
  "File_Manager_status_bar": [126, 726, 702, 747],
  "mouse_cursor": [724, 403, 744, 423]
}'''
        bounding_boxes = json.loads(a)

        input_image_path = "/home/mehdi/all/repositories/github.com/orchestrator/orchestrator_data/local/2/test.png"

        output_image_path = "/home/mehdi/all/repositories/github.com/orchestrator/orchestrator_data/local/2/image_with_rectangles.png"

        # فراخوانی تابع برای رسم مستطیل‌ها
        self.draw_scaled_rectangles_on_image(input_image_path, bounding_boxes, output_image_path)

    def draw_yolo_boxes(self, image_path, label_path, output_path=None):
        """
        رسم مستطیل‌های YOLO روی تصویر

        پارامترها:
            image_path (str): مسیر تصویر ورودی
            label_path (str): مسیر فایل متنی حاوی برچسب‌های YOLO
            output_path (str): مسیر ذخیره تصویر خروجی (اختیاری)
        """
        # خواندن تصویر
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"تصویر در مسیر {image_path} یافت نشد")

        height, width = image.shape[:2]

        # خواندن فایل برچسب‌ها
        with open(label_path, 'r') as file:
            lines = file.readlines()

        # رنگ‌ها (به فرمت BGR)
        colors = {
            'button': (0, 255, 0),  # سبز
            'clock': (255, 0, 0),  # آبی
            'selected_app_minimize_button': (0, 0, 255),  # قرمز
            # می‌توانید رنگ‌های بیشتری اضافه کنید
        }

        default_color = (255, 255, 0)  # زرد برای کلاس‌های تعریف نشده

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            class_name = parts[0]
            x_center, y_center, box_width, box_height = map(float, parts[1:5])

            # تبدیل مختصات YOLO به مختصات OpenCV
            x_center *= width
            y_center *= height
            box_width *= width
            box_height *= height

            x_min = int(x_center - box_width / 2)
            y_min = int(y_center - box_height / 2)
            x_max = int(x_center + box_width / 2)
            y_max = int(y_center + box_height / 2)

            # انتخاب رنگ بر اساس کلاس
            color = colors.get(class_name, default_color)

            # رسم مستطیل
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)

            # نمایش نام کلاس
            cv2.putText(image, class_name, (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # نمایش یا ذخیره تصویر
        if output_path:
            cv2.imwrite(output_path, image)
            print(f"تصویر با مستطیل‌ها در {output_path} ذخیره شد")
        else:
            cv2.imshow('Image with Bounding Boxes', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def test_yolo(self):
        image_path = "/home/mehdi/all/repositories/github.com/orchestrator/orchestrator/services/yolo/my_dataset_raw/image1.png"  # مسیر تصویر خود را وارد کنید
        label_path = "/home/mehdi/all/repositories/github.com/orchestrator/orchestrator/services/yolo/my_dataset_raw/image1.txt"  # مسیر فایل برچسب‌ها را وارد کنید
        output_path = "/home/mehdi/all/repositories/github.com/orchestrator/orchestrator/services/yolo/my_dataset_raw/image1_out.png"  # اختیاری

        self.draw_yolo_boxes(image_path, label_path, output_path)

    # def test_divar(self):
    #     model_name = ''
    #
    #     response = g4f.Client(PollinationsAI).chat.completions.create('', '').choices[0].message.content
    #     print("test")
    def test_divar(self):
        instance = DivarContest('tpsg-JHWxxG3qsmc4t7tAtv6AfGQOfIOKrjw')
        # print(instance.capture_the_flag('do what image says at { https://divar-contest.darkube.app/fyvkr93-public.png }'))
        # print(instance.capture_the_flag('find the first laptop in this url: https://divar-contest.darkube.app/divar_sample.html and return the price in numbers.'))
        print(instance.capture_the_flag(".tfel nruter tsuj t'nod tub rewsna eht sa \"tfel\" drow eht fo etisoppo eht etirw ,ecnetnes siht dnatsrednu uoy fI"))
        # print(instance.capture_the_flag("Write a Python script that initializes a shared counter to zero. Then, create and run a number of concurrent tasks (10 for threads and 10 for processes) that each increment this shared counter a significant number of times (100,000 times per task). Finally, print the final value of the counter after all tasks have completed."))
        # print(instance.capture_the_flag("task_id 3 - I'm making a shopping list for my dad, but he's a professor at Sharif and he's a real stickler when it comes to Buy things. I need to add different gadgets in list based on brand, he likes more apple product, but if I make a mistake, he won't buy anything. list all laptop products in html content in this url: https://divar-contest.darkube.app/divar_sample.html and sum price of them, and return the sum price in numbers"))
        # print(instance.capture_the_flag(
        #     "At UOD (University of Divar), your professor has assigned you a task, You are given a small codebase located in `{https://divar-contest.darkube.app/public-code-bug-fix.zip}` and a logs.txt file. Analyze the logs, identify and fix the bug in the codebase, then run the program (main.py) and return its output as the flag."))
        # print(instance.capture_the_flag("find wikipedia link in html content in this url: https://divar-contest.darkube.app/divar_sample.html and based on information on that link return how many users open app annually according to the new annual report of Divar, do not round the number, return only the number without any dots or commas"))

    if __name__ == '__main__':
        unittest.main()
