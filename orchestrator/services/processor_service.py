import os
import tempfile
import cv2
import numpy as np
import pytesseract
import re

from PIL import Image, ImageDraw
from sqlalchemy import or_
from copy import deepcopy
from PIL import Image
import json
from datetime import datetime
import time
from typing import Optional
import traceback

from sympy.codegen.ast import continue_

from orchestrator.models import SystemUser, Message
from orchestrator.models.WebSocketType import WebSocketType
from orchestrator.models.base import SessionLocal
from orchestrator.models.task import TaskStatus, Task, TaskMessage
from orchestrator.services.command_service import CommandService
from orchestrator.services.config_service import grounding_model, vision_model, action_model, position_model
from orchestrator.services.container_service import ContainerService
from orchestrator.services.data_service import DataService
from orchestrator.services.element_memory_service import get_element_memory_service
from orchestrator.models.memory import Memory
from orchestrator.services.group_service import GroupService
from orchestrator.services.log_service import logService
from orchestrator.services.rag_service import RAGSystem
from orchestrator.services.shared_learning_service import SharedLearningService
from orchestrator.models.user import User
from orchestrator.services.user_service import create_user
from orchestrator.services.websocket_service import websocket_manager, WebSocketManager

TYPING_DELAY_MS = 12
TYPING_GROUP_SIZE = 5
# MODEL_WIDTH = 1420.0
# MODEL_HEIGHT = 650.0
# OS_WIDTH = 1920.0
# OS_HEIGHT = 915.0

tools = {
    "complete_task": {
        "description": "Mark the current task as completed",
        "parameters": {
            "description": "Reason for completion",
            "last_action_result": "What was the result of the last action",
            "image_width": "The screenshot width size",
            "image_height": "The screenshot height size",
        }
    },
    # "stop": {"description": "Signal task completion", "parameters: {}},
    "wait": {
        "description": "Pause execution",
        "parameters": {
            "seconds": "Wait duration", "description": "Reason",
        }
    },
    "single_click": {
        "description": "Click UI element one time",
        "parameters": {
            "x": "X", "y": "Y",
        }
    },
    "double_click": {
        "description": "Double-click UI element",
        "parameters": {
            "x": "X", "y": "Y",
        }
    },
    "move_mouse": {
        "description": "Move mouse",
        "parameters": {
            "x": "X", "y": "Y",
        }
    },
    "right_click": {
        "description": "Right-click UI element",
        "parameters": {
            "x": "X", "y": "Y",
        }
    },
    "type_text": {
        "description": "Input text",
        "parameters": {
            "text": "Text to type",
        }
    },
    "scroll_down": {
        "description": "Scroll down the page or window that current mouse is in there",
        "parameters": {
            "repeat": "The times to scroll",
        }
    },
    "scroll_up": {
        "description": "Scroll up the page or window that current mouse is in there",
        "parameters": {
            "repeat": "The times to scroll",
        }
    },
    "send_key": {
        "description": "Send keyboard key(s). Can send single keys or key combinations (like Ctrl+L). For combinations, pass multiple keys.",
        "parameters": {
            "keys": "List of keys to press. Single key: ['Enter'] or ['Escape']. Combinations: ['Control_L', 'l'] for Ctrl+L, ['Alt_L', 'Tab'] for Alt+Tab, ['Shift_L', 'a'] for Shift+A. Common keys: Enter, Escape, Tab, BackSpace, Delete, Return. Modifiers: Control_L, Alt_L, Shift_L, Super_L",
        }
    },
    "create_task": {
        "description": "Create new task",
        "parameters": {
            "task": "Task description",
            "priority": "Priority level",
        }
    },
    "create_employee": {
        "description": "Create new employee",
        "parameters": {
            "name": "Full name of the user",
            "employee_description": "Detail of the employee in full detail to create by these description"
        }
    },
    "read_and_respond_messages": {
        "description": "Read manager messages from RocketChat and respond appropriately",
        "parameters": {
            "description": "Reason",
        }
    }
}


class ProcessorService:
    def __init__(self, user_id: int, system_user: SystemUser):
        self.db = SessionLocal()  # HACK: need to get clear
        user = self.db.query(User).get(user_id)
        if user is None:
            raise PermissionError()  # HACK: change to not found
        if user.group.system_user_id != system_user.id:
            raise PermissionError()  # HACK: Get this part better
        self.system_user = system_user
        self.user = user
        self.user_id = user.id
        self.memory = Memory(user_id=self.user_id)
        self.current_task: Optional[Task] = None
        self.latest_screenshot: Optional[str] = None
        self.model_width = None
        self.model_height = None
        self.task_rag = RAGSystem()
        self.shared_learning = SharedLearningService()
        self.step_id = 0
        self.websocket_manager = websocket_manager
        
        # برای ذخیره آخرین action و expected outcome
        self.last_action_info = {}

        # اضافه کردن checkpoint system
        self.checkpoint_data = {}
        self.processing_stage = "start"  # start, position, vision, action, complete

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
        user = self.db.query(User).filter(User.id == self.user_id).first()
        user.status = "IN_PROGRESS"
        self.db.commit()
        self.websocket_manager.send_to_user_with_format(self.system_user.id, WebSocketType.USER_UPDATE, user.summary())
        try:
            self.current_task = self.db.query(Task).filter(
                Task.user_id == self.user_id,
                or_(
                    Task.status == TaskStatus.NEW,
                    Task.status == TaskStatus.IN_PROGRESS,
                    Task.status == TaskStatus.PAUSED,
                )).order_by(Task.priority.desc()).first()

            if not self.current_task:
                return False

            self.current_task.status = TaskStatus.IN_PROGRESS
            self.db.commit()
            if not self.current_task:
                return

            for _ in range(20):
                continue_next_step = self.process_task_step()
                if not continue_next_step:
                    break
                user = self.db.query(User).filter(User.id == self.user_id).first()
                user_can_continue = user.continue_automatically
                if not user_can_continue:
                    break
            return True
        finally:
            user = self.db.query(User).filter(User.id == self.user_id).first()
            user.status = "IDLE"
            self.db.commit()
            self.websocket_manager.send_to_user_with_format(self.system_user.id, WebSocketType.USER_UPDATE,
                                                            user.summary())

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
        real_width, real_height = self.get_image_dimensions(DataService().get_user_path_screenshot(self.user_id))
        width = float(x) * (real_width / self.model_width)
        height = float(y) * (real_height / self.model_height)
        self.draw_red_circle_on_image(DataService().get_user_path_screenshot(self.user_id),
                                      [width, height],
                                      DataService().get_user_path_screenshot_with_point(
                                          self.user_id, self.step_id))
        logService.append_log(self.step_id, self.current_task.id, f'single_click {width}, {height}', 'action_mouse',
                              DataService().get_user_path_screenshot_with_point(self.user_id, self.step_id))
        return CommandService.click(width,
                                    height,
                                    self.user_id)

    def scroll_up(self, repeat, description):
        return CommandService.scroll_up(repeat, self.user_id)

    def scroll_down(self, repeat, description):
        return CommandService.scroll_down(repeat, self.user_id)

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
        real_width, real_height = self.get_image_dimensions(DataService().get_user_path_screenshot(self.user_id))
        width = float(x) * (real_width / self.model_width)
        height = float(y) * (real_height / self.model_height)
        self.draw_red_circle_on_image(DataService().get_user_path_screenshot(self.user_id),
                                      [width, height],
                                      DataService().get_user_path_screenshot_with_point(
                                          self.user_id, self.step_id))
        logService.append_log(self.step_id, self.current_task.id, f'double_click {width}, {height}', 'action_mouse',
                              DataService().get_user_path_screenshot_with_point(self.user_id, self.step_id))
        return CommandService.double_click(width,
                                           height,
                                           self.user_id)

    def move_mouse(self, x, y, description):
        real_width, real_height = self.get_image_dimensions(DataService().get_user_path_screenshot(self.user_id))
        width = float(x) * (real_width / self.model_width)
        height = float(y) * (real_height / self.model_height)
        self.draw_red_circle_on_image(DataService().get_user_path_screenshot(self.user_id),
                                      [width, height],
                                      DataService().get_user_path_screenshot_with_point(
                                          self.user_id, self.step_id))
        logService.append_log(self.step_id, self.current_task.id, f'move_mouse {width}, {height}', 'action_mouse',
                              DataService().get_user_path_screenshot_with_point(self.user_id, self.step_id))
        return CommandService.move_mouse(width, height, self.user_id)

    def right_click(self, x, y, description):
        real_width, real_height = self.get_image_dimensions(DataService().get_user_path_screenshot(self.user_id))
        width = float(x) * (real_width / self.model_width)
        height = float(y) * (real_height / self.model_height)
        return CommandService.right_click(width, height, self.user_id)

    def type_text(self, text, description):
        CommandService.typing(text, self.user_id)
        return f"Typed: {text[:50]}..."

    def send_key(self, keys, description):
        """
        Send keyboard key(s) via VNC.
        
        Args:
            keys: List of keys to press (e.g., ['Control_L', 'l'] for Ctrl+L)
            description: Reason for sending keys
        """
        # Ensure keys is a list
        if isinstance(keys, str):
            keys = [keys]
        return CommandService.send_key(keys, self.user_id)

    def append_screenshot(self):
        time.sleep(1)
        CommandService.screenshot(self.user_id)
        
        # دریافت context از RAG (سرچ سمانتیک)
        rag_context = self.task_rag.query_context(
            self.user_id, f"Memory data: {self.current_task.description}"
        )
        
        # دریافت ۱۰ event اخیر بر اساس timestamp
        recent_events = self.task_rag.get_recent_events(self.user_id, limit=10)
        
        # ترکیب اطلاعات برای enhanced_prompt
        enhanced_prompt = ""
        
        if rag_context:
            enhanced_prompt += f"\n=== Related Context from Memory ===\n{rag_context}\n"
        
        if recent_events:
            enhanced_prompt += "\n=== Recent 10 Events (Chronological) ===\n"
            for i, event in enumerate(recent_events, 1):
                metadata = event.get('metadata', {})
                task_info = f"[Task {metadata.get('task_id')}: {metadata.get('task_description', 'N/A')}]" if metadata.get('task_id') else "[No Task Info]"
                event_type = metadata.get('event_type', 'unknown')
                timestamp = metadata.get('timestamp', 'N/A')
                content = event.get('content', '')
                enhanced_prompt += f"{i}. {task_info} [{event_type}] {timestamp}\n   {content}\n"
            enhanced_prompt += "===\n"

        detector_raw_data = json.loads(self.getCoordinatesFromDetector())
        
        # استخراج ابعاد صفحه از detector data
        desktop_info = detector_raw_data.get('desktop_info', {})
        screen_resolution = desktop_info.get('screen_resolution', {})
        if screen_resolution.get('width') and screen_resolution.get('height'):
            self.model_width = float(screen_resolution['width'])
            self.model_height = float(screen_resolution['height'])
            print(f"Screen dimensions from detector: {self.model_width}x{self.model_height}")
        
        # Pre-filter detector data before compression
        detector_coordinates = self.prefilter_detector_data(detector_raw_data, self.current_task.description)


        # رسم rectangles برای لاگ (Detector coordinates)
        detector_output_path = DataService().get_user_path_screenshot_with_coordinates(self.user_id, self.step_id)
        details = self.draw_rectangles_on_image(
            DataService().get_user_path_screenshot(self.user_id),
            detector_raw_data,  # Use original data for visualization
            detector_output_path
        )

        # لاگ اطلاعات Detector coordinates
        real_width, real_height = self.get_image_dimensions(DataService().get_user_path_screenshot(self.user_id))
        if details != {}:
            logService.append_log(self.step_id, self.current_task.id,
                                  f'real_width: {real_width}, real_height: {real_height}, Detector detected elements: {len(detector_coordinates)}, detail: {details}',
                                  'detector_coordinates',
                                  detector_output_path)

        # Discovery Phase: اجرای discovery و ساخت enhanced coordinates برای vision
        # enhanced_coordinates = self.perform_element_discovery_for_vision(detector_coordinates,
        #                                                                  DataService().get_user_path_screenshot(
        #                                                                      self.user_id))

        user = self.db.query(User).filter(User.id == self.user_id).first()
        username = user.name.replace(' ', '_').lower()

        # Compress detector coordinates for vision model
        compressed_coordinates = self.compress_coordinates_for_vision(detector_coordinates, self.current_task.description)
        
        # Parse compressed data to create readable summary
        coord_summary = self.create_coordinate_summary(compressed_coordinates)
        
        # دریافت shared mistakes مرتبط با task
        relevant_mistakes = self.shared_learning.get_relevant_mistakes(
            self.current_task.description, 
            top_k=3
        )
        mistakes_context = self.shared_learning.format_mistakes_for_prompt(relevant_mistakes)
        
        # ساخت قسمت Last Action Validation
        last_action_validation = ""
        if self.last_action_info:
            last_action_validation = f'''
=== LAST ACTION VALIDATION (CRITICAL - Must Answer First) ===
Previous Action: {self.last_action_info.get('action_type', 'N/A')} at coordinates {self.last_action_info.get('coordinates', 'N/A')}
Expected Outcome: {self.last_action_info.get('expected_outcome', 'N/A')}

MANDATORY QUESTIONS:
1. Did the expected outcome happen? (yes/no/partially)
2. What actually happened? (describe current screen state)
3. Does the situation need correction? (yes/no)

IF ANSWER TO Q3 IS YES:
→ Your FIRST action in "Next Actions" MUST be a corrective action
→ Examples: 
  - Close unwanted tab: send_key(["Control_L", "w"])
  - Switch to correct tab: single_click on the tab element
  - Dismiss popup: send_key(["Escape"])
→ Do NOT proceed with original task until correction is done

'''
        
        vision_prompt = f'''Analyze this screenshot to complete the task.

        TASK: {self.current_task.description}
        PRIORITY: {self.current_task.priority}
{last_action_validation}
{mistakes_context}

        AVAILABLE UI ELEMENTS (from detector):
        {coord_summary}

        FULL ELEMENT DATA (JSON):
        {compressed_coordinates}

        ANALYSIS REQUIREMENTS:
        
        1. Active Window Analysis:
           - What is currently focused/active
           - List interactive elements with their exact coordinates
           - Current state and what user is doing
        
        2. Desktop Context:
           - Visible desktop icons (if relevant to task)
           - Other windows (if task requires them)
        
        3. Element Details:
           Format: "name (role) at center [x,y]"
           - Only mention elements relevant to task
           - Use exact coordinates from detector data
           - Focus on interactive elements

        4. Task Status:
           - Is task complete? (yes/no/in_progress)
           - What percentage complete?
           - Any blockers?
        
        5. Outcome Validation (if last action info provided):
           - outcome_validation: yes/no/partially (did expected outcome happen?)
           - actual_outcome: describe what actually happened
           - needs_correction: yes/no (does situation need fixing?)

        CRITICAL RULE FOR NEXT ACTIONS:
        - IF outcome_validation is "no" or "partially" AND needs_correction is "yes":
          → The FIRST action MUST be a corrective action to fix the problem
          → Examples: close unexpected tab/window, switch to correct tab, dismiss popup
          → ONLY after correction, proceed with the original task
        - IF outcome_validation is "yes":
          → Continue with normal task actions

        Next Actions (simplified format):

        1. Action: {{
            "type": "single_click/double_click/type_text/send_key/scroll_up/scroll_down/wait/complete_task",
            "target": {{
                "name": "element name from detector",
                "role": "element role",
                "center": [x, y],  // Use center from detector data
                "app": "application name",
                "window": "window title"
            }},
            "reason": "why this action",
            "expected": "what should happen"
        }}
        
        Note: Choose action type carefully:
        - double_click: for icons, files, folders, desktop items
        - single_click: for buttons, menus, links, UI controls
        - send_key: MUST use list format for keys (e.g., ["Control_L", "l"] not "Ctrl+L")

        KEY RULES (simplified):
        1. Use EXACT coordinates from detector data above
        2. For clicks: use "center" point from element data
        3. Focus on active window first
        4. Desktop icons: look in caja "Desktop" window
        5. Only interact with interactive elements (i:true)
        6. For text input: click element first, then type
        7. Use scroll when needed to see more content
        8. Check FTP (ftp://ftp-server:20) for files if needed
        9. Check RocketChat messenger (http://rocketchat:3000) - IMPORTANT: RocketChat is a web-based messenger, you MUST use Firefox browser to access it
        10. Username: {username}, Password: mypassword (for both)
        11. MANAGER MESSAGES & NOTIFICATIONS:
            - If you see RocketChat notification badge or unread message indicator
            - Navigate to RocketChat in Firefox browser
            - Check for new messages from your manager
            - Read messages carefully - some may be tasks, some may be just chat
            - If task "Check and respond to manager messages" exists, you should:
              1. Open Firefox and go to RocketChat (http://rocketchat:3000)
              2. Click on the direct message with your manager
              3. Read all unread messages
              4. The messages will be automatically processed (task creation if needed)
        12. KEYBOARD SHORTCUTS: Always use list format:
            - Ctrl+L (address bar): ["Control_L", "l"]
            - Ctrl+T (new tab): ["Control_L", "t"]
            - Ctrl+W (close tab): ["Control_L", "w"]
            - Alt+Tab (switch window): ["Alt_L", "Tab"]
            - Enter key: ["Return"]
        12. CORRECTIVE ACTIONS (Critical - Execute IMMEDIATELY if something went wrong):
            BROWSER-SPECIFIC CORRECTIONS:
            - Unexpected tab/window opened (e.g., Password Manager, New Tab):
              → send_key(["Control_L", "w"]) to close current unwanted tab
              → Then click on the correct tab to return to original page
            - Wrong page loaded in current tab:
              → Click on correct browser tab to switch back
              → OR use send_key(["Alt_L", "Left"]) to go back in history
            - Browser popup/autocomplete appeared:
              → send_key(["Escape"]) to dismiss popup
            - Multiple unwanted tabs opened:
              → Repeatedly use send_key(["Control_L", "w"]) until back to correct tab
            
            GENERAL CORRECTIONS:
            - Action had no effect: Try alternative element or method
            - Element not found: Scroll to find it, or use keyboard navigation
            - Dialog/modal blocking: Press Escape or click close button
            
            IMPORTANT: Corrective actions take priority over task continuation!
            Always learn from mistakes listed above and avoid repeating them
        
        IMPORTANT CLICK RULES:
        - DOUBLE_CLICK: Use for opening icons, files, folders, applications (role: icon, label with file/app names)
        - SINGLE_CLICK: Use for buttons, menu items, links, tabs, text fields (role: button, menu item, link, tab, text, entry)
        - Examples: 
          * Desktop icons → double_click
          * File manager files/folders → double_click
          * Application icons → double_click
          * Buttons/Menu items → single_click

        DESKTOP CONTEXT:
        - mate-panel: Top/bottom panels (system only, rarely clickable)
        - caja Desktop: Desktop icons (interactive)
        - caja root: File browser window
        '''
        logService.append_log(self.step_id, self.current_task.id, enhanced_prompt + vision_prompt, 'vision_request',
                              DataService().get_user_path_screenshot(self.user_id))
        # Recent actions: {recent_actions}
        self.websocket_manager.send_to_user_with_format(self.system_user.id, WebSocketType.VISION_REQUEST, {
            'user_id': self.user_id,
            'request': vision_prompt + enhanced_prompt,
        })
        # تبدیل به فرمت استرینگ برای سازگاری با OllamaProvider
        full_prompt = enhanced_prompt + vision_prompt
        response = vision_model.call(full_prompt, DataService().get_user_path_screenshot(self.user_id))
        logService.append_log(self.step_id, self.current_task.id, response, 'vision_response')
        print(f'screenshot response: {response}')
        self.websocket_manager.send_to_user_with_format(self.system_user.id, WebSocketType.VISION_RESPONSE, {
            'user_id': self.user_id,
            'response': response,
        })
        
        # پردازش outcome validation و ذخیره mistake در صورت لزوم
        self._process_outcome_validation(response)
        
        # Desktop visualization disabled (using detector coordinates instead)
        # Visualization is already done by draw_rectangles_on_image above
        
        return response

    def getCoordinatesInLocal(self):
        import sys
        import os

        # اضافه کردن مسیر sam.py
        sam_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "sam.py")
        sys.path.append(os.path.dirname(sam_path))

        try:
            from sam import load_sam_model, load_image, segment_image
        except ImportError:
            # fallback به YOLO اگر SAM در دسترس نباشد
            return self._fallback_to_yolo()

        img_path = DataService().get_user_path_screenshot(self.user_id)

        try:
            # بارگذاری مدل SAM
            sam_model = load_sam_model()

            # بارگذاری و پردازش تصویر
            image = load_image(img_path)
            original_image = cv2.imread(img_path)

            # انجام segmentation
            masks = segment_image(image, sam_model)

            # ساخت coordinates با polygon format
            coordinates = {
                "desktop": [[0, 0], [original_image.shape[1], 0], [original_image.shape[1], original_image.shape[0]],
                            [0, original_image.shape[0]]]}

            # اضافه کردن OCR برای یافتن فایل‌های احتمالی
            ocr_detected_files = self.detect_files_with_ocr(original_image)

            # پردازش masks و تبدیل به polygon coordinates
            for i, mask in enumerate(masks):
                try:
                    # دریافت polygon از mask
                    polygon_coords = self.mask_to_polygon(mask['segmentation'])

                    if polygon_coords and len(polygon_coords) >= 3:  # حداقل 3 نقطه برای polygon معتبر
                        # طبقه‌بندی المان
                        bbox = mask['bbox']
                        element_name = self.classify_sam_element(
                            original_image,
                            polygon_coords,
                            bbox,
                            f"segment_{i}",
                            ocr_detected_files
                        )

                        coordinates[element_name] = polygon_coords

                        # تجزیه تفصیلی برای المان‌های متنی
                        if self.is_text_element(element_name, polygon_coords, original_image):
                            detailed_coords = self.analyze_polygon_element_detailed(
                                original_image,
                                polygon_coords,
                                element_name
                            )
                            coordinates.update(detailed_coords)

                except Exception as e:
                    print(f"Error processing mask {i}: {e}")
                    continue

            # اضافه کردن فایل‌های OCR که overlap ندارند
            for file_info in ocr_detected_files:
                file_name = file_info['name']
                file_bbox = file_info['coords']  # [x1, y1, x2, y2]

                # تبدیل bbox به polygon
                file_polygon = [[file_bbox[0], file_bbox[1]], [file_bbox[2], file_bbox[1]],
                                [file_bbox[2], file_bbox[3]], [file_bbox[0], file_bbox[3]]]

                # بررسی overlap با polygons موجود
                overlap_found = False
                for existing_coords in coordinates.values():
                    if isinstance(existing_coords, list) and len(existing_coords) > 0 and isinstance(existing_coords[0],
                                                                                                     list):
                        if self.check_polygon_overlap(file_polygon, existing_coords, threshold=0.5):
                            overlap_found = True
                            break

                if not overlap_found:
                    coordinates[file_name] = file_polygon

            return json.dumps(coordinates, ensure_ascii=False)

        except Exception as e:
            print(f"Error in SAM processing: {e}")
            # fallback به YOLO در صورت خطا
            return self._fallback_to_yolo()

    def detect_files_with_ocr(self, image):
        """
        استفاده از OCR برای یافتن فایل‌های احتمالی در تصویر
        """
        detected_files = []

        try:
            # تبدیل به تصویر خاکستری
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # استخراج متن با OCR
            data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT, config='--psm 6')

            # یافتن کلمات که احتمال دارد نام فایل باشند
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                confidence = int(data['conf'][i])

                if confidence > 50 and text:
                    # بررسی اینکه آیا متن شبیه نام فایل است
                    if self.looks_like_filename(text):
                        x = data['left'][i]
                        y = data['top'][i]
                        w = data['width'][i]
                        h = data['height'][i]

                        # اضافه کردن padding برای در نظر گیری icon فایل
                        padding = 20
                        x1 = max(0, x - padding)
                        y1 = max(0, y - padding)
                        x2 = min(image.shape[1], x + w + padding)
                        y2 = min(image.shape[0], y + h + padding)

                        file_name = self.generate_file_element_name(text)
                        detected_files.append({
                            'name': file_name,
                            'coords': [x1, y1, x2, y2],
                            'original_text': text,
                            'confidence': confidence
                        })

        except Exception as e:
            print(f"OCR file detection error: {e}")

        return detected_files

    def looks_like_filename(self, text):
        """
        بررسی اینکه آیا متن شبیه نام فایل است
        """
        # الگوهای مختلف فایل
        file_patterns = [
            r'.*\.(txt|pdf|doc|docx|jpg|jpeg|png|gif|exe|zip|rar|mp3|mp4|avi)$',  # فایل‌های با extension
            r'^[a-zA-Z0-9_\-\.]+\.(txt|pdf|doc|docx|jpg|jpeg|png|gif|exe|zip|rar|mp3|mp4|avi)$',  # نام فایل کامل
            r'^test\.txt$',  # فایل خاص که در task ذکر شده
        ]

        # بررسی task context برای نام‌های خاص
        if self.current_task and 'test.txt' in self.current_task.description.lower():
            if 'test.txt' in text.lower() or 'test' in text.lower():
                return True

        import re
        for pattern in file_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return True

        return False

    def generate_file_element_name(self, filename):
        """
        تولید نام element برای فایل
        """
        # حذف extension و ایجاد نام مناسب
        base_name = filename.lower()
        if '.' in base_name:
            name_part, ext_part = base_name.rsplit('.', 1)
            return f"{name_part}_{ext_part}_file"
        else:
            return f"{base_name}_file"

    def classify_element_type(self, image, coords, class_name, index, ocr_files):
        """
        طبقه‌بندی نوع المان: فایل، فولدر یا UI element
        """
        x1, y1, x2, y2 = coords
        element_image = image[y1:y2, x1:x2]

        # بررسی اینکه آیا این المان با فایل‌های OCR مطابقت دارد
        for file_info in ocr_files:
            if self.check_overlap(coords, file_info['coords'], threshold=0.3):
                return file_info['name']

        # طبقه‌بندی بر اساس class name و محل قرارگیری
        if class_name.lower() in ['file', 'document', 'text_file']:
            return f"unknown_file_{index}"
        elif class_name.lower() in ['folder', 'directory']:
            return f"folder_{index}"
        elif class_name.lower() in ['icon'] and self.is_on_desktop(coords, image.shape):
            # icon های روی desktop احتمال بیشتری دارد که فایل باشند
            return f"desktop_icon_{index}"
        elif class_name.lower() in ['text', 'label']:
            # بررسی محل قرارگیری برای تشخیص UI text vs file name
            if self.is_ui_context(coords, image.shape):
                return f"ui_{class_name}_{index}"
            else:
                return f"text_element_{index}"
        else:
            return f"{class_name}_{index}"

    def is_on_desktop(self, coords, image_shape):
        """
        بررسی اینکه آیا المان روی desktop قرار دارد
        """
        x1, y1, x2, y2 = coords
        height, width = image_shape[:2]

        # اگر المان در نیمه بالایی صفحه و نه در نوار پایینی باشد
        if y1 < height * 0.7 and y2 < height * 0.9:
            return True
        return False

    def is_ui_context(self, coords, image_shape):
        """
        بررسی اینکه آیا المان در context UI قرار دارد (مثل terminal، taskbar)
        """
        x1, y1, x2, y2 = coords
        height, width = image_shape[:2]

        # نوار پایینی (taskbar)
        if y1 > height * 0.9:
            return True

        # گوشه‌های صفحه (status areas)
        if (x1 < width * 0.1 and y1 < height * 0.1) or (x1 > width * 0.9 and y1 < height * 0.1):
            return True

        return False

    def check_overlap(self, coords1, coords2, threshold=0.5):
        """
        بررسی overlap بین دو مستطیل
        """
        x1_1, y1_1, x2_1, y2_1 = coords1
        x1_2, y1_2, x2_2, y2_2 = coords2

        # محاسبه ناحیه overlap
        overlap_x1 = max(x1_1, x1_2)
        overlap_y1 = max(y1_1, y1_2)
        overlap_x2 = min(x2_1, x2_2)
        overlap_y2 = min(y2_1, y2_2)

        if overlap_x1 >= overlap_x2 or overlap_y1 >= overlap_y2:
            return False

        overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

        overlap_ratio = overlap_area / min(area1, area2)
        return overlap_ratio >= threshold

    def mask_to_polygon(self, mask):
        """
        تبدیل SAM mask به polygon coordinates
        """
        try:
            import cv2
            import numpy as np

            # تبدیل mask به uint8
            mask_uint8 = (mask * 255).astype(np.uint8)

            # پیدا کردن contours
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                return None

            # انتخاب بزرگترین contour
            largest_contour = max(contours, key=cv2.contourArea)

            # ساده‌سازی polygon (کاهش تعداد نقاط)
            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            simplified_contour = cv2.approxPolyDP(largest_contour, epsilon, True)

            # تبدیل به فرمت [[x, y], [x, y], ...]
            polygon_points = []
            for point in simplified_contour:
                x, y = point[0]
                polygon_points.append([int(x), int(y)])

            return polygon_points

        except Exception as e:
            print(f"Error converting mask to polygon: {e}")
            return None

    def classify_sam_element(self, image, polygon_coords, bbox, default_name, ocr_files):
        """
        طبقه‌بندی المان‌های SAM
        """
        try:
            # محاسبه bounding box از polygon
            if not polygon_coords:
                return default_name

            # بررسی تطابق با فایل‌های OCR
            polygon_bbox = self.polygon_to_bbox(polygon_coords)
            for file_info in ocr_files:
                if self.check_overlap(polygon_bbox, file_info['coords'], threshold=0.3):
                    return file_info['name']

            # طبقه‌بندی بر اساس ویژگی‌های geometric
            area = self.polygon_area(polygon_coords)
            aspect_ratio = self.polygon_aspect_ratio(polygon_coords)

            if area < 500:
                return f"small_element_{default_name}"
            elif area > 50000:
                return f"large_element_{default_name}"
            elif aspect_ratio > 3:
                return f"text_element_{default_name}"
            elif aspect_ratio < 0.5:
                return f"button_element_{default_name}"
            else:
                return f"ui_element_{default_name}"

        except Exception as e:
            return default_name

    def is_text_element(self, element_name, polygon_coords, image):
        """
        تشخیص اینکه آیا المان متنی است
        """
        if any(keyword in element_name.lower() for keyword in ['text', 'input', 'field', 'label', 'button']):
            return True

        # بررسی aspect ratio
        aspect_ratio = self.polygon_aspect_ratio(polygon_coords)
        return aspect_ratio > 2.5  # المان‌های بلند و باریک احتمالاً متن هستند

    def analyze_polygon_element_detailed(self, image, polygon_coords, element_name):
        """
        تجزیه تفصیلی المان‌های polygon
        """
        detailed_coords = {}

        try:
            # تبدیل polygon به bbox برای OCR
            bbox = self.polygon_to_bbox(polygon_coords)

            # استفاده از تابع موجود با bbox
            bbox_detailed = self.analyze_element_detailed(image, bbox, element_name)

            # تبدیل نتایج bbox به polygon format در صورت نیاز
            detailed_coords.update(bbox_detailed)

        except Exception as e:
            print(f"Error analyzing polygon element {element_name}: {e}")

        return detailed_coords

    def check_polygon_overlap(self, poly1, poly2, threshold=0.5):
        """
        بررسی overlap بین دو polygon
        """
        try:
            # تبدیل به bbox برای محاسبه ساده
            bbox1 = self.polygon_to_bbox(poly1)
            bbox2 = self.polygon_to_bbox(poly2)

            return self.check_overlap(bbox1, bbox2, threshold)

        except Exception as e:
            return False

    def polygon_to_bbox(self, polygon_coords):
        """
        تبدیل polygon به bounding box [x1, y1, x2, y2]
        """
        if not polygon_coords:
            return [0, 0, 0, 0]

        x_coords = [point[0] for point in polygon_coords]
        y_coords = [point[1] for point in polygon_coords]

        return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]

    def polygon_area(self, polygon_coords):
        """
        محاسبه مساحت polygon
        """
        if len(polygon_coords) < 3:
            return 0

        area = 0
        n = len(polygon_coords)
        for i in range(n):
            j = (i + 1) % n
            area += polygon_coords[i][0] * polygon_coords[j][1]
            area -= polygon_coords[j][0] * polygon_coords[i][1]

        return abs(area) / 2

    def polygon_aspect_ratio(self, polygon_coords):
        """
        محاسبه aspect ratio polygon
        """
        bbox = self.polygon_to_bbox(polygon_coords)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]

        if height == 0:
            return float('inf')

        return width / height

    def _fallback_to_yolo(self):
        """
        Fallback به YOLO در صورت عدم دسترسی به SAM
        """
        try:
            from ultralytics import YOLO
            import os

            img_path = DataService().get_user_path_screenshot(self.user_id)

            # بارگذاری مدل YOLO
            model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                      "best.pt")
            if not os.path.exists(model_path):
                model_path = "yolov8n.pt"

            model = YOLO(model_path)
            model.overrides['conf'] = 0.3
            model.overrides['iou'] = 0.5
            model.overrides['max_det'] = 50

            results = model(img_path)
            image = cv2.imread(img_path)

            # تبدیل نتایج YOLO به polygon format
            coordinates = {
                "desktop": [[0, 0], [image.shape[1], 0], [image.shape[1], image.shape[0]], [0, image.shape[0]]]}

            for result in results:
                if result.boxes is not None:
                    for i, box in enumerate(result.boxes):
                        class_id = int(box.cls)
                        class_name = model.names[class_id] if class_id in model.names else f"class_{class_id}"
                        coords = box.xyxy[0].tolist()
                        confidence = box.conf.item()

                        if confidence > 0.3:
                            # تبدیل bbox به polygon
                            x1, y1, x2, y2 = [int(c) for c in coords]
                            polygon = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

                            element_name = f"yolo_{class_name}_{i}"
                            coordinates[element_name] = polygon

            return json.dumps(coordinates, ensure_ascii=False)

        except Exception as e:
            print(f"YOLO fallback failed: {e}")
            # حداقل desktop را برگردان
            return json.dumps({"desktop": [[0, 0], [1920, 0], [1920, 1080], [0, 1080]]}, ensure_ascii=False)

    def analyze_element_detailed(self, image, coords, element_name):
        """
        تجزیه تفصیلی یک المان برای استخراج کاراکترها و محتویات
        """
        detailed_coords = {}

        try:
            # برش المان از تصویر
            x1, y1, x2, y2 = coords
            element_image = image[y1:y2, x1:x2]

            if element_image.size == 0:
                return detailed_coords

            # تبدیل به تصویر خاکستری
            gray = cv2.cvtColor(element_image, cv2.COLOR_BGR2GRAY)

            # preprocessing برای بهبود OCR
            # افزایش کنتراست
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)

            # حذف نویز
            denoised = cv2.medianBlur(enhanced, 3)

            # threshold برای تشخیص بهتر متن
            _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # تشخیص متن با جزئیات
            try:
                # استخراج اطلاعات کاراکتر به کاراکتر
                data = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT, config='--psm 6')

                # تجمیع اطلاعات کاراکترها
                characters = []
                words = []
                current_word = ""
                current_word_coords = []

                for i in range(len(data['text'])):
                    if int(data['conf'][i]) > 30:  # فقط کاراکترهای با اطمینان بالا
                        char_text = data['text'][i].strip()
                        if char_text:
                            # مختصات کاراکتر نسبت به المان
                            char_x = data['left'][i]
                            char_y = data['top'][i]
                            char_w = data['width'][i]
                            char_h = data['height'][i]

                            # مختصات مطلق کاراکتر
                            abs_char_x1 = x1 + char_x
                            abs_char_y1 = y1 + char_y
                            abs_char_x2 = abs_char_x1 + char_w
                            abs_char_y2 = abs_char_y1 + char_h

                            # اضافه کردن کاراکتر
                            char_info = {
                                'text': char_text,
                                'coordinates': [abs_char_x1, abs_char_y1, abs_char_x2, abs_char_y2],
                                'confidence': data['conf'][i],
                                'font_size': char_h,
                                'relative_position': len(characters)
                            }
                            characters.append(char_info)

                            # ساخت کلید برای کاراکتر
                            char_key = f"{element_name}_char_{len(characters) - 1}_{char_text}"
                            detailed_coords[char_key] = [abs_char_x1, abs_char_y1, abs_char_x2, abs_char_y2]

                            # جمع‌آوری کلمات
                            if data['word_num'][i] == data['word_num'][i - 1] if i > 0 else True:
                                current_word += char_text
                                current_word_coords.extend([abs_char_x1, abs_char_y1, abs_char_x2, abs_char_y2])
                            else:
                                # کلمه جدید شروع شد
                                if current_word:
                                    word_key = f"{element_name}_word_{len(words)}_{current_word}"
                                    word_x1 = min(current_word_coords[::4])
                                    word_y1 = min(current_word_coords[1::4])
                                    word_x2 = max(current_word_coords[2::4])
                                    word_y2 = max(current_word_coords[3::4])
                                    detailed_coords[word_key] = [word_x1, word_y1, word_x2, word_y2]
                                    words.append(current_word)

                                current_word = char_text
                                current_word_coords = [abs_char_x1, abs_char_y1, abs_char_x2, abs_char_y2]

                # اضافه کردن آخرین کلمه
                if current_word:
                    word_key = f"{element_name}_word_{len(words)}_{current_word}"
                    word_x1 = min(current_word_coords[::4])
                    word_y1 = min(current_word_coords[1::4])
                    word_x2 = max(current_word_coords[2::4])
                    word_y2 = max(current_word_coords[3::4])
                    detailed_coords[word_key] = [word_x1, word_y1, word_x2, word_y2]

                # اضافه کردن اطلاعات کلی المان
                full_text = ' '.join([char['text'] for char in characters])
                if full_text.strip():
                    detailed_coords[f"{element_name}_full_text"] = coords
                    detailed_coords[f"{element_name}_text_content"] = full_text.strip()
                    detailed_coords[f"{element_name}_char_count"] = len(characters)
                    detailed_coords[f"{element_name}_word_count"] = len(words)

                # تشخیص cursor (اگر المان فعال باشد)
                cursor_pos = self.detect_cursor_position(element_image, characters, x1, y1)
                if cursor_pos:
                    cursor_key = f"{element_name}_cursor"
                    detailed_coords[cursor_key] = [
                        x1 + cursor_pos[0],
                        y1 + cursor_pos[1],
                        x1 + cursor_pos[0] + 2,
                        y1 + cursor_pos[1] + cursor_pos[3]
                    ]

            except Exception as e:
                # در صورت خطا در OCR، حداقل متن کلی را استخراج کن
                try:
                    text = pytesseract.image_to_string(thresh, config='--psm 6').strip()
                    if text:
                        detailed_coords[f"{element_name}_text_content"] = text
                except:
                    pass

        except Exception as e:
            print(f"Error analyzing element {element_name}: {e}")

        return detailed_coords

    def detect_cursor_position(self, element_image, characters, element_x1=0, element_y1=0):
        """
        تشخیص موقعیت cursor در یک المان متنی
        """
        try:
            # تبدیل به تصویر خاکستری
            gray = cv2.cvtColor(element_image, cv2.COLOR_BGR2GRAY)

            # تشخیص خطوط عمودی (cursor معمولاً یک خط عمودی است)
            # استفاده از kernel عمودی برای تشخیص خطوط عمودی
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))
            vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel)

            # پیدا کردن contour ها
            contours, _ = cv2.findContours(vertical_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                # cursor معمولاً یک خط باریک و بلند است
                if w <= 3 and h >= 10:  # cursor criteria
                    return [x, y, w, h]

            # اگر cursor مستقیماً یافت نشد، موقعیت احتمالی بعد از آخرین کاراکتر
            if characters:
                last_char = characters[-1]
                return [last_char['coordinates'][2] - element_x1, last_char['coordinates'][1] - element_y1, 2, last_char['font_size']]

        except Exception as e:
            print(f"Error detecting cursor: {e}")

        return None

    def analyze_text_field_state(self, element_coords, element_name):
        """
        تجزیه وضعیت یک text field شامل focus، selection و cursor
        """
        try:
            img_path = DataService().get_user_path_screenshot(self.user_id)
            image = cv2.imread(img_path)

            x1, y1, x2, y2 = element_coords
            field_image = image[y1:y2, x1:x2]

            state_info = {
                'element_name': element_name,
                'coordinates': element_coords,
                'is_focused': False,
                'has_selection': False,
                'cursor_position': None,
                'text_content': '',
                'character_positions': [],
                'word_positions': [],
                'field_type': 'unknown'
            }

            # تشخیص focus (معمولاً با border رنگی یا highlight)
            state_info['is_focused'] = self.detect_focus_state(field_image)

            # تشخیص selection (معمولاً با background رنگی)
            selection_info = self.detect_text_selection(field_image)
            state_info['has_selection'] = selection_info['has_selection']
            state_info['selection_range'] = selection_info.get('range', None)

            # تشخیص نوع فیلد
            state_info['field_type'] = self.detect_field_type(field_image)

            return state_info

        except Exception as e:
            print(f"Error analyzing text field state: {e}")
            return None

    def detect_focus_state(self, field_image):
        """
        تشخیص وضعیت focus یک text field
        """
        try:
            # تبدیل به HSV برای تشخیص بهتر رنگ‌ها
            hsv = cv2.cvtColor(field_image, cv2.COLOR_BGR2HSV)

            # تشخیص border های رنگی (معمولاً آبی برای focus)
            # محدوده رنگ آبی
            lower_blue = np.array([100, 50, 50])
            upper_blue = np.array([130, 255, 255])
            blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

            # اگر مقدار قابل توجهی از border آبی یافت شد
            blue_pixels = cv2.countNonZero(blue_mask)
            total_pixels = field_image.shape[0] * field_image.shape[1]

            if blue_pixels > total_pixels * 0.02:  # 2% از pixels آبی باشند
                return True

            # تشخیص سایر نشانه‌های focus
            gray = cv2.cvtColor(field_image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)

            # اگر edge های زیادی در حاشیه وجود دارد، احتمالاً focused است
            border_edges = np.sum(edges[0, :]) + np.sum(edges[-1, :]) + np.sum(edges[:, 0]) + np.sum(edges[:, -1])

            return border_edges > 100

        except Exception as e:
            return False

    def detect_text_selection(self, field_image):
        """
        تشخیص انتخاب متن در یک text field
        """
        try:
            # تبدیل به HSV
            hsv = cv2.cvtColor(field_image, cv2.COLOR_BGR2HSV)

            # تشخیص ناحیه‌های highlight شده (معمولاً آبی روشن)
            lower_highlight = np.array([100, 30, 150])
            upper_highlight = np.array([130, 100, 255])
            highlight_mask = cv2.inRange(hsv, lower_highlight, upper_highlight)

            # پیدا کردن مناطق highlight
            contours, _ = cv2.findContours(highlight_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            selection_info = {
                'has_selection': False,
                'range': None
            }

            if contours:
                # بزرگترین ناحیه highlight را پیدا کن
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)

                # اگر ناحیه به اندازه کافی بزرگ باشد
                if w > 10 and h > 5:
                    selection_info['has_selection'] = True
                    selection_info['range'] = [x, y, x + w, y + h]

            return selection_info

        except Exception as e:
            return {'has_selection': False, 'range': None}

    def detect_field_type(self, field_image):
        """
        تشخیص نوع text field (input, textarea, password, etc.)
        """
        try:
            height, width = field_image.shape[:2]

            # تشخیص بر اساس ابعاد
            if height > width * 0.3:  # اگر ارتفاع بیش از 30% عرض باشد
                return "textarea"
            elif height < 30:
                return "input"
            else:
                return "input"

            # تشخیص password field (کاراکترهای mask شده)
            gray = cv2.cvtColor(field_image, cv2.COLOR_BGR2GRAY)

            # تشخیص نقاط یا ستاره‌های تکراری
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=2, maxRadius=8)

            if circles is not None and len(circles[0]) > 2:
                return "password"

        except Exception as e:
            pass

        return "input"

    def draw_red_circle_on_image(self, input_image_path, point, output_image_path=None, circle_radius=20):
        """
        روی عکس ورودی یک دایره قرمز در نقطه مشخص شده رسم می‌کند
        """
        # باز کردن عکس
        if isinstance(input_image_path, str):
            img = Image.open(input_image_path)
        else:
            img = input_image_path.copy()

        # ایجاد شیء Draw به صورت صحیح
        draw = ImageDraw.Draw(img)  # اینجا باید از ماژول ImageDraw استفاده شود

        # مختصات نقطه
        x, y = point

        # محاسبه مختصات دایره
        bounding_box = [
            (x - circle_radius, y - circle_radius),
            (x + circle_radius, y + circle_radius)
        ]

        # رسم دایره
        draw.ellipse(bounding_box, outline="red", width=4)

        # ذخیره یا نمایش
        if output_image_path:
            os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
            img.save(output_image_path)
        else:
            img.show()

        return img

    def get_image_dimensions(self, image_path):
        """
        خواندن ابعاد واقعی عکس از فایل
        """
        with Image.open(image_path) as img:
            return img.width, img.height

    def draw_polygons_on_image(self, input_image_path, coordinates, output_image_path=None):
        """
        روی عکس ورودی polygon های قرمز برای هر المان در مختصات مشخص شده رسم می‌کند
        """
        # باز کردن عکس
        if isinstance(input_image_path, str):
            img = Image.open(input_image_path)
        else:
            img = input_image_path.copy()

        # ایجاد شیء Draw
        draw = ImageDraw.Draw(img)

        # خواندن ابعاد واقعی عکس
        real_width, real_height = self.get_image_dimensions(input_image_path)

        detail = {}
        for element_name, coords in coordinates.items():
            try:
                if element_name != "desktop":  # از رسم polygon برای دسکتاپ صرف نظر می‌کنیم
                    if isinstance(coords, list) and len(coords) > 0 and isinstance(coords[0], list):
                        # polygon format: [[x1, y1], [x2, y2], ...]
                        polygon_points = []
                        for point in coords:
                            x, y = point[0], point[1]
                            polygon_points.extend([x, y])

                        detail[element_name] = coords

                        # رسم polygon
                        if len(polygon_points) >= 6:  # حداقل 3 نقطه (6 مختصات)
                            draw.polygon(polygon_points, outline="red", width=2)

                        # اضافه کردن نام المان در center polygon
                        center_x = sum(point[0] for point in coords) // len(coords)
                        center_y = sum(point[1] for point in coords) // len(coords)
                        draw.text((center_x, center_y - 15), element_name, fill="red")

                    elif isinstance(coords, list) and len(coords) == 4 and isinstance(coords[0], (int, float)):
                        # rectangle format: [x1, y1, x2, y2] (fallback for YOLO)
                        x1, y1, x2, y2 = coords
                        detail[element_name] = ((x1, y1), (x2, y2))
                        draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=2)
                        draw.text((x1, y1 - 15), element_name, fill="red")

            except Exception as e:
                print(f"Error drawing element {element_name}: {e}")
                pass

        # ذخیره یا نمایش
        if output_image_path:
            os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
            img.save(output_image_path)
        else:
            img.show()

        return detail

    def process_task_step(self):
        try:
            self.step_id = int(time.time() * 1000)

            # بررسی checkpoint موجود
            checkpoint = self.load_checkpoint()

            # مقادیر پیش‌فرض
            screenshot_thought = None
            enhanced_prompt = None
            coordinates = None

            # تعیین مرحله شروع بر اساس checkpoint
            if checkpoint:
                self.processing_stage = checkpoint.get('processing_stage', 'start')
                print(f"Resuming from checkpoint: {self.processing_stage}")
            else:
                self.processing_stage = "start"

            # مرحله 1: Screenshot و Discovery Phase
            if self.processing_stage in ["start", "discovery"]:
                try:
                    self.processing_stage = "discovery"
                    screenshot_thought = self.append_screenshot()
                    print(f"screenshot: {screenshot_thought}")

                    # ذخیره checkpoint پس از screenshot و discovery
                    self.save_checkpoint("discovery_completed", {
                        'screenshot_thought': screenshot_thought
                    })

                except Exception as e:
                    print(f"Error in discovery stage: {e}")
                    traceback.format_exc()
                    raise e
            else:
                # بازیابی از checkpoint
                screenshot_thought = self.checkpoint_data.get('screenshot_thought')
                print(f"Restored screenshot from checkpoint")

            # مرحله 2: Enhanced Prompt
            if self.processing_stage in ["start", "discovery", "enhanced_prompt"]:
                try:
                    self.processing_stage = "enhanced_prompt"
                    
                    # دریافت context از RAG (سرچ سمانتیک)
                    rag_context = self.task_rag.query_context(
                        self.user_id, f"Memory data: {self.current_task.description}"
                    )
                    
                    # دریافت ۱۰ event اخیر بر اساس timestamp
                    recent_events = self.task_rag.get_recent_events(self.user_id, limit=10)
                    
                    # ترکیب اطلاعات برای enhanced_prompt
                    enhanced_prompt = ""
                    
                    if rag_context:
                        enhanced_prompt += f"\n=== Related Context from Memory ===\n{rag_context}\n"
                    
                    if recent_events:
                        enhanced_prompt += "\n=== Recent 10 Events (Chronological) ===\n"
                        for i, event in enumerate(recent_events, 1):
                            metadata = event.get('metadata', {})
                            task_info = f"[Task {metadata.get('task_id')}: {metadata.get('task_description', 'N/A')}]" if metadata.get('task_id') else "[No Task Info]"
                            event_type = metadata.get('event_type', 'unknown')
                            timestamp = metadata.get('timestamp', 'N/A')
                            content = event.get('content', '')
                            enhanced_prompt += f"{i}. {task_info} [{event_type}] {timestamp}\n   {content}\n"
                        enhanced_prompt += "===\n"

                    # ذخیره checkpoint پس از enhanced prompt
                    self.save_checkpoint("enhanced_prompt_completed", {
                        'screenshot_thought': screenshot_thought,
                        'enhanced_prompt': enhanced_prompt
                    })

                except Exception as e:
                    print(f"Error in enhanced_prompt stage: {e}")
                    raise e
            else:
                # بازیابی از checkpoint
                enhanced_prompt = self.checkpoint_data.get('enhanced_prompt')
                print(f"Restored enhanced_prompt from checkpoint")
            # مرحله 3: Action Request
            if self.processing_stage in ["start", "discovery", "enhanced_prompt", "action"]:
                try:
                    self.processing_stage = "action"

                    action_prompt = f"""Based on the recent actions and screenshot analysis and task objective, determine the next action to take.

                    Screenshot analysis:
                    {screenshot_thought}

                    CRITICAL RULE - EXECUTE VISION ACTIONS EXACTLY:
                    - The screenshot analysis above contains specific actions proposed by vision analysis
                    - You MUST execute the FIRST action exactly as specified in the "Next Actions" section
                    - Do NOT create new tasks unless the vision analysis explicitly recommends create_task
                    - Do NOT modify or ignore the action type specified (e.g., if vision says "double_click", use double_click)
                    - Only use create_task if there's an error or the vision analysis explicitly mentions it

                    Rules to follow:
                    1. If a task with a higher priority number (higher urgency) is created via create_task:
                        - Immediately terminate the current task's action list.
                        - Only output the new task (no other actions or tasks).
                        - Do NOT add subsequent actions to the original task.
                    2. CONTEXT-AWARE ACTION EXECUTION:
                        - ALWAYS check Recent Actions first before executing ANY action
                        - If the same action type with similar parameters was executed in the last 3-5 actions, SKIP it
                        - This includes: window focus clicks, application launches, navigation clicks
                        - Example: If "single_click terminal_window" was done recently, don't repeat it for focus
                        - Exception: Only repeat if the previous action clearly failed or context significantly changed
                    3. Pre-actions are CRITICAL – They determine if the main action can execute, but MUST be context-aware:
                        - Check if the required state (window focus, app launch, etc.) is already achieved from recent actions
                        - Only execute pre-actions that are actually needed based on current context
                        - Avoid redundant pre-actions that were recently completed successfully
                    4. Window Focus Requirement:
                        - If pre-actions specify focusing on a particular window:
                            - First check if the window was already focused in recent actions (last 3-5 steps)
                            - If already focused recently and no context change occurred, SKIP the focus action
                            - Only physically click on the **window itself** when focus is actually needed
                            - Example: For "LX Terminal," click the terminal window, not the LX Terminal icon.
                            - Verify focus by checking window attributes (title, active state) before proceeding.
                    5. KEYBOARD KEY COMBINATIONS (IMPORTANT):
                        - send_key accepts a LIST of keys, not a string
                        - For single keys: ["Return"] or ["Enter"] or ["Escape"] or ["Tab"]
                        - For key combinations: ["Control_L", "l"] for Ctrl+L, ["Alt_L", "Tab"] for Alt+Tab
                        - Key mappings:
                            * Ctrl → "Control_L"
                            * Alt → "Alt_L"
                            * Shift → "Shift_L"
                            * Super/Windows → "Super_L"
                            * Enter/Return → "Return"
                            * Backspace → "BackSpace"
                            * Delete → "Delete"
                        - NEVER use strings like "Ctrl+L" or "ctrl+l", always use list format: ["Control_L", "l"]
                    6. If you want apps opened in the taskbar you should say it's name that you hve in screenshot analysis
                    7. For ensure the apps is focused you need to click on them before the main action. But ONLY if they weren't focused recently
                    8. For each action of screenshot thought you should give me one action
                    9. For click or open something click on center of it
                    10. INTELLIGENT PRE-ACTION FILTERING:
                        - Before executing any Pre-Action, analyze Recent Actions for similar operations
                        - Skip pre-actions if: same action type + similar target + executed within last 3-5 steps + no failure occurred
                        - Execute pre-actions only if: required state is not achieved OR context significantly changed OR previous attempt failed
                    11. Try to use EXACTLY the actions that I told you in the prompt. double_click => double_click
                    12. For more accuracy you should just answer one action.(just for pre-focus on windows you can use one more action before the real action IF needed)
                    13. In some scenarios like type(with have single_click or hover in their pre-action) you can use several actions in SAFE mode. 
                    14. For clicking in text fields or typing in somewhere you should have two action for that. One for click on it then type in it
                    15. I give you some region for GUI elements for any type of clicking you should click on the midlle of that origin ( For example if I gave you this ((1, 2),(3, 4))) The center of that will be ((1+3/2)= 2,(2+4) / 2 = 3) => (2,3)
                    16. DO NOT ADD ANY COMMENT IN THE RESPONSE
                    17. memory_data is REQUIRED. You MUST say it in the response
                    18. AVOID REDUNDANT INPUT FIELD INTERACTIONS:
                        - Before clicking on any text input field (e.g., URL bar, search input):
                            - Check if the field is already focused OR its contents are already selected (based on context or screenshot cues).
                            - If the field is already active (cursor blinking OR text selected) and no context change occurred:
                                - SKIP the click action.
                            - Only click if:
                                - The field is not selected/focused, OR
                                - Previous attempt failed, OR
                                - Focus was lost due to an interrupt or task switch.
                    
                    Current task objective: {self.current_task.description}
                    Priority: {self.current_task.priority}
                    
                    CONTEXT ANALYSIS REQUIRED:
                    - The enhanced_prompt above contains recent actions and memory data from RAG system
                    - Review this context carefully to avoid repetition
                    - Identify any repeated patterns or similar actions that were already performed
                    - Skip actions that were recently performed successfully
                    - Only proceed if the action is truly needed or context has changed
                    
                    Determine next action using available tools.
"""

                    for x, y in tools.items():
                        y['parameters']["last_action_result"] = "What was the result of the last action"
                        y['parameters']["image_width"] = "The screenshot width size"
                        y['parameters']["image_height"] = "The screenshot height size"
                        y['parameters']["description"] = "Reason"
                        y['parameters']["memory_data"] = "The things need to save in memory // it's REQUIRED"
                        y['parameters']["expected_outcome"] = "What you expect to happen after this action (be specific)"

                    logService.append_log(self.step_id, self.current_task.id, enhanced_prompt + action_prompt,
                                          'action_request',
                                          DataService().get_user_path_screenshot(self.user_id))
                    self.websocket_manager.send_to_user_with_format(self.system_user.id, WebSocketType.ACTION_REQUEST, {
                        'user_id': self.user_id,
                        'request': enhanced_prompt + action_prompt,
                    })
                    # تبدیل به فرمت استرینگ برای سازگاری با OllamaProvider
                    full_prompt = enhanced_prompt + action_prompt
                    response = action_model.call(full_prompt, DataService().get_user_path_screenshot(self.user_id),
                                                 tools)
                    logService.append_log(self.step_id, self.current_task.id, json.dumps(response), 'action_response')
                    print(f'ACTION MODEL RESPONSE: {response}')
                    self.websocket_manager.send_to_user_with_format(self.system_user.id, WebSocketType.ACTION_RESPONSE,
                                                                    {
                                                                        'user_id': self.user_id,
                                                                        'response': response[1],
                                                                    })

                    # ذخیره checkpoint پس از دریافت response
                    self.save_checkpoint("action_completed", {
                        'screenshot_thought': screenshot_thought,
                        'enhanced_prompt': enhanced_prompt,
                        'action_response': response
                    })

                except Exception as e:
                    print(f"Error in action stage: {e}")
                    raise e
            else:
                # بازیابی از checkpoint
                response = self.checkpoint_data.get('action_response')
                print(f"Restored action_response from checkpoint")

            can_continue = True
            for tool_call in response[1]:
                action = None
                try:
                    # بررسی ساختار tool_call و استخراج action و args
                    if 'name' in tool_call and 'parameters' in tool_call:
                        # ساختار استاندارد: {'name': 'action_name', 'parameters': {...}}
                        action = tool_call['name']
                        args = deepcopy(tool_call['parameters'])
                    else:
                        # ساختار جایگزین: {'action_name': {'parameters': {...}}}
                        # پیدا کردن اولین کلید که یک dict با 'parameters' است
                        found = False
                        for key, value in tool_call.items():
                            if isinstance(value, dict) and 'parameters' in value:
                                action = key
                                args = deepcopy(value['parameters'])
                                found = True
                                break
                        
                        if not found:
                            print(f"Warning: Could not extract action from tool_call structure: {tool_call}")
                            continue
                    
                    if not action:
                        print(f"Warning: Action name is None, skipping tool_call: {tool_call}")
                        continue
                    
                    memory_data = args.get('memory_data')
                    if memory_data:
                        # ذخیره event با metadata تسک
                        task_metadata = {
                            'task_id': self.current_task.id if self.current_task else None,
                            'task_description': self.current_task.description if self.current_task else None,
                            'timestamp': datetime.now().isoformat(),
                            'event_type': 'memory_data'
                        }
                        self.task_rag.add_text(
                            self.user_id, 
                            "event: " + str(memory_data), 
                            metadata=task_metadata
                        )
                    
                    # استخراج expected_outcome برای validation در screenshot بعدی
                    expected_outcome = args.get('expected_outcome', '')
                    
                    print(f'action: {action} args: {args}')

                    if hasattr(self, action):
                        # استفاده از ابعاد از LLM parameters فقط به عنوان fallback (اگر از detector نیامده باشد)
                        if args.get('image_width') and args.get('image_height'):
                            llm_width = float(args['image_width'])
                            llm_height = float(args['image_height'])
                            # اگر ابعاد از detector تنظیم نشده، از LLM استفاده کن
                            if not hasattr(self, 'model_width') or not hasattr(self, 'model_height'):
                                self.model_width = llm_width
                                self.model_height = llm_height
                                print(f"Using screen dimensions from LLM (fallback): {self.model_width}x{self.model_height}")
                            else:
                                print(f"Screen dimensions already set from detector: {self.model_width}x{self.model_height}, LLM provided: {llm_width}x{llm_height}")
                        
                        # حذف پارامترهای اضافی قبل از اجرای action
                        args.pop('image_height', None)
                        args.pop('image_width', None)
                        args.pop('last_action_result', None)
                        args.pop('memory_data', None)
                        args.pop('expected_outcome', None)
                        
                        # ذخیره اطلاعات action قبل از اجرا برای validation بعدی
                        action_coordinates = None
                        if 'x' in args and 'y' in args:
                            action_coordinates = (args['x'], args['y'])
                        
                        # اجرای action
                        result = getattr(self, action)(**args)
                        
                        # ذخیره اطلاعات action برای validation در vision بعدی
                        self.last_action_info = {
                            'action_type': action,
                            'coordinates': action_coordinates,
                            'expected_outcome': expected_outcome,
                            'timestamp': datetime.now().isoformat(),
                            'args': args.copy()
                        }

                        # ذخیره action با metadata تسک
                        action_metadata = {
                            'task_id': self.current_task.id if self.current_task else None,
                            'task_description': self.current_task.description if self.current_task else None,
                            'timestamp': datetime.now().isoformat(),
                            'event_type': 'action',
                            'action_name': action
                        }
                        self.task_rag.add_text(
                            self.user_id,
                            "action: " + action + " args: " + " ".join(
                                f'{k}: {v}' for k, v in args.items()),
                            metadata=action_metadata
                        )

                        # اگر action ایجاد تسک بود و تسک جدید اولویت بالاتری داشت، باید ادامه ندهیم
                        if action == "create_task" and "switched to higher priority task" in result:
                            can_continue = False
                    else:
                        print(f"Warning: Action '{action}' not found as a method in processor service")

                    # ذخیره پیام تسک
                    self.current_task.task_messages.append(
                        TaskMessage(content=json.dumps({
                            "role": "assistant",
                            "content": json.dumps(tool_call),
                        }))
                    )

                    self.db.commit()

                    if not can_continue:
                        # در صورت عدم ادامه، checkpoint را پاک کن
                        self.clear_checkpoint()
                        return False

                except Exception as e:
                    traceback.print_exc()
                    action_str = action if action else "unknown"
                    print(f"Error processing tool call {action_str}: {str(e)}")
                    print(f"Tool call structure: {tool_call}")
                    continue

            # در صورت تکمیل موفق، checkpoint را پاک کن
            self.clear_checkpoint()
            
            # چک کردن پیغام‌های جدید بعد از هر step
            self.check_and_create_message_task()
            
            return can_continue
        except Exception as e:
            self.websocket_manager.send_to_user_with_format(self.system_user.id, WebSocketType.ERROR_IN_PROCESS, {
                'user_id': self.user_id
            })
            traceback.print_exc()
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

    def create_employee(self, name, employee_description, description):
        create_user(self.db, name, self.user_id, None, employee_description)
        self.db.commit()
        GroupService().update_group_for_websockets(self.user)

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

            self.current_task = new_task
            self.websocket_manager.send_to_user_with_format(self.system_user.id, WebSocketType.USER_UPDATE,
                                                            new_task.summary())
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
        self.websocket_manager.send_to_user_with_format(self.system_user.id, WebSocketType.USER_UPDATE,
                                                        new_task.summary())
        return f"Created task {new_task.id} (current task continues)"

    def check_and_create_message_task(self):
        """
        چک کردن پیغام‌های خوانده نشده و ایجاد task در صورت لزوم
        این متد بعد از هر step اجرا میشه
        """
        try:
            unread_count = self.db.query(Message).filter(
                Message.user_id == self.user_id,
                Message.is_read == False
            ).count()
            
            if unread_count > 0:
                # چک کن آیا task مربوط به check messages وجود داره
                existing_task = self.db.query(Task).filter(
                    Task.user_id == self.user_id,
                    Task.description.like("%Check and respond to manager messages%"),
                    Task.status.in_([TaskStatus.NEW, TaskStatus.IN_PROGRESS])
                ).first()
                
                if not existing_task:
                    # Task جدید با priority بالا بساز
                    message_task = Task(
                        description=f"Check and respond to manager messages ({unread_count} unread)",
                        user_id=self.user_id,
                        priority=1000,  # Priority بالا
                        status=TaskStatus.NEW
                    )
                    self.db.add(message_task)
                    self.db.commit()
                    
                    print(f"✓ Created message task for user {self.user_id}: {unread_count} unread messages")
                    
                    # ارسال notification به WebSocket
                    self.websocket_manager.send_to_user_with_format(
                        self.system_user.id, 
                        WebSocketType.USER_UPDATE,
                        message_task.summary()
                    )
        except Exception as e:
            print(f"Error in check_and_create_message_task: {str(e)}")
            traceback.print_exc()

    def read_and_respond_messages(self, description):
        """
        خواندن پیغام‌های مدیر و پاسخ دادن هوشمندانه
        این متد توسط agent وقتی task "Check messages" رو میبینه، اجرا میشه
        """
        try:
            # 1. دریافت پیغام‌های unread
            unread_messages = self.db.query(Message).filter(
                Message.user_id == self.user_id,
                Message.is_read == False
            ).order_by(Message.created_at).all()
            
            if not unread_messages:
                return "No unread messages"
            
            print(f"Processing {len(unread_messages)} unread messages for user {self.user_id}")
            
            # 2. Agent باید RocketChat رو باز کنه و پیغام‌ها رو ببینه
            # این قسمت باید توسط vision و action model انجام بشه
            # برای الان فرض میکنیم که agent رفته و پیغام‌ها رو دیده
            
            # 3. تحلیل پیغام‌ها با AI و تصمیم‌گیری
            tasks_created = []
            for message in unread_messages:
                try:
                    response = self.analyze_manager_message(message)
                    
                    # اگر نیاز به task بود، ساخته میشه
                    if response.get('needs_task'):
                        from orchestrator.services.task_service import TaskService
                        new_task = TaskService.create_task(
                            self.db, 
                            response['task_description'], 
                            self.user_id
                        )
                        tasks_created.append(new_task.id)
                        print(f"  ✓ Created task {new_task.id}: {response['task_description']}")
                    
                    # پیغام رو به عنوان read علامت بزن
                    message.is_read = True
                    message.is_processed = True
                    
                except Exception as e:
                    print(f"Error processing message {message.id}: {str(e)}")
                    continue
            
            self.db.commit()
            
            result_msg = f"Processed {len(unread_messages)} messages"
            if tasks_created:
                result_msg += f", created {len(tasks_created)} new tasks"
            
            print(f"✓ {result_msg}")
            return result_msg
            
        except Exception as e:
            print(f"Error in read_and_respond_messages: {str(e)}")
            traceback.print_exc()
            return f"Error processing messages: {str(e)}"

    def analyze_manager_message(self, message: Message):
        """
        تحلیل پیغام مدیر با AI برای تشخیص نیاز به task یا پاسخ ساده
        
        Args:
            message: Message object
        
        Returns:
            dict با کلیدهای needs_task, task_description, response_message
        """
        try:
            prompt = f'''Analyze this manager message and determine if it requires action:

Manager: "{message.content}"

Determine:
1. Is this a task/request that needs action? (yes/no)
2. If yes, what should the task description be?
3. What is the appropriate response message?

Examples:
- "Can you check the sales report?" → needs_task: true, task: "Review and analyze sales report"
- "Good job on yesterday's work!" → needs_task: false, response: "Thank you!"
- "Please install Python on your machine" → needs_task: true, task: "Install Python programming language"

Respond in valid JSON format:
{{
    "needs_task": true/false,
    "task_description": "detailed task description if needs_task is true, otherwise empty",
    "response_message": "appropriate response to manager"
}}
'''
            
            # استفاده از action_model برای تحلیل
            # فرض: action_model.call میتونه بدون screenshot هم کار کنه برای text analysis
            response = action_model.call(prompt, screenshot_path=None, tools={})
            
            # پارس کردن JSON response
            if isinstance(response, tuple):
                response = response[0]
            
            # تلاش برای parse JSON
            try:
                result = json.loads(response)
            except json.JSONDecodeError:
                # اگر JSON نبود، سعی کن از text استخراج کنی
                import re
                
                needs_task = 'true' in response.lower() and 'needs_task' in response.lower()
                
                # سعی در استخراج task_description
                task_match = re.search(r'"task_description":\s*"([^"]+)"', response)
                task_desc = task_match.group(1) if task_match else message.content
                
                result = {
                    "needs_task": needs_task,
                    "task_description": task_desc if needs_task else "",
                    "response_message": "Message received and processed"
                }
            
            return result
            
        except Exception as e:
            print(f"Error analyzing message: {str(e)}")
            traceback.print_exc()
            # Fallback: فرض کن همه پیغام‌ها task هستند
            return {
                "needs_task": True,
                "task_description": message.content,
                "response_message": "I'll work on this"
            }

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

    def _classify_element(self, element_image):
        """
        طبقه‌بندی نوع المان UI بر اساس ویژگی‌های تصویر
        """
        # تبدیل به تصویر خاکستری
        gray = cv2.cvtColor(element_image, cv2.COLOR_BGR2GRAY)

        # محاسبه ویژگی‌های ساده
        aspect_ratio = element_image.shape[1] / element_image.shape[0]
        area = element_image.shape[0] * element_image.shape[1]

        # تشخیص نوع المان بر اساس ویژگی‌ها
        if aspect_ratio > 3:
            return "textfield"
        elif area < 1000:
            return "icon"
        elif aspect_ratio < 0.5:
            return "button"
        else:
            return "window"

    def save_checkpoint(self, stage, data):
        """
        ذخیره checkpoint برای امکان resume در صورت خطا
        """
        try:
            checkpoint_path = DataService().get_user_checkpoint_path(self.user_id, self.step_id)
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

            checkpoint = {
                'stage': stage,
                'step_id': self.step_id,
                'task_id': self.current_task.id if self.current_task else None,
                'timestamp': datetime.now().isoformat(),
                'data': data,
                'processing_stage': self.processing_stage,
                'model_width': self.model_width,
                'model_height': self.model_height
            }

            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(checkpoint, f, ensure_ascii=False, indent=2)

            logService.append_log(self.step_id, self.current_task.id if self.current_task else 0,
                                  f'checkpoint_saved: {stage}', 'checkpoint')

        except Exception as e:
            print(f"Error saving checkpoint: {e}")

    def load_checkpoint(self):
        """
        بارگذاری آخرین checkpoint در صورت وجود
        """
        try:
            checkpoint_path = DataService().get_user_checkpoint_path(self.user_id, self.step_id)

            if os.path.exists(checkpoint_path):
                with open(checkpoint_path, 'r', encoding='utf-8') as f:
                    checkpoint = json.load(f)

                self.processing_stage = checkpoint.get('processing_stage', 'start')
                self.checkpoint_data = checkpoint.get('data', {})
                self.model_width = checkpoint.get('model_width')
                self.model_height = checkpoint.get('model_height')

                logService.append_log(self.step_id, checkpoint.get('task_id', 0),
                                      f'checkpoint_loaded: {checkpoint.get("stage")}', 'checkpoint')

                return checkpoint

        except Exception as e:
            print(f"Error loading checkpoint: {e}")

        return None

    def clear_checkpoint(self):
        """
        پاک کردن checkpoint پس از تکمیل موفق
        """
        try:
            checkpoint_path = DataService().get_user_checkpoint_path(self.user_id, self.step_id)
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)

            self.checkpoint_data = {}
            self.processing_stage = "start"

        except Exception as e:
            print(f"Error clearing checkpoint: {e}")

    def crop_element_image(self, image_path: str, coordinates) -> np.ndarray:
        """
        برش المان از تصویر اصلی - پشتیبانی از polygon و rectangle
        """
        try:
            image = cv2.imread(image_path)

            # تشخیص نوع coordinates
            if isinstance(coordinates, list) and len(coordinates) > 0:
                if isinstance(coordinates[0], list):
                    # polygon format: [[x1, y1], [x2, y2], ...]
                    bbox = self.polygon_to_bbox(coordinates)
                    x1, y1, x2, y2 = bbox
                elif len(coordinates) == 4 and isinstance(coordinates[0], (int, float)):
                    # rectangle format: [x1, y1, x2, y2]
                    x1, y1, x2, y2 = coordinates
                else:
                    raise ValueError("Invalid coordinates format")
            else:
                raise ValueError("Invalid coordinates")

            # اطمینان از مختصات معتبر
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2 = min(image.shape[1], int(x2))
            y2 = min(image.shape[0], int(y2))

            if x2 <= x1 or y2 <= y1:
                return None

            cropped = image[y1:y2, x1:x2]

            # اضافه کردن padding اگر المان خیلی کوچک باشد
            if cropped.shape[0] < 32 or cropped.shape[1] < 32:
                padding = 10
                x1_padded = max(0, x1 - padding)
                y1_padded = max(0, y1 - padding)
                x2_padded = min(image.shape[1], x2 + padding)
                y2_padded = min(image.shape[0], y2 + padding)
                cropped = image[y1_padded:y2_padded, x1_padded:x2_padded]

            return cropped

        except Exception as e:
            print(f"Error cropping element: {e}")
            return None

    def create_discovery_image(self, original_image_path: str, element_crop: np.ndarray, coordinates: list) -> str:
        """
        ایجاد تصویر ترکیبی برای discovery (تصویر اصلی + المان crop شده)
        """
        try:
            # بارگذاری تصویر اصلی
            original = cv2.imread(original_image_path)

            # resize المان crop شده
            crop_height, crop_width = element_crop.shape[:2]
            max_crop_size = 200

            if crop_height > max_crop_size or crop_width > max_crop_size:
                scale = max_crop_size / max(crop_height, crop_width)
                new_width = int(crop_width * scale)
                new_height = int(crop_height * scale)
                element_crop = cv2.resize(element_crop, (new_width, new_height))

            # ایجاد border قرمز دور المان crop شده
            bordered_crop = cv2.copyMakeBorder(
                element_crop, 5, 5, 5, 5,
                cv2.BORDER_CONSTANT,
                value=[0, 0, 255]  # قرمز
            )

            # resize تصویر اصلی اگر لازم باشد
            orig_height, orig_width = original.shape[:2]
            max_original_width = 800

            if orig_width > max_original_width:
                scale = max_original_width / orig_width
                new_width = int(orig_width * scale)
                new_height = int(orig_height * scale)
                original = cv2.resize(original, (new_width, new_height))

            # رسم شکل قرمز در موقعیت المان در تصویر اصلی (polygon یا rectangle)
            if isinstance(coordinates, list) and len(coordinates) > 0:
                if isinstance(coordinates[0], list):
                    # polygon format: [[x1, y1], [x2, y2], ...]
                    scaled_points = []
                    if orig_width > max_original_width:
                        scale = max_original_width / orig_width
                        for point in coordinates:
                            scaled_x = int(point[0] * scale)
                            scaled_y = int(point[1] * scale)
                            scaled_points.append([scaled_x, scaled_y])
                    else:
                        scaled_points = coordinates

                    # تبدیل به format مناسب cv2
                    polygon_points = np.array(scaled_points, np.int32)
                    polygon_points = polygon_points.reshape((-1, 1, 2))
                    cv2.polylines(original, [polygon_points], True, (0, 0, 255), 3)

                elif len(coordinates) == 4:
                    # rectangle format: [x1, y1, x2, y2]
                    x1, y1, x2, y2 = coordinates
                    if orig_width > max_original_width:
                        scale = max_original_width / orig_width
                        x1, y1, x2, y2 = int(x1 * scale), int(y1 * scale), int(x2 * scale), int(y2 * scale)

                    cv2.rectangle(original, (x1, y1), (x2, y2), (0, 0, 255), 3)

            # ترکیب دو تصویر کنار هم
            crop_resized_height, crop_resized_width = bordered_crop.shape[:2]
            original_height, original_width = original.shape[:2]

            # تنظیم ارتفاع برای ترکیب
            if original_height != crop_resized_height:
                if original_height > crop_resized_height:
                    # pad کردن crop
                    pad_top = (original_height - crop_resized_height) // 2
                    pad_bottom = original_height - crop_resized_height - pad_top
                    bordered_crop = cv2.copyMakeBorder(
                        bordered_crop, pad_top, pad_bottom, 0, 0,
                        cv2.BORDER_CONSTANT, value=[255, 255, 255]
                    )
                else:
                    # resize کردن original
                    scale = crop_resized_height / original_height
                    new_width = int(original_width * scale)
                    original = cv2.resize(original, (new_width, crop_resized_height))

            # ترکیب تصاویر
            combined = np.hstack((original, bordered_crop))

            # ذخیره تصویر ترکیبی
            timestamp = str(int(time.time()))
            discovery_image_path = f"orchestrator_data/discovery/discovery_{self.user_id}_{timestamp}.png"
            os.makedirs(os.path.dirname(discovery_image_path), exist_ok=True)
            cv2.imwrite(discovery_image_path, combined)

            return discovery_image_path

        except Exception as e:
            print(f"Error creating discovery image: {e}")
            return None

    def discover_unknown_element(self, element_name: str, coordinates: list, image_path: str) -> dict:
        """
        فرآیند Discovery: شناسایی المان‌های ناشناخته با vision model
        """
        try:
            # ذخیره عکس crop شده برای لاگ شروع discovery
            element_crop = self.crop_element_image(image_path, coordinates)
            element_crop_start_path = None
            if element_crop is not None:
                element_crop_start_path = f"orchestrator_data/discovery/start_crop_{self.user_id}_{self.step_id}_{element_name}.png"
                os.makedirs(os.path.dirname(element_crop_start_path), exist_ok=True)
                cv2.imwrite(element_crop_start_path, element_crop)

            logService.append_log(
                self.step_id,
                self.current_task.id if self.current_task else 0,
                f"Starting discovery for element: {element_name} at coordinates: {coordinates}",
                'discovery_start',
                element_crop_start_path
            )

            # WebSocket notification
            self.websocket_manager.send_to_user_with_format(
                self.system_user.id,
                WebSocketType.DISCOVERY_ELEMENT_START,
                {'user_id': self.user_id, 'element_name': element_name, 'coordinates': coordinates}
            )

            # بررسی memory برای المان - استفاده از element_crop که قبلاً ایجاد شده
            if element_crop is None:
                return {"error": "Could not crop element"}

            memory_service = get_element_memory_service()
            element_signature = memory_service.generate_element_signature(element_crop, coordinates)

            # بررسی وجود در memory
            cached_element = memory_service.check_element_in_memory(element_signature)
            if cached_element:
                logService.append_log(
                    self.step_id,
                    self.current_task.id if self.current_task else 0,
                    f"Element found in memory: {json.dumps(cached_element, indent=2, ensure_ascii=False)}",
                    'discovery_memory_hit',
                    None
                )

                # WebSocket notification
                self.websocket_manager.send_to_user_with_format(
                    self.system_user.id,
                    WebSocketType.DISCOVERY_MEMORY_HIT,
                    {
                        'user_id': self.user_id,
                        'element_name': element_name,
                        'element_type': cached_element.get('element_type', 'unknown')
                    }
                )

                return {
                    "source": "memory",
                    "element_info": cached_element,
                    "signature": element_signature
                }

            # Element در memory نیست، باید discovery کنیم
            # ذخیره عکس crop شده برای لاگ
            element_crop_path = f"orchestrator_data/discovery/element_crop_{self.user_id}_{self.step_id}_{element_name}.png"
            os.makedirs(os.path.dirname(element_crop_path), exist_ok=True)
            cv2.imwrite(element_crop_path, element_crop)

            logService.append_log(
                self.step_id,
                self.current_task.id if self.current_task else 0,
                f"Element not found in memory. Starting AI discovery for signature: {element_signature}",
                'discovery_memory_miss',
                element_crop_path
            )

            # WebSocket notification
            self.websocket_manager.send_to_user_with_format(
                self.system_user.id,
                WebSocketType.DISCOVERY_MEMORY_MISS,
                {'user_id': self.user_id, 'element_name': element_name, 'signature': element_signature}
            )

            # ایجاد تصویر ترکیبی برای discovery
            discovery_image_path = self.create_discovery_image(image_path, element_crop, coordinates)
            if not discovery_image_path:
                return {"error": "Could not create discovery image"}

            # پرامپت discovery
            discovery_prompt = f"""Analyze this UI element carefully. I need you to identify what type of element this is and what it represents.

The image shows:
1. LEFT SIDE: Full screenshot with the element highlighted in a RED RECTANGLE
2. RIGHT SIDE: Cropped view of the specific element with a red border

Please tell me:
1. What type of UI element is this? (button, icon, text, file, folder, menu item, input field, etc.)
2. What is its purpose or function?
3. What text or content does it contain (if any)?
4. Is it a clickable element or just visual?
5. What category does it belong to? (navigation, action, content, decoration, etc.)

Context: This element was detected at coordinates {coordinates} in a desktop environment. The task context is: {self.current_task.description if self.current_task else 'General UI interaction'}.

Be specific and describe exactly what you see. This information will be used to interact with this element programmatically.

Respond in JSON format:
{{
    "element_type": "...",
    "purpose": "...",
    "text_content": "...",
    "is_clickable": true/false,
    "category": "...",
    "description": "...",
    "confidence": 0.0-1.0
}}"""

            logService.append_log(
                self.step_id,
                self.current_task.id if self.current_task else 0,
                discovery_prompt,
                'discovery_prompt',
                discovery_image_path
            )

            # WebSocket notification - AI Request
            self.websocket_manager.send_to_user_with_format(
                self.system_user.id,
                WebSocketType.DISCOVERY_AI_REQUEST,
                {'user_id': self.user_id, 'element_name': element_name}
            )

            # فراخوانی vision model
            response = vision_model.call(discovery_prompt, discovery_image_path)

            # WebSocket notification - AI Response
            self.websocket_manager.send_to_user_with_format(
                self.system_user.id,
                WebSocketType.DISCOVERY_AI_RESPONSE,
                {'user_id': self.user_id, 'element_name': element_name}
            )

            logService.append_log(
                self.step_id,
                self.current_task.id if self.current_task else 0,
                response,
                'discovery_response',
                None
            )

            # پردازش پاسخ
            try:
                element_info = json.loads(response)

                # اعتبارسنجی و تمیز کردن پاسخ
                element_info.setdefault("element_type", "unknown")
                element_info.setdefault("purpose", "")
                element_info.setdefault("text_content", "")
                element_info.setdefault("is_clickable", False)
                element_info.setdefault("category", "unknown")
                element_info.setdefault("description", "")
                element_info.setdefault("confidence", 0.5)

                # تمیز کردن اطلاعات برای ذخیره در memory (حذف metadata اضافی)
                clean_element_info = self.clean_element_info_for_memory(element_info)

                # ذخیره در memory
                memory_service.save_element_to_memory(element_signature, clean_element_info)

                logService.append_log(
                    self.step_id,
                    self.current_task.id if self.current_task else 0,
                    f"Discovery completed successfully. Element saved to memory: {json.dumps(element_info, indent=2, ensure_ascii=False)}",
                    'discovery_success',
                    discovery_image_path
                )

                # WebSocket notification - Element Complete
                self.websocket_manager.send_to_user_with_format(
                    self.system_user.id,
                    WebSocketType.DISCOVERY_ELEMENT_COMPLETE,
                    {
                        'user_id': self.user_id,
                        'element_name': element_name,
                        'discovered_type': element_info.get('element_type', 'unknown'),
                        'success': True
                    }
                )

                return {
                    "source": "discovery",
                    "element_info": element_info,
                    "signature": element_signature,
                    "discovery_image_path": discovery_image_path
                }

            except json.JSONDecodeError:
                # اگر پاسخ JSON نبود، پاسخ ساده ایجاد کن
                element_info = {
                    "element_type": "unknown",
                    "purpose": "Unknown element",
                    "text_content": "",
                    "is_clickable": False,
                    "category": "unknown",
                    "description": response[:200],  # اول 200 کاراکتر
                    "confidence": 0.3,
                    "raw_response": response
                }

                # تمیز کردن اطلاعات برای ذخیره در memory
                clean_element_info = self.clean_element_info_for_memory(element_info)

                memory_service.save_element_to_memory(element_signature, clean_element_info)

                logService.append_log(
                    self.step_id,
                    self.current_task.id if self.current_task else 0,
                    f"Discovery completed with non-JSON response. Element saved: {json.dumps(element_info, indent=2, ensure_ascii=False)}",
                    'discovery_partial_success',
                    discovery_image_path
                )

                # WebSocket notification - Element Complete (Partial)
                self.websocket_manager.send_to_user_with_format(
                    self.system_user.id,
                    WebSocketType.DISCOVERY_ELEMENT_COMPLETE,
                    {
                        'user_id': self.user_id,
                        'element_name': element_name,
                        'discovered_type': element_info.get('element_type', 'unknown'),
                        'success': True,
                        'partial': True
                    }
                )

                return {
                    "source": "discovery",
                    "element_info": element_info,
                    "signature": element_signature,
                    "discovery_image_path": discovery_image_path
                }

        except Exception as e:
            error_msg = f"Error in discovery process: {str(e)}\n{traceback.format_exc()}"
            logService.append_log(
                self.step_id,
                self.current_task.id if self.current_task else 0,
                error_msg,
                'discovery_error',
                None
            )
            return {"error": error_msg}

    def perform_element_discovery(self, coordinates: dict, image_path: str):
        """
        اجرای فرآیند discovery برای المان‌های ناشناخته
        """
        try:
            logService.append_log(
                self.step_id,
                self.current_task.id if self.current_task else 0,
                f"Starting element discovery phase for {len(coordinates)} elements",
                'discovery_phase_start',
                None
            )

            # WebSocket notification
            self.websocket_manager.send_to_user_with_format(
                self.system_user.id,
                WebSocketType.DISCOVERY_PHASE_START,
                {'user_id': self.user_id, 'total_elements': len(coordinates)}
            )

            # شناسایی المان‌های ناشناخته (غیر از desktop)
            unknown_elements = []
            for element_name, coords in coordinates.items():
                if element_name == "desktop":
                    continue

                # تشخیص اینکه آیا المان ناشناخته است
                if self.is_unknown_element(element_name, coords):
                    unknown_elements.append((element_name, coords))

            logService.append_log(
                self.step_id,
                self.current_task.id if self.current_task else 0,
                f"Found {len(unknown_elements)} unknown elements to discover: {[elem[0] for elem in unknown_elements]}",
                'discovery_unknown_count',
                None
            )

            # اجرای discovery برای هر المان ناشناخته
            discovered_elements = {}
            for element_name, coords in unknown_elements:
                try:
                    discovery_result = self.discover_unknown_element(element_name, coords, image_path)

                    if "error" not in discovery_result:
                        element_info = discovery_result.get("element_info", {})

                        # بهبود نام المان بر اساس discovery
                        improved_name = self.generate_improved_element_name(element_name, element_info)

                        # اگر نام بهبود یافت، مختصات را با نام جدید ذخیره کن
                        if improved_name != element_name:
                            discovered_elements[improved_name] = coords
                            discovered_elements[f"{improved_name}_discovery_info"] = element_info

                            logService.append_log(
                                self.step_id,
                                self.current_task.id if self.current_task else 0,
                                f"Element '{element_name}' improved to '{improved_name}' based on discovery",
                                'discovery_element_improved',
                                None
                            )
                        else:
                            # نام تغییر نکرد، فقط اطلاعات discovery را اضافه کن
                            discovered_elements[f"{element_name}_discovery_info"] = element_info

                except Exception as e:
                    logService.append_log(
                        self.step_id,
                        self.current_task.id if self.current_task else 0,
                        f"Error discovering element '{element_name}': {str(e)}",
                        'discovery_element_error',
                        None
                    )

            # اضافه کردن المان‌های discovered به مختصات اصلی
            coordinates.update(discovered_elements)

            logService.append_log(
                self.step_id,
                self.current_task.id if self.current_task else 0,
                f"Discovery phase completed. Added {len(discovered_elements)} new entries to coordinates",
                'discovery_phase_complete',
                None
            )

            # WebSocket notification
            self.websocket_manager.send_to_user_with_format(
                self.system_user.id,
                WebSocketType.DISCOVERY_PHASE_COMPLETE,
                {
                    'user_id': self.user_id,
                    'discovered_count': len(discovered_elements),
                    'total_unknown': len(unknown_elements)
                }
            )

        except Exception as e:
            logService.append_log(
                self.step_id,
                self.current_task.id if self.current_task else 0,
                f"Error in discovery phase: {str(e)}\n{traceback.format_exc()}",
                'discovery_phase_error',
                None
            )

    def is_unknown_element(self, element_name: str, coordinates: list) -> bool:
        """
        تشخیص اینکه آیا المان ناشناخته است یا نه
        """
        # المان‌های عمومی که نیاز به discovery ندارند
        known_patterns = [
            'desktop', 'browser_window', 'terminal_window', 'start_menu',
            'system_clock', 'taskbar', 'notification_area'
        ]

        # بررسی الگوهای شناخته شده
        for pattern in known_patterns:
            if pattern in element_name.lower():
                return False

        # المان‌های YOLO اصلی (با شماره) معمولاً ناشناخته هستند
        if any(pattern in element_name for pattern in ['button_', 'text_element_', 'ui_', 'desktop_icon_']):
            return True

        # المان‌های با نام‌های غیرواضح
        generic_patterns = ['element', 'object', 'item', 'unknown', 'detected']
        if any(pattern in element_name.lower() for pattern in generic_patterns):
            return True

        return False

    def generate_improved_element_name(self, original_name: str, element_info: dict) -> str:
        """
        تولید نام بهبود یافته برای المان بر اساس اطلاعات discovery
        """
        try:
            element_type = element_info.get("element_type", "").lower()
            text_content = element_info.get("text_content", "").strip()
            purpose = element_info.get("purpose", "").lower()

            # اگر متن مشخصی دارد، از آن استفاده کن
            if text_content:
                # تمیز کردن متن برای استفاده در نام
                clean_text = "".join(c for c in text_content if c.isalnum() or c in ['_', '-']).lower()
                if clean_text:
                    if element_type in ['file', 'folder']:
                        return f"{clean_text}_{element_type}"
                    elif element_type == 'button':
                        return f"{clean_text}_button"
                    else:
                        return f"{clean_text}_{element_type}"

            # اگر نوع مشخصی دارد
            if element_type and element_type != "unknown":
                if "button" in original_name:
                    return f"{element_type}_button"
                elif "icon" in original_name:
                    return f"{element_type}_icon"
                else:
                    return f"{element_type}_element"

            # اگر purpose مشخصی دارد
            if purpose and "unknown" not in purpose:
                purpose_clean = "".join(c for c in purpose if c.isalnum() or c in ['_', '-']).lower()
                if purpose_clean:
                    return f"{purpose_clean}_element"

            # اگر هیچ بهبودی ممکن نیست، نام اصلی را برگردان
            return original_name
        except Exception as e:
            return original_name

    def filter_coordinates_for_vision(self, coordinates: dict) -> dict:
        """
        فیلتر کردن coordinates برای vision model - حذف discovery metadata و اطلاعات اضافی
        """
        filtered_coords = {}

        for element_name, coords in coordinates.items():
            # حذف discovery info entries
            if "_discovery_info" in element_name:
                continue

            # حذف detailed character/word analysis
            if any(suffix in element_name for suffix in [
                "_char_", "_word_", "_full_text", "_text_content",
                "_char_count", "_word_count", "_cursor"
            ]):
                continue

            # نگه داشتن فقط main elements با coordinates (polygon یا rectangle)
            if isinstance(coords, list) and len(coords) > 0:
                try:
                    if isinstance(coords[0], list):
                        # polygon format: [[x1, y1], [x2, y2], ...]
                        clean_coords = []
                        for point in coords:
                            if len(point) == 2:
                                clean_coords.append([int(point[0]), int(point[1])])
                        if len(clean_coords) >= 3:  # حداقل 3 نقطه برای polygon معتبر
                            filtered_coords[element_name] = clean_coords
                    elif len(coords) == 4 and isinstance(coords[0], (int, float)):
                        # rectangle format: [x1, y1, x2, y2]
                        clean_coords = [int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])]
                        filtered_coords[element_name] = clean_coords
                except (ValueError, TypeError):
                    # اگر coords قابل تبدیل به int نباشند، skip کن
                    continue

        return filtered_coords

    def clean_element_info_for_memory(self, element_info: dict) -> dict:
        """
        تمیز کردن اطلاعات المان قبل از ذخیره در memory - حذف metadata اضافی
        """
        # فیلدهای مجاز برای ذخیره در memory
        allowed_fields = {
            "element_type",
            "purpose",
            "text_content",
            "is_clickable",
            "category",
            "description",
            "confidence"
        }

        clean_info = {}
        for key, value in element_info.items():
            if key in allowed_fields:
                clean_info[key] = value

        return clean_info

    def perform_element_discovery_for_vision(self, coordinates: dict, image_path: str) -> dict:
        """
        اجرای فرآیند discovery برای vision model - برگرداندن enhanced coordinates با توضیحات
        پشتیبانی از desktop, mobile, tablet و polygon coordinates از SAM
        """
        try:
            logService.append_log(
                self.step_id,
                self.current_task.id if self.current_task else 0,
                f"Starting enhanced element discovery for vision with {len(coordinates)} elements",
                'discovery_for_vision_start',
                None
            )

            # WebSocket notification
            self.websocket_manager.send_to_user_with_format(
                self.system_user.id,
                WebSocketType.DISCOVERY_PHASE_START,
                {'user_id': self.user_id, 'total_elements': len(coordinates), 'for_vision': True}
            )

            enhanced_coords = {}
            discovery_results = {}

            # تشخیص نوع platform بر اساس ابعاد تصویر
            real_width, real_height = self.get_image_dimensions(image_path)
            platform_type = self.detect_platform_type(real_width, real_height)

            # Scale coordinates from SAM model size to original image size
            scaled_coordinates = self.scale_sam_coordinates_to_original(coordinates, real_width, real_height)

            # رسم polygons برای coordinates مقیاس‌شده و ذخیره در لاگ
            scaled_output_path = DataService().get_user_path_screenshot_with_coordinates_scaled(self.user_id,
                                                                                                self.step_id)
            scaled_details = self.draw_polygons_on_image(
                DataService().get_user_path_screenshot(self.user_id),
                scaled_coordinates,
                scaled_output_path
            )

            # لاگ اطلاعات scaled coordinates
            if scaled_details != {}:
                logService.append_log(self.step_id, self.current_task.id,
                                      f'Scaled coordinates: real_width: {real_width}, real_height: {real_height}, Scaled elements: {len(scaled_coordinates)}, detail: {scaled_details}',
                                      'scaled_coordinates',
                                      scaled_output_path)

            logService.append_log(
                self.step_id,
                self.current_task.id if self.current_task else 0,
                f"Detected platform type: {platform_type} (resolution: {real_width}x{real_height}), scaled {len(scaled_coordinates)} coordinates from SAM",
                'discovery_platform_detection',
                None
            )

            # شناسایی المان‌های ناشناخته و discovery
            for element_name, coords in scaled_coordinates.items():
                # فیلتر پلتفرم‌های مختلف - نه فقط desktop
                platform_elements = ["desktop", "mobile", "tablet", "large_element_segment"]
                if any(platform in element_name.lower() for platform in platform_elements):
                    # برای المان‌های پلتفرم، ساختار ساده برگردان
                    enhanced_coords[element_name] = self.create_platform_element_structure(element_name, coords,
                                                                                           platform_type)
                    continue

                # فیلتر کردن detailed analysis از YOLO/SAM (char_, word_, etc.)
                if any(suffix in element_name for suffix in [
                    "_char_", "_word_", "_full_text", "_text_content",
                    "_char_count", "_word_count", "_cursor"
                ]):
                    continue

                # پردازش coordinates - پشتیبانی از polygon و rectangle
                processed_coords = self.process_coordinates(coords, element_name)
                if not processed_coords:
                    continue

                # تشخیص اینکه آیا المان نیاز به discovery دارد
                if self.is_unknown_element(element_name, processed_coords):
                    try:
                        discovery_result = self.discover_unknown_element(element_name, processed_coords, image_path)

                        if "error" not in discovery_result:
                            element_info = discovery_result.get("element_info", {})

                            # بهبود نام المان بر اساس platform type
                            improved_name = self.generate_improved_element_name_with_platform(element_name,
                                                                                              element_info,
                                                                                              platform_type)

                            # ساخت enhanced element با توضیحات کامل
                            enhanced_element = {
                                "coordinates": processed_coords,
                                "original_coordinates": coords,  # نگه داشتن مختصات اصلی
                                "original_name": element_name,
                                "improved_name": improved_name,
                                "element_type": element_info.get("element_type", "unknown"),
                                "description": element_info.get("description", ""),
                                "purpose": element_info.get("purpose", ""),
                                "text_content": element_info.get("text_content", ""),
                                "is_clickable": element_info.get("is_clickable", False),
                                "category": element_info.get("category", "unknown"),
                                "confidence": element_info.get("confidence", 0.5),
                                "discovery_source": discovery_result.get("source", "unknown"),
                                "platform_type": platform_type,
                                "coordinate_type": "polygon" if self.is_polygon_coordinates(coords) else "rectangle",
                                "scaled_from_sam": True
                            }

                            # استفاده از نام بهبود یافته به عنوان کلید
                            enhanced_coords[improved_name] = enhanced_element
                            discovery_results[element_name] = discovery_result

                            logService.append_log(
                                self.step_id,
                                self.current_task.id if self.current_task else 0,
                                f"Enhanced element '{element_name}' → '{improved_name}': {element_info.get('element_type', 'unknown')} (Platform: {platform_type})",
                                'discovery_element_enhanced',
                                discovery_result.get("discovery_image_path")
                            )
                        else:
                            # اگر discovery خطا داشت، المان اصلی را نگه دار
                            enhanced_coords[element_name] = self.create_fallback_element_structure(element_name,
                                                                                                   processed_coords,
                                                                                                   coords,
                                                                                                   platform_type,
                                                                                                   "discovery_failed")

                    except Exception as e:
                        logService.append_log(
                            self.step_id,
                            self.current_task.id if self.current_task else 0,
                            f"Error discovering element '{element_name}': {str(e)}",
                            'discovery_element_error',
                            None
                        )
                        # در صورت خطا، المان اصلی را نگه دار
                        enhanced_coords[element_name] = self.create_fallback_element_structure(element_name,
                                                                                               processed_coords, coords,
                                                                                               platform_type,
                                                                                               "discovery_error")
                else:
                    # المان شناخته شده - فقط coordinates اضافه کن
                    enhanced_coords[element_name] = {
                        "coordinates": processed_coords,
                        "original_coordinates": coords,
                        "original_name": element_name,
                        "improved_name": element_name,
                        "element_type": self.classify_known_element_type(element_name),
                        "description": f"Known UI element: {element_name}",
                        "purpose": "System UI element",
                        "text_content": "",
                        "is_clickable": True,
                        "category": "system_ui",
                        "confidence": 0.9,
                        "discovery_source": "known",
                        "platform_type": platform_type,
                        "coordinate_type": "polygon" if self.is_polygon_coordinates(coords) else "rectangle",
                        "scaled_from_sam": True
                    }

            # WebSocket notification
            self.websocket_manager.send_to_user_with_format(
                self.system_user.id,
                WebSocketType.DISCOVERY_PHASE_COMPLETE,
                {
                    'user_id': self.user_id,
                    'discovered_count': len(discovery_results),
                    'total_enhanced': len(enhanced_coords),
                    'platform_type': platform_type,
                    'for_vision': True
                }
            )

            logService.append_log(
                self.step_id,
                self.current_task.id if self.current_task else 0,
                f"Enhanced discovery completed. Platform: {platform_type}, Enhanced {len(enhanced_coords)} elements, discovered {len(discovery_results)} new elements",
                'discovery_for_vision_complete',
                None
            )

            return enhanced_coords

        except Exception as e:
            logService.append_log(
                self.step_id,
                self.current_task.id if self.current_task else 0,
                f"Error in enhanced discovery for vision: {str(e)}\n{traceback.format_exc()}",
                'discovery_for_vision_error',
                None
            )
            # در صورت خطا، coordinates اصلی را برگردان
            return self.filter_coordinates_for_vision(coordinates)

    def scale_sam_coordinates_to_original(self, coordinates: dict, real_width: int, real_height: int) -> dict:
        """
        Scale کردن coordinates از سایز مدل SAM به سایز واقعی تصویر
        """
        try:
            # محاسبه سایز مدل SAM (بر اساس کد sam.py)
            sam_max_size = 1024

            # محاسبه scale factor مدل SAM
            original_max_dim = max(real_width, real_height)
            if original_max_dim > sam_max_size:
                sam_scale_factor = sam_max_size / original_max_dim
                sam_width = int(real_width * sam_scale_factor)
                sam_height = int(real_height * sam_scale_factor)
            else:
                # اگر تصویر اصلی کوچک‌تر از max_size باشد، scale نشده
                sam_width = real_width
                sam_height = real_height

            # محاسبه scale factors برای بازگرداندن به سایز اصلی
            scale_x = real_width / sam_width
            scale_y = real_height / sam_height

            logService.append_log(
                self.step_id,
                self.current_task.id if self.current_task else 0,
                f"SAM scaling info: Original({real_width}x{real_height}) -> SAM({sam_width}x{sam_height}) -> Scale factors(x:{scale_x:.3f}, y:{scale_y:.3f})",
                'sam_coordinate_scaling',
                None
            )

            scaled_coordinates = {}

            for element_name, coords in coordinates.items():
                try:
                    if self.is_polygon_coordinates(coords):
                        # Scale polygon coordinates: [[x1, y1], [x2, y2], ...]
                        scaled_polygon = []
                        for point in coords:
                            if len(point) >= 2:
                                scaled_x = int(point[0] * scale_x)
                                scaled_y = int(point[1] * scale_y)
                                # اطمینان از اینکه coordinates در محدوده تصویر باشند
                                scaled_x = max(0, min(real_width - 1, scaled_x))
                                scaled_y = max(0, min(real_height - 1, scaled_y))
                                scaled_polygon.append([scaled_x, scaled_y])

                        if len(scaled_polygon) >= 3:  # حداقل 3 نقطه برای polygon معتبر
                            scaled_coordinates[element_name] = scaled_polygon

                    elif isinstance(coords, tuple) and len(coords) == 2:
                        # Scale tuple coordinates: ((x1, y1), (x2, y2))
                        x1, y1 = coords[0]
                        x2, y2 = coords[1]

                        scaled_x1 = int(x1 * scale_x)
                        scaled_y1 = int(y1 * scale_y)
                        scaled_x2 = int(x2 * scale_x)
                        scaled_y2 = int(y2 * scale_y)

                        # اطمینان از مختصات معتبر
                        scaled_x1 = max(0, min(real_width - 1, scaled_x1))
                        scaled_y1 = max(0, min(real_height - 1, scaled_y1))
                        scaled_x2 = max(0, min(real_width - 1, scaled_x2))
                        scaled_y2 = max(0, min(real_height - 1, scaled_y2))

                        scaled_coordinates[element_name] = ((scaled_x1, scaled_y1), (scaled_x2, scaled_y2))

                    elif isinstance(coords, list) and len(coords) == 4:
                        # Scale rectangle coordinates: [x1, y1, x2, y2]
                        x1, y1, x2, y2 = coords

                        scaled_x1 = int(x1 * scale_x)
                        scaled_y1 = int(y1 * scale_y)
                        scaled_x2 = int(x2 * scale_x)
                        scaled_y2 = int(y2 * scale_y)

                        # اطمینان از مختصات معتبر
                        scaled_x1 = max(0, min(real_width - 1, scaled_x1))
                        scaled_y1 = max(0, min(real_height - 1, scaled_y1))
                        scaled_x2 = max(0, min(real_width - 1, scaled_x2))
                        scaled_y2 = max(0, min(real_height - 1, scaled_y2))

                        scaled_coordinates[element_name] = [scaled_x1, scaled_y1, scaled_x2, scaled_y2]

                    else:
                        # فرمت ناشناخته، coordinate اصلی را نگه دار
                        scaled_coordinates[element_name] = coords

                except Exception as e:
                    logService.append_log(
                        self.step_id,
                        self.current_task.id if self.current_task else 0,
                        f"Error scaling coordinates for element '{element_name}': {str(e)}",
                        'coordinate_scaling_error',
                        None
                    )
                    # در صورت خطا، coordinate اصلی را نگه دار
                    scaled_coordinates[element_name] = coords

            return scaled_coordinates

        except Exception as e:
            logService.append_log(
                self.step_id,
                self.current_task.id if self.current_task else 0,
                f"Error in coordinate scaling: {str(e)}",
                'coordinate_scaling_general_error',
                None
            )
            # در صورت خطا کلی، coordinates اصلی را برگردان
            return coordinates

    def detect_platform_type(self, width: int, height: int) -> str:
        """
        تشخیص نوع platform بر اساس resolution
        """
        aspect_ratio = width / height

        # Mobile (Portrait)
        if width < 800 and aspect_ratio < 1.0:
            return "mobile_portrait"

        # Mobile (Landscape)
        elif height < 800 and aspect_ratio > 1.5:
            return "mobile_landscape"

        # Tablet (Portrait)
        elif 800 <= width <= 1200 and aspect_ratio < 1.2:
            return "tablet_portrait"

        # Tablet (Landscape)
        elif 800 <= height <= 1200 and 1.2 <= aspect_ratio <= 2.0:
            return "tablet_landscape"

        # Desktop/Large screens
        elif width >= 1200 or height >= 900:
            return "desktop"

        # Default
        return "unknown"

    def is_polygon_coordinates(self, coords) -> bool:
        """
        تشخیص اینکه coordinates به صورت polygon هستند یا rectangle
        """
        if not isinstance(coords, list):
            return False

        # اگر اولین element خودش list باشد، polygon است
        if len(coords) > 0 and isinstance(coords[0], list):
            return True

        return False

    def process_coordinates(self, coords, element_name: str):
        """
        پردازش coordinates - پشتیبانی از polygon و rectangle
        """
        try:
            if self.is_polygon_coordinates(coords):
                # Polygon format: [[x1, y1], [x2, y2], ...]
                if len(coords) < 3:  # حداقل 3 نقطه برای polygon
                    return None

                # محاسبه bounding box از polygon
                x_coords = [point[0] for point in coords if len(point) >= 2]
                y_coords = [point[1] for point in coords if len(point) >= 2]

                if not x_coords or not y_coords:
                    return None

                min_x, max_x = min(x_coords), max(x_coords)
                min_y, max_y = min(y_coords), max(y_coords)

                # برگرداندن به صورت [x1, y1, x2, y2] برای سازگاری
                return [int(min_x), int(min_y), int(max_x), int(max_y)]

            else:
                # Rectangle format: [x1, y1, x2, y2] یا ((x1, y1), (x2, y2))
                if isinstance(coords, tuple) and len(coords) == 2:
                    # ((x1, y1), (x2, y2)) format
                    return [int(coords[0][0]), int(coords[0][1]), int(coords[1][0]), int(coords[1][1])]
                elif isinstance(coords, list) and len(coords) == 4:
                    # [x1, y1, x2, y2] format
                    return [int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])]
                else:
                    return None

        except (ValueError, TypeError, IndexError):
            return None

    def create_platform_element_structure(self, element_name: str, coords, platform_type: str) -> dict:
        """
        ایجاد ساختار برای المان‌های platform
        """
        processed_coords = self.process_coordinates(coords, element_name)
        return {
            "coordinates": processed_coords if processed_coords else coords,
            "original_coordinates": coords,
            "original_name": element_name,
            "improved_name": f"{platform_type}_{element_name}",
            "element_type": "platform_background",
            "description": f"Platform background element for {platform_type}",
            "purpose": "Background/container element",
            "text_content": "",
            "is_clickable": False,
            "category": "platform",
            "confidence": 1.0,
            "discovery_source": "platform_detection",
            "platform_type": platform_type,
            "coordinate_type": "polygon" if self.is_polygon_coordinates(coords) else "rectangle"
        }

    def create_fallback_element_structure(self, element_name: str, processed_coords, original_coords,
                                          platform_type: str, error_type: str) -> dict:
        """
        ایجاد ساختار fallback برای المان‌هایی که discovery شان خطا داشته
        """
        return {
            "coordinates": processed_coords,
            "original_coordinates": original_coords,
            "original_name": element_name,
            "improved_name": element_name,
            "element_type": "unknown",
            "description": f"Element discovery failed: {error_type}",
            "purpose": "Unknown element",
            "text_content": "",
            "is_clickable": False,
            "category": "unknown",
            "confidence": 0.1,
            "discovery_source": error_type,
            "platform_type": platform_type,
            "coordinate_type": "polygon" if self.is_polygon_coordinates(original_coords) else "rectangle"
        }

    def generate_improved_element_name_with_platform(self, element_name: str, element_info: dict,
                                                     platform_type: str) -> str:
        """
        بهبود نام المان با در نظر گیری platform type
        """
        base_name = self.generate_improved_element_name(element_name, element_info)
        element_type = element_info.get("element_type", "unknown")

        # اضافه کردن prefix platform برای بهتر شدن نام‌گذاری
        if element_type in ["button", "icon", "menu", "file", "folder"]:
            return f"{platform_type}_{base_name}"

        return base_name

    def classify_known_element_type(self, element_name: str) -> str:
        """
        طبقه‌بندی المان‌های شناخته شده بر اساس نام
        """
        element_name_lower = element_name.lower()

        if 'desktop' in element_name_lower:
            return 'desktop'
        elif 'browser' in element_name_lower or 'firefox' in element_name_lower:
            return 'browser'
        elif 'terminal' in element_name_lower:
            return 'terminal'
        elif 'button' in element_name_lower:
            return 'button'
        elif 'icon' in element_name_lower:
            return 'icon'
        elif 'window' in element_name_lower:
            return 'window'
        elif 'menu' in element_name_lower:
            return 'menu'
        elif 'clock' in element_name_lower or 'time' in element_name_lower:
            return 'clock'
        elif 'taskbar' in element_name_lower:
            return 'taskbar'
        else:
            return 'ui_element'

    def getCoordinatesFromDetector(self):
        """
        استفاده از detector script در container برای دریافت اطلاعات المان‌های UI
        """
        try:
            CommandService.run_command_via_container(
                'gsettings set org.gnome.desktop.interface toolkit-accessibility true', self.user_id)
            CommandService.run_command_via_container('export XAUTHORITY=/home/ubuntu/.Xauthority', self.user_id)
            output = CommandService.run_command_via_container("python3 /root/Desktop/orchestrator/detector.py",
                                                              self.user_id)

            logService.append_log(self.step_id, self.current_task.id if self.current_task else 0,
                                  f'Detector script output: {output}',
                                  'detector_script_output',
                                  None)

            if output and isinstance(output, str):
                idx = output.find("{")
                output = output[idx:]
                detector_data = json.loads(output.strip())
                logService.append_log(self.step_id, self.current_task.id if self.current_task else 0,
                                      f'Processed {len(detector_data)} elements from detector',
                                      'detector_processing_complete',
                                      None)

                return json.dumps(detector_data, ensure_ascii=False)
            else:
                logService.append_log(self.step_id, self.current_task.id if self.current_task else 0,
                                      f'Invalid detector output: {output}',
                                      'detector_error',
                                      None)
                raise Exception("Invalid detector output")

        except json.JSONDecodeError as e:
            logService.append_log(self.step_id, self.current_task.id if self.current_task else 0,
                                  f'JSON parse error from detector: {str(e)}',
                                  'detector_json_error',
                                  None)
            raise e

        except Exception as e:
            logService.append_log(self.step_id, self.current_task.id if self.current_task else 0,
                                  f'Error running detector: {str(e)}',
                                  'detector_script_error',
                                  None)
            raise e





    def draw_rectangles_on_image(self, input_image_path, data, output_image_path=None):
        """
        روی عکس ورودی rectangle های رنگی برای applications و elements رسم می‌کند
        بر اساس ساختار داده جدید
        """
        try:
            # باز کردن عکس
            if isinstance(input_image_path, str):
                img = Image.open(input_image_path)
            else:
                img = input_image_path.copy()

            # ایجاد شیء Draw
            draw = ImageDraw.Draw(img)

            # خواندن ابعاد واقعی عکس
            real_width, real_height = self.get_image_dimensions(input_image_path)

            detail = {}
            drawn_count = 0

            # تعریف رنگ‌ها برای انواع مختلف
            colors = {
                "application": "blue",        # اپلیکیشن - آبی
                "interactive": "red",         # المان‌های تعاملی - قرمز
                "frame": "lime",              # فریم - سبز روشن
                "panel": "green",             # پنل‌ها - سبز
                "menu_bar": "purple",         # منو بار - بنفش
                "tool_bar": "cyan",           # تول بار - فیروزه‌ای
                "filler": "yellow",           # فیلر - زرد
                "element": "orange",          # المان‌های عادی - نارنجی
                "unknown": "gray"             # المان‌های ناشناخته - خاکستری
            }

            # دریافت اطلاعات desktop
            desktop_info = data.get("desktop_info", {})
            active_window_id = desktop_info.get("active_window_id", "")
            screen_resolution = desktop_info.get("screen_resolution", {})
            
            logService.append_log(self.step_id, self.current_task.id if self.current_task else 0,
                                  f'Drawing rectangles for screen resolution: {screen_resolution}, active window: {active_window_id}',
                                  'draw_rectangles_start',
                                  None)

            # بررسی اینکه داده شامل applications است
            if "applications" in data:
                applications = data["applications"]
                
                for app_idx, application in enumerate(applications):
                    try:
                        app_name = application.get("app_name", "unknown")
                        elements = application.get("elements", [])
                        element_count = application.get("element_count", 0)
                        interactive_count = application.get("interactive_count", 0)
                        window_info = application.get("window_info", {})

                        # رسم window اصلی اگر window_info موجود باشد
                        if window_info and "bounds" in window_info:
                            bounds = window_info["bounds"]
                            window_title = window_info.get("title", "")
                            layer = window_info.get("layer", 0)
                            
                            x = bounds.get("x", 0)
                            y = bounds.get("y", 0)
                            width = bounds.get("width", 0)
                            height = bounds.get("height", 0)
                            
                            x1, y1 = x, y
                            x2, y2 = x + width, y + height

                            # اطمینان از اینکه مختصات در محدوده تصویر هستند
                            x1 = max(0, min(real_width - 1, x1))
                            y1 = max(0, min(real_height - 1, y1))
                            x2 = max(0, min(real_width, x2))
                            y2 = max(0, min(real_height, y2))

                            if x2 > x1 and y2 > y1:
                                # رسم مستطیل اپلیکیشن
                                color = colors["application"]
                                line_width = 3
                                
                                draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=line_width)
                                
                                # اضافه کردن نام اپلیکیشن و اطلاعات
                                display_name = f"{app_name}"
                                if window_title and window_title != app_name:
                                    display_name += f"_{window_title}"
                                display_name = display_name[:30]  # محدود کردن طول نام
                                
                                # اضافه کردن تعداد elements
                                info_text = f"{display_name} ({element_count}e/{interactive_count}i)"
                                draw.text((x1 + 5, y1 + 5), info_text, fill=color)
                                
                                detail[f"{app_name}_application_{app_idx}"] = {
                                    "coords": [x1, y1, x2, y2],
                                    "center_point": [(x1 + x2) // 2, (y1 + y2) // 2],
                                    "app_name": app_name,
                                    "title": window_title,
                                    "layer": layer,
                                    "element_count": element_count,
                                    "interactive_count": interactive_count
                                }
                                drawn_count += 1

                        # رسم elements داخل اپلیکیشن
                        for elem_idx, element in enumerate(elements):
                            try:
                                # استخراج اطلاعات element با ساختار جدید
                                position = element.get("position", [0, 0])
                                size = element.get("size", [0, 0])
                                center = element.get("center", [0, 0])
                                depth = element.get("depth", 1)
                                layer = element.get("layer", 1)
                                is_interactive = element.get("is_interactive", False)
                                visible = element.get("visible", True)
                                name = element.get("name", "")
                                role = element.get("role", "unknown")
                                window_title = element.get("window_title", "")
                                
                                # محاسبه مختصات از position و size
                                elem_x, elem_y = position[0], position[1]
                                elem_width, elem_height = size[0], size[1]
                                center_x, center_y = center[0], center[1]
                                
                                if elem_width > 2 and elem_height > 2 and visible:  # فقط elements قابل مشاهده و با اندازه مناسب
                                    ex1, ey1 = elem_x, elem_y
                                    ex2, ey2 = elem_x + elem_width, elem_y + elem_height

                                    # اطمینان از محدوده تصویر
                                    ex1 = max(0, min(real_width - 1, ex1))
                                    ey1 = max(0, min(real_height - 1, ey1))
                                    ex2 = max(0, min(real_width, ex2))
                                    ey2 = max(0, min(real_height, ey2))

                                    if ex2 > ex1 and ey2 > ey1:
                                        # تعیین رنگ بر اساس role و interactive بودن
                                        if is_interactive:
                                            elem_color = colors["interactive"]
                                            elem_width_line = 3
                                        elif role == "frame":
                                            elem_color = colors["frame"]
                                            elem_width_line = 2
                                        elif role == "panel":
                                            elem_color = colors["panel"]
                                            elem_width_line = 1
                                        elif role == "menu bar":
                                            elem_color = colors["menu_bar"]
                                            elem_width_line = 2
                                        elif role == "tool bar":
                                            elem_color = colors["tool_bar"]
                                            elem_width_line = 2
                                        elif role == "filler":
                                            elem_color = colors["filler"]
                                            elem_width_line = 1
                                        elif role == "unknown":
                                            elem_color = colors["unknown"]
                                            elem_width_line = 1
                                        else:
                                            elem_color = colors["element"]
                                            elem_width_line = 1
                                        
                                        # رسم مستطیل element
                                        draw.rectangle([(ex1, ey1), (ex2, ey2)], outline=elem_color, width=elem_width_line)
                                        
                                        # رسم نقطه مرکزی برای elements تعاملی یا مهم
                                        if is_interactive or role == "frame":
                                            # اطمینان از اینکه center در محدوده تصویر است
                                            center_x_bounded = max(0, min(real_width - 1, center_x))
                                            center_y_bounded = max(0, min(real_height - 1, center_y))
                                            
                                            # رسم دایره کوچک در مرکز
                                            circle_size = 4 if is_interactive else 2
                                            draw.ellipse([
                                                center_x_bounded - circle_size, center_y_bounded - circle_size,
                                                center_x_bounded + circle_size, center_y_bounded + circle_size
                                            ], fill=elem_color)
                                        
                                        # اضافه کردن نام element اگر وجود دارد و مهم است
                                        if name and len(name.strip()) > 0 and (is_interactive or role == "frame"):
                                            name_text = name[:20]  # محدود کردن طول نام
                                            # محاسبه موقعیت متن تا از مرز خارج نشود
                                            text_x = min(ex1 + 2, real_width - 100)
                                            text_y = min(ey1 + 2, real_height - 20)
                                            draw.text((text_x, text_y), name_text, fill=elem_color)
                                        
                                        detail[f"{app_name}_element_{elem_idx}"] = {
                                            "coords": [ex1, ey1, ex2, ey2],
                                            "center_point": [center_x, center_y],
                                            "position": position,
                                            "size": size,
                                            "role": role,
                                            "name": name,
                                            "depth": depth,
                                            "layer": layer,
                                            "is_interactive": is_interactive,
                                            "visible": visible,
                                            "window_title": window_title,
                                            "app_name": app_name
                                        }
                                        drawn_count += 1

                            except Exception as e:
                                logService.append_log(self.step_id, self.current_task.id if self.current_task else 0,
                                                      f'Error drawing element {elem_idx} in application {app_name}: {str(e)}',
                                                      'draw_element_error',
                                                      None)
                                continue

                    except Exception as e:
                        logService.append_log(self.step_id, self.current_task.id if self.current_task else 0,
                                              f'Error drawing application {app_idx}: {str(e)}',
                                              'draw_application_error',
                                              None)
                        continue

            # اضافه کردن اطلاعات summary به detail
            summary = data.get("summary", {})
            detail["summary"] = {
                "total_applications": summary.get("total_applications", 0),
                "total_elements": summary.get("total_elements", 0),
                "total_interactive_elements": summary.get("total_interactive_elements", 0),
                "drawn_rectangles": drawn_count,
                "screen_resolution": screen_resolution,
                "active_window_id": active_window_id
            }

            # ذخیره یا نمایش
            if output_image_path:
                os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
                img.save(output_image_path)

                logService.append_log(self.step_id, self.current_task.id if self.current_task else 0,
                                      f'Successfully drew {drawn_count} rectangles on image. Applications: {summary.get("total_applications", 0)}, Elements: {summary.get("total_elements", 0)}, Interactive: {summary.get("total_interactive_elements", 0)}. Saved to: {output_image_path}',
                                      'draw_rectangles_complete',
                                      output_image_path)
            else:
                img.show()

            return detail

        except Exception as e:
            logService.append_log(self.step_id, self.current_task.id if self.current_task else 0,
                                  f'Error in draw_rectangles_on_image: {str(e)}',
                                  'draw_rectangles_error',
                                  None)
            return {}



    def prefilter_detector_data(self, detector_data: dict, task_description: str = "") -> dict:
        """
        Pre-filter detector data to reduce size before compression
        Focus on interactive elements and elements relevant to task
        """
        try:
            filtered_data = {
                'desktop_info': detector_data.get('desktop_info', {}),
                'applications': [],
                'summary': {}
            }
            
            active_window_id = detector_data.get('desktop_info', {}).get('active_window_id', '')
            task_lower = task_description.lower()
            
            # Keywords for relevance check
            task_keywords = set(task_lower.split())
            
            for app in detector_data.get('applications', []):
                app_name = app.get('app_name', '').lower()
                window_info = app.get('window_info', {})
                window_title = window_info.get('title', '').lower()
                
                # Check if this is the active window
                is_active_window = (window_info.get('bounds', {}) and 
                                   active_window_id != '')
                
                # Check if app/window is relevant to task
                is_relevant = (
                    any(keyword in app_name for keyword in task_keywords if len(keyword) > 2) or
                    any(keyword in window_title for keyword in task_keywords if len(keyword) > 2) or
                    is_active_window or
                    'caja' in app_name  # Always include file manager (desktop icons)
                )
                
                if not is_relevant:
                    continue
                
                filtered_elements = []
                for element in app.get('elements', []):
                    element_name = element.get('name', '').lower()
                    element_role = element.get('role', '').lower()
                    is_interactive = element.get('is_interactive', False)
                    
                    # Include element if:
                    # 1. Interactive
                    # 2. Has meaningful name relevant to task
                    # 3. Important roles (frame, icon, button, entry, menu)
                    # 4. From active window
                    should_include = (
                        is_interactive or
                        element_role in ['frame', 'icon', 'button', 'push button', 'entry', 'menu', 'menu item', 'text'] or
                        any(keyword in element_name for keyword in task_keywords if len(keyword) > 2) or
                        (is_active_window and element_role not in ['filler', 'panel', 'scroll pane'])
                    )
                    
                    if should_include:
                        # Simplify element data - keep only essential fields
                        filtered_element = {
                            'name': element.get('name', ''),
                            'role': element_role,
                            'position': element.get('position', [0, 0]),
                            'size': element.get('size', [0, 0]),
                            'center': element.get('center', [0, 0]),
                            'is_interactive': is_interactive,
                            'window_title': element.get('window_title', '')
                        }
                        filtered_elements.append(filtered_element)
                
                if filtered_elements:
                    filtered_app = {
                        'app_name': app.get('app_name', ''),
                        'elements': filtered_elements,
                        'element_count': len(filtered_elements),
                        'interactive_count': len([e for e in filtered_elements if e.get('is_interactive', False)]),
                        'window_info': window_info
                    }
                    filtered_data['applications'].append(filtered_app)
            
            # Update summary
            total_elements = sum(app['element_count'] for app in filtered_data['applications'])
            total_interactive = sum(app.get('interactive_count', 0) for app in filtered_data['applications'])
            
            filtered_data['summary'] = {
                'total_applications': len(filtered_data['applications']),
                'total_elements': total_elements,
                'total_interactive_elements': total_interactive
            }
            
            # Log filtering results
            original_apps = len(detector_data.get('applications', []))
            original_elements = sum(app.get('element_count', 0) for app in detector_data.get('applications', []))
            filtered_apps = len(filtered_data['applications'])
            filtered_elements = total_elements
            
            reduction_ratio = round((1 - filtered_elements / max(original_elements, 1)) * 100, 1)
            
            logService.append_log(self.step_id, self.current_task.id if self.current_task else 0,
                                  f"Pre-filtered detector data: {original_apps} apps ({original_elements} elements) -> {filtered_apps} apps ({filtered_elements} elements) - {reduction_ratio}% reduction",
                                  "detector_prefiltering",
                                  None)
            
            return filtered_data
            
        except Exception as e:
            logService.append_log(self.step_id, self.current_task.id if self.current_task else 0,
                                  f"Error in prefilter_detector_data: {str(e)}. Using original data.",
                                  "prefiltering_error",
                                  None)
            return detector_data

    def create_coordinate_summary(self, compressed_json: str) -> str:
        """
        Create human-readable summary from compressed coordinates
        """
        try:
            data = json.loads(compressed_json)
            
            summary_lines = []
            summary_lines.append(f"Screen: {data.get('screen', {}).get('width', 0)}x{data.get('screen', {}).get('height', 0)}")
            
            # Active window summary
            active = data.get('active_window')
            if active:
                summary_lines.append(f"\nActive Window: {active.get('title', active.get('app', 'Unknown'))}")
                interactive_count = sum(1 for e in active.get('elements', []) if e.get('i', False))
                summary_lines.append(f"  - {len(active.get('elements', []))} elements ({interactive_count} interactive)")
                
                # List key interactive elements
                for elem in active.get('elements', [])[:10]:  # Max 10 elements
                    if elem.get('i') or elem.get('n'):
                        name = elem.get('n', 'unnamed')
                        role = elem.get('r', 'unknown')
                        center = elem.get('c', [0, 0])
                        summary_lines.append(f"    • {name} ({role}) at center {center}")
            
            # Other windows summary (only if interactive)
            other_windows = data.get('windows', [])
            if other_windows:
                summary_lines.append(f"\nOther Windows ({len(other_windows)}):")
                for window in other_windows[:3]:  # Max 3 windows
                    interactive_count = sum(1 for e in window.get('elements', []) if e.get('i', False))
                    if interactive_count > 0:
                        summary_lines.append(f"  - {window.get('title', window.get('app', 'Unknown'))}: {interactive_count} interactive elements")
            
            return '\n'.join(summary_lines)
            
        except Exception as e:
            return f"Coordinate data available (parsing error: {str(e)[:50]})"

    def _process_outcome_validation(self, vision_response: str):
        """
        پردازش outcome validation از vision response و ذخیره mistake در صورت لزوم
        """
        try:
            # اگر اطلاعات last action وجود نداشت، چیزی برای validate کردن نیست
            if not self.last_action_info or not self.last_action_info.get('expected_outcome'):
                return
            
            response_lower = vision_response.lower()
            
            # تشخیص outcome_validation از response
            # جستجوی کلمات کلیدی که نشان دهنده validation منفی هستند
            negative_indicators = [
                'outcome_validation: no',
                'outcome_validation: partially',
                'outcome_validation:no',
                'outcome_validation:partially',
                'did the expected outcome happen? no',
                'did the expected outcome happen? partially',
                'expected outcome: no',
                'validation: no',
                'validation: partially',
                'needs_correction: yes',
                'needs_correction:yes',
                'unexpected',
                'wrong page',
                'wrong window',
                'wrong tab',
                'unwanted tab',
                'unwanted window',
                'new tab opened',
                'blank tab',
                'opened password',
                'opened passwords',
                'password manager',
                'passwords — mozilla firefox',
                'mozilla firefox password'
            ]
            
            is_negative = any(indicator in response_lower for indicator in negative_indicators)
            
            if is_negative:
                # استخراج actual_outcome از response
                actual_outcome = self._extract_actual_outcome(vision_response)
                
                # استخراج window title از vision response (اگر موجود باشد)
                window_title_after = self._extract_window_title(vision_response)
                
                # ساخت action_context برای ذخیره mistake
                action_context = {
                    'action_type': self.last_action_info.get('action_type', 'unknown'),
                    'coordinates': self.last_action_info.get('coordinates'),
                    'window_title_after': window_title_after,
                    'timestamp': self.last_action_info.get('timestamp')
                }
                
                # استخراج suggested_solution از response (اگر وجود داشت)
                suggested_solution = self._extract_corrective_action(vision_response)
                
                # ذخیره mistake
                mistake_id = self.shared_learning.save_mistake(
                    action_context=action_context,
                    expected_outcome=self.last_action_info.get('expected_outcome', ''),
                    actual_outcome=actual_outcome,
                    problem_description=f"Action {action_context['action_type']} did not produce expected outcome",
                    suggested_solution=suggested_solution
                )
                
                print(f"✗ Mistake detected and saved: {mistake_id}")
                print(f"  Expected: {self.last_action_info.get('expected_outcome')}")
                print(f"  Actual: {actual_outcome}")
                
                # لاگ کردن mistake
                logService.append_log(
                    self.step_id, 
                    self.current_task.id if self.current_task else 0,
                    f"Mistake detected - ID: {mistake_id}, Expected: {self.last_action_info.get('expected_outcome')}, Actual: {actual_outcome}",
                    'mistake_detected',
                    None
                )
            else:
                print(f"✓ Action outcome validated successfully")
                
        except Exception as e:
            print(f"Error in outcome validation processing: {e}")
            traceback.print_exc()
    
    def _extract_actual_outcome(self, vision_response: str) -> str:
        """استخراج actual_outcome از vision response"""
        # جستجوی patterns مختلف برای actual outcome
        patterns = [
            r'actual_outcome:\s*(.+?)(?:\n|$)',
            r'actually happened:\s*(.+?)(?:\n|$)',
            r'what actually happened:\s*(.+?)(?:\n|$)',
            r'instead:\s*(.+?)(?:\n|$)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, vision_response, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).strip()
        
        # اگر pattern پیدا نشد، سعی کن از context استنباط کنی
        response_lower = vision_response.lower()
        
        if 'password' in response_lower and 'manager' in response_lower:
            return "Password manager page opened instead of expected action"
        elif 'new tab' in response_lower:
            return "New tab opened unexpectedly"
        elif 'blank' in response_lower and 'tab' in response_lower:
            return "Blank tab opened instead of expected action"
        elif 'tab' in response_lower and ('opened' in response_lower or 'unexpected' in response_lower):
            return "Unexpected tab opened"
        elif 'wrong' in response_lower and ('page' in response_lower or 'window' in response_lower):
            return "Wrong page/window displayed"
        
        return "Unexpected outcome occurred"
    
    def _extract_window_title(self, vision_response: str) -> str:
        """استخراج window title از vision response"""
        # جستجوی patterns برای window title
        patterns = [
            r'active window:\s*["\']?([^"\'\n]+)["\']?',
            r'window:\s*["\']?([^"\'\n]+)["\']?',
            r'title:\s*["\']?([^"\'\n]+)["\']?'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, vision_response, re.IGNORECASE)
            if match:
                title = match.group(1).strip()
                if len(title) > 5:  # حداقل طول معقول
                    return title
        
        return "Unknown"
    
    def _extract_corrective_action(self, vision_response: str) -> str:
        """استخراج corrective action از vision response"""
        # جستجوی patterns برای solution
        patterns = [
            r'corrective action[s]?:\s*(.+?)(?:\n\n|$)',
            r'solution:\s*(.+?)(?:\n\n|$)',
            r'to fix:\s*(.+?)(?:\n\n|$)',
            r'should:\s*(.+?)(?:\n\n|$)',
            r'first action.*?:\s*(.+?)(?:\n|$)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, vision_response, re.IGNORECASE | re.DOTALL)
            if match:
                solution = match.group(1).strip()[:200]
                if solution and len(solution) > 10:  # حداقل طول معقول
                    return solution
        
        # راه‌حل‌های پیش‌فرض بر اساس context و window title
        response_lower = vision_response.lower()
        
        # Firefox password manager
        if 'password' in response_lower and ('manager' in response_lower or 'firefox' in response_lower):
            return "Close password manager tab using send_key(['Control_L', 'w']) and return to login page"
        
        # New Tab opened
        if 'new tab' in response_lower or 'blank tab' in response_lower:
            return "Close new tab using send_key(['Control_L', 'w']) to return to previous page"
        
        # Generic unexpected tab/window
        if ('tab' in response_lower or 'window' in response_lower) and ('unexpected' in response_lower or 'wrong' in response_lower or 'opened' in response_lower):
            return "Close unwanted tab/window using send_key(['Control_L', 'w'])"
        
        # Popup or dialog
        if 'popup' in response_lower or 'dialog' in response_lower or 'modal' in response_lower:
            return "Dismiss popup using send_key(['Escape'])"
        
        # Wrong page loaded
        if 'wrong page' in response_lower or 'incorrect page' in response_lower:
            return "Click on correct browser tab to switch back, or use send_key(['Alt_L', 'Left']) to go back"
        
        return "Review situation and use appropriate corrective action (close tab, dismiss popup, or switch tab)"

    def compress_coordinates_for_vision(self, detector_coordinates: dict, task_description: str = "") -> str:
        """
        Compress detector coordinates to a minimal structured format for vision model
        Already pre-filtered data, now create human-readable compact summary
        """
        try:
            # Build compact summary structure
            compressed = {
                'screen': detector_coordinates.get('desktop_info', {}).get('screen_resolution', {}),
                'active_window': None,
                'windows': []
            }
            
            active_window_id = detector_coordinates.get('desktop_info', {}).get('active_window_id', '')
            
            for app in detector_coordinates.get('applications', []):
                window_info = app.get('window_info', {})
                
                # Create window summary
                window_summary = {
                    'app': app.get('app_name', ''),
                    'title': window_info.get('title', ''),
                    'bounds': window_info.get('bounds', {}),
                    'elements': []
                }
                
                # Add compact element info
                for element in app.get('elements', []):
                    # Create ultra-compact element representation
                    elem_compact = {
                        'n': element.get('name', '')[:30],  # name (truncated)
                        'r': element.get('role', ''),  # role
                        'p': element.get('position', []),  # position
                        's': element.get('size', []),  # size
                        'c': element.get('center', []),  # center
                        'i': element.get('is_interactive', False)  # interactive
                    }
                    
                    # Only include if has useful info
                    if elem_compact['n'] or elem_compact['i']:
                        window_summary['elements'].append(elem_compact)
                
                # Determine if this is active window
                is_active = (window_info.get('layer', 0) == 0 or 
                           'active' in window_info.get('title', '').lower())
                
                if is_active and not compressed['active_window']:
                    compressed['active_window'] = window_summary
                else:
                    # Only add if has interactive elements
                    if any(e['i'] for e in window_summary['elements']):
                        compressed['windows'].append(window_summary)
            
            # Convert to compact JSON string
            compressed_str = json.dumps(compressed, separators=(',', ':'), ensure_ascii=False)
            
            # Log compression results
            original_size = len(json.dumps(detector_coordinates))
            compressed_size = len(compressed_str)
            compression_ratio = round((1 - compressed_size / original_size) * 100, 1)
            
            logService.append_log(self.step_id, self.current_task.id if self.current_task else 0,
                                  f"Compressed coordinates: {original_size} -> {compressed_size} chars ({compression_ratio}% reduction)",
                                  "coordinates_compression",
                                  None)
            
            return compressed_str
            
        except Exception as e:
            logService.append_log(self.step_id, self.current_task.id if self.current_task else 0,
                                  f"Error compressing coordinates: {str(e)}. Using simplified fallback.",
                                  "compression_error",
                                  None)
            # Fallback: ultra-minimal format
            return json.dumps({
                'apps': [app.get('app_name', '') for app in detector_coordinates.get('applications', [])]
            }, separators=(',', ':'))

