import os
import tempfile
import cv2
import numpy as np
import pytesseract

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

from orchestrator.models import SystemUser
from orchestrator.models.WebSocketType import WebSocketType
from orchestrator.models.base import SessionLocal
from orchestrator.models.task import TaskStatus, Task, TaskMessage
from orchestrator.services.command_service import CommandService
from orchestrator.services.config_service import grounding_model, vision_model, action_model, position_model
from orchestrator.services.data_service import DataService
from orchestrator.services.element_memory_service import get_element_memory_service
from orchestrator.models.memory import Memory
from orchestrator.services.group_service import GroupService
from orchestrator.services.log_service import logService
from orchestrator.services.rag_service import RAGSystem
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
        "description": "Send keyboard key. The main keys are: Enter Escape Shift Alt Ctrl Tab",
        "parameters": {
            "name": "Key name",
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
        self.step_id = 0
        self.websocket_manager = websocket_manager
        
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

    def send_key(self, name, description):
        return CommandService.send_key(name, self.user_id)

    def append_screenshot(self):
        time.sleep(1)
        CommandService.screenshot(self.user_id)
        enhanced_prompt = self.task_rag.query_context(
            self.user_id, f"Memory data: {self.current_task.description}"
        )

        # پردازش درخواست position با استفاده از تابع process_position_request
        coordinates = self.process_position_request(
            self.user_id,
            DataService().get_user_path_screenshot(self.user_id),
            DataService().get_user_path_screenshot_with_coordinates(self.user_id, self.step_id)
        )

        # Discovery Phase: شناسایی المان‌های ناشناخته
        # این مرحله پس از position_request انجام میشه تا discovery data در position تداخل نکنه
        self.perform_element_discovery(coordinates, DataService().get_user_path_screenshot(self.user_id))
        
        # فیلتر کردن coordinates برای vision model (حذف discovery metadata)
        clean_coordinates = self.filter_coordinates_for_vision(coordinates)

        user = self.db.query(User).filter(User.id == self.user_id).first()
        username = user.name.replace(' ', '_').lower()
        vision_prompt = f'''Analyze this screenshot with extreme precision and provide a detailed response in this EXACT format:

        ENHANCED CHARACTER-LEVEL ANALYSIS:
        You have access to detailed coordinate information including:
        - Individual character positions and content
        - Word-level coordinates and text
        - Cursor positions when visible
        - Text field states (focused/unfocused, selection)
        - Element-level content analysis
        
        Use this information to provide:
        - Exact text content for each element
        - Character-by-character positioning data
        - Cursor location within text fields
        - Text selection ranges if any
        - Focus states and visual indicators
        - Content validation states
        - Character count and text length for fields
        - Word boundaries and spacing information

        CRITICAL FILE IDENTIFICATION RULES:
        1. DISTINGUISH FILES FROM UI TEXT:
           - Look for file icons with extensions (.txt, .pdf, .doc, etc.)
           - Desktop files are typically smaller icons (16x16 to 64x64) with file names
           - UI text labels are part of application interface (terminal prompts, status text, etc.)
           - File manager or desktop context indicates actual files
           - Check for file-like visual appearance (document icons, folder icons)
        
        2. TASK-AWARE FILE DETECTION:
           - Current task mentions: "{self.current_task.description}"
           - Prioritize identifying files mentioned in the task objective
           - Look specifically for files with names matching task requirements
           - Consider file location context (desktop vs application UI)
        
        3. VALIDATION CRITERIA:
           - Files should have file extensions in their visual representation
           - Files are typically on desktop, in file managers, or folder windows
           - Terminal/application text labels are NOT files - they are UI elements
           - Files can be double-clicked to open, labels are static text

        Current Objective: {self.current_task.description}
        Priority: {self.current_task.priority}

        Screen Analysis:
        - [Windows]: [
            [
                "window_name": "exact name of the window",
                "state": "active/minimized/maximized/hidden",
                "coordinates": [100, 50, 800, 600],
                "content": "detailed description of window content",
                "interactive_elements": [
                    "element_name at coordinates 150,75",
                    "element_name at coordinates 200,100"
                ],
                "status": "ready/busy/error"
            ]
        ]
        
        - [ALL UI Elements]: [
            [
                "element_name": "exact name of the element",
                "type": "button/textfield/icon/link/file/folder/label/etc",
                "coordinates": [50, 100, 200, 150],
                "state": "enabled/disabled/selected/hovered",
                "visibility": "fully_visible/partially_visible/hidden",
                "interaction_method": "click/double_click/type/etc",
                "text_content": "exact text content if any",
                "file_properties": [
                    "is_actual_file": "true/false",
                    "file_extension": "file extension if applicable",
                    "file_name": "complete file name if it's a file",
                    "location_context": "desktop/file_manager/application_ui",
                    "visual_file_indicator": "description of file icon or visual cues"
                ],
                "element_classification": [
                    "element_category": "file/folder/ui_label/button/icon/window/etc",
                    "confidence_level": "high/medium/low",
                    "classification_reason": "why this element is classified this way",
                    "task_relevance": "high/medium/low/none"
                ],
                "character_details": [
                    [
                        "char": "individual character",
                        "position": "character position index",
                        "coordinates": [55, 105, 65, 125],
                        "font_size": "estimated font size",
                        "confidence": "OCR confidence level"
                    ]
                ],
                "word_details": [
                    [
                        "word": "complete word",
                        "word_index": "word position index",
                        "coordinates": [55, 105, 120, 125],
                        "character_count": "number of characters"
                    ]
                ],
                "cursor_info": [
                    "position": "cursor position in text (character index)",
                    "coordinates": [75, 110, 77, 130],
                    "visible": "true/false",
                    "blinking": "true/false"
                ],
                "selection_info": [
                    "has_selection": "true/false",
                    "start_position": "selection start character index",
                    "end_position": "selection end character index",
                    "selected_text": "text that is selected",
                    "selection_coordinates": [60, 110, 150, 130]
                ],
                "focus_state": [
                    "is_focused": "true/false",
                    "focus_indicators": ["list of visual focus indicators"],
                    "border_color": "border color if focused",
                    "background_color": "background color"
                ],
                "field_properties": [
                    "field_type": "input/textarea/password/rich_editor",
                    "placeholder_text": "placeholder content if visible",
                    "required": "true/false",
                    "readonly": "true/false",
                    "multiline": "true/false",
                    "max_length": "maximum character limit if visible",
                    "current_length": "current character count",
                    "validation_state": "valid/invalid/error",
                    "error_message": "error message if any",
                    "auto_complete": "auto-complete suggestions if visible"
                ]
            ]
        ]
        
        - [Files and Folders Detected]: [
            [
                "file_name": "exact file name with extension",
                "file_type": "file/folder",
                "coordinates": [250, 300, 350, 330],
                "location": "desktop/file_manager/folder_path",
                "file_extension": "extension if applicable",
                "task_relevance": "mentioned_in_task/related/unrelated",
                "visual_indicators": ["file icon", "extension visible", "file-like appearance"],
                "accessibility": "can_double_click/read_only/locked"
            ]
        ]
        
        - [Result of last action]: {{
            "success": true/false,
            "details": "detailed description of what happened",
            "problems": ["list of any issues encountered"],
            "suggestions": ["list of potential solutions"],
            "rollback_steps": ["steps to undo if needed"]
        }}
        
        - [Screen Dimensions]: {{
            "width": "exact width in pixels",
            "height": "exact height in pixels",
            "resolution": "screen resolution",
            "scale_factor": "UI scaling factor if any"
        }}

        Task Status: {{
            "status": "complete/not_complete/in_progress/error",
            "completion_percentage": "estimated completion percentage",
            "blockers": ["list of any blocking issues"],
            "next_milestone": "next major step to complete"
        }}
        
        Memory Storage: {{
            "learned_patterns": ["patterns discovered during task execution"],
            "successful_actions": ["actions that worked well"],
            "failed_actions": ["actions that didn't work"],
            "environment_state": "current state of the system",
            "user_preferences": "any user-specific preferences discovered"
        }}

        Next Actions (list ALL required actions in order):

        1. Action: {{
            "type": "single_click/double_click/right_click/type_text/send_key/wait/create_task/stop/move_mouse/create_employee/scroll_down/scroll_up",
            "target": {{
                "element_name": "exact name of target",
                "coordinates": [400, 500, 550, 530],
                "interaction_point": "475,515",
                "character_position": "specific character index for cursor placement",
                "word_position": "specific word index for interaction",
                "text_content": "current text content of target",
                "cursor_location": "current cursor position if applicable",
                "element_type": "file/folder/button/textfield/icon/label/etc",
                "file_validation": {{
                    "is_actual_file": "true/false",
                    "matches_task_requirement": "true/false",
                    "file_extension_visible": "true/false",
                    "location_appropriate": "true/false"
                }}
            }},
            "detail": "precise description of the action",
            "reason": "detailed explanation of why this action is needed",
            "text_editing_details": {{
                "edit_type": "insert/replace/append/delete/select",
                "target_text": "text to insert or modify",
                "cursor_placement": "before_char_X/after_char_X/at_word_X/start/end",
                "selection_range": "start_char_X_to_char_Y",
                "preserve_formatting": "true/false",
                "validate_after": "true/false"
            }},
            "pre_actions": [
                {{
                    "type": "action type",
                    "target": "target details",
                    "reason": "why this pre-action is needed",
                    "cursor_requirements": "cursor positioning needs",
                    "focus_requirements": "focus state needs"
                }}
            ],
            "post_actions": [
                {{
                    "type": "action type",
                    "target": "target details",
                    "reason": "why this post-action is needed",
                    "validation_steps": "validation actions needed",
                    "cursor_final_position": "where cursor should end up"
                }}
            ],
            "thought": "detailed reasoning behind this action",
            "expected_outcome": "what should happen after this action",
            "fallback_plan": "what to do if this action fails",
            "character_level_precision": {{
                "required": "true/false",
                "target_character_index": "specific character position",
                "insertion_point": "600,650",
                "selection_precision": "character-level selection details"
            }}
        }}

        CRITICAL RULES:
        1. ALWAYS maintain this exact format
        2. List ALL required actions in proper sequence
        3. Include ALL necessary pre/post actions
        4. PRIORITIZE files mentioned in task context over generic UI text
        5. VERIFY file authenticity before treating as file
        6. For terminal interactions:
           - Ensure terminal is focused first
           - Include appropriate wait times
           - Verify command execution
        7. For system commands:
           - Check for errors in response
           - Verify command success
        8. Be specific about targets and reasons
        9. For open apps and files you MUST use double_click instead of single_click
        10. For errors and something like them you can use create_task to create new task with higher priority
        11. If a task with higher priority is created via create_task:
            - Immediately terminate the current task's action list
            - Only output the new task (no other actions or tasks)
            - Do NOT add subsequent actions to the original task
        12. When describing coordinates or locations, provide all four corner points (top-left, top-right, bottom-right, bottom-left)
        13. Include ALL details in screen analysis (press enter, clicks, etc.)
        14. Consider system state changes (windows closed, OS shutdown, etc.)
        15. Report ALL command line content in window details
        16. Actions must be explicit and deterministic—no conditionals
        17. For opening apps, use application menu, desktop, or terminal
        18. Review Recent Actions before executing any UI or system changes
        19. For downloading big files, check ftp://ftp-server:20 (username: myuser, password: mypassword)
        20. Monitor http://rocketchat:3000 for messages (username: {username}, password: mypassword)
        21. Include single_click in pre-actions when cursor position is critical
        22. Verify element visibility and accessibility before interaction
        23. Consider element hierarchy and relationships
        24. Account for UI scaling and resolution
        25. Verify action success before proceeding
        26. Include error handling and recovery steps
        27. Consider system performance and response times
        28. ALWAYS report mouse position separately
        29. DO NOT confuse mouse cursor with other UI elements
        30. Verify element types before reporting
        31. Use exact coordinates for all elements
        32. Double-check element names and types
        33. You SHOULD use scroll that page when we need to see the other part of the page
        34. CHARACTER-LEVEL TEXT EDITING RULES:
            - When editing text, use exact character positions from coordinate data
            - For inserting text at specific positions, reference character indices
            - For selecting text, use start and end character positions
            - For cursor placement, specify exact character location
            - Always report current text content and cursor position
            - Use word boundaries for more efficient text operations
            - Consider text field validation states before editing
            - Preserve existing formatting when possible
            - Report character-level changes in action results
        35. PRECISION REQUIREMENTS:
            - Always use character-level coordinates for text editing
            - Report exact cursor position after each text operation
            - Validate text content after modifications
            - Use word-level operations when character-level is not needed
            - Consider multi-byte characters and unicode support
            - Handle text selection ranges with character precision
            - Account for text field boundaries and limits
        36. TEXT FIELD STATE MANAGEMENT:
            - Always check focus state before text operations
            - Verify cursor visibility and position
            - Handle text selection states appropriately
            - Consider readonly and disabled field states
            - Respect field validation requirements
            - Monitor text length limits and restrictions
        37. FILE DETECTION VALIDATION:
            - Verify that identified files have appropriate visual file indicators
            - Confirm file names match task requirements before targeting
            - Distinguish between file icons and UI text elements
            - Check file location context (desktop vs application UI)
            - Validate file extensions are visible or implied by icon type

        Current task objective: {self.current_task.description}
        Priority: {self.current_task.priority}
        Coordinates of the picture: {json.dumps(clean_coordinates, indent=2, ensure_ascii=False)}
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
        return response

    def process_position_request(self, user_id: int, screenshot_path: str, output_path: str = None) -> dict:
        localCoordinates = self.getCoordinatesInLocal()
        
        # اضافه کردن task context برای کمک به تشخیص بهتر
        task_context = ""
        if self.current_task:
            task_context = f"\nTASK CONTEXT: {self.current_task.description}"
        
        positionTextRequest = f'''CRITICAL: Return ONLY valid JSON. No text before or after.

        Task: {task_context if task_context else "General UI detection"}

        MANDATORY REQUIREMENT - PRESERVE ALL ELEMENTS:
        You MUST output EXACTLY one element for each main input element. Count verification required.
        Input main elements: Elements WITHOUT these suffixes: _char_, _word_, _full_text, _text_content, _char_count, _word_count
        
        STRICT PROCESSING STEPS:
        1. SCAN all input elements
        2. IDENTIFY main elements (exclude breakdown details)
        3. FOR EACH main element, create exactly ONE output element with better name
        4. NEVER skip any main element
        5. NEVER add elements not in input

        ELEMENT FILTERING RULES:
        KEEP: "desktop", "web_browser_0", "terminal_1", "text_element_2", "desktop_icon_3", "file_manager_button_4", "start_menu_5", "button_6", "button_8", "desktops_button_9", "desktop_icon_10", "button_11", "text_element_12", "text_element_13", "clock_14", "ui_text_15", "desktop_icon_16", "firefox_minized_app_17"
        
        REMOVE: Any element with these suffixes: "_char_0_", "_word_0_", "_full_text", "_text_content", "_char_count", "_word_count"

        SMART NAME MAPPING:
        - "text_element_12" + text_content="test.txt" → "test_txt_file"
        - "text_element_13" → "text_label_13"  
        - "desktop_icon_3" → "firefox_icon"
        - "desktop_icon_7" → "terminal_icon"
        - "desktop_icon_10" → "folder_icon"
        - "desktop_icon_16" → "terminal_app_icon"
        - "web_browser_0" → "browser_window"
        - "terminal_1" → "terminal_window" 
        - "start_menu_5" → "start_menu_button"
        - "clock_14" → "system_clock"
        - "button_6" → "ui_button_6"
        - "button_8" → "ui_button_8"
        - "button_11" → "ui_button_11"
        - "file_manager_button_4" → "file_manager_button"
        - "desktops_button_9" → "desktops_button"
        - "ui_text_15" → "ui_text_label"
        - "firefox_minized_app_17" → "firefox_minimized_app"

        VERIFICATION CHECKLIST:
        ✓ Desktop element first
        ✓ Count main input elements vs output elements (must match)
        ✓ All element names improved but coordinates preserved
        ✓ No duplicate names in output
        ✓ JSON valid and parseable

        INPUT YOLO DATA:
        ''' + localCoordinates + '''

        OUTPUT: Clean JSON with ALL main elements, better names, original coordinates:'''

        path_local_coordinates = DataService().get_user_path_screenshot_with_coordinates(self.user_id, self.step_id)
        details = self.draw_rectangles_on_image(
            screenshot_path,
            json.loads(localCoordinates),
            path_local_coordinates
        )
        real_width, real_height = self.get_image_dimensions(screenshot_path)
        if details != {}:
            logService.append_log(self.step_id, self.current_task.id,
                                  f'real_width: {real_width}, real_height: {real_height}, detail: {details}',
                                  'local_coordinates',
                                  path_local_coordinates)
        logService.append_log(self.step_id, self.current_task.id, positionTextRequest, 'position_request',
                              DataService().get_user_path_screenshot(self.user_id))
        
        self.websocket_manager.send_to_user_with_format(self.system_user.id, WebSocketType.POSITION_REQUEST, {
            'user_id': self.user_id,
            'request': positionTextRequest,
        })
        
        coordinates_string = position_model.call(positionTextRequest, screenshot_path)
        
        # فرمت کردن JSON برای نمایش بهتر در لاگ
        try:
            parsed_coordinates = json.loads(coordinates_string)
            formatted_coordinates_string = json.dumps(parsed_coordinates, indent=2, ensure_ascii=False)
        except json.JSONDecodeError:
            formatted_coordinates_string = coordinates_string
        
        logService.append_log(self.step_id, self.current_task.id, formatted_coordinates_string, 'position_response', None)
        
        self.websocket_manager.send_to_user_with_format(self.system_user.id, WebSocketType.POSITION_RESPONSE, {
            'user_id': self.user_id,
            'response': coordinates_string,
        })
        coordinates = json.loads(coordinates_string)

        # رسم مستطیل‌ها روی عکس
        if output_path:
            details = self.draw_rectangles_on_image(
                screenshot_path,
                coordinates,
                output_path
            )
            real_width, real_height = self.get_image_dimensions(screenshot_path)
            if details != {}:
                logService.append_log(self.step_id, self.current_task.id,
                                      f'real_width: {real_width}, real_height: {real_height}, detail: {details}',
                                      'coordinates',
                                      output_path)

        return coordinates

    def getCoordinatesInLocal(self):
        from ultralytics import YOLO
        import os
        
        img_path = DataService().get_user_path_screenshot(self.user_id)
        
        # بارگذاری مدل YOLO کوچک‌تر
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "best.pt")
        if not os.path.exists(model_path):
            # اگر مدل سفارشی وجود نداشت، از مدل پیش‌فرض استفاده کن
            model_path = "yolov8n.pt"
        
        model = YOLO(model_path)
        
        # تنظیمات بهینه برای کاهش مصرف حافظه
        model.overrides['conf'] = 0.3  # افزایش آستانه اطمینان
        model.overrides['iou'] = 0.5   # تنظیم IoU
        model.overrides['max_det'] = 50  # کاهش تعداد تشخیص‌ها
        
        # انجام پیش‌بینی روی تصویر
        results = model(img_path)
        
        # خواندن ابعاد تصویر
        image = cv2.imread(img_path)
        coordinates = {"desktop": [0, 0, image.shape[1], image.shape[0]]}
        
        # اضافه کردن OCR کلی برای یافتن فایل‌های احتمالی
        ocr_detected_files = self.detect_files_with_ocr(image)
        
        # پردازش نتایج YOLO
        for result in results:
            if result.boxes is not None:
                for i, box in enumerate(result.boxes):
                    class_id = int(box.cls)
                    class_name = model.names[class_id] if class_id in model.names else f"class_{class_id}"
                    coords = box.xyxy[0].tolist()  # مختصات [x1, y1, x2, y2]
                    confidence = box.conf.item()
                    
                    # فقط المان‌هایی با اطمینان بالا را در نظر بگیریم
                    if confidence > 0.3:
                        # بررسی اینکه آیا این المان فایل است یا UI text
                        element_name = self.classify_element_type(
                            image, 
                            [int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])],
                            class_name, 
                            i,
                            ocr_detected_files
                        )
                        
                        coordinates[element_name] = [int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])]
                        
                        # تجزیه تفصیلی برای text field ها و المان‌های متنی
                        if any(keyword in class_name.lower() for keyword in ['text', 'input', 'field', 'label', 'button']):
                            detailed_coords = self.analyze_element_detailed(
                                image, 
                                [int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])],
                                element_name
                            )
                            coordinates.update(detailed_coords)
        
        # اضافه کردن فایل‌های شناسایی شده با OCR که توسط YOLO یافت نشده‌اند
        for file_info in ocr_detected_files:
            file_name = file_info['name']
            file_coords = file_info['coords']
            
            # بررسی اینکه آیا این فایل قبلاً توسط YOLO شناسایی شده یا نه
            overlap_found = False
            for existing_coords in coordinates.values():
                if self.check_overlap(file_coords, existing_coords, threshold=0.5):
                    overlap_found = True
                    break
            
            if not overlap_found:
                coordinates[file_name] = file_coords
        
        return json.dumps(coordinates)

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
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
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
                            char_key = f"{element_name}_char_{len(characters)-1}_{char_text}"
                            detailed_coords[char_key] = [abs_char_x1, abs_char_y1, abs_char_x2, abs_char_y2]
                            
                            # جمع‌آوری کلمات
                            if data['word_num'][i] == data['word_num'][i-1] if i > 0 else True:
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
                cursor_pos = self.detect_cursor_position(element_image, characters)
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

    def detect_cursor_position(self, element_image, characters):
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
                return [last_char['coordinates'][2] - x1, last_char['coordinates'][1] - y1, 2, last_char['font_size']]
            
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

    def draw_rectangles_on_image(self, input_image_path, coordinates, output_image_path=None):
        """
        روی عکس ورودی مستطیل‌های قرمز برای هر المان در مختصات مشخص شده رسم می‌کند
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
        desktop = None
        for coordinate in coordinates.keys():
            if 'desktop' == coordinate:
                desktop = coordinates[coordinate]
        if desktop is None:
            return {}
        scale_x = real_width / desktop[2]  # تقسیم عرض واقعی بر عرض مدل
        scale_y = real_height / desktop[3]  # تقسیم ارتفاع واقعی بر ارتفاع مدل

        detail = {}
        for element_name, coords in coordinates.items():
            try:
                if element_name != "desktop":  # از رسم مستطیل برای دسکتاپ صرف نظر می‌کنیم
                    # مقیاس‌بندی مختصات
                    x1 = coords[0] * scale_x
                    y1 = coords[1] * scale_y
                    x2 = coords[2] * scale_x
                    y2 = coords[3] * scale_y
                    detail[element_name] = ((x1, y1), (x2, y2))
                    # رسم مستطیل
                    draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=2)

                    # اضافه کردن نام المان
                    draw.text((x1, y1 - 15), element_name, fill="red")
            except Exception as e:
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
            
            # مرحله 1: Screenshot و Position Request
            if self.processing_stage in ["start", "position"]:
                try:
                    self.processing_stage = "position"
                    screenshot_thought = self.append_screenshot()
                    print(f"screenshot: {screenshot_thought}")
                    
                    # ذخیره checkpoint پس از screenshot
                    self.save_checkpoint("position_completed", {
                        'screenshot_thought': screenshot_thought
                    })
                    
                except Exception as e:
                    print(f"Error in position stage: {e}")
                    # حفظ checkpoint برای تلاش مجدد
                    raise e
            else:
                # بازیابی از checkpoint
                screenshot_thought = self.checkpoint_data.get('screenshot_thought')
                print(f"Restored screenshot from checkpoint")

            # مرحله 2: Enhanced Prompt
            if self.processing_stage in ["start", "position", "enhanced_prompt"]:
                try:
                    self.processing_stage = "enhanced_prompt"
                    enhanced_prompt = self.task_rag.query_context(
                        self.user_id, f"Memory data: {self.current_task.description}"
                    )
                    
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
            if self.processing_stage in ["start", "position", "enhanced_prompt", "action"]:
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
                    5. In send_key for enter something you should use `Return` word
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

                    logService.append_log(self.step_id, self.current_task.id, enhanced_prompt + action_prompt, 'action_request',
                                          DataService().get_user_path_screenshot(self.user_id))
                    self.websocket_manager.send_to_user_with_format(self.system_user.id, WebSocketType.ACTION_REQUEST, {
                        'user_id': self.user_id,
                        'request': enhanced_prompt + action_prompt,
                    })
                    # تبدیل به فرمت استرینگ برای سازگاری با OllamaProvider
                    full_prompt = enhanced_prompt + action_prompt
                    response = action_model.call(full_prompt, DataService().get_user_path_screenshot(self.user_id), tools)
                    logService.append_log(self.step_id, self.current_task.id, json.dumps(response), 'action_response')
                    print(f'ACTION MODEL RESPONSE: {response}')
                    self.websocket_manager.send_to_user_with_format(self.system_user.id, WebSocketType.ACTION_RESPONSE, {
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
                try:
                    action = tool_call['name']
                    args = deepcopy(tool_call['parameters'])
                    memory_data = args['memory_data']
                    self.task_rag.add_text(self.user_id,
                                           "event: " + str(memory_data) + " timestamp: " + datetime.now().isoformat())
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

                        # اگر action ایجاد تسک بود و تسک جدید اولویت بالاتری داشت، باید ادامه ندهیم
                        if action == "create_task" and "switched to higher priority task" in result:
                            can_continue = False

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
                    print(f"Error processing tool call {action}: {str(e)}")
                    continue

            # در صورت تکمیل موفق، checkpoint را پاک کن
            self.clear_checkpoint()
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

    def crop_element_image(self, image_path: str, coordinates: list) -> np.ndarray:
        """
        برش المان از تصویر اصلی
        """
        try:
            image = cv2.imread(image_path)
            x1, y1, x2, y2 = coordinates
            
            # اطمینان از مختصات معتبر
            x1, y1 = max(0, x1), max(0, y1)
            x2 = min(image.shape[1], x2)
            y2 = min(image.shape[0], y2)
            
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
            
            # رسم مستطیل قرمز در موقعیت المان در تصویر اصلی
            x1, y1, x2, y2 = coordinates
            if orig_width > max_original_width:
                scale = max_original_width / orig_width
                x1, y1, x2, y2 = int(x1*scale), int(y1*scale), int(x2*scale), int(y2*scale)
            
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
            logService.append_log(
                self.step_id, 
                self.current_task.id if self.current_task else 0,
                f"Starting discovery for element: {element_name} at coordinates: {coordinates}",
                'discovery_start',
                None
            )
            
            # WebSocket notification
            self.websocket_manager.send_to_user_with_format(
                self.system_user.id, 
                WebSocketType.DISCOVERY_ELEMENT_START, 
                {'user_id': self.user_id, 'element_name': element_name, 'coordinates': coordinates}
            )
            
            # بررسی memory برای المان
            element_crop = self.crop_element_image(image_path, coordinates)
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
            logService.append_log(
                self.step_id, 
                self.current_task.id if self.current_task else 0,
                f"Element not found in memory. Starting AI discovery for signature: {element_signature}",
                'discovery_memory_miss',
                None
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
                    None
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
                    None
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
            
            # نگه داشتن فقط main elements با coordinates
            if isinstance(coords, list) and len(coords) == 4:
                # اطمینان از اینکه coordinates لیست 4 عنصری هست
                try:
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

