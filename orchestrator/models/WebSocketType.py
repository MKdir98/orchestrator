from enum import Enum


class WebSocketType(Enum):
    GROUP_UPDATE = 'group_update'
    POSITION_REQUEST = 'position_request'
    POSITION_RESPONSE = 'position_response'
    VISION_REQUEST = 'vision_request'
    VISION_RESPONSE = 'vision_response'
    ACTION_REQUEST = 'action_request'
    ACTION_RESPONSE = 'action_response'
    ERROR_IN_PROCESS = 'error_in_process'
    USER_UPDATE = 'user_update'
    PING = 'ping'
    
    # Discovery Phase Events
    DISCOVERY_PHASE_START = 'discovery_phase_start'
    DISCOVERY_ELEMENT_START = 'discovery_element_start'
    DISCOVERY_MEMORY_HIT = 'discovery_memory_hit'
    DISCOVERY_MEMORY_MISS = 'discovery_memory_miss'
    DISCOVERY_AI_REQUEST = 'discovery_ai_request'
    DISCOVERY_AI_RESPONSE = 'discovery_ai_response'
    DISCOVERY_ELEMENT_COMPLETE = 'discovery_element_complete'
    DISCOVERY_PHASE_COMPLETE = 'discovery_phase_complete'