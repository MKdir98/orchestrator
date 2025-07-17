class OrchestratorException(Exception):
    
    def __init__(self, message, status_code=500, persian_message=None):
        self.message = message
        self.status_code = status_code
        self.persian_message = persian_message or message
        super().__init__(self.message)
    
    def to_dict(self):
        """تبدیل اکسپشن به دیکشنری برای ارسال به کاربر"""
        return {
            "error": self.message,
            "persian_error": self.persian_message,
            "status_code": self.status_code
        }


class AccessDeniedException(OrchestratorException):
    
    def __init__(self, message="Access denied", persian_message="دسترسی غیرمجاز"):
        super().__init__(message, status_code=403, persian_message=persian_message)


class ResourceNotFoundException(OrchestratorException):
    
    def __init__(self, resource_type, resource_id, message=None, persian_message=None):
        if message is None:
            message = f"{resource_type} with ID {resource_id} not found"
        if persian_message is None:
            persian_message = f"{resource_type} با شناسه {resource_id} یافت نشد"
        super().__init__(message, status_code=404, persian_message=persian_message)


class ValidationException(OrchestratorException):
    
    def __init__(self, message="Validation error", persian_message="خطای اعتبارسنجی"):
        super().__init__(message, status_code=400, persian_message=persian_message)


class AuthenticationException(OrchestratorException):
    
    def __init__(self, message="Authentication failed", persian_message="خطای احراز هویت"):
        super().__init__(message, status_code=401, persian_message=persian_message) 