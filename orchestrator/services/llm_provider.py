from openai import OpenAI
from anthropic import Anthropic

from PIL import Image
import io
import json
import re
import base64
import g4f


def Message(content, role="assistant"):
    return {"role": role, "content": content}


def Text(text):
    return {"type": "text", "text": text}


def parse_json(s):
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        print(f"Error decoding JSON for tool call arguments: {s}")
        return None


class LLMProvider:
    """
    The LLM provider is used to make calls to an LLM given a provider and model name, with optional tool use support
    """

    # Class attributes for base URL and API key
    base_url = None
    api_key = None

    # Mapping of model aliases
    aliases = {}

    # Initialize the API client
    def __init__(self, model):
        self.model = self.aliases.get(model, model)
        print(f"Using {self.__class__.__name__} with {self.model}")
        self.client = self.create_client()

    # Convert our function schema to the provider's required format
    def create_function_schema(self, definitions):
        functions = []

        for name, details in definitions.items():
            properties = {}
            required = []

            for param_name, param_desc in details["params"].items():
                properties[param_name] = {"type": "string", "description": param_desc}
                required.append(param_name)

            function_def = self.create_function_def(name, details, properties, required)
            functions.append(function_def)

        return functions

    # Represent a tool call as an object
    def create_tool_call(self, name, parameters):
        return {
            "type": "function",
            "name": name,
            "parameters": parameters,
        }

    # Wrap a content block in a text or an image object
    def wrap_block(self, block):
        if isinstance(block, bytes):
            # Pass raw bytes so that imghdr can detect the image type properly.
            return self.create_image_block(block)
        else:
            return Text(block)

    # Wrap all blocks in a given input message
    def transform_message(self, message):
        content = message["content"]
        if isinstance(content, list):
            wrapped_content = [self.wrap_block(block) for block in content]
            return {**message, "content": wrapped_content}
        else:
            return message

    # Create a chat completion using the API client
    def completion(self, messages, **kwargs):
        # Skip the tools parameter if it's None
        filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        # Wrap content blocks in image or text objects if necessary
        new_messages = [self.transform_message(message) for message in messages]
        # Call the inference provider
        completion = self.client.create(
            messages=new_messages, model=self.model, **filtered_kwargs
        )
        # Check for errors in the response
        if hasattr(completion, "error"):
            raise Exception("Error calling model: {}".format(completion.error))
        return completion




class OpenAIBaseProvider(LLMProvider):

    def create_client(self):
        return OpenAI(base_url=self.base_url, api_key=self.api_key).chat.completions

    def create_function_def(self, name, details, properties, required):
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": details["description"],
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def create_image_block(self, image_data: bytes):
        # Use Pillow to detect the image type
        image_type = "png"  # Default to PNG if detection fails
        try:
            with Image.open(io.BytesIO(image_data)) as img:
                image_type = img.format.lower()
        except Exception as e:
            print(f"Error detecting image type: {e}")

        # Base64-encode the raw image bytes.
        encoded = base64.b64encode(image_data).decode("utf-8")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/{image_type};base64,{encoded}"},
        }

    def call(self, messages, functions=None):
        # If functions are provided, only return actions
        tools = self.create_function_schema(functions) if functions else None
        completion = self.completion(messages, tools=tools)
        message = completion.choices[0].message
        print(completion)
        print(message)

        # Return response text and tool calls separately
        if functions:
            tool_calls = message.tool_calls or []
            combined_tool_calls = [
                self.create_tool_call(
                    tool_call.function.name, parse_json(tool_call.function.arguments)
                )
                for tool_call in tool_calls
                if parse_json(tool_call.function.arguments) is not None
            ]

            # Sometimes, function calls are returned unparsed by the inference provider. This code parses them manually.
            if message.content and not tool_calls:
                tool_call_matches = re.search(r"\{.*\}", message.content)
                if tool_call_matches:
                    tool_call = parse_json(tool_call_matches.group(0))
                    # Some models use "arguments" as the key instead of "parameters"
                    parameters = tool_call.get("parameters", tool_call.get("arguments"))
                    if tool_call.get("name") and parameters:
                        combined_tool_calls.append(
                            self.create_tool_call(tool_call.get("name"), parameters)
                        )
                        return None, combined_tool_calls

            return message.content, combined_tool_calls

        # Only return response text
        else:
            return message.content


class AnthropicBaseProvider(LLMProvider):

    def create_client(self):
        return Anthropic(api_key=self.api_key).messages

    def create_function_def(self, name, details, properties, required):
        return {
            "name": name,
            "description": details["description"],
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }

    def create_image_block(self, base64_image):
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": base64_image,
            },
        }

    def call(self, messages, functions=None):
        tools = self.create_function_schema(functions) if functions else None

        # Move all messages with the system role to a system parameter
        system = "\n".join(
            msg.get("content") for msg in messages if msg.get("role") == "system"
        )
        messages = [msg for msg in messages if msg.get("role") != "system"]

        # Call the Anthropic API
        completion = self.completion(
            messages, system=system, tools=tools, max_tokens=4096
        )
        text = "".join(getattr(block, "text", "") for block in completion.content)

        # Return response text and tool calls separately
        if functions:
            tool_calls = [
                self.create_tool_call(block.name, block.input)
                for block in completion.content
                if block.type == "tool_use"
            ]
            return text, tool_calls

        # Only return response text
        else:
            return text


class MistralBaseProvider(OpenAIBaseProvider):
    def create_function_def(self, name, details, properties, required):
        # If description is wrapped in a dict, extract the inner string
        if isinstance(details.get("description"), dict):
            details["description"] = details["description"].get("description", "")
        return super().create_function_def(name, details, properties, required)

    def call(self, messages, functions=None):
        if messages and messages[-1].get("role") == "assistant":
            prefix = messages.pop()["content"]
            if messages and messages[-1].get("role") == "user":
                messages[-1]["content"] = (
                        prefix + "\n" + messages[-1].get("content", "")
                )
            else:
                messages.append({"role": "user", "content": prefix})
        return super().call(messages, functions)

class G4FProvider(LLMProvider):

    def create_client(self):
        # استفاده از g4f به عنوان کلاینت
        return g4f.ChatCompletion

    def create_function_def(self, name, details, properties, required):
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": details["description"],
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def call(self, messages, functions=None):
        """
        ارسال درخواست به مدل و پردازش پاسخ.

        Args:
            messages (list): لیستی از پیام‌ها (هر پیام یک دیکشنری با کلیدهای 'role' و 'content').
            functions (dict, optional): دیکشنری توابعی که مدل می‌تواند فراخوانی کند.

        Returns:
            tuple: یک تاپل شامل (content, tool_calls) که:
                - content (str): متن پاسخ مدل.
                - tool_calls (list): لیستی از فراخوانی‌های تابع (هر تابع یک دیکشنری با کلیدهای 'name' و 'arguments').
        """
        # print(messages)
        # اگر توابع ارائه شده‌اند، آن‌ها را به مدل بفهمانید
        if functions:
            # بررسی اینکه functions یک دیکشنری باشد
            if not isinstance(functions, dict):
                raise ValueError("The 'functions' parameter must be a dictionary.")

            # تبدیل دیکشنری functions به لیست
            functions_list = []
            for func_name, func_details in functions.items():
                if not isinstance(func_details, dict):
                    raise ValueError(f"Function '{func_name}' details must be a dictionary.")
                if "description" not in func_details or "params" not in func_details:
                    raise ValueError(f"Function '{func_name}' must have 'description' and 'params' keys.")
                
                # تبدیل params به properties و required
                properties = {}
                required = []
                for param_name, param_desc in func_details["params"].items():
                    properties[param_name] = {
                        "type": "string",  # فرض می‌کنیم همه پارامترها رشته هستند
                        "description": param_desc,
                    }
                    required.append(param_name)
                
                # ایجاد تعریف تابع
                functions_list.append({
                    "name": func_name,
                    "description": func_details["description"],
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                })

            system_prompt = {
                "role": "system",
                "content": """
                You are a helpful assistant. Your response must be a valid JSON array of tool calls. Each tool call should have a 'name' and 'parameters' field.
                Important rules:
                1. Your response must **ONLY** contain a valid JSON array. Do not include any additional text, explanations, or formatting outside the JSON array.
                2. The JSON array must follow this structure:
                [
                    {
                        "name": "tool_name",
                        "parameters": {
                            "param1": "value1",
                            "param2": "value2"
                        }
                    }
                ]
                3. If no tools are needed, return an empty array: [].
                4. **Never** add any comments, explanations, or additional text outside the JSON array.
                """
            }
            messages.insert(0, system_prompt)

            # اضافه کردن توابع به پیام‌ها
            messages.append({"role": "user", "content": "Please generate tool calls based on the following functions: " + json.dumps(functions_list)})

        # ایجاد completion با استفاده از g4f
        try:
            print(messages)
            completion = self.create_client().create(
                model=self.model,  # یا هر مدل دیگری که پشتیبانی می‌شود
                messages=messages,
            )
        except Exception as e:
            print(f"Error during completion: {e}")
            return "", []  # برگرداندن مقادیر پیش‌فرض در صورت خطا

        # پردازش پاسخ
        response = completion
        # print(response)

        # اگر توابع ارائه شده‌اند، پاسخ را به‌صورت JSON پارس کنید
        if functions:
            try:
                # حذف ```json و ``` از ابتدا و انتهای response (اگر وجود داشته باشد)
                response = re.sub(r'^```json\s*|\s*```$', '', response, flags=re.MULTILINE).strip()
                
                # تلاش برای پارس کردن JSON کامل
                tool_calls = json.loads(response)
                
                # بررسی اینکه آیا tool_calls یک لیست است
                if isinstance(tool_calls, list):
                    # بررسی ساختار هر آیتم در لیست
                    for tool_call in tool_calls:
                        if not isinstance(tool_call, dict):
                            raise ValueError("Each tool call must be a dictionary.")
                        if "name" not in tool_call or "arguments" not in tool_call:
                            raise ValueError("Each tool call must have 'name' and 'arguments' keys.")
                    return response, tool_calls
                else:
                    # اگر tool_calls یک دیکشنری است، آن را به لیست تبدیل کنید
                    if isinstance(tool_calls, dict):
                        if "name" in tool_calls and "arguments" in tool_calls:
                            return response, [tool_calls]
                        else:
                            raise ValueError("The tool call must have 'name' and 'arguments' keys.")
                    else:
                        raise ValueError("The response must be a list or a dictionary.")
            
            except json.JSONDecodeError:
                # اگر JSON کامل نبود، به‌صورت دستی JSON را استخراج کنید
                try:
                    tool_call_matches = re.findall(r"\{.*?\}", response)
                    if tool_call_matches:
                        tool_calls = []
                        for match in tool_call_matches:
                            try:
                                tool_call = json.loads(match)
                                if isinstance(tool_call, dict) and "name" in tool_call and "arguments" in tool_call:
                                    tool_calls.append(tool_call)
                            except json.JSONDecodeError:
                                continue  # اگر JSON نامعتبر بود، از آن صرف‌نظر کنید
                        
                        if tool_calls:
                            return response, tool_calls
                        else:
                            print("Error: No valid tool calls found in the response.")
                            return response, []
                    else:
                        print("Error: Model did not return valid JSON.")
                        return response, []
                except Exception as e:
                    print(f"Error parsing JSON: {e}")
                    return response, []
        else:
            return response, []  # برگرداندن محتوا و یک لیست خالی از tool_calls
