from g4f import ProviderType
from g4f.Provider import OIVSCodeSer0501, PollinationsAI
from gradio_client import Client, handle_file
from openai import OpenAI
from anthropic import Anthropic
import traceback
import requests

from PIL import Image
import io
import json
import re
import base64
import g4f
import os



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
                return None, None
                # tool_call_matches = re.search(r"\{.*\}", message.content)
                # if tool_call_matches:
                #     tool_call = parse_json(tool_call_matches.group(0))
                #     # Some models use "arguments" as the key instead of "parameters"
                #     parameters = tool_call.get("parameters", tool_call.get("arguments"))
                #     if tool_call.get("name") and parameters:
                #         combined_tool_calls.append(
                #             self.create_tool_call(tool_call.get("name"), parameters)
                #         )
                #         return None, combined_tool_calls

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
        # return g4f.Client(NewHarProvider).chat.completions
        # return g4f.Client(OIVSCodeSer5).chat.completions
        return g4f.Client(PollinationsAI).chat.completions
        # return g4f.Client(OIVSCodeSer0501).chat.completions

    def call(self, message, image, tools=None):
        imageData = [[open(image, "rb"), 'picture.png']]
        if tools is not None:
            exampleFakeData = '''
                [{
                    "name":"click" , // you MUST use 'name'
                    "parameters": { // DONT ANSWER WITH 'params' AS KEY JUST USE "parameters"
                        "x": "111", "y": "222",
                        "last_action_result": "test last action result",
                        "image_width": "1234",
                        "image_height": "4321",
                        "description": "Fake reason"
                    }
                }, ...]
                '''
            tool_calls_prompt = (
                    'The format of the response is should be EXACT like this: Example of response with fake data: ' +
                    exampleFakeData + '. These are the list of functions that always you MUST use in your json response: ' + json.dumps(
                tools) + '')
            message = message + tool_calls_prompt

        # response = self.client.create(
        #     message, 'openai-large', images=imageData).choices[0].message.content  ## That was not great

        response = self.client.create(
            message, 
            # 'qwen2.5-vl-32b-instruct',
            'o4-mini',
            # '',
            # 'o3-2025-04-16',
            # 'claude-3-7-sonnet-20250219',
            # 'gemini-2.5-flash-preview-04-17',
            images=imageData).choices[0].message.content  ## That was not great
        
        print(message, response)
        if tools is not None:
            if response[0:3] == '```':
                return ['', json.loads(response[7:-3])]
            else:
                return ['', json.loads(response)]
        else:
            if response[0:3] == '```':
                return response[7:-3]
            else:
                return response


class HuggingFaceProvider(LLMProvider):

    def call(self, message, image, tools=None):
        if (tools != None):
            exampleFakeData = '''
                    [{
                        "name":"click" , // you MUST use 'name'
                        "parameters": { // DONT answer with 'params'
                            "x": "111", "y": "222",
                            "last_action_result": "test last action result",
                            "image_width": "1234",
                            "image_height": "4321",
                            "description": "Fake reason"
                        }
                    }]
                    '''
            tool_calls_prompt = (
                    'The format of the response (array of jsonObject) is should be EXACT like this: Example of response with fake data: ' +
                    exampleFakeData + '. These are the list of functions that always you MUST use in your json response: ' + json.dumps(
                tools) + '')
            message = message + tool_calls_prompt

            # messages[1]['content'] = messages[1]['content'] + tool_calls_prompt

        response = self.client.predict(
            message={"text": message, "files": [
                handle_file(image)]},
            api_name="/chat"
        )

        print(message, response)
        if tools is not None:
            if response[0:3] == '```':
                return ['', json.loads(response[7:-3])]
            else:
                return ['', json.loads(response)]
        else:
            if response[0:3] == '```':
                return response[7:-3]
            else:
                return response

    def create_client(self):
        return Client("prithivMLmods/Qwen2.5-VL-7B-Instruct", hf_token=HF_TOKEN)


class OllamaProvider(LLMProvider):
    """
    Provider for local Ollama models with image support
    Supports Qwen and other vision models running locally
    """
    
    def __init__(self, model="qwen2.5vl:3b"):
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = model
        print(f"Using OllamaProvider with {self.model} at {self.base_url}")
        self.client = self.create_client()
    
    def create_client(self):
        """Create Ollama client - using requests for simplicity"""
        return None  # We'll use requests directly
    
    def create_image_block(self, image_data: bytes):
        """Convert image to base64 for Ollama"""
        # Use Pillow to detect the image type
        image_type = "png"  # Default to PNG if detection fails
        try:
            with Image.open(io.BytesIO(image_data)) as img:
                image_type = img.format.lower()
        except Exception as e:
            print(f"Error detecting image type: {e}")

        # Base64-encode the raw image bytes
        encoded = base64.b64encode(image_data).decode("utf-8")
        return {
            "type": "image",
            "data": encoded
        }
    
    def transform_messages_for_ollama(self, messages):
        """Transform messages to Ollama format"""
        ollama_messages = []
        
        for message in messages:
            if message.get("role") == "system":
                # Ollama doesn't have system messages, convert to user message
                ollama_messages.append({
                    "role": "user",
                    "content": f"System: {message['content']}"
                })
            elif message.get("role") in ["user", "assistant"]:
                content = message["content"]
                
                # Handle multimodal content (text + images)
                if isinstance(content, list):
                    # Combine text and images
                    text_parts = []
                    images = []
                    
                    for block in content:
                        if isinstance(block, dict):
                            if block.get("type") == "text":
                                text_parts.append(block["text"])
                            elif block.get("type") == "image_url":
                                # Extract base64 from data URL
                                url = block["image_url"]["url"]
                                if url.startswith("data:image/"):
                                    base64_data = url.split(",", 1)[1]
                                    images.append(base64_data)
                            elif block.get("type") == "image":
                                images.append(block["data"])
                        else:
                            text_parts.append(str(block))
                    
                    # Combine text parts
                    combined_text = " ".join(text_parts)
                    
                    # Create Ollama message
                    ollama_message = {
                        "role": message["role"],
                        "content": combined_text
                    }
                    
                    # Add images if present
                    if images:
                        ollama_message["images"] = images
                    
                    ollama_messages.append(ollama_message)
                else:
                    # Simple text message
                    ollama_messages.append({
                        "role": message["role"],
                        "content": str(content)
                    })
        
        return ollama_messages
    
    def call(self, messages, image_path=None, tools=None):
        """
        Call Ollama API with support for images and tools
        
        Args:
            messages: List of message dictionaries
            image_path: Optional path to image file
            tools: Optional tools/functions definition
        """
        try:
            # Handle both string and list formats
            if isinstance(messages, str):
                # Simple string message
                ollama_messages = [{"role": "user", "content": messages}]
            else:
                # List of message dictionaries
                ollama_messages = self.transform_messages_for_ollama(messages)
            
            # Prepare request payload
            payload = {
                "model": self.model,
                "messages": ollama_messages,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_predict": 2048
                }
            }
            
            # Add tools if provided
            if tools:
                # Convert tools to Ollama format
                ollama_tools = []
                for name, details in tools.items():
                    tool_def = {
                        "name": name,
                        "description": details.get("description", ""),
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                    }
                    
                    # Convert parameters
                    for param_name, param_desc in details.get("params", {}).items():
                        tool_def["parameters"]["properties"][param_name] = {
                            "type": "string",
                            "description": param_desc
                        }
                        tool_def["parameters"]["required"].append(param_name)
                    
                    ollama_tools.append(tool_def)
                
                payload["tools"] = ollama_tools
            
            # Add image if provided
            if image_path and os.path.exists(image_path):
                with open(image_path, "rb") as f:
                    image_data = f.read()
                    base64_image = base64.b64encode(image_data).decode("utf-8")
                    
                    # Add image to the last user message
                    if ollama_messages and ollama_messages[-1]["role"] == "user":
                        if "images" not in ollama_messages[-1]:
                            ollama_messages[-1]["images"] = []
                        ollama_messages[-1]["images"].append(base64_image)
                        payload["messages"] = ollama_messages
            
            # Make request to Ollama
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=120
            )
            
            if response.status_code != 200:
                raise Exception(f"Ollama API error: {response.status_code} - {response.text}")
            
            result = response.json()
            
            # Extract response
            if "message" in result:
                content = result["message"].get("content", "")
                
                # Handle tool calls if present
                if tools and "tool_calls" in result["message"]:
                    tool_calls = []
                    for tool_call in result["message"]["tool_calls"]:
                        tool_calls.append({
                            "type": "function",
                            "name": tool_call["name"],
                            "parameters": tool_call.get("args", {})
                        })
                    return content, tool_calls
                
                return content
            
            return ""
            
        except Exception as e:
            print(f"Error calling Ollama: {e}")
            traceback.print_exc()
            return f"Error: {str(e)}"
    
    def completion(self, messages, **kwargs):
        """Compatibility method for the base class"""
        return self.call(messages)
