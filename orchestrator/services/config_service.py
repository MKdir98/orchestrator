from orchestrator.services.llm_provider import G4FProvider, HuggingFaceProvider, OllamaProvider
from orchestrator.services.osatlas_service import OSAtlasProvider
from orchestrator.services.providers import *

grounding_model = None
# grounding_model = providers.ShowUIProvider()

# مدل‌های کوچک و بهینه برای سیستم 16GB RAM
# vision_model = providers.FireworksProvider("llama-3.2")
# vision_model = providers.OpenAIProvider("gpt-4o")
# vision_model = providers.AnthropicProvider("claude-3.5-sonnet")
# vision_model = providers.MoonshotProvider("moonshot-v1-vision")
# vision_model = providers.MistralProvider("pixtral")
# vision_model = providers.GroqProvider("llama-3.2")
# vision_model = OpenRouterProvider("qwen-2.5-vl")

# مدل‌های محلی Ollama (برای سیستم‌های قوی‌تر):
vision_model = G4FProvider("")  # مدل Qwen 3B محلی
# vision_model = G4FProvider("qwen2.5-vl:7b")  # مدل Qwen Vision 7B محلی
# vision_model = providers.LocalG4FProvider("")  # با alias
# vision_model = G4FProvider("")  # مدل‌های رایگان و کوچک

# action_model = FireworksProvider("llama-3.3")
# action_model = providers.OpenAIProvider("gpt-4o")
# action_model = providers.AnthropicProvider("claude-3.5-sonnet")
# vision_model = providers.MoonshotProvider("moonshot-v1-vision")
# action_model = MistralProvider("mistral")
# action_model = G4FProvider('blackboxai')
# action_model = GroqProvider("llama-3.2")
# action_model = OpenRouterProvider("google/gemini-2.0-flash-exp:free")

# مدل‌های محلی Ollama:
action_model = G4FProvider("")  # مدل Qwen 3B محلی
# action_model = providers.LocalG4FProvider("")  # با alias
# action_model = G4FProvider('')  # مدل‌های رایگان و کوچک

position_model = G4FProvider("")  # مدل Qwen 3B محلی
# position_model = G4FProvider('')  # مدل‌های رایگان و کوچک