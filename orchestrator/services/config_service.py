from orchestrator.services.llm_provider import G4FProvider
from orchestrator.services.osatlas_service import OSAtlasProvider
from orchestrator.services.providers import *

grounding_model = OSAtlasProvider()
# grounding_model = providers.ShowUIProvider()

# vision_model = providers.FireworksProvider("llama-3.2")
# vision_model = providers.OpenAIProvider("gpt-4o")
# vision_model = providers.AnthropicProvider("claude-3.5-sonnet")
# vision_model = providers.MoonshotProvider("moonshot-v1-vision")
# vision_model = providers.MistralProvider("pixtral")
#vision_model = providers.GroqProvider("llama-3.2")
vision_model = OpenRouterProvider("qwen-2.5-vl")

# action_model = FireworksProvider("llama-3.3")
# action_model = providers.OpenAIProvider("gpt-4o")
# action_model = providers.AnthropicProvider("claude-3.5-sonnet")
# vision_model = providers.MoonshotProvider("moonshot-v1-vision")
# action_model = MistralProvider("mistral")
# action_model = G4FProvider('blackboxai')
action_model = GroqProvider("llama-3.3")
