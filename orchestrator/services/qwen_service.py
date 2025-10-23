rom gradio_client import Client, handle_file

import os

from orchestrator.services.grounding_service import extract_bbox_midpoint

OSATLAS_HUGGINGFACE_SOURCE = "Qwen/Qwen2.5-VL-72B-Instruct"
OSATLAS_HUGGINGFACE_MODEL = "OS-Copilot/OS-Atlas-Base-7B"
OSATLAS_HUGGINGFACE_API = "/run_example"

HF_TOKEN = os.getenv("HF_TOKEN")


class QwenProvider:
    """
    The OS-Atlas provider is used to make calls to OS-Atlas.
    """

    def __init__(self):
        self.client = Client(OSATLAS_HUGGINGFACE_SOURCE, hf_token=HF_TOKEN)

    def call(self, prompt, image_data):
        # Initialize conversation history
        client = Client("prithivMLmods/Qwen2.5-VL-7B-Instruct", hf_token=HF_TOKEN)
        final_result = client.predict(
            message={"text": prompt, "files": [
                handle_file(image_data)]},
            api_name="/chat"
        )
        print(final_result)

        position = extract_bbox_midpoint(final_result[1])
        image_url = final_result[2]
        print(f"bbox {image_url}", "gray")
        return position
