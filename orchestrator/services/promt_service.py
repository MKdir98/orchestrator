import g4f
from g4f.Provider.Blackbox import Blackbox

class PromptService:
    """
    Service class for sending prompts to the GPT-4 model using g4f.
    """

    @staticmethod
    def send_prompt_to_model(prompt: str, image_path: str) -> str:
        """
        Send a prompt to the GPT-4 model using g4f and return the response.

        Args:
            prompt (str): The prompt to send to the model.

        Returns:
            str: The model's response.
        """
        try:
            # ارسال پرامپت به مدل
            response = g4f.ChatCompletion.create(
                model="gemini-1.5-flash",  # استفاده از مدل GPT-4
                messages=[{"role": "user", "content": prompt}],
                provider=Blackbox,
                image=open(image_path, 'rb')

            )
            # print(response)
            if response:
                return response
            else:
                raise Exception("No response received from the model.")
        except Exception as e:
            raise Exception(f"Failed to send prompt to the model: {str(e)}")
