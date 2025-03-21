import g4f


class GPTService:
    @staticmethod
    def get_commands(task, screenshot_path):
        try:
            with open(screenshot_path, "rb") as f:
                screenshot_data = f.read()
            response = g4f.generate(
                model="gpt-4",
                prompt=f"Task: {task}\nHere is a screenshot of the current state. What commands should I execute?",
                max_tokens=100,
            )
            return response.strip().split("\n")
        except Exception as e:
            print(f"Error sending data to GPT: {e}")
            return []
