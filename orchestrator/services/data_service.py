import tempfile, os


class DataService:
    def get_user_path_screenshot(self, user_id: int):
        return self.get_user_data_path(user_id) + 'screenshot.png'

    def get_user_data_path(self, user_id: int):
        path = self.get_orchestrator_path() + str(user_id) + '/'
        os.makedirs(path, exist_ok=True)  # Create directory if not exists
        return path

    def get_orchestrator_path(self):
        path = tempfile.gettempdir() + '/orchestrator/'
        os.makedirs(path, exist_ok=True)  # Create directory if not exists
        return path
