import os


class DataService:
    def get_user_path_screenshot(self, user_id: int):
        return self.get_user_data_path(user_id) + 'screenshot.png'

    def get_user_path_screenshot_with_point(self, user_id: int, step_id: int):
        return self.get_user_data_path(user_id) + 'screenshot_clicked.png'

    def get_user_path_screenshot_with_coordinates(self, user_id: int, step_id: int):
        return self.get_user_data_path(user_id) + 'screenshot_coordinates.png'

    def get_user_path_screenshot_with_local_coordinates(self, user_id: int, step_id: int):
        return self.get_user_data_path(user_id) + 'screenshot_local_coordinates.png'

    def get_user_data_path(self, user_id: int):
        path = self.get_orchestrator_path() + str(user_id) + '/'
        os.makedirs(path, exist_ok=True)
        return path

    def get_orchestrator_path(self):
        current_file = os.path.abspath(__file__)
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
        path = base_path + '/orchestrator_data/' + os.getenv('MODE') + '/'
        os.makedirs(path, exist_ok=True)
        print(path)
        return path

    def get_log_path(self, log_id: int):
        path = self.get_logs_path() + str(log_id) + '.json'
        return path

    def get_logs_path(self):
        path = self.get_orchestrator_path() + 'logs/'
        os.makedirs(path, exist_ok=True)
        return path

    def get_logs_image_path(self):
        path = self.get_logs_path() + 'images/'
        os.makedirs(path, exist_ok=True)
        return path

    def get_user_checkpoint_path(self, user_id: int, step_id: int):
        """
        مسیر فایل checkpoint برای یک کاربر و مرحله خاص
        """
        checkpoints_dir = os.path.join(self.get_user_data_path(user_id), 'checkpoints')
        os.makedirs(checkpoints_dir, exist_ok=True)
        return os.path.join(checkpoints_dir, f'checkpoint_{step_id}.json')
