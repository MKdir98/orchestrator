import json, requests, os
from dotenv import load_dotenv

from orchestrator.models.base import SessionLocal
from orchestrator.models import User
from orchestrator.services.processor_service import ProcessorService
from orchestrator.services.task_service import TaskService

load_dotenv()


class ChatService:
    def check_and_create_users_in_chat(self):
        db = SessionLocal()
        try:
            users = db.query(User).all()
            usernames = self.users()
            for user in users:
                username = user.name.replace(' ', '_').lower()
                if not username in usernames:
                    print('create user in chat for ' + user.name)
                    self.create_user(user.name, username, username + '@orchestrator.com')
                    if user.parent_user_id is None:
                        task = 'Your leader is ' + os.getenv('ROCKET_CHAT_ADMIN_NAME') + ' with id=' + os.getenv(
                            'ROCKET_CHAT_ADMIN_USER_ID') + '. Text to him\her in messanger and introduce yourself'
                        TaskService.create_task(db, task, user.id)
                    else:
                        parentUser = db.query(User).filter(User.id == user.parent_user_id).first()
                        task = 'Your leader is ' + parentUser.name + ' with id=' + parentUser.name.replace(' ',
                                                                                                           '_').lower() \
                               + '. Text to him\her in messanger and introduce yourself'
                        TaskService.create_task(db, task, user.id)
        finally:
            db.close()

    def create_user(self, name, username, email):
        url = os.getenv('ROCKET_CHAT_ADDRESS') + "/api/v1/users.create"

        payload = json.dumps({
            "roles": [
                "user"
            ],
            "name": name,
            "password": "mypassword",
            "username": username,
            "bio": "",
            "nickname": "",
            "email": email,
            "verified": False,
            "setRandomPassword": False,
            "requirePasswordChange": False,
            "customFields": {},
            "statusText": "",
            "joinDefaultChannels": True,
            "sendWelcomeEmail": False,
            "fields": ""
        })
        headers = {
            'X-Auth-Token': os.getenv("ROCKET_CHAT_AUTH_TOKEN"),
            'sec-ch-ua-platform': '"Linux"',
            'X-User-Id': os.getenv("ROCKET_CHAT_USER_ID"),
            'sec-ch-ua': '"Not A(Brand";v="8", "Chromium";v="132", "Google Chrome";v="132"',
            'sec-ch-ua-mobile': '?0',
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
        response = requests.request("POST", url, headers=headers, data=payload)
        print(response.text)

    def users(self):
        url = os.getenv(
            'ROCKET_CHAT_ADDRESS') + "/api/v1/users.listByStatus?count=100&offset=0&searchTerm=&sort=%7B%20%22name%22%3A%201%20%7D"

        payload = {}
        headers = {
            'X-Auth-Token': os.getenv("ROCKET_CHAT_AUTH_TOKEN"),
            'sec-ch-ua-platform': '"Linux"',
            'X-User-Id': os.getenv("ROCKET_CHAT_USER_ID"),
            'Referer': 'http://localhost:3000/admin/users/info/2MigT6uDksKqXMuPJ',
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36',
            'sec-ch-ua': '"Not A(Brand";v="8", "Chromium";v="132", "Google Chrome";v="132"',
            'sec-ch-ua-mobile': '?0'
        }

        response = requests.request("GET", url, headers=headers, data=payload)
        print(response.text)
        return map(lambda x: x['username'], json.loads(response.text)['users'])
