import json, requests, os
from datetime import datetime
from sqlalchemy.testing.suite.test_reflection import users

from orchestrator.models.base import SessionLocal
from orchestrator.models import User, Message
from orchestrator.services.processor_service import ProcessorService
from orchestrator.services.task_service import TaskService


class ChatService:
    def check_and_create_users_in_chat(self):
        db = SessionLocal()
        try:
            users = db.query(User).all()
            usernames = list(map(lambda user: user['username'], self.users()))
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
        users_list = list(map(lambda x: {'username': x['username'], 'id': x['_id'], 'name': x['name']},
                              filter(lambda x: 'admin' not in x['roles'] and x['type'] == 'user',
                                     json.loads(response.text)['users'])))
        return users_list

    def user_delete(self, userId: str):
        url = os.getenv(
            'ROCKET_CHAT_ADDRESS') + "/api/v1/users.delete"

        payload = json.dumps({
            "userId": userId,
            "confirmRelinquish": False
        })
        headers = {
            'X-Auth-Token': os.getenv("ROCKET_CHAT_AUTH_TOKEN"),
            'sec-ch-ua-platform': '"Linux"',
            'X-User-Id': os.getenv("ROCKET_CHAT_USER_ID"),
            'Referer': 'http://localhost:3000/admin/users',
            'sec-ch-ua': '"Not A(Brand";v="8", "Chromium";v="132", "Google Chrome";v="132"',
            'sec-ch-ua-mobile': '?0',
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }

        response = requests.request("POST", url, headers=headers, data=payload)

        print(response.text)

    def get_direct_messages(self, username: str, since_timestamp=None):
        """
        دریافت پیغام‌های direct message برای یک user از RocketChat
        
        Args:
            username: نام کاربری agent در RocketChat
            since_timestamp: تاریخ آخرین پیغام (برای دریافت فقط پیغام‌های جدید)
        
        Returns:
            لیست پیغام‌های جدید
        """
        try:
            # ابتدا باید room ID رو پیدا کنیم (direct message room بین admin و این user)
            # فرض: admin پیغام میده، پس باید direct message room بین admin و username رو پیدا کنیم
            
            # دریافت لیست direct messages
            url = os.getenv('ROCKET_CHAT_ADDRESS') + "/api/v1/im.list"
            headers = {
                'X-Auth-Token': os.getenv("ROCKET_CHAT_AUTH_TOKEN"),
                'X-User-Id': os.getenv("ROCKET_CHAT_USER_ID"),
                'Content-Type': 'application/json'
            }
            
            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                print(f"Error fetching IM list for {username}: {response.text}")
                return []
            
            ims = response.json().get('ims', [])
            
            # پیدا کردن room که شامل username است
            target_room = None
            for im in ims:
                # چک کردن usernames در room
                if 'usernames' in im and username in im['usernames']:
                    target_room = im
                    break
            
            if not target_room:
                # هنوز room ایجاد نشده (پیغامی رد و بدل نشده)
                return []
            
            room_id = target_room.get('_id')
            
            # دریافت پیغام‌های این room
            messages_url = os.getenv('ROCKET_CHAT_ADDRESS') + f"/api/v1/im.messages?roomId={room_id}"
            
            # اگر since_timestamp داده شده، فقط پیغام‌های بعد از آن تاریخ
            if since_timestamp:
                # تبدیل به ISO format
                if isinstance(since_timestamp, datetime):
                    since_iso = since_timestamp.isoformat()
                else:
                    since_iso = since_timestamp
                messages_url += f"&oldest={since_iso}"
            
            messages_response = requests.get(messages_url, headers=headers)
            if messages_response.status_code != 200:
                print(f"Error fetching messages for room {room_id}: {messages_response.text}")
                return []
            
            messages = messages_response.json().get('messages', [])
            
            # فیلتر کردن: فقط پیغام‌هایی که از admin اومده (نه از خود agent)
            admin_username = os.getenv('ROCKET_CHAT_ADMIN_NAME', 'admin').replace(' ', '_').lower()
            filtered_messages = []
            for msg in messages:
                msg_username = msg.get('u', {}).get('username', '')
                # فقط پیغام‌های admin
                if msg_username == admin_username:
                    filtered_messages.append(msg)
            
            return filtered_messages
            
        except Exception as e:
            print(f"Exception in get_direct_messages for {username}: {str(e)}")
            return []

    def get_last_message_time(self, db, user_id):
        """
        دریافت زمان آخرین پیغام ذخیره شده برای یک user
        
        Args:
            db: Database session
            user_id: ID کاربر
        
        Returns:
            datetime آخرین پیغام یا None
        """
        try:
            last_message = db.query(Message).filter(
                Message.user_id == user_id
            ).order_by(Message.rocketchat_timestamp.desc()).first()
            
            if last_message and last_message.rocketchat_timestamp:
                return last_message.rocketchat_timestamp
            
            return None
        except Exception as e:
            print(f"Error getting last message time for user {user_id}: {str(e)}")
            return None

    def save_new_messages_to_db(self):
        """
        برای تمام user های فعال، پیغام‌های جدید رو بگیر و در DB ذخیره کن
        این متد توسط scheduler هر 2 دقیقه یکبار فراخوانی میشه
        """
        db = SessionLocal()
        try:
            users = db.query(User).all()
            total_new_messages = 0
            
            for user in users:
                try:
                    username = user.name.replace(' ', '_').lower()
                    last_message_time = self.get_last_message_time(db, user.id)
                    
                    new_messages = self.get_direct_messages(username, since_timestamp=last_message_time)
                    
                    for msg in new_messages:
                        # چک کنیم که این پیغام قبلاً ذخیره نشده باشه
                        msg_id = msg.get('_id')
                        existing = db.query(Message).filter(
                            Message.rocketchat_message_id == msg_id
                        ).first()
                        
                        if existing:
                            continue  # این پیغام قبلاً ذخیره شده
                        
                        # ساخت Message جدید
                        msg_timestamp = msg.get('ts')
                        if isinstance(msg_timestamp, dict) and '$date' in msg_timestamp:
                            # فرمت RocketChat: {"$date": 1234567890123}
                            timestamp_ms = msg_timestamp['$date']
                            msg_datetime = datetime.fromtimestamp(timestamp_ms / 1000.0)
                        elif isinstance(msg_timestamp, str):
                            # فرمت ISO
                            msg_datetime = datetime.fromisoformat(msg_timestamp.replace('Z', '+00:00'))
                        else:
                            msg_datetime = datetime.utcnow()
                        
                        new_message = Message(
                            rocketchat_message_id=msg_id,
                            user_id=user.id,
                            sender_username=msg.get('u', {}).get('username', 'unknown'),
                            content=msg.get('msg', ''),
                            is_read=False,
                            is_processed=False,
                            rocketchat_timestamp=msg_datetime
                        )
                        
                        db.add(new_message)
                        total_new_messages += 1
                    
                except Exception as e:
                    print(f"Error processing messages for user {user.name} (id={user.id}): {str(e)}")
                    continue
            
            db.commit()
            
            if total_new_messages > 0:
                print(f"✓ Saved {total_new_messages} new messages to database")
            
        except Exception as e:
            print(f"Error in save_new_messages_to_db: {str(e)}")
            db.rollback()
        finally:
            db.close()
