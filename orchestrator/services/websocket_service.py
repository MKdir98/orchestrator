import threading
import json

from orchestrator.models.WebSocketType import WebSocketType


class WebSocketManager:
    def __init__(self):
        self.active_connections = {}
        self.lock = threading.Lock()
        self.socketio = None
        self.app = None  # Flask app instance for context

    def add_connection(self, user_id, sid):
        with self.lock:
            self.active_connections[user_id] = sid

    def remove_connection(self, user_id):
        with self.lock:
            if user_id in self.active_connections:
                del self.active_connections[user_id]

    def send_to_user_with_format(self, system_user_id, event_type: WebSocketType, data: str):
        connection = self.active_connections.get(system_user_id)
        
        if not connection or not self.socketio:
            return
        
        try:
            data_json = json.dumps({'event_type': event_type.value, 'data': data})
            
            # Use server.emit directly (bypasses Flask context issues)
            if hasattr(self.socketio, 'server'):
                self.socketio.server.emit('message', data_json, room=connection, namespace='/')
            else:
                self.socketio.emit('message', data_json, to=connection, namespace='/')
                    
        except Exception as e:
            print(f"Error sending message to user {system_user_id}: {e}")

    def broadcast(self, message):
        with self.lock:
            for user_id, connection in self.active_connections.items():
                try:
                    if hasattr(connection, "send_json"):
                        connection.send_json(message)
                    else:
                        import json
                        connection.send(json.dumps(message))
                except Exception as e:
                    print(f"Error broadcasting to user {user_id}: {e}")


websocket_manager = WebSocketManager()
