import sys
import os
import threading
import traceback
from datetime import timedelta

from dotenv import load_dotenv

from orchestrator.services.command_service import CommandService

load_dotenv()

from orchestrator.services.user_service import UserService

from orchestrator.models.WebSocketType import WebSocketType

from flask import Flask, render_template, jsonify, request, send_file
from flask_jwt_extended import jwt_required, create_access_token, JWTManager, get_jwt_identity, verify_jwt_in_request, \
    decode_token
from flask_socketio import SocketIO, emit, disconnect, join_room, leave_room
from flask import request
from werkzeug.exceptions import HTTPException

from orchestrator.services.group_service import GroupService, group_service
from orchestrator.services.websocket_service import websocket_manager
from orchestrator.services.log_service import logService, DBLogService
from orchestrator.services.data_service import DataService

from orchestrator.models.sysytem_user import SystemUser
from orchestrator.models.user import User
from orchestrator.models.task import Task
from orchestrator.models.exceptions import OrchestratorException
from orchestrator.services.chat_service import ChatService
from orchestrator.services.processor_service import ProcessorService

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# from models.base import init_db
from models.base import SessionLocal

from apscheduler.schedulers.background import BackgroundScheduler
from services.group_service import get_groups
from services.user_service import check_and_create_containers, create_user, get_users_by_group
from services.task_service import TaskService

from flask_cors import CORS
import logging

logging.basicConfig(level=logging.ERROR)

app = Flask(__name__)
CORS(app)

app.config['JWT_SECRET_KEY'] = os.getenv("JWT_SECRET_KEY")
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(days=7)
jwt = JWTManager(app)

socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')


# socketio.init_app(app, cors_allowed_origins="*", logger=True,
#                   engineio_logger=True)


def get_current_user(db):
    verify_jwt_in_request()
    user_id = get_jwt_identity()
    return db.query(SystemUser).filter(SystemUser.id == user_id).first()


@socketio.on('connect')
def handle_connect():
    db = SessionLocal()
    token = request.args.get('token')
    if not token:
        return False
    try:
        decoded_token = decode_token(token)
        user_id = int(decoded_token.get('sub'))
        system_user = db.query(SystemUser).filter(SystemUser.id == user_id).first()
        if system_user:
            # Join user to their own room (using SID as room name)
            join_room(request.sid)
            
            websocket_manager.add_connection(system_user.id, request.sid)
            websocket_manager.send_to_user_with_format(system_user.id, WebSocketType.PING, {
                'system_user_id': system_user.id
            })
        else:
            return False
    except Exception as e:
        print(f"Token verification failed: {e}")
        return False


@socketio.on('disconnect')
def handle_disconnect():
    user_id = request.args.get('user_id')
    if user_id:
        websocket_manager.remove_connection(user_id)


@app.errorhandler(Exception)
def handle_exception(e):
    traceback.print_exc()

    # اگر اکسپشن از نوع OrchestratorException باشد
    if isinstance(e, OrchestratorException):
        return jsonify(e.to_dict()), e.status_code

    status_code = 500
    if isinstance(e, HTTPException):
        status_code = e.code

    response = {
        "error": "Internal Server Error",
        "status_code": status_code
    }

    return jsonify(response), status_code


# @app.route("/")
# def index():
#     return render_template("index.html")


@jwt_required
@app.route("/api/groups/<int:group_id>/users", methods=["GET"])
def get_users_by_group_api(group_id):
    db = SessionLocal()
    users = get_users_by_group(db, group_id)
    db.close()
    return jsonify([user.summary() for user in users])


@jwt_required
@app.route("/api/groups", methods=["GET"])
def get_groups_api():
    db = SessionLocal()
    groups = get_groups(db, get_current_user(db))
    db.close()
    return jsonify([{"id": group.id, "name": group.name, "root_user": group.root_user} for group in groups])


@jwt_required
@app.route("/api/groups", methods=["POST"])
def create_group_api():
    data = request.json
    db = SessionLocal()
    try:
        group = group_service.create_group(db, data["name"], data["root_user"], data['description'],
                                           get_current_user(db))

        group_data = {
            "id": group.id,
            "name": group.name,
            "root_user": group.root_user
        }

        db.commit()
        return jsonify(group_data)
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()


@jwt_required
@app.route("/api/users", methods=["POST"])
def create_user_api():
    data = request.json
    db = SessionLocal()
    user = create_user(db, data["name"], data["parent_user_id"], data["group_id"], data["vnc_port"])
    db.close()
    return jsonify(
        {"id": user.id, "name": user.name, "parent_user_id": user.parent_user_id, "vnc_port": user.novnc_port})


@jwt_required
@app.route("/api/users/<int:parent_user_id>/children", methods=["POST"])
def create_child_user_api(parent_user_id):
    data = request.json
    db = SessionLocal()
    user = create_user(db, data["name"], parent_user_id, None, data['description'])
    db.close()
    return jsonify(
        {"id": user.id, "name": user.name, "parent_user_id": user.parent_user_id, "vnc_port": user.novnc_port})


@jwt_required
@app.route("/api/users/<int:user_id>/tasks", methods=["GET"])
def get_tasks_for_user_api(user_id):
    db = SessionLocal()
    tasks = TaskService.get_tasks_by_user(db, user_id)
    db.close()
    return jsonify([task.summary() for task in tasks])


@jwt_required
@app.route("/api/users/<int:user_id>/tasks", methods=["POST"])
def add_task_api(user_id):
    data = request.json
    db = SessionLocal()
    task = TaskService.create_task(db, data["description"], user_id)
    db.close()
    return jsonify({"id": task.id, "description": task.description, "user_id": task.user_id})


@jwt_required
@app.route("/api/users/<int:user_id>/process_tasks", methods=["POST"])
def process_tasks_api(user_id):
    db = SessionLocal()
    try:
        service = ProcessorService(user_id, get_current_user(db))
        process_next_task = service.process_next_task
        threading.Thread(target=process_next_task, daemon=True).start()
        return jsonify({
            "message": "Task submitted to process",
        })
    except Exception as e:
        logging.error("Error processing tasks:", exc_info=True)
        return jsonify({"error": "An error occurred while processing tasks."}), 500
    finally:
        db.close()


@app.route('/signup', methods=['POST'])
def signup():
    session = SessionLocal()
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        if session.query(SystemUser).filter(SystemUser.email == email).first():
            return jsonify({"msg": "This email get registered"}), 400

        new_user = SystemUser(email=email)
        new_user.set_password(password)

        session.add(new_user)
        session.commit()

        return jsonify({"msg": "Register gets completed"}), 200
    finally:
        session.close()


@app.route('/login', methods=['POST'])
def login():
    db = SessionLocal()
    try:
        data = request.get_json()
        user = db.query(SystemUser).filter(SystemUser.email == data.get('email')).first()

        if not user or not user.check_password(data.get('password')):
            return jsonify({"msg": "incorrect username or password"}), 401

        access_token = create_access_token(identity=str(user.id))
        return jsonify(access_token=access_token), 200
    finally:
        db.close()


@jwt_required
@app.route("/api/users/<int:user_id>", methods=["PATCH"])
def update_user_settings(user_id):
    db = SessionLocal()
    try:
        data = request.get_json()
        service = UserService()
        updated_user = service.patch_user(db, data, user_id, get_current_user(db))
        return jsonify({
            "message": "User settings updated successfully",
            "user": updated_user.summary()
        })
    except Exception as e:
        logging.error("Error updating user settings:", exc_info=True)
        return jsonify({"error": "An error occurred while updating user settings."}), 500
    finally:
        db.close()


@jwt_required
@app.route("/api/users/<int:user_id>", methods=["GET"])
def get_user(user_id):
    db = SessionLocal()
    try:
        service = UserService()
        user = service.get_user(db, user_id, get_current_user(db))
        return jsonify(user.summary())
    except Exception as e:
        logging.error("Error updating user settings:", exc_info=True)
        return jsonify({"error": "An error occurred while updating user settings."}), 500
    finally:
        db.close()


@jwt_required
@app.route("/api/logs/images/<path:filename>", methods=["GET"])
def get_log_image(filename):
    image_path = os.path.join(DataService().get_logs_image_path(), filename)
    if os.path.exists(image_path):
        return send_file(image_path)
    else:
        return jsonify({"error": "Image not found"}), 404


@app.route("/api/tasks/<int:task_id>/logs", methods=["GET"])
@jwt_required()
def get_task_logs(task_id):
    try:
        db = SessionLocal()
        
        # دریافت query parameters
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 50, type=int)
        log_types_str = request.args.get('types', None)
        
        # پارس کردن log types (comma separated)
        log_types = None
        if log_types_str:
            log_types = [t.strip() for t in log_types_str.split(',') if t.strip()]
        
        step_logs = logService.get_logs_by_task_step(
            task_id, 
            get_current_user(db), 
            include_images=True,
            page=page,
            per_page=per_page,
            log_types=log_types
        )
        return jsonify(step_logs)
    except Exception as e:
        raise e
    finally:
        db.close()


# # اگر لازم داری پیام از کلاینت دریافت کنی
# @socketio.on('client_message')
# def handle_client_message(data):
#     print("پیام از کلاینت:", data)
#
#
# # متد ارسال پیام به کاربر خاص
# def send_ws_message_to_user(user_id, event, data):
#     sid = websocket_manager.active_connections.get(user_id)
#     if sid:
#         socketio.emit(event, data, room=sid)


def create_app():
    # init_db()
    
    # Reset all in_progress users to IDLE on startup
    db = SessionLocal()
    try:
        in_progress_users = db.query(User).filter(User.status == "IN_PROGRESS").all()
        for user in in_progress_users:
            user.status = "IDLE"
        db.commit()
        if in_progress_users:
            print(f"Reset {len(in_progress_users)} user(s) from IN_PROGRESS to IDLE on startup")
    except Exception as e:
        print(f"Error resetting user statuses on startup: {e}")
        db.rollback()
    finally:
        db.close()
    
    scheduler = BackgroundScheduler()
    chatService = ChatService()
    scheduler.add_job(func=check_and_create_containers, trigger="interval", minutes=1)
    scheduler.add_job(func=chatService.check_and_create_users_in_chat, trigger="interval", minutes=1)
    scheduler.add_job(func=chatService.save_new_messages_to_db, trigger="interval", minutes=2)
    scheduler.start()
    websocket_manager.socketio = socketio
    websocket_manager.app = app
    return app


def main():
    app = create_app()
    socketio.run(app, allow_unsafe_werkzeug=True)
    # sock = Sock(app)


main()
