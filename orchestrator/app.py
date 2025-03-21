import sys
import os

from flask import Flask, render_template, jsonify, request

from orchestrator.services.processor_service import ProcessorService

# API to process tasks for a user

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.base import init_db
from models.base import SessionLocal

from apscheduler.schedulers.background import BackgroundScheduler
from services.group_service import create_group, get_groups
from services.user_service import check_and_create_containers, create_user, get_users_by_group
from services.task_service import TaskService

from flask_cors import CORS
import logging

logging.basicConfig(level=logging.ERROR)  # فقط خطاها را لاگ کنید

app = Flask(__name__)
CORS(app)  # فعال‌سازی CORS برای تمام روت‌ها


# Route for the main page
@app.route("/")
def index():
    return render_template("index.html")


# API to get all groups
@app.route("/api/groups", methods=["GET"])
def get_groups_api():
    db = SessionLocal()
    groups = get_groups(db)
    db.close()
    return jsonify([{"id": group.id, "name": group.name, "root_user": group.root_user} for group in groups])


# API to create a new group
@app.route("/api/groups", methods=["POST"])
def create_group_api():
    data = request.json
    db = SessionLocal()
    try:
        # ایجاد گروه
        group = create_group(db, data["name"], data["root_user"], data['description'])

        # ذخیره اطلاعات گروه قبل از بستن Session
        group_data = {
            "id": group.id,
            "name": group.name,
            "root_user": group.root_user
        }

        db.commit()  # تغییرات را commit کنید
        return jsonify(group_data)
    except Exception as e:
        db.rollback()  # در صورت خطا، تغییرات را rollback کنید
        raise e
    finally:
        db.close()  # در هر صورت Session را ببندید


# API to get users by group
@app.route("/api/groups/<int:group_id>/users", methods=["GET"])
def get_users_by_group_api(group_id):
    db = SessionLocal()
    users = get_users_by_group(db, group_id)
    db.close()
    return jsonify(
        [{"id": user.id, "name": user.name, "parent_user_id": user.parent_user_id, "vnc_port": user.novnc_port} for user
         in users])


# API to create a new user
@app.route("/api/users", methods=["POST"])
def create_user_api():
    data = request.json
    db = SessionLocal()
    user = create_user(db, data["name"], data["parent_user_id"], data["group_id"], data["vnc_port"])
    db.close()
    return jsonify({"id": user.id, "name": user.name, "parent_user_id": user.parent_user_id, "vnc_port": user.novnc_port})


# API to create a child user
@app.route("/api/users/<int:parent_user_id>/children", methods=["POST"])
def create_child_user_api(parent_user_id):
    data = request.json
    db = SessionLocal()
    user = create_user(db, data["name"], parent_user_id, None, data['description'])
    db.close()
    return jsonify({"id": user.id, "name": user.name, "parent_user_id": user.parent_user_id, "vnc_port": user.novnc_port})


# API to get tasks for a user
@app.route("/api/users/<int:user_id>/tasks", methods=["GET"])
def get_tasks_for_user_api(user_id):
    db = SessionLocal()
    tasks = TaskService.get_tasks_by_user(db, user_id)
    db.close()
    return jsonify([{"id": task.id, "description": task.description, "user_id": task.user_id} for task in tasks])


# API to add a task
@app.route("/api/users/<int:user_id>/tasks", methods=["POST"])
def add_task_api(user_id):
    data = request.json
    db = SessionLocal()
    task = TaskService.create_task(db, data["description"], user_id)
    db.close()
    return jsonify({"id": task.id, "description": task.description, "user_id": task.user_id})


@app.route("/api/users/<int:user_id>/process_tasks", methods=["POST"])
def process_tasks_api(user_id):
    db = SessionLocal()
    try:
        tasks = TaskService.get_tasks_by_user(db, user_id)
        service = ProcessorService(db, tasks[0])
        service.process_task()
        # prompt, output = TaskService.process_tasks(db, user_id)
        return jsonify({
            "message": "Tasks processed successfully",
            "prompt": "",
            "output": ""
        })
    except Exception as e:
        # ثبت کامل خطا در لاگ
        logging.error("Error processing tasks:", exc_info=True)
        return jsonify({"error": "An error occurred while processing tasks."}), 500
    finally:
        db.close()  # در هر صورت Session را ببندید


def main():
    """
    Start the Flask server.
    """
    init_db()
    scheduler = BackgroundScheduler()
    scheduler.add_job(func=check_and_create_containers, trigger="interval", minutes=1)
    scheduler.start()
    app.run(debug=True)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
