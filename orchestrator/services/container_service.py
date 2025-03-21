import docker
import os
import socket

from orchestrator.models.base import SessionLocal, get_db


class ContainerService:
    def __init__(self):
        self.client = docker.from_env()

    
    def find_container_by_user(self, user):
        try:
            container = self.client.containers.get(f"orchestrator_container_{user.id}")
            if container.status != "running":
                container.start()
            return container
        except:
            return None

    def create_container(self, user, session):
        try:
            vnc_port = self.find_free_port()
            novnc_port = self.find_free_port()
            user_data_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "data", str(user.id)
            )
            os.makedirs(user_data_dir, exist_ok=True)
            container = self.client.containers.run(
                image="karam_orchestrator:latest",
                command="sleep infinity",
                detach=True,
                name=f"orchestrator_container_{user.id}",
                ports={
                    "80/tcp": novnc_port,
                    "5900/tcp": vnc_port
                },
                volumes={
                    user_data_dir: {
                        "bind": "/root/Desktop/orchestrator/data",
                        "mode": "rw",
                    }
                },
            )
            user.vnc_port = vnc_port
            
            user.novnc_port = novnc_port
            session.add(user)
            session.commit()
            return container
        except Exception as e:
            print(f"Failed to create container for user '{user.name}': {e}")
            return None

    def find_free_port(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            s.listen(1)
            return s.getsockname()[1]

    def exec_command_in_container(self, user_id: int, command: str) -> str:
        """
        Execute a command in the Docker container associated with the user.

        Args:
            user_id (int): The ID of the user.
            command (str): The command to execute in the container.

        Returns:
            str: The output of the command, or an error message if the container is not found.
        """
        container = self.client.containers.get(f"orchestrator_container_{user_id}")
        
        exec_result = container.exec_run(command)
        print(exec_result)
        
        return exec_result.output.decode("utf-8")
