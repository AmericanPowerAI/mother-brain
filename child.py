import docker
import hashlib
import json
import logging
from typing import Dict, Any
import requests
import tempfile
import subprocess
from pathlib import Path

class ChildNode:
    def __init__(self, mother_url: str, node_id: str):
        self.mother_url = mother_url
        self.node_id = node_id
        self.sandbox = docker.from_env().containers.run(
            "python:3.9-slim",
            detach=True,
            tty=True,
            mem_limit="100m",
            network_mode="none",
            volumes={'/tmp': {'bind': '/tmp', 'mode': 'ro'}}
        )
        self.logger = self._setup_logging()
        
    def _setup_logging(self):
        logging.basicConfig(
            filename=f'child_{self.node_id}.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger()

    def heartbeat(self) -> Dict[str, Any]:
        try:
            resp = requests.post(
                f"{self.mother_url}/heartbeat",
                json={"node_id": self.node_id},
                timeout=5
            )
            return resp.json()
        except Exception as e:
            self.logger.error(f"Heartbeat failed: {str(e)}")
            return {"status": "offline"}

    def execute_command(self, command: Dict) -> Dict:
        # Security validation
        if not self._validate_command(command):
            return {"error": "Invalid command structure"}
            
        try:
            # Isolated execution
            result = self._run_in_sandbox(command)
            
            # Checksum verification
            result['integrity'] = hashlib.sha256(
                json.dumps(result).encode()
            ).hexdigest()
            
            return result
        except Exception as e:
            self.logger.error(f"Execution failed: {str(e)}")
            return {"error": str(e), "command": command}

    def _validate_command(self, command: Dict) -> bool:
        required = ['action', 'signature', 'timestamp']
        return all(k in command for k in required)

    def _run_in_sandbox(self, command: Dict):
        # Write to temp file in sandbox
        with tempfile.NamedTemporaryFile() as tmp:
            tmp.write(json.dumps(command).encode())
            tmp.flush()
            
            # Execute in container with resource limits
            exit_code, output = self.sandbox.exec_run(
                f"python -c 'import json; cmd=json.load(open(\"{tmp.name}\")); print(\"EXEC:\", cmd[\"action\"])'",
                demux=True
            )
            
            return {
                "exit_code": exit_code,
                "output": output.decode() if output else "",
                "node_id": self.node_id
            }

if __name__ == "__main__":
    child = ChildNode(
        mother_url="https://mother-brain.example.com",
        node_id="child_01"
    )
    
    while True:
        cmd = child.heartbeat()
        if cmd.get("command"):
            result = child.execute_command(cmd["command"])
            requests.post(
                f"{child.mother_url}/report",
                json=result
            )
