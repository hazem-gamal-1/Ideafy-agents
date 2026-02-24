import uuid
from fastapi import FastAPI, UploadFile
from utils.config import IdeaValidationAgentConfig, OrchestratorAgentConfig
from orchestrator import OrchestratorAgent

app = FastAPI()

app.post("/analyze")


async def run(file: UploadFile, prompt: str, actions: str):
    file_bytes = await file.read()
    thread_id = str(uuid.uuid4())
    validation_config = IdeaValidationAgentConfig(actions=actions)
    orchestrator = OrchestratorAgent(
        OrchestratorAgentConfig(
            file_bytes, thread_id, validation_config=validation_config
        )
    )
    result = orchestrator.run(prompt)
    return result
