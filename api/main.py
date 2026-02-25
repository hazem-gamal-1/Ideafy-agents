import uuid
from fastapi import FastAPI, UploadFile, Query
from fastapi.responses import StreamingResponse
from utils.config import IdeaValidationAgentConfig, OrchestratorAgentConfig
from api.orchestrator import OrchestratorAgent
import json

app = FastAPI()


@app.post("/analyze")
async def run(
    file: UploadFile, prompt: str = Query(...), actions: list[str] = Query(...)
):
    file_bytes = await file.read()
    thread_id = str(uuid.uuid4())
    validation_config = IdeaValidationAgentConfig(actions=actions)
    orchestrator = OrchestratorAgent(
        OrchestratorAgentConfig(
            file_bytes, thread_id, validation_config=validation_config
        )
    )

    def stream_result():
        for step, data in orchestrator.stream(prompt):
            # Convert content_blocks into strings
            content_blocks = data["messages"][-1].content_blocks
            for block in content_blocks:
                yield json.dumps({"step": step, "content": block}) + "\n"

    return StreamingResponse(stream_result(), media_type="application/json")
