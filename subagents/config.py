from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import SystemMessage
from langchain.agents.structured_output import ToolStrategy
from pydantic import Field, BaseModel
from typing import Literal


class IdeaValidationOutput(BaseModel):
    decision: Literal["proceed", "iterate", "reject"]

    market_score: float = Field(ge=0, le=10)
    competition_score: float = Field(ge=0, le=10)
    feasibility_score: float = Field(ge=0, le=10)

    risks: list[str] = Field(max_items=3)

    next_steps: list[str] = Field(..., max_items=3)

    confidence: float = Field(..., ge=0, le=1)
    iteration: int


class IdeaValidationAgentConfig:
    def __init__(
        self,
        model: ChatGoogleGenerativeAI | None = None,
        system_prompt: SystemMessage | None = None,
    ):
        self.model = model or ChatGoogleGenerativeAI(model="gemini-2.5-flash")
        self.system_prompt = system_prompt or SystemMessage(
            "You are an idea validation agent. "
            "Return concise, structured output only."
        )
        
        self.response_format = ToolStrategy(IdeaValidationOutput)
