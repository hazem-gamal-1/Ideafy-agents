from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import SystemMessage
from langchain.agents.structured_output import ToolStrategy
from pydantic import Field, BaseModel
from typing import Literal


class IdeaValidationOutput(BaseModel):
    market_score: float = Field(ge=0, le=10)
    competition_score: float = Field(ge=0, le=10)
    risks: list[str] = Field(max_items=3)
    summary: str = Field(description="Summary of the analysis")


class IdeaValidationAgentConfig:
    def __init__(
        self,
        model: ChatGoogleGenerativeAI | None = None,
        system_prompt: SystemMessage | None = None,
    ):
        self.model = model or ChatGoogleGenerativeAI(model="gemini-2.5-flash")
        self.system_prompt = system_prompt or SystemMessage(
            "You are an idea validation agent" \
            "Return concise, structured output only."\
            "you can use provided tools to check market trends and competitors"
        )

        self.response_format = ToolStrategy(IdeaValidationOutput)
