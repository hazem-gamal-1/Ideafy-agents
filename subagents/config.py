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
        model = None,
        system_prompt=None,
    ):
        self.model = model or ChatGoogleGenerativeAI(model="gemini-2.5-flash")
        self.system_prompt = system_prompt or SystemMessage(
            "You are an idea validation agent" \
            "Return concise, structured output only."\
            "you can use provided tools to check market trends and competitors"
        )

        self.response_format = ToolStrategy(IdeaValidationOutput)





class LegalAgentOutput(BaseModel):
    legal_risks: list[str] = Field(max_items=3, description="Top 3 legal risks")
    recommended_steps: list[str] = Field(max_items=5, description="Step-by-step actions to handle legal operations")
    summary: str = Field(description="Concise summary of legal considerations")


class LegalAgentConfig:
    def __init__(
        self,
        model = None,
        system_prompt=None,
    ):
        self.model = model or ChatGoogleGenerativeAI(model="gemini-2.5-flash")
        self.system_prompt = system_prompt or SystemMessage(
            "You are a legal analysis agent. "
            "Return concise, structured output only. "
            "For each legal risk, suggest actionable steps to mitigate or resolve it."
        )
        self.response_format = ToolStrategy(LegalAgentOutput)

