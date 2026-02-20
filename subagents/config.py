from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import SystemMessage
from langchain.agents.structured_output import ToolStrategy
from pydantic import Field, BaseModel


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
        system_prompt=None
        ):
        self.model = model or ChatGoogleGenerativeAI(model="gemini-2.5-flash")
        self.system_prompt = system_prompt or SystemMessage(
            "You are a legal analysis agent. "
            "Return concise, structured output only. "
            "For each legal risk, suggest actionable steps to mitigate or resolve it."
        )
        self.response_format = ToolStrategy(LegalAgentOutput)





class SWOTAnalyzerAgentOutput(BaseModel):
    strengths: list[str] = Field(max_items=3, description="Key strengths of the startup")
    weaknesses: list[str] = Field(max_items=3, description="Key weaknesses or vulnerabilities")
    opportunities: list[str] = Field(max_items=3, description="Opportunities to grow or leverage")
    threats: list[str] = Field(max_items=3, description="External threats, legal, market, tech, etc.")
    scenarios: list[str] = Field(
        max_items=3,
        description="Possible future scenarios with context and impact"
    )
    summary: str = Field(description="Concise summary combining SWOT and scenario analysis")




class SWOTAnalyzerAgentConfig:
    def __init__(self, model=None, system_prompt=None):
        self.model = model or ChatGoogleGenerativeAI(model="gemini-2.5-flash")
        self.system_prompt = system_prompt or SystemMessage(
            "You are a Risk Analysis agent. "
            "Analyze startup ideas or operations using SWOT (Strengths, Weaknesses, Opportunities, Threats) and generate future scenarios. "
            "Return concise, structured output only. "
            "Include recommended steps to mitigate risks and leverage opportunities."
        )
        self.response_format = ToolStrategy(SWOTAnalyzerAgentOutput)