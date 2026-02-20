
from langchain.messages import SystemMessage
from langchain.agents.structured_output import ToolStrategy
from pydantic import Field, BaseModel
from langchain_openai import ChatOpenAI
import os



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
        self.model = model or  ChatOpenAI(
                        model="gpt-4o",
                        api_key=os.getenv("GITHUB_TOKEN"),
                        base_url="https://models.inference.ai.azure.com",
                        temperature=0.2,
                    )
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
        self.model = model or ChatOpenAI(
                        model="gpt-4o",
                        api_key=os.getenv("GITHUB_TOKEN"),
                        base_url="https://models.inference.ai.azure.com",
                        temperature=0.2,
                    )
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
        self.model = model or ChatOpenAI(
                        model="gpt-4o",
                        api_key=os.getenv("GITHUB_TOKEN"),
                        base_url="https://models.inference.ai.azure.com",
                        temperature=0.2,
                    )
        self.system_prompt = system_prompt or SystemMessage(
            "You are a Risk Analysis agent. "
            "Analyze startup ideas or operations using SWOT (Strengths, Weaknesses, Opportunities, Threats) and generate future scenarios. "
            "Return concise, structured output only. "
            "Include recommended steps to mitigate risks and leverage opportunities."
        )
        self.response_format = ToolStrategy(SWOTAnalyzerAgentOutput)





class OrchestratorAgentOutput(BaseModel):
    idea_validation: IdeaValidationOutput
    legal_analysis: LegalAgentOutput
    swot_analysis: SWOTAnalyzerAgentOutput
    overall_summary: str = Field(
        description="Concise, unified summary combining all sub-agent insights"
    )


class OrchestratorAgentConfig:
    def __init__(self,thread_id,model=None,system_prompt=None,validation_onfig=None,swot_config=None,legal_config=None):
        self.validation_onfig=IdeaValidationAgentConfig()
        self.swot_config=SWOTAnalyzerAgentConfig()
        self.legal_config=LegalAgentConfig()
        self.model=ChatOpenAI(
                        model="gpt-4o",
                        api_key=os.getenv("GITHUB_TOKEN"),
                        base_url="https://models.inference.ai.azure.com",
                        temperature=0.2,
                    )
        self.system_prompt= SystemMessage(
                                "You are an Orchestrator Agent overseeing startup evaluation and risk management. "
                                "Your role is to coordinate sub-agents for comprehensive analysis. "
                                "You have access to three sub-agents: "
                                "1. Idea Validation Agent :  assesses market potential, competition, and risks; "
                                "2. Legal Agent  : analyzes legal risks and provides actionable steps; "
                                "3. SWOT Analyzer Agent : identifies strengths, weaknesses, opportunities, threats, and future scenarios. "
                                "Always return concise, structured output combining all sub-agent analyses. "
                                "Use the provided tools to call sub-agents when needed."
                            )
        self.response_format=ToolStrategy(OrchestratorAgentOutput)
        self.thread_id=thread_id
                                    