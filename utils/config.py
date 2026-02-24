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
    def __init__(self, model=None, system_prompt=None, actions=None):
        self.model = model or ChatOpenAI(
            model="gpt-4o",
            api_key=os.getenv("GITHUB_TOKEN"),
            base_url="https://models.inference.ai.azure.com",
            temperature=0.2,
        )
        self.system_prompt = system_prompt or SystemMessage(
            "You are the Idea Validation Agent. Your task is to evaluate startup ideas for market potential, competition, and risks. "
            "Before generating your structured output, first determine exactly what information you need from the startup knowledge base (market trends, competitors, risks) to answer this query. "
            "Then retrieve relevant context using the 'retrieve_context' tool. "
            "Finally, return ONLY JSON following the IdeaValidationOutput schema. "
            "Do not include explanations, commentary, or any text outside the JSON."
        )

        self.response_format = ToolStrategy(IdeaValidationOutput)
        self.actions = actions


class LegalAgentOutput(BaseModel):
    legal_risks: list[str] = Field(max_items=3, description="Top 3 legal risks")
    recommended_steps: list[str] = Field(
        max_items=5, description="Step-by-step actions to handle legal operations"
    )
    summary: str = Field(description="Concise summary of legal considerations")


class LegalAgentConfig:
    def __init__(self, model=None, system_prompt=None):
        self.model = model or ChatOpenAI(
            model="gpt-4o",
            api_key=os.getenv("GITHUB_TOKEN"),
            base_url="https://models.inference.ai.azure.com",
            temperature=0.2,
        )
        self.system_prompt = system_prompt or SystemMessage(
            "You are the Legal Analysis Agent. Your task is to identify the top legal risks for a startup and provide actionable steps. "
            "Before generating output, first decide exactly what laws, regulations, or compliance context you need to retrieve to make an informed legal assessment. "
            "Use the 'retrieve_context' tool to gather this information. "
            "Then return ONLY JSON following the LegalAgentOutput schema. "
            "Do not include any text outside the JSON."
        )
        self.response_format = ToolStrategy(LegalAgentOutput)


class SWOTAnalyzerAgentOutput(BaseModel):
    strengths: list[str] = Field(
        max_items=3, description="Key strengths of the startup"
    )
    weaknesses: list[str] = Field(
        max_items=3, description="Key weaknesses or vulnerabilities"
    )
    opportunities: list[str] = Field(
        max_items=3, description="Opportunities to grow or leverage"
    )
    threats: list[str] = Field(
        max_items=3, description="External threats, legal, market, tech, etc."
    )
    scenarios: list[str] = Field(
        max_items=3, description="Possible future scenarios with context and impact"
    )
    summary: str = Field(
        description="Concise summary combining SWOT and scenario analysis"
    )


class SWOTAnalyzerAgentConfig:
    def __init__(self, model=None, system_prompt=None):
        self.model = model or ChatOpenAI(
            model="gpt-4o",
            api_key=os.getenv("GITHUB_TOKEN"),
            base_url="https://models.inference.ai.azure.com",
            temperature=0.2,
        )
        self.system_prompt = system_prompt or SystemMessage(
            "You are the SWOT Analyzer Agent. Your task is to analyze strengths, weaknesses, opportunities, threats, and future scenarios for a startup. "
            "Before generating your response, first identify exactly what market, competitor, and operational information you need from the knowledge base. "
            "Use the 'retrieve_context' tool to collect this information. "
            "Then return ONLY JSON following the SWOTAnalyzerAgentOutput schema. "
            "Do not include explanations, commentary, or text outside the JSON."
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
    def __init__(
        self,
        file_bytes,
        thread_id,
        model=None,
        system_prompt=None,
        validation_config=None,
        swot_config=None,
        legal_config=None,
    ):
        self.validation_config = validation_config or IdeaValidationAgentConfig()
        self.swot_config = swot_config or SWOTAnalyzerAgentConfig()
        self.legal_config = legal_config or LegalAgentConfig()
        self.model = model or ChatOpenAI(
            model="gpt-4o",
            api_key=os.getenv("GITHUB_TOKEN"),
            base_url="https://models.inference.ai.azure.com",
            temperature=0.2,
        )
        self.system_prompt = system_prompt or SystemMessage(
            "You are the Orchestrator Agent. Your role is to coordinate three sub-agents: Idea Validation, Legal Analysis, and SWOT Analysis. "
            "For each sub-agent, follow these steps: "
            "1) Let the sub-agent determine what context it needs to answer the query. "
            "2) Use the 'retrieve_context' tool to provide relevant information. "
            "3) Have the sub-agent generate its structured JSON output. "
            "After collecting all sub-agent outputs, combine them into a single JSON with keys: idea_validation, legal_analysis, swot_analysis, overall_summary. "
            "Return ONLY JSON following the OrchestratorAgentOutput schema. "
            "Do not add explanations, commentary, or any text outside the JSON."
        )
        self.response_format = ToolStrategy(OrchestratorAgentOutput)
        self.file_bytes = file_bytes
        self.thread_id = thread_id
