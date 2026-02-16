from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import SystemMessage
from langchain.agents.structured_output import ToolStrategy
from pydantic import Field, BaseModel


class IdeaValidationAgentOutputFormat(BaseModel):
    pass


class IdeaValidationAgentConfig:
    def __init__(
        self,
        model=ChatGoogleGenerativeAI(model="gemini-2.5-flash"),
        system_prompt=SystemMessage(""),
        response_format=IdeaValidationAgentOutputFormat,
    ):
        self.model = model
        self.system_prompt = system_prompt
        self.response_format = ToolStrategy(response_format)
