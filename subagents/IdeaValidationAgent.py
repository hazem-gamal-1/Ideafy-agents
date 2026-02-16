from langchain.agents import create_agent
from langchain.tools import tool
from dotenv import load_dotenv
from config import IdeaValidationAgentConfig
from langchain.messages import HumanMessage

load_dotenv()


class IdeaValidationAgent:
    def __init__(self):
        self._config = IdeaValidationAgentConfig()
        self._agent = create_agent(
            model=self._config.model,
            system_prompt=self._config.system_prompt,
            # tools=[
            #     self._check_market_trends,
            #     self._lookup_competitors,
            #     self._score_feasibility,
            #     self._validate_survey_responses,
            # ],
            response_format=self._config.response_format,
        )

    def validate_idea(self):
        idea = input("Enter your idea")
        return self._agent.invoke({"messages": [HumanMessage(f"my idea {idea}")]})

    @tool
    def _check_market_trends(self):
        """
        Returns a trend score and top keywords.
        Human can:
        - approve trends
        - adjust score
        - flag for further research
        - add resources (for RAG)
        """
        pass

    @tool
    def _lookup_competitors(self):
        """
        Returns top competitors with funding and market share.
        Human can:
        - approve competitor list
        - add missing competitors
        - adjust numeric metrics
        - add resources (for RAG)
        """
        pass

    @tool
    def _score_feasibility(self):
        """
        Returns feasibility score, risk factors, and next steps.
        Human can:
        - approve or reject the score
        - adjust numeric score
        - edit risk factors and next steps
        - add resources (for RAG)
        """

    @tool
    def _validate_survey_responses(self):
        """
        Summarizes survey responses.
        Human can:
        - approve summary
        - adjust counts
        - add qualitative notes
        - add resources (for RAG)
        """
        pass


print(IdeaValidationAgent().validate_idea())
