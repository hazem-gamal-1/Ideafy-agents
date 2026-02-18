from langchain.agents import create_agent
from langchain.tools import tool
from dotenv import load_dotenv
from config import IdeaValidationAgentConfig
from langchain.messages import HumanMessage
from langchain.agents.middleware import HumanInTheLoopMiddleware

load_dotenv()


class IdeaValidationAgent:
    def __init__(self):
        self._config = IdeaValidationAgentConfig()
        self._main_agent = create_agent(
            model=self._config.model,
            system_prompt=self._config.system_prompt,
            tools=[
                self._check_market_trends,
                self._lookup_competitors,
            ],
            middleware=[
                HumanInTheLoopMiddleware(
                    interrupt_on={
                        "_lookup_competitors": {
                            "allowed_decisions": ["edit"],
                            "description": "Would you like to add market competitors ?",
                        },
                        "_check_market_trends": {
                            "allowed_decisions": ["edit"],
                            "description": "would you like to add market trends ?",
                        },
                    }
                )
            ],
            response_format=self._config.response_format,
        )

    def validate_idea(self, user_prompt):
        return self._main_agent.invoke(
            {"messages": [HumanMessage(f"my idea {user_prompt}")]}
        )

    @tool
    def _check_market_trends(self, trends):

        return f"Market Trends {trends}" if trends else "No Trends"

    @tool
    def _lookup_competitors(self, competitors):

        return f"Market Trends {competitors}" if competitors else "No Trends"


print(IdeaValidationAgent().validate_idea())
