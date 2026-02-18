from langchain.agents import create_agent
from langchain.tools import tool
from dotenv import load_dotenv
from config import IdeaValidationAgentConfig
from langchain.messages import HumanMessage
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver

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
            checkpointer=InMemorySaver(),
        )

    def validate_idea(self, user_prompt, thread_id="default"):
        config = {"configurable": {"thread_id": thread_id}}
        return self._main_agent.invoke(
            {"messages": [HumanMessage(f"my idea {user_prompt}")]}, config
        )

    @tool
    def _check_market_trends(self, trends: str | None) -> str:
        """ "Use this tool to get market trends"""
        return f"Market Trends {trends}" if trends else "No Trends"

    @tool
    def _lookup_competitors(self, competitors: str | None) -> str:
        """use this tool to get market competitors"""
        return f"Market Trends {competitors}" if competitors else "No Trends"


if __name__ == "__main__":
    agent = IdeaValidationAgent()
    print(agent.validate_idea("my start up idea is car washer"))
