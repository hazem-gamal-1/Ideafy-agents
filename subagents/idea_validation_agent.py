from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import HumanMessage
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command


class IdeaValidationAgent:
    def __init__(self, config, retrieve_context_tool):
        self._config = config
        self._agent = create_agent(
            model=self._config.model,
            system_prompt=self._config.system_prompt,
            tools=[
                tool(retrieve_context_tool),
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

    def validate_idea(self, prompt, thread_id="default"):
        config = {"configurable": {"thread_id": thread_id}}
        result = self._agent.invoke({"messages": [HumanMessage(f"{prompt}")]}, config)

        decisions = []

        for action in result["__interrupt__"][0].value["action_requests"]:
            print(action["name"])
            value = input(f"{action["description"]} ")
            key, _ = next(iter(action["args"].items()))
            decisions.append(
                {
                    "type": "edit",
                    "edited_action": {"name": action["name"], "args": {key: value}},
                }
            )

        final_result = self._agent.invoke(
            Command(resume={"decisions": decisions}), config=config
        )
        return final_result["structured_response"]

    @staticmethod
    @tool
    def _check_market_trends(trends: str | None) -> str:
        """ "Use this tool to get market trends"""
        return f"{trends}"

    @staticmethod
    @tool
    def _lookup_competitors(competitors: str | None) -> str:
        """use this tool to get market competitors"""
        return f"{competitors}"
