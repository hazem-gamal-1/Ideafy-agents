from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.messages import HumanMessage
from langchain.tools import tool


class SWOTAnalyzerAgent:
    def __init__(self, config, retrieve_context_tool):
        self._config = config
        self._agent = create_agent(
            model=self._config.model,
            tools=[tool(retrieve_context_tool)],
            system_prompt=self._config.system_prompt,
            response_format=self._config.response_format,
            checkpointer=InMemorySaver(),
        )

    def run_scenario_analysis(self, prompt, thread_id="default"):
        config = {"configurable": {"thread_id": thread_id}, "max_concurrency": 1}
        result = self._agent.invoke({"messages": [HumanMessage(f"{prompt}")]}, config)
        return result["structured_response"]
