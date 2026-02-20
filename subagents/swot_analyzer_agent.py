from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.messages import HumanMessage
from dotenv import load_dotenv
load_dotenv()

class SWOTAnalyzerAgent:
    def __init__(self,config):
        self._config = config
        self._agent = create_agent(
            model=self._config.model,
            system_prompt=self._config.system_prompt,
            response_format=self._config.response_format,
            checkpointer=InMemorySaver(),
        )

    def run_scenario_analysis(self, prompt, thread_id="default"):
        config = {"configurable": {"thread_id": thread_id}}
        result=self._agent.invoke(
            {"messages": [HumanMessage(f"{prompt}")]}, config
        )
        return result['structured_response']




