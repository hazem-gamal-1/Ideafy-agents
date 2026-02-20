from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.messages import HumanMessage
from config import SWOTAnalyzerAgentConfig
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
load_dotenv()

class SWOTAnalyzerAgent:
    def __init__(self,config):
        self._config = config
        self._main_agent = create_agent(
            model=self._config.model,
            system_prompt=self._config.system_prompt,
            response_format=self._config.response_format,
            checkpointer=InMemorySaver(),
        )

    def run_scenario_analysis(self, user_prompt, thread_id="default"):
        config = {"configurable": {"thread_id": thread_id}}
        result=self._main_agent.invoke(
            {"messages": [HumanMessage(f"my idea {user_prompt}")]}, config
        )
        return result





if __name__ == "__main__":
    config=SWOTAnalyzerAgentConfig(model=ChatOpenAI(
    model="gpt-4o",
    api_key=os.getenv("GITHUB_TOKEN"),
    base_url="https://models.inference.ai.azure.com",
    temperature=0.2,
    ))
    agent = SWOTAnalyzerAgent(config)
    result=agent.run_scenario_analysis("my start up idea is car washer")
    print(result)
