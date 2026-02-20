from click import prompt
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.messages import HumanMessage
from dotenv import load_dotenv
from langchain.tools import tool
load_dotenv()

class OrchestratorAgent:
    def __init__(self,config):
        self._config = config
        self._main_agent = create_agent(
            model=self._config.model,
            system_prompt=self._config.system_prompt,
            response_format=self._config.response_format,
            checkpointer=InMemorySaver(),
        )


    @staticmethod
    @tool
    def _run_idea_validation_agent(self,prompt):
        """Run the Idea Validation Agent."""
        

    @staticmethod
    @tool
    def _run_legal_agent(self,prompt):
        """ Run the Legal Agent to identify potential legal risks."""



    @staticmethod
    @tool
    def _run_swot_analyzer_agent(self,prompt):
        """ Run the SWOT Analyzer Agent to assess strategic positioning."""


    def run(self,thread_id="default"):
        config = {"configurable": {"thread_id": thread_id}}
        prompt=" "
        result=self._main_agent.invoke(
            {"messages": [HumanMessage(f"{prompt}")]}, config
        )
        return result['structured_response']




