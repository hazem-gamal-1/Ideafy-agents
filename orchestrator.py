from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.messages import HumanMessage
from dotenv import load_dotenv
from langchain.tools import tool
from config import OrchestratorAgentConfig
from subagents.idea_validation_agent import IdeaValidationAgent
from subagents.legal_agent import LegalAgent
from subagents.swot_analyzer_agent import SWOTAnalyzerAgent
load_dotenv()

class OrchestratorAgent:
    def __init__(self,config):
        self._config = config
        self._orchestrator_agent = create_agent(
            model=self._config.model,
            tools=[tool(self._run_idea_validation_agent),tool(self._run_swot_analyzer_agent),tool(self._run_legal_agent)]
            ,
            system_prompt=self._config.system_prompt,
            response_format=self._config.response_format,
            checkpointer=InMemorySaver(),
        )
        self._idea_validation_agent=IdeaValidationAgent(self._config.validation_onfig)
        self._legal_agent=LegalAgent(self._config.legal_config)
        self._swot_agent=SWOTAnalyzerAgent(self._config.swot_config)


  
    def _run_idea_validation_agent(self,prompt):
        """Run the Idea Validation Agent."""
        return self._idea_validation_agent.validate_idea(prompt,self._config.thread_id)
        


    def _run_legal_agent(self,prompt):
        """ Run the Legal Agent to identify potential legal risks."""
        return self._legal_agent.analyze_legal_risks(prompt,self._config.thread_id)



  
    def _run_swot_analyzer_agent(self,prompt):
        """ Run the SWOT Analyzer Agent to assess strategic positioning."""
        return self._swot_agent.run_scenario_analysis(prompt,self._config.thread_id)


    def run(self,prompt):
        config = {"configurable": {"thread_id": self._config.thread_id}}
        result=self._orchestrator_agent.invoke(
            {"messages": [HumanMessage(f"{prompt}")]}, config
        )
        return result['structured_response']




if __name__=="__main__":

    result=OrchestratorAgent(OrchestratorAgentConfig("3")).run("car washing")
    print(result)