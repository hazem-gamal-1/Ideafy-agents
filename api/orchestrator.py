from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.messages import HumanMessage
from dotenv import load_dotenv
from langchain.tools import tool
from subagents.idea_validation_agent import IdeaValidationAgent
from subagents.legal_agent import LegalAgent
from subagents.swot_analyzer_agent import SWOTAnalyzerAgent
from utils.utils import ContextRetrieval

load_dotenv()


class OrchestratorAgent:
    def __init__(self, config):
        self._config = config
        self._retrieve_context_tool = ContextRetrieval(
            self._config.file_bytes
        ).retrieve_context
        self._orchestrator_agent = create_agent(
            model=self._config.model,
            tools=[
                tool(self._retrieve_context_tool),
                tool(self._run_idea_validation_agent),
                tool(self._run_swot_analyzer_agent),
                tool(self._run_legal_agent),
            ],
            system_prompt=self._config.system_prompt,
            response_format=self._config.response_format,
            checkpointer=InMemorySaver(),
        )
        self._idea_validation_agent = IdeaValidationAgent(
            self._config.validation_config, self._retrieve_context_tool
        )
        self._legal_agent = LegalAgent(
            self._config.legal_config, self._retrieve_context_tool
        )
        self._swot_agent = SWOTAnalyzerAgent(
            self._config.swot_config, self._retrieve_context_tool
        )

    def _run_idea_validation_agent(self, prompt):
        """Run the Idea Validation Agent."""
        return self._idea_validation_agent.validate_idea(prompt, self._config.thread_id)

    def _run_legal_agent(self, prompt):
        """Run the Legal Agent to identify potential legal risks."""
        return self._legal_agent.analyze_legal_risks(prompt, self._config.thread_id)

    def _run_swot_analyzer_agent(self, prompt):
        """Run the SWOT Analyzer Agent to assess strategic positioning."""
        return self._swot_agent.run_scenario_analysis(prompt, self._config.thread_id)

    def invoke(self, prompt):
        config = {"configurable": {"thread_id": self._config.thread_id}, "max_concurrency": 1}
        result = self._orchestrator_agent.invoke(
            {"messages": [HumanMessage(f"{prompt}")]}, config
        )
        return result

    def stream(self, prompt):
        config = {"configurable": {"thread_id": self._config.thread_id}, "max_concurrency": 1}
        for chunk in self._orchestrator_agent.stream(
            {"messages": [HumanMessage(f"{prompt}")]},
            config,
            stream_mode="updates",
        ):
            for step, data in chunk.items():
                yield step, data
