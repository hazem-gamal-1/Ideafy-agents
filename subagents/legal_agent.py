from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

class LegalAgent:
    def __init__(self,config):
        self._config = config
        self._main_agent = create_agent(
            model=self._config.model,
            system_prompt=self._config.system_prompt,
            tools=[

            ],
            middleware=[
               
            ],
            response_format=self._config.response_format,
            checkpointer=InMemorySaver(),
        )

    def analyze_legal_risks(self, user_prompt, thread_id="default"):
        pass




if __name__ == "__main__":
    pass
    
