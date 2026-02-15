from langchain.agents import create_agent 
from langchain.messages import HumanMessage,SystemMessage
from langchain.tools import tool

from dotenv import load_dotenv
load_dotenv()



class IdeaValidationAgent:
    def __init__(self):
        self.system_prompt=SystemMessage("")



    @tool
    def check_market_trends():
        pass

    @tool
    def lookup_competitors():
        pass


    @tool
    def score_feasibility():
        pass

    @tool
    def validate_survey_responses():
        pass

    



    
    







