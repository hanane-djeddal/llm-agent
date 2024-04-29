from agents import Agent
from tools import SearchTool

if __name__=='_main_':

    question = 'what is the capital of ..'
    tools = [SearchTool(name = 'search', index='robust04', start_token='boq', end_token='eoq')]
    agent = Agent( model = 'mymodel', tools = tools)
    kwargs = {'do_sample': True, 'top_p': 0.9}
    answer = agent.generate(question , **kwargs)
