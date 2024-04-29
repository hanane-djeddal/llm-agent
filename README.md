# llm-agent
Small minimalist library for llm based agent

Only for inference

Models need to be finetuned on tool usage

Current Agent: Load a pretrained causal LLM and can call multiple tools during generation (until eos)

Current SearchTool: retrieve document in pyserini index

```py
from agents import Agent
from tools import SearchTool
question = 'what is the capital of France'
tools = [SearchTool(name = 'search', index='robust04', start_token='boq', end_token='eoq')]
agent = Agent( model = 'mymodel', tools = tools)
kwargs = {'do_sample': True, 'top_p': 0.9}
answer = agent.generate(question , **kwargs)
```
