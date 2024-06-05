
from tqdm import tqdm
import time
import pandas as pd



from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
from transformers import set_seed

    
def parse(message, begin, end):
    """
    This function parses a message to find all substrings between
    a given begin_token and end_token.

    Args:
        message: The message to be parsed.
        begin_token: The starting token (inclusive).
        end_token: The ending token (inclusive).

    Returns:
        A list of all substrings found between the begin_token and end_token.
    """
    substrings = []
    start_index = 0
    while True:
        begin_loc = message.find(begin, start_index)
        if begin_loc == -1:
            break
        end_loc = message.find(end, begin_loc + len(begin))
        if end_loc == -1:
            break
        substring = message[begin_loc + len(begin):end_loc]
        substrings.append(substring)
        start_index = end_loc + len(end)
    return substrings


from agent import Agent
from tools import SearchTool

def main():
    SEED= 42
    set_seed(SEED)

    config = PeftConfig.from_pretrained("erbacher/zephyr-rag-agent", load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained("HuggingFaceH4/zephyr-7b-beta",   device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
    model = PeftModel.from_pretrained(model, "erbacher/zephyr-rag-agent",   device_map="auto")

    model = model.merge_and_unload()

    start = time.time()
    tools = [SearchTool(name = 'search', index='miracl-v1.0-en', start_token='[SEARCH]', end_token='[/SEARCH]')]
    agent = Agent( model = model,
                tokenizer = tokenizer,
                    tools = tools, rounds=4, use_tools = False, num_docs=2)


    kwargs = {'do_sample': True, "top_p": 0.5,'max_new_tokens' : 1000}

    dataset = datasets.load_dataset("miracl/hagrid", split="dev")

    results = []
    for row in tqdm(dataset):
        docids, docs_text, answer = agent.generate(row["query"] , **kwargs)
        results.append(
                    {
                        "query": row["query"],
                        "generated_text": answer,
                        "docids": docids,
                        "docs_text":docs_text,
                        "gold_truth": row["answers"],
                        "gold_quotes": row["quotes"],
                    }
                )
    end = time.time()

    execution_time = (end - start) / 60
    results_df = pd.DataFrame.from_dict(results)
    results_file = "hagrid_dev_agent_noRetrieval_stopCNoToolDetectedmax4.csv"
    results_df.to_csv(results_file)
    print("Result file:", results_file)
    print("execution_time:", execution_time)

def test():    
    config = PeftConfig.from_pretrained("erbacher/zephyr-rag-agent", load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained("HuggingFaceH4/zephyr-7b-beta",   device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
    model = PeftModel.from_pretrained(model, "erbacher/zephyr-rag-agent",   device_map="auto")

    model = model.merge_and_unload()
    tools = [SearchTool(name = 'search', index='miracl-v1.0-en', start_token='[SEARCH]', end_token='[/SEARCH]')]
    agent = Agent( model = model,
                tokenizer = tokenizer,
                    tools = tools,rounds=4, use_tools = False, num_docs=2)
    kwargs = {'do_sample': True, "top_p": 0.5,'max_new_tokens' : 1000}
    _, _, answer = agent.generate("What was the first modern cruise ship?" , **kwargs)

    a = parse(answer, '[ANSWER]', '[/ANSWER]')
    q = parse(answer, '[SEARCH]', '[/SEARCH]')
    print(a)
    print(q)
    print(answer)


main()