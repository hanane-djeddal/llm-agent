from tqdm import tqdm
import time
import pandas as pd
import json
import argparse
import src.index
import src.contriever
import src.utils
import src.slurm
import src.data
from src.evaluation import calculate_matches
import src.normalize_text

import os

#os.environ["HTTP_PROXY"] = "http://hacienda:3128"
#os.environ["HTTPS_PROXY"] = "http://hacienda:3128"
os.environ['HF_HOME'] = os.environ['WORK'] + '/.cache/huggingface'
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
from transformers import set_seed

RAGAGENT_MODEL_NAME = "/lustre/fswork/projects/rech/fiz/udo61qq/llm-jz/llama/llama-2-chat-hagrid-att-rag-agent-13b" #llama-2-chat-hagrid-rag-agent-7b/" #zephyr-webgpt-deduplicated-rag-agent-7b/" #"/lustre/fswork/projects/rech/fiz/udo61qq/llm-jz/data/zephyr-hagrid-rag-agent-7b/"  #zephyr-webgpt-rag-agent-3b-v2/"  #zephyr-webgpt-deduplicated-rag-agent-3b-v2/checkpoint-10500/" #zephyr-webgpt-rag-agent-3b/" #/linkhome/rech/geniri01/udo61qq/Code/llm-jz/data/zephyr-hagrid-deduplicated-rag-agent-7b" #zephyr-hagrid-deduplicated-rag-agent-3b" #zephyr-hagrid-deduplicated-rag-agent-7b" #zephyr-hagrid-rag-agent-7b/" #zephyr-hagrid-rag-agent-7b/" #zephyr-hagrid-deduplicated-rag-agent-3b" #zephyr-hagrid-deduplicated-finetuned-with-search-3b"#zephyr-hagrid-rag-agent-3b"#"/lustre/fswork/projects/rech/fiz/udo61qq/zephyr-rag-agent-webgpt" #   "erbacher/zephyr-rag-agent-webgpt"
TRAINING_CORPUS =  "HAGRID" #WEBGPT HAGRID


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
            end_loc = len(message)
        offset = 0
        if message[begin_loc + len(begin)] == ":":
            offset = 1
        substring = message[begin_loc + len(begin) + offset : end_loc]
        substrings.append(substring)
        start_index = end_loc + len(end)
    return substrings


from agent import Agent
from tools import SearchTool, SearchToolWithinDocs
from tools_alce import SearchToolALCE



def test():
    #set_seed(42)
    config = PeftConfig.from_pretrained(RAGAGENT_MODEL_NAME, load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-chat-hf", device_map="auto", cache_dir=os.environ['WORK'] + '/.cache/huggingface/hub')  #meta-llama/Llama-2-7b-chat-hf stabilityai/stablelm-zephyr-3b"  HuggingFaceH4/zephyr-7b-beta"
    tokenizer = AutoTokenizer.from_pretrained(RAGAGENT_MODEL_NAME,cache_dir=os.environ['WORK'] + '/.cache/huggingface/hub')
    model = PeftModel.from_pretrained(model,RAGAGENT_MODEL_NAME, device_map="auto")
    
    model = model.merge_and_unload()
    tools = [
        SearchTool(
            name="search",
            index="wikipedia-dpr", #"miracl-v1.0-en",
            start_token="[SEARCH]",
            end_token="[/SEARCH]",
            reranker= "GTR"
        )
    ]
    agent = Agent(
        model=model,
        tokenizer=tokenizer,
        tools=tools,
        rounds=6,
        use_tools=True,
        num_docs=3,
        adjusted=False, #True,
        train_corpus=TRAINING_CORPUS,  #"HAGRID",
        model_params = "7B",
        manual_stop_words= False,
    )
    kwargs = {"do_sample": True, "top_p": 0.5, "max_new_tokens": 1000}
    docs,_, answer = agent.generate("Who has the highest goals in men's world international football?", **kwargs) #What was the first modern cruise ship? Where did Jehovah's Witnesses originate?
    print(answer)
    a = parse(answer, "[ANSWER]", "[/ANSWER]")
    q = parse(answer, "[SEARCH]", "[/SEARCH]")
    # docs = parse(answer, "[DOCS]", "[/DOCS]")
    print("answer", a)
    print("queries", q)
    print(answer)
    print(docs[0])



def test_alce_docs_gtr():
    SEED = 42
    set_seed(SEED)
    parser = argparse.ArgumentParser()
    dataset_name = "ALCE"  # "HAGRID"   "ALCE"    args = parser.parse_args()

    parser.add_argument(
        "--query_file",
        type=str,
        default=None,
        help=".json file containing question and answers, similar format to reader data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Results are written to outputdir with data suffix",
    )
    args = parser.parse_args()
    src.slurm.init_distributed_mode(args)
    tools = [
        SearchToolWithinDocs(
            name="search", start_token="[SEARCH]", end_token="[/SEARCH]"
        )
    ]

    config = PeftConfig.from_pretrained(
        RAGAGENT_MODEL_NAME, load_in_8bit=True, force_download=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceH4/zephyr-7b-beta", device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
    model = PeftModel.from_pretrained(
        model, RAGAGENT_MODEL_NAME, device_map="auto", force_download=True
    )

    model = model.merge_and_unload()

    agent = Agent(
        model=model,
        tokenizer=tokenizer,
        tools=tools,
        rounds=5,
        use_tools=True,
        num_docs=3,
        train_corpus=TRAINING_CORPUS,
    )

    kwargs = {"do_sample": True, "top_p": 0.5, "max_new_tokens": 1000}

    if dataset_name == "HAGRID":
        dataset = datasets.load_dataset("miracl/hagrid", split="dev")
        query_column = "query"
    else:
        with open(args.query_file) as f:
            dataset = json.load(f)
        query_column = "question"

    start = time.time()
    results = []
    for itera, row in enumerate(tqdm(dataset)):
        docs_text, scores, answer = agent.generate(
            row[query_column], docs=row["docs"], **kwargs
        )
        docids_per_sub_query = []
        if TRAINING_CORPUS == "HAGRID":
            docs = []
            kept_docids = []
            for statement_docs in docs_text:
                docids = []
                for doc in statement_docs:
                    docids.append(doc["docid"])
                    docid = doc["docid"]
                    if docid not in kept_docids:
                        doc["indice"] = len(docs) + 1
                        docs.append(doc)
                        kept_docids.append(docid)
            docids_per_sub_query.append(docids)
        else:
            docs = docs_text

        parsed_answers = parse(answer, "[ANSWER]", "[/ANSWER]")
        if parsed_answers:
            output = " ".join(parsed_answers)
        else:
            position = answer.find("[ANSWER]")

            if position != -1:
                start = position + len("[ANSWER]")
                output = answer[start:]
        #### replace docids in answer by indices
        if TRAINING_CORPUS == "HAGRID":
            for i in range(len(docs)):
                if docs[i]["docid"] in output:
                    output = output.replace(docs[i]["docid"], str(i + 1))

        if dataset_name == "HAGRID":
            annotations = []
            for a in row["answers"]:
                annotations.append({"long_answer": a["answer"]})
            if len(annotations) < 2:
                annotations.append({"long_answer": a["answer"]})

            row["output"] = output
            row["docs"] = docs
            row["generated_text"] = answer
            row["answer"] = row["answers"][0]["answer"]
            row["annotations"] = annotations
            row["scores"] = scores
            row["docids_per_sub_query"] = docids_per_sub_query
            results.append(row)
        else:
            row["output"] = output
            row["docs"] = docs
            row["scores"] = scores
            row["generated_text"] = answer
            results.append(row)
    end = time.time()

    execution_time = (end - start) / 60
    results_df = {"data": results}
    # results_df = pd.DataFrame.from_dict(results)
    # results_df.to_csv(results_file)
    results_file = (
        args.output_dir
        if args.output_dir
        else (
            "agent_alce_asqa_3doc_rerank_GTR_5rounds_hagrid.json"  # "agent_hagrid_3doc_2rounds.csv"
        )
    )
    with open(results_file, "w") as writer:
        json.dump(results_df, writer)

    print("Result file:", results_file)
    print("execution_time:", execution_time)


if __name__ == "__main__":
    test()
    #test_alce_docs_gtr()
