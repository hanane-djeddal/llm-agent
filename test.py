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

os.environ["HTTP_PROXY"] = "http://hacienda:3128"
os.environ["HTTPS_PROXY"] = "http://hacienda:3128"

from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
from transformers import set_seed

RAGAGENT_MODEL_NAME = "erbacher/zephyr-rag-agent"  # "erbacher/zephyr-rag-agent-webgpt"
TRAINING_CORPUS = "HAGRID"


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


def main():
    SEED = 42
    set_seed(SEED)
    dataset_name = "HAGRID"  # "HAGRID"   "ALCE"
    input_file = None  # "ALCE_data/datasets/asqa_eval_dpr_top100.json"

    config = PeftConfig.from_pretrained(RAGAGENT_MODEL_NAME, load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceH4/zephyr-7b-beta", device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
    model = PeftModel.from_pretrained(model, RAGAGENT_MODEL_NAME, device_map="auto")

    model = model.merge_and_unload()

    start = time.time()
    tools = [
        SearchTool(
            name="search",
            index="miracl-v1.0-en",
            start_token="[SEARCH]",
            end_token="[/SEARCH]",
        )
    ]
    agent = Agent(
        model=model,
        tokenizer=tokenizer,
        tools=tools,
        rounds=4,
        use_tools=True,
        num_docs=3,
        train_corpus="HAGRID",
    )

    kwargs = {"do_sample": True, "top_p": 0.5, "max_new_tokens": 1000}

    if dataset_name == "HAGRID":
        dataset = datasets.load_dataset("miracl/hagrid", split="dev")
        query_column = "query"
    else:
        with open(input_file) as f:
            dataset = json.load(f)
        query_column = "question"

    results = []
    for _, row in enumerate(tqdm(dataset)):
        docs_text, answer = agent.generate(row[query_column], **kwargs)
        parsed_answers = parse(answer, "[ANSWER]", "[/ANSWER]")
        if parsed_answers:
            output = " ".join(parsed_answers)
        else:
            position = answer.find("[ANSWER]")

            if position != -1:
                start = position + len("[ANSWER]")
                output = answer[start:]
        if dataset_name == "HAGRID":
            annotations = []
            for a in row["answers"]:
                annotations.append({"long_answer": a["answer"]})
            if len(annotations) < 2:
                annotations.append({"long_answer": a["answer"]})
            results.append(
                {
                    "question": row[query_column],
                    "generated_text": answer,
                    "output": output,
                    "docs": docs_text,
                    "gold_truth": row["answers"],
                    "gold_quotes": row["quotes"],
                    "answer": row["answers"][0]["answer"],
                    "annotations": annotations,
                }
            )
        else:
            results.append(
                {
                    "question": row[query_column],
                    "generated_text": answer,
                    "output": output,
                    "docs": docs_text,
                    "answer": row["answer"],
                    "annotations": row["annotations"],
                }
            )
    end = time.time()

    execution_time = (end - start) / 60
    results_df = {"data": results}
    # results_df = pd.DataFrame.from_dict(results)
    # results_df.to_csv(results_file)
    results_file = "agent_hagrid_3doc_4rounds.json"  # "agent_hagrid_3doc_2rounds.csv"
    with open(results_file, "w") as writer:
        json.dump(results_df, writer)

    print("Result file:", results_file)
    print("execution_time:", execution_time)


def test():
    config = PeftConfig.from_pretrained(RAGAGENT_MODEL_NAME, load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceH4/zephyr-7b-beta", device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
    model = PeftModel.from_pretrained(model, RAGAGENT_MODEL_NAME, device_map="auto")

    model = model.merge_and_unload()
    tools = [
        SearchTool(
            name="search",
            index="miracl-v1.0-en",
            start_token="[SEARCH]",
            end_token="[/SEARCH]",
        )
    ]
    agent = Agent(
        model=model,
        tokenizer=tokenizer,
        tools=tools,
        rounds=4,
        use_tools=True,
        num_docs=2,
        train_corpus="WEBGPT",
    )
    kwargs = {"do_sample": True, "top_p": 0.5, "max_new_tokens": 1000}
    docs, answer = agent.generate("What was the first modern cruise ship?", **kwargs)

    a = parse(answer, "[ANSWER]", "[/ANSWER]")
    q = parse(answer, "[SEARCH]", "[/SEARCH]")
    # docs = parse(answer, "[DOCS]", "[/DOCS]")
    print("answer", a)
    print("queries", q)
    print(answer)
    print(docs[0])


def alce_data():
    SEED = 42
    set_seed(SEED)
    parser = argparse.ArgumentParser()
    dataset_name = "ALCE"  # "HAGRID"   "ALCE"

    parser.add_argument(
        "--query_file",
        type=str,
        default=None,
        help=".json file containing question and answers, similar format to reader data",
    )
    parser.add_argument(
        "--passages", type=str, default=None, help="Path to passages (.tsv file)"
    )
    parser.add_argument(
        "--passages_embeddings",
        type=str,
        default=None,
        help="Glob path to encoded passages",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Results are written to outputdir with data suffix",
    )
    parser.add_argument(
        "--n_docs",
        type=int,
        default=100,
        help="Number of documents to retrieve per questions",
    )
    parser.add_argument(
        "--validation_workers",
        type=int,
        default=32,
        help="Number of parallel processes to validate results",
    )
    parser.add_argument(
        "--per_gpu_batch_size",
        type=int,
        default=64,
        help="Batch size for question encoding",
    )
    parser.add_argument(
        "--save_or_load_index",
        action="store_true",
        help="If enabled, save index and load index if it exists",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="path to directory containing model weights and config file",
    )
    parser.add_argument("--no_fp16", action="store_true", help="inference in fp32")
    parser.add_argument(
        "--question_maxlength",
        type=int,
        default=512,
        help="Maximum number of tokens in a question",
    )
    parser.add_argument(
        "--indexing_batch_size",
        type=int,
        default=1000000,
        help="Batch size of the number of passages indexed",
    )
    parser.add_argument("--projection_size", type=int, default=768)
    parser.add_argument(
        "--n_subquantizers",
        type=int,
        default=0,
        help="Number of subquantizer used for vector quantization, if 0 flat index is used",
    )
    parser.add_argument(
        "--n_bits", type=int, default=8, help="Number of bits per subquantizer"
    )
    parser.add_argument("--lang", nargs="+")
    parser.add_argument("--dataset", type=str, default="none")
    parser.add_argument(
        "--lowercase", action="store_true", help="lowercase text before encoding"
    )
    parser.add_argument("--normalize_text", action="store_true", help="normalize text")

    args = parser.parse_args()
    src.slurm.init_distributed_mode(args)
    tools = [
        SearchToolALCE(
            name="search", start_token="[SEARCH]", end_token="[/SEARCH]", args=args
        )
    ]

    config = PeftConfig.from_pretrained(RAGAGENT_MODEL_NAME, load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceH4/zephyr-7b-beta", device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
    model = PeftModel.from_pretrained(model, RAGAGENT_MODEL_NAME, device_map="auto")

    model = model.merge_and_unload()

    agent = Agent(
        model=model,
        tokenizer=tokenizer,
        tools=tools,
        rounds=5,
        use_tools=True,
        num_docs=3,
        train_corpus="HAGRID",
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
    for _, row in enumerate(tqdm(dataset)):
        docs_text, answer = agent.generate(row[query_column], **kwargs)
        docs = []
        kept_docids = []
        for statement_docs in docs_text:
            for doc in statement_docs:
                docid = doc["docid"]
                if docid not in kept_docids:
                    doc["indice"] = len(docs) + 1
                    docs.append(doc)
                    kept_docids.append(docid)

        parsed_answers = parse(answer, "[ANSWER]", "[/ANSWER]")
        if parsed_answers:
            output = " ".join(parsed_answers)
        else:
            position = answer.find("[ANSWER]")

            if position != -1:
                start = position + len("[ANSWER]")
                output = answer[start:]
        #### replace docids in answer by indices
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
            results.append(row)
        else:
            row["output"] = output
            row["docs"] = docs
            row["generated_text"] = answer
            results.append(row)
    end = time.time()

    execution_time = (end - start) / 60
    results_df = {"data": results}
    # results_df = pd.DataFrame.from_dict(results)
    # results_df.to_csv(results_file)
    results_file = (
        "agent_alce_asqa_3doc_5rounds_trainH.json"  # "agent_hagrid_3doc_2rounds.csv"
    )
    with open(results_file, "w") as writer:
        json.dump(results_df, writer)

    print("Result file:", results_file)
    print("execution_time:", execution_time)


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
    # main()
    # alce_data()
    # test()
    test_alce_docs_gtr()
