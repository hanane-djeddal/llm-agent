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
import argparse
import os
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


os.environ["HTTP_PROXY"] = "http://hacienda:3128"
os.environ["HTTPS_PROXY"] = "http://hacienda:3128"
#os.environ['HF_HOME'] = os.environ['WORK'] + '/.cache/huggingface'
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
from transformers import set_seed

global RAGAGENT_MODEL_NAME,  TRAINING_CORPUS, model_result_file
RAGAGENT_MODEL_NAME ="/lustre/fswork/projects/rech/fiz/udo61qq/llm-jz/llama/llama-2-chat-hagrid-att-rag-agent-13b" #zephyr-hagrid-deduplicated-rag-agent-3b/" #zephyr-hagrid-deduplicated-rag-agent-7b/" #zephyr-hagrid-deduplicated-rag-agent-3b" #zephyr-hagrid-deduplicated-finetuned-with-search-3b"#zephyr-hagrid-rag-agent-3b"#"/lustre/fswork/projects/rech/fiz/udo61qq/zephyr-rag-agent-webgpt" #   "erbacher/zephyr-rag-agent-webgpt"
TRAINING_CORPUS =  "HAGRID" #WEBGPT
model_result_file= "llama-2-chat-hagrid-att-rag-agent-13b"

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
import re

def reposition_period_after_citation(text):
    result = re.sub(r'\.\s*((\[[^\]]+\])+)(?!\S)', r' \1.', text)
    return result

def main():
    global RAGAGENT_MODEL_NAME,  TRAINING_CORPUS, model_result_file
    parser = argparse.ArgumentParser()
    parser.add_argument("--nb_rounds", type=int, default=4)
    parser.add_argument("--nb_docs", type=int, default=3)
    parser.add_argument("--resume_from_file", type=str, default=None)
    parser.add_argument("--ranker", type=str, default="GTR", choices=["GTR","MonoT5"])
    parser.add_argument("--retrieval", action="store_true")
    parser.add_argument("--inference_variant", type=str, default=None, choices=["normal", "without_query"])
    parser.add_argument(
        "--validating_code",
        action="store_true",
        help="Run short iteration to test code",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Results are written to outputdir with data suffix",
    ) 
    parser.add_argument(
        "--ragnroll_model_name",
        default= RAGAGENT_MODEL_NAME,
        type = str,
        help="Tested Model",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Tag to add in the results file",
    )
    parser.add_argument(
        "--training_corpus",
        type=str,
        default =TRAINING_CORPUS,
        help="Corpus used for training",
    )
    parser.add_argument(
        "--add_instruction",
        action="store_true",
        help="Adds instruction to prompt",
    )
    parser.add_argument(
        "--retrieve_with_answer",
        action="store_true",
        help="use generated answers for retrieval",
    )
    parser.add_argument(
        "--gen_config",
        type=str,
        default =None,
        help="Config to pass to the generator",
    )
    parser.add_argument(
        "--diverse_query_only",
        action="store_true",
        help="Only apply diversity config to query",
    )
    parser.add_argument(
        "--add_user_query",
        action="store_true",
        help="adding user query to the subqueries",
    )    
    args = parser.parse_args()
    results_path = os.environ['WORK'] +"/llm-agent/llama13/" 
    results_dir = args.output_dir if args.output_dir else results_path
    RAGAGENT_MODEL_NAME = args.ragnroll_model_name
    TRAINING_CORPUS = args.training_corpus

    logger.info(f"Loading Language model...{RAGAGENT_MODEL_NAME}")
    logger.info(f"Using HAGRID TEST SET")
    logger.info(f"Retrieval : BM25 + {args.ranker}")
    logger.info(f"Appending user query to subqueries...{args.add_user_query}")
    tag = args.tag if args.tag else  args.ragnroll_model_name.split('/')[-1]
    tag = tag+"_instruction_prompt" if args.add_instruction else tag
    tag = tag +"_using_answer_for_retrieval_" if args.retrieve_with_answer else tag
    logger.info(f"Used Tag {tag}")
    logger.info(f"Inference without query : {args.inference_variant}")
    if args.validating_code:
        logger.info(f"Only running two iterations to test")
        tag = tag + "code_validation"

    if args.gen_config:
        kwargs = json.loads(args.gen_config)
    else:
        kwargs = {"do_sample": True, "top_p": 0.5, "max_new_tokens": 1000}
    logger.info(f"Generator config...{kwargs} to query only {args.diverse_query_only}")
    SEED = 42
    set_seed(SEED)
    dataset_name = "HAGRID"  # "HAGRID"   "ALCE"
    input_file = None  # "ALCE_data/datasets/asqa_eval_dpr_top100.json"
    print("testing model :",RAGAGENT_MODEL_NAME)

    config = PeftConfig.from_pretrained(RAGAGENT_MODEL_NAME, load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-13b-chat-hf", device_map="auto" #"HuggingFaceH4/zephyr-7b-beta   stabilityai/stablelm-zephyr-3b
    )
    tokenizer = AutoTokenizer.from_pretrained(RAGAGENT_MODEL_NAME) #"HuggingFaceH4/zephyr-7b-beta")
    model = PeftModel.from_pretrained(model, RAGAGENT_MODEL_NAME, device_map="auto")

    model = model.merge_and_unload()

    start = time.time()
    if args.inference_variant == "without_query":
        retireval_start_token ="[ANSWER]"
        retireval_end_token = "[/ANSWER]"
    elif args.retrieve_with_answer:
        retireval_start_token ="[ANSWER]"
        retireval_end_token = "[/SEARCH]"
    else:
        retireval_start_token = "[SEARCH]"
        retireval_end_token = "[/SEARCH]"
    tools = [
        SearchTool(
            name="search",
            index="miracl-v1.0-en",
            start_token=retireval_start_token,
            end_token=retireval_end_token,
            reranker=args.ranker,
        )
    ]
    agent = Agent(
        model=model,
        tokenizer=tokenizer,
        tools=tools,
        rounds=args.nb_rounds, #*2,
        use_tools=True, #args.retrieval,
        num_docs=args.nb_docs,
        train_corpus=TRAINING_CORPUS,
        adjusted= False, #True,
        model_params = "7B",
        manual_stop_words= False,
        without_query_gen = args.inference_variant == "without_query",
        add_instruction = args.add_instruction,
        diverse_query_only = args.diverse_query_only,
        add_user_query = args.add_user_query
    )
    print("Adjusted", False)

    if dataset_name == "HAGRID":
        dataset = datasets.load_from_disk(os.environ["WORK"] + "/hagrid-dev") #load_dataset("miracl/hagrid", split="dev")
        query_column = "query"
    else:
        with open(input_file) as f:
            dataset = json.load(f)
        query_column = "question"

    start_idx = 0
    if args.resume_from_file:
        with open(args.resume_from_file) as f:
            data_with_config = json.load(f)
        results = data_with_config["data"]
        start_idx = len(results)
        print("Resuming test from file:",args.resume_from_file)
        print("Starting Iteration:",start_idx)
    else:
        results = []
    for nb_row, row in enumerate(tqdm(dataset)):
        if nb_row < start_idx:
            continue 
        if args.validating_code and nb_row == 2:
            break
        docs_text, scores,answer = agent.generate(row[query_column], **kwargs)
        parsed_answers = parse(answer, "[ANSWER]", "[/ANSWER]")
        output = None
        if parsed_answers:
            output = " ".join(parsed_answers)
        else:
            position = answer.find("[ANSWER]")

            if position != -1:
                start = position + len("[ANSWER]")
                output = answer[start:]

        if output is None:
            output = ""
        if TRAINING_CORPUS == "HAGRID":
             docs = []
             docids = []
             for statement_docs in docs_text:
                  for doc in statement_docs:
                       docid = doc["docid"]
                       if docid not in docids:
                            docs.append(doc)
                            docids.append(docid)
                  for i in range(len(docs)):
                       if docs and docs[i]["docid"] in output:
                            output = output.replace(docs[i]["docid"],str(i+1))
        output = reposition_period_after_citation(output)
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
                    "docs": docs,
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
                    "docs": docs,
                    "answer": row["answer"],
                    "annotations": row["annotations"],
                }
            )
        if (nb_row+1) % 150 == 0:
            results_df = {"data": results}
            results_file = results_dir + "intr_testHagrid_"+tag+"_"+str(args.nb_rounds)+"rounds_"+str(args.nb_docs)+"docs.json"  # "agent_hagrid_3doc_2rounds.csv"
            with open(results_file, "w") as writer:
                json.dump(results_df, writer)
    end = time.time()

    execution_time = (end - start) / 60
    results_df = {"data": results, "params":vars(args)}
    results_file = results_dir + "all_testHagrid_"+tag+"_"+str(args.nb_rounds)+"rounds_"+str(args.nb_docs)+"docs.json"  # "agent_hagrid_3doc_2rounds.csv"
    with open(results_file, "w") as writer:
        json.dump(results_df, writer)

    print("Result file:", results_file)
    print("execution_time:", execution_time)


def test():
    config = PeftConfig.from_pretrained(RAGAGENT_MODEL_NAME, load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained("stabilityai/stablelm-zephyr-3b", device_map="auto")
    #    "HuggingFaceH4/zephyr-7b-beta", device_map="auto"
   # )
    tokenizer = AutoTokenizer.from_pretrained("stabilityai/stablelm-zephyr-3b")#"HuggingFaceH4/zephyr-7b-beta")
    model = PeftModel.from_pretrained(model,RAGAGENT_MODEL_NAME, device_map="auto")
    
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
        train_corpus=TRAINING_CORPUS,  #"HAGRID",
        adjusted= True
    )
    kwargs = {"do_sample": True, "top_p": 0.5, "max_new_tokens": 1000}
    docs,_, answer = agent.generate("What was the first modern cruise ship?", **kwargs)
    #print(answer)
    a = parse(answer, "[ANSWER]", "[/ANSWER]")
    q = parse(answer, "[SEARCH]", "[/SEARCH]")
    # docs = parse(answer, "[DOCS]", "[/DOCS]")
    print("answer", a)
    print("queries", q)
    print(answer)
    #print(docs[0])


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
    parser.add_argument(
        "--gen_config",
        type=str,
        default =None,
        help="Config to pass to the generator",
    )
    args = parser.parse_args()

    if args.gen_config:
        kwargs = json.loads(args.gen_config)
    else:
        kwargs = {"do_sample": True, "top_p": 0.5, "max_new_tokens": 1000}
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
    results_df = {"data": results, "params":vars(args)}

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
    main()
    # alce_data()
    #test()
    #test_alce_docs_gtr()
