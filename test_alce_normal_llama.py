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
import re
import os

#os.environ["HTTP_PROXY"] = "http://hacienda:3128"
#os.environ["HTTPS_PROXY"] = "http://hacienda:3128"
os.environ['HF_HOME'] = os.environ['WORK'] + '/.cache/huggingface/'
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
from transformers import set_seed
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

global RAGAGENT_MODEL_NAME,  TRAINING_CORPUS, model_result_file
RAGAGENT_MODEL_NAME = "/lustre/fswork/projects/rech/fiz/udo61qq/llm-jz/llama/llama-2-chat-hagrid-att-param-knw-rag-agent-13b" #llama-2-chat-hagrid-att-rag-agent-13b"  #zephyr-hagrid-deduplicated-finetuned-with-search-3b"#zephyr-hagrid-rag-agent-3b"#"/lustre/fswork/projects/rech/fiz/udo61qq/zephyr-rag-agent-webgpt" #   "erbacher/zephyr-rag-agent-webgpt"
TRAINING_CORPUS =  "HAGRID" #WEBGPT
model_result_file= "llama-2-chat-hagrid-att-param-knw-rag-agent-13b"

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

def reposition_period_after_citation(text):
    result = re.sub(r'\.\s*((\[[^\]]+\])+)(?!\S)', r' \1.', text)
    return result


def test_alce_docs_gtr():
    print(os.environ.get('HF_HOME'))
    SEED = 42
    set_seed(SEED)
    results_path= "/lustre/fswork/projects/rech/fiz/udo61qq/llm-agent/llama13/"
    parser = argparse.ArgumentParser()
    dataset_name = "ALCE"  # "HAGRID"   "ALCE"    args = parser.parse_args()
    global RAGAGENT_MODEL_NAME,  TRAINING_CORPUS, model_result_file
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
        "--validating_code",
        action="store_true",
        help="Run short iteration to test code",
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
    
    parser.add_argument("--ranker", type=str, default="GTR", choices=["GTR","MonoT5"])
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
    parser.add_argument(
        "--startindex",
        type=int,
        default=0,
        help="Index to start iterations",
    )
    parser.add_argument(
        "--stopindex",
        type=int,
        default=None,
        help="Index to stop iterations",
    )
    parser.add_argument("--nb_rounds", type=int, default=4)
    parser.add_argument("--nb_docs", type=int, default=3)
    parser.add_argument("--resume_from_file", type=str, default=None)
    args = parser.parse_args()
    src.slurm.init_distributed_mode(args)

    results_dir = args.output_dir if args.output_dir else results_path
    RAGAGENT_MODEL_NAME = args.ragnroll_model_name
    TRAINING_CORPUS = args.training_corpus

    logger.info(f"Loading Language model...{RAGAGENT_MODEL_NAME}")
    logger.info(f"Using ASAQ TEST SET")
    logger.info(f"Retrieval : {args.ranker} top 100 reranked")
    logger.info(f"Adding instruction...{args.add_instruction}")
    logger.info(f"Using answer for retrieval...{args.retrieve_with_answer}")
    logger.info(f"Appending user query to subqueries...{args.add_user_query}")
    tag = args.tag if args.tag else  args.ragnroll_model_name.split('/')[-1]
    tag = tag+"_instruction_prompt" if args.add_instruction else tag
    tag = tag +"_using_answer_for_retrieval" if args.retrieve_with_answer else tag
    tag = tag +"_diverse_query_only" if args.diverse_query_only else tag
    logger.info(f"Used Tag {tag}")
    if args.validating_code:
        logger.info(f"Only running two iterations to test")
        tag = tag + "code_validation" 
    if args.retrieve_with_answer:
        retireval_start_token ="[ANSWER]"
        retireval_end_token = "[/SEARCH]"
    else: 
        retireval_start_token ="[SEARCH]"
        retireval_end_token = "[/SEARCH]"
    if args.gen_config:
        kwargs = json.loads(args.gen_config)
    else:
        kwargs = {"do_sample": True, "top_p": 0.5, "max_new_tokens": 1000}
    logger.info(f"Generator config...{kwargs} to query only {args.diverse_query_only}")

    tools = [
        SearchToolWithinDocs(
            name="search", start_token=retireval_start_token,end_token=retireval_end_token, reranker=args.ranker,
        )
    ]
    config = PeftConfig.from_pretrained(
        RAGAGENT_MODEL_NAME, load_in_8bit=True, force_download=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-13b-chat-hf", device_map="auto", cache_dir=os.environ['WORK'] + '/.cache/huggingface/hub' # HuggingFaceH4/zephyr-7b-beta stabilityai/stablelm-zephyr-3b
    )
    tokenizer = AutoTokenizer.from_pretrained(RAGAGENT_MODEL_NAME)
    model = PeftModel.from_pretrained(
        model, RAGAGENT_MODEL_NAME, device_map="auto", force_download=True
    )

    model = model.merge_and_unload()

    agent = Agent(
        model=model,
        tokenizer=tokenizer,
        tools=tools,
        rounds=args.nb_rounds,
        use_tools=True,
        num_docs=args.nb_docs,
        train_corpus=TRAINING_CORPUS,
        adjusted= False,
        model_params = "7B",
        manual_stop_words= False,
        add_instruction = args.add_instruction,
        diverse_query_only = args.diverse_query_only,
        add_user_query = args.add_user_query,
        add_answer = args.retrieve_with_answer,
    )


    if dataset_name == "HAGRID":
        dataset = datasets.load_dataset("miracl/hagrid", split="dev")
        query_column = "query"
    else:
        with open(args.query_file) as f:
            dataset = json.load(f)
        query_column = "question"

    start = time.time()
    start_idx = args.startindex
    if args.resume_from_file:
        with open(args.resume_from_file) as f:
            data_with_config = json.load(f)
        results = data_with_config["data"]
        start_idx = len(results)
        print("Resuming test from file:",args.resume_from_file)
    else:
        results = []
        
    print("Starting Iteration:",start_idx)
    for itera, row in enumerate(tqdm(dataset)):
        if itera < start_idx:
            continue
        if args.validating_code:
            if itera == 2:
                break
        docs_text, scores, answer = agent.generate(
            row[query_column], docs=row["docs"], **kwargs
        )
        docids_per_sub_query = []
        #if TRAINING_CORPUS == "HAGRID":
        #    docs = []
        #    kept_docids = []
        #    for statement_docs in docs_text:
        #        docids = []
        #        for doc in statement_docs:
        #            docids.append(doc["docid"])
        #            docid = doc["docid"]
        #            if docid not in kept_docids:
        #                doc["indice"] = len(docs) + 1
        #                docs.append(doc)
        #                kept_docids.append(docid)
        #    docids_per_sub_query.append(docids)
        #else:
        docs = docs_text
        output = ""
        parsed_answers = parse(answer, "[ANSWER]", "[/ANSWER]")
        if parsed_answers:
            output = " ".join(parsed_answers)
        else:
            position = answer.find("[ANSWER]")

            if position != -1:
                start = position + len("[ANSWER]")
                output = answer[start:]
        ### replace docids in answer by indices
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
        #    for i in range(len(docs)):
        #        if docs[i]["docid"] in output:
        #            output = output.replace(docs[i]["docid"], str(i + 1))
        output = reposition_period_after_citation(output)
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
        if (itera+1) % 150 == 0:
            results_df = {"data": results}
            results_file = results_dir+"intr_testasqa_"+tag+"_"+str(args.nb_rounds)+"rounds_"+str(args.nb_docs)+"docs.json"  # "agent_hagrid_3doc_2rounds.csv"
            with open(results_file, "w") as writer:
                json.dump(results_df, writer)

        if args.stopindex and args.stopindex == itera:
            logger.info(f"Stoping after {args.stopindex - args.startindex + 1} iterartion, index {args.stopindex} finished")
            break
    end = time.time()

    execution_time = (end - start) / 60
    results_df = {"data": results, "params":vars(args)}
    if args.stopindex:
        #assert (args.stopindex == itera)
        results_file = results_dir+str(args.startindex)+"-"+str(args.stopindex)+"intr_testasqa_"+tag+"_"+str(args.nb_rounds)+"rounds_"+str(args.nb_docs)+"docs.json"  # "agent_hagrid_3doc_2rounds.csv"
        with open(results_file, "w") as writer:
            json.dump(results_df, writer)
    else:
        results_file = results_dir+"all_testasqa_"+tag+"_"+str(args.nb_rounds)+"rounds_"+str(args.nb_docs)+"docs.json" 
        with open(results_file, "w") as writer:
            json.dump(results_df, writer)

    print("Result file:", results_file)
    print("execution_time:", execution_time)


if __name__ == "__main__":
    #main()
    test_alce_docs_gtr()
