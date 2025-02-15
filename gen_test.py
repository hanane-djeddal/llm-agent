import os
import torch
os.environ['HF_HOME'] = os.environ['WORK'] + '/.cache/huggingface'
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import StoppingCriteriaList, StoppingCriteria, StoppingCriteriaList




RAGAGENT_MODEL_NAME ="/lustre/fswork/projects/rech/fiz/udo61qq/llm-jz/llama/llama-2-chat-hagrid-rag-agent-7b/" #"/linkhome/rech/geniri01/udo61qq/Code/llm-jz/data/zephyr-hagrid-deduplicated-rag-agent-3b" #zephyr-hagrid-deduplicated-rag-agent-7b" #zephyr-webgpt-deduplicated-rag-agent-7b" #zephyr-hagrid-rag-agent-7b"  #zephyr-hagrid-rag-agent-3b"

config = PeftConfig.from_pretrained(RAGAGENT_MODEL_NAME, load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", device_map="auto")  #stabilityai/stablelm-zephyr-3b  #HuggingFaceH4/zephyr-7b-beta meta-llama/Llama-2-7b-chat-hf

tokenizer = AutoTokenizer.from_pretrained(RAGAGENT_MODEL_NAME)  #AutoTokenizer.from_pretrained("stabilityai/stablelm-zephyr-3b")
model = PeftModel.from_pretrained(model,RAGAGENT_MODEL_NAME, device_map="auto")
model = model.merge_and_unload()

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        # Store the stop tokens in CUDA (if necessary)
        self.stops = [stop.to("cuda") for stop in stops]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs):
        for stop in self.stops:
            # Compare the last part of input_ids with the stop sequence
            if input_ids.shape[1] >= len(stop):  # Ensure the input is long enough
                print("stop criteria",tokenizer.decode(input_ids[0], skip_special_tokens=False), input_ids[0][-len(stop):])
                if torch.all(stop == input_ids[0][-len(stop):]).item():
                #if (stop in tokenizer.decode(input_ids[0], skip_special_tokens=True)):
                    return True
        return False

kwargs = {"do_sample": True, "top_p": 0.5, "max_new_tokens": 55}

question = "What was the first modern cruise ship?"
message = [{"role": "user", "content": question}]

inputs = tokenizer.apply_chat_template(message, tokenize=True, add_generation_prompt=True, return_tensors="pt")
#output = model.generate(inputs.to(model.device),**kwargs)
#output = tokenizer.batch_decode(output)[0]

#print("--- output: ",output)


stop_words = ["[/SEARCH]"]
#stop_words_ids = [torch.tensor([32871,  2354, 20756,  7082])] #28792, 28748,  1151, 17046,  3328])] #733, 28748,  1151, 17046, 28793])] #28792,28748,  1151, 17046,3328]),torch.tensor([16,  2354, 20756,    62]), torch.tensor([32871,  2354, 20756,  7082])]#tokenizer(stop_word, return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze() for stop_word in stop_words]
stop_words_ids= [tokenizer(stop_word, return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze() for stop_word in stop_words]
print("Input ids for [/SAERCH]", tokenizer("[/SEARCH]", return_tensors="pt", add_special_tokens=False)["input_ids"])
print("Input ids for [SEARCH]", tokenizer("[SEARCH]", return_tensors="pt", add_special_tokens=False)["input_ids"])
print("Input ids for ]", tokenizer("]", return_tensors="pt", add_special_tokens=False)["input_ids"])
print("detokensize", tokenizer.batch_decode([torch.tensor([32871,  2354, 20756,  7082])])) #28792, 28748,  1151, 17046,  3328])]))  #28792,28748,  1151, 17046,3328
stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
output = model.generate(inputs.to(model.device),stopping_criteria=stopping_criteria,**kwargs)
output = tokenizer.batch_decode(output)[0]
print("--- output with stop words: ",output)
