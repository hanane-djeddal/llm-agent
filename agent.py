
from transformers import StoppingCriteriaList, StoppingCriteria, StoppingCriteriaList
import torch


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops = [], encounters=1):
        super().__init__()
        self.stops = [stop.to("cuda") for stop in stops]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False



class Agent:
    def __init__(self, model, tokenizer, tools, rounds = 6, use_tools = True, num_docs =2):

        self.model = model
        self.tokenizer = tokenizer
        self.eos_token_id = self.tokenizer.eos_token_id
        self.tools = tools
        self.rounds = rounds
        self.num_docs = num_docs
        self.use_tools = use_tools
        stop_words = self.get_stop_token()
        stop_words_ids = [self.tokenizer(stop_word, return_tensors='pt', add_special_tokens = False )['input_ids'].squeeze() for stop_word in stop_words]
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
        
    def detect_tool(self, message):
        """
        Finds the ID of the latest substring in a string.

        Args:
            string: The string to search.
            substrings: A list of substrings to search for.

        Returns:
            The ID of the latest substring found in the string, or None if no substrings are found.
        """
        latest_id = None
        latest_start = -1
        substrings = [tool.end_token for tool in self.tools]
        for i, substring in enumerate(substrings):
            start = message.rfind(substring)
            if start != -1 and start > latest_start:
                latest_id = i
                latest_start = start
        return latest_id


    def get_stop_token(self):
        list_end_gen = []
        for tool in self.tools:
            list_end_gen.append(tool.end_token)
        return list_end_gen
    

    def generate(self, question, **kwargs):
        message = [{'role': 'user', 'content': question}]
        inputs = self.tokenizer.apply_chat_template(message, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        last_gen = 0
        all_docids= []
        all_docs_text = []
        for i in range(self.rounds):
            output = self.model.generate( inputs.to(self.model.device), stopping_criteria=self.stopping_criteria, **kwargs)
            output = self.tokenizer.batch_decode(output)[0]
            cuurent_output = output[last_gen:]
            last_gen = len(output)
            tool_id = self.detect_tool(cuurent_output)
            if tool_id is not None:
                if self.use_tools:
                    docids, docs_text, inputs = self.tools[tool_id](output, k=self.num_docs)
                    all_docids.append(docids)
                    all_docs_text.append(docs_text)
                else:
                    ### without retrieval
                    inputs = output +  f"\n[DOCS] {[]} [/DOCS]\n"
                inputs = self.tokenizer(inputs, return_tensors = 'pt', add_special_tokens = False)['input_ids'] 
            else:
                break
        return all_docids, all_docs_text, output
