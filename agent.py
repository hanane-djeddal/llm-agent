from transformers import StoppingCriteriaList, StoppingCriteria, StoppingCriteriaList
import torch
import re

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        # Store the stop tokens in CUDA (if necessary)
        self.stops = [stop.to("cuda") for stop in stops]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs):
        for stop in self.stops:
            # Compare the last part of input_ids with the stop sequence
            if input_ids.shape[1] >= len(stop):  # Ensure the input is long enough
                #print("stop criteria",tokenizer.decode(input_ids[0], skip_special_tokens=True), input_ids[0][-len(stop):])
                if torch.all(stop == input_ids[0][-len(stop):]).item():
                #if (stop in tokenizer.decode(input_ids[0], skip_special_tokens=True)):
                    return True
        return False
    #def __init__(self, stops=[], encounters=1):
    #    super().__init__()
    #    self.stops = [stop.to("cuda") for stop in stops]

    #def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
    #    for stop in self.stops:
    #        if torch.all((stop[1:] == input_ids[0][-len(stop) + 1 :])).item():
    #            return True
    #    return False


class Agent:
    def __init__(
        self,
        model,
        tokenizer,
        tools,
        rounds=6,
        use_tools=True,
        num_docs=2,
        train_corpus="HAGRID",
        adjusted=False
    ):

        self.model = model
        self.tokenizer = tokenizer
        self.eos_token_id = self.tokenizer.eos_token_id
        self.tools = tools
        self.adjusted = adjusted
        self.rounds = rounds
        self.num_docs = num_docs
        self.use_tools = use_tools
        self.train_corpus = train_corpus
        stop_words = self.get_stop_token()
        stop_words_ids = stop_words_ids = [torch.tensor([16,  2354, 20756,    62]), torch.tensor([32871,  2354, 20756,  7082])] # [
            #self.tokenizer(stop_word, return_tensors="pt", add_special_tokens=False)[
            #    "input_ids"
            #].squeeze()
            #for stop_word in stop_words
        #]
        self.stopping_criteria = StoppingCriteriaList(
            [StoppingCriteriaSub(stops=stop_words_ids)]
        )

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

    def generate(self, question, docs=None, **kwargs):
        all_docs = []
        all_scores = []
        used_docids = {}
        pattern = r'\[DOCS\].*?\[/DOCS\]'
        message = [{"role": "user", "content": question}]
        inputs = self.tokenizer.apply_chat_template(
            message, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        last_gen = 0
        for i in range(self.rounds):
            print("round:",i)
            output = self.model.generate(
                inputs.to(self.model.device),
                stopping_criteria=self.stopping_criteria,
                **kwargs,
            )
            output = self.tokenizer.batch_decode(output)[0]
            if self.adjusted:
                output  =  re.sub(pattern, '', output)
            cuurent_output = output[last_gen:]
            last_gen = len(output)
            tool_id = self.detect_tool(cuurent_output)
            if tool_id is not None:
                if self.use_tools:
                    if docs:
                        docs_text, scores, inputs = self.tools[tool_id](
                            output, k=self.num_docs, initial_docs=docs
                        )
                    else:
                        docs_text, scores, inputs = self.tools[tool_id](
                            output, k=self.num_docs
                        )
                    all_scores.append(scores)
                    ### replcaing docids with indices
                    if self.train_corpus == "WEBGPT":
                        for doc in docs_text:
                            docid = doc["docid"]
                            if docid not in used_docids:
                                used_docids[docid] = len(used_docids) + 1
                                doc["indice"] = used_docids[docid]
                                all_docs.append(doc)
                                inputs = inputs.replace(
                                    str(docid), str(used_docids[docid])
                                )
                    else:
                        all_docs.append(docs_text)

                else:
                    ### without retrieval
                    inputs = output + f"\n[DOCS] {[]} [/DOCS]\n"
                inputs = self.tokenizer(
                    inputs, return_tensors="pt", add_special_tokens=False
                )["input_ids"]
            else:
              break 
        return all_docs, all_scores, output
