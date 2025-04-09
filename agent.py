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
        adjusted=False,
        model_params = "7B",
        manual_stop_words= False,
        without_query_gen = None,
        add_instruction = None,
        diverse_query_only = False,
        add_user_query = None,
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
        self.without_query_gen = without_query_gen
        self.add_instruction = add_instruction
        self.diverse_query_only = diverse_query_only
        self.add_user_query =add_user_query
        stop_words = self.get_stop_token()
        stop_words_ids = [
            self.tokenizer(stop_word, return_tensors="pt", add_special_tokens=False)[
                "input_ids"
            ].squeeze()
            for stop_word in stop_words
        ]
        if manual_stop_words:
            if model_params == "7B":
                stop_words_ids.append(torch.tensor([28792, 28748,  1151, 17046, 3328]))  # 7b-huggingface-beta adjsuted
            if model_params == "3B":        
                #stop_words_ids.append(torch.tensor([16,  2354, 20756,    62])) ##stablelm 3b normal
                if self.adjusted:
                    stop_words_ids.append(torch.tensor([32871,  2354, 20756,  7082]))   #stablelm 3b adjusted 
                else:
                    stop_words_ids.append(torch.tensor([16,  2354, 20756,    62])) ##stablelm 3b normal
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
        query_pattern = r'\[SEARCH\](.*?)\[/SEARCH\]'
        if self.add_instruction:
            instruction= "Given the user query, provide a long answer that tackles different related aspects. To construct your answer, you will alternate between generating a subquery between [SEARCH][/SEARCH] tokens that describes what you will talk about, then use the provided doucments [DOCS][/DOCS] to generate an answer to the subquery and cite the documents you use. Repeat the process until the query is fully answered. Use your generated answer to generate the next subquery based on what you intend to tackle next. The subqueries should be diverse and different from previous ones and allow you to gather new information."
            message = [{"role": "system", "content":instruction},{"role": "user", "content": question}]
            print("Adding system instruction:", instruction)
        else:
            message = [{"role": "user", "content": question}]
        inputs = self.tokenizer.apply_chat_template(
            message, tokenize=True, add_generation_prompt=True, return_tensors="pt", truncation=True
        )
        max_length = self.tokenizer.model_max_length
        #print("max length:", self.tokenizer.model_max_length)
        last_gen = 0
        generated_tool = False
        if self.diverse_query_only:
            query_gen_kw = kwargs
            standard_gen = {"do_sample": True, "top_p": 0.5, "max_new_tokens": 1000}
        for i in range(self.rounds):
            #print("round:",i)
            #print("~~~~~~~~ gen kw",kwargs)
            output = self.model.generate(
                inputs.to(self.model.device),
                stopping_criteria=self.stopping_criteria,
                **kwargs,
            )
            if self.diverse_query_only:
                kwargs = standard_gen
            output = self.tokenizer.batch_decode(output)[0]
            if output[-1] == "[":
                output= output[:-1]
            #print("unmodified output for round ",i, ":", output)    
            if i == 0:
                output=output.replace("[SEARCH]","[ANSWER][SEARCH]")
            if self.adjusted:
                #print("unmodified output for round ",i, ":", output)
                hallucinated_docs = output.find("[DOCS]")
                if generated_tool == False and hallucinated_docs != -1:
                    output = output[:hallucinated_docs]
                output  =  re.sub(pattern, '', output)
                #print("output for round ",i, ":", output)
            cuurent_output = output[last_gen:]
            #if cuurent_output in output[:last_gen]:
                #print("Redundunt output",cuurent_output)
                #break
            #hallucinated_docs = cuurent_output.find("[DOCS]")
            #if generated_tool == False and hallucinated_docs != -1:
                #cuurent_output = cuurent_output[:hallucinated_docs]
                #print("output for round ",i, ":", output)
                #output = output[last_gen:] + cuurent_output
                #print("current output for round ",i, ":", cuurent_output)
            if self.adjusted:
                patternA = r'\[ANSWER\](.*?)\[/ANSWER\]'
                matches = re.findall(patternA, cuurent_output, re.DOTALL)
                if len(matches) >= 3:
                    a_idx = cuurent_output.find("[ANSWER]")
                    #print("cuurent output before editing:", cuurent_output)
                    cuurent_output = cuurent_output[:a_idx] +"[ANSWER]" + matches[0] +"[/ANSWER][ANSWER]"+matches[1]+"[/ANSWER]"
                    output = output[:last_gen] + cuurent_output
                    #print("cuurent output after editing:", cuurent_output)
                    #print("output after editing:", output)            
            last_gen = len(output)
            tool_id = self.detect_tool(cuurent_output)
            if tool_id is not None:
                if self.add_user_query:
                    matches = re.findall(query_pattern, cuurent_output, re.DOTALL)
                    if len(matches):
                        new_subquery = "[SEARCH]"+question+" "+matches[-1]+"[/SEARCH]"
                        output=re.sub(query_pattern,new_subquery,cuurent_output)
                        print("adding user query : ",output)
                start = output.rfind(self.tools[tool_id].start_token)
                if start == -1:
                    break 
                generated_tool = True
                if self.use_tools:
                    if docs:
                        docs_text, scores, inputs = self.tools[tool_id](
                            output, k=self.num_docs, initial_docs=docs
                        )
                    else:
                        docs_text, scores, inputs = self.tools[tool_id](
                            output, k=self.num_docs
                        )
                    #inputs = inputs.replace("[[","[")
                    #print("Adding documents to output for next round ",i, ":", inputs)
                    all_scores.append(scores)
                    ### replcaing docids with indices
                    if self.train_corpus == "OLDWEBGPT":
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
                if self.add_user_query:
                    if len(matches):
                        inputs=inputs.replace(new_subquery,"[SEARCH]"+matches[-1]+"[/SEARCH]")
                        print("readjusting  query", inputs)
            else:
                if self.adjusted:
                    inputs = output
                    generated_tool = False
                else:
                    inputs = output
                    generated_tool = False
                    inputs = inputs.replace("<|endoftext|>","")
                    inputs = inputs.replace("</s>","")
                    inputs = inputs +"[SEARCH]"
                    #print("modified output without tool",inputs)
                    #break
                if self.diverse_query_only:
                    kwargs = query_gen_kw
            if i == 0:
                inputs=inputs.replace("[ANSWER][SEARCH]","[SEARCH]")
            inputs = inputs.replace("<|endoftext|>","")
            inputs = inputs.replace("</s>","")
            #print("---inputs", inputs)
            inputs = self.tokenizer(
                 inputs, return_tensors="pt", add_special_tokens=False, truncation=True
            )["input_ids"] 
            #print("---inputs tokenized", inputs.size(1))
            if inputs.size(1) > max_length:
                print("Exceeding Max length: ", max_length)
                inputs = inputs[:, :max_length]
                break
        return all_docs, all_scores, output
