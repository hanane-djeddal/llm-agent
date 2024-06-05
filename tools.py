from typing import Any
from pyserini.search.lucene import LuceneSearcher
import torch
import transformers
import numpy as np

def batch(docs: list, nb: int = 10):
    batches = []
    batch = []
    for d in docs:
        batch.append(d)
        if len(batch) == nb:
            batches.append(batch)
            batch = []
    if len(batch) > 0:
        batches.append(batch)
    return batches


def greedy_decode(model, input_ids, length, attention_mask, return_last_logits=True):
    decode_ids = torch.full(
        (input_ids.size(0), 1), model.config.decoder_start_token_id, dtype=torch.long
    ).to(input_ids.device)
    encoder_outputs = model.get_encoder()(input_ids, attention_mask=attention_mask)
    next_token_logits = None
    for _ in range(length):
        model_inputs = model.prepare_inputs_for_generation(
            decode_ids,
            encoder_outputs=encoder_outputs,
            past=None,
            attention_mask=attention_mask,
            use_cache=True,
        )
        outputs = model(**model_inputs)  # (batch_size, cur_len, vocab_size)
        next_token_logits = outputs[0][:, -1, :]  # (batch_size, vocab_size)
        decode_ids = torch.cat(
            [decode_ids, next_token_logits.max(1)[1].unsqueeze(-1)], dim=-1
        )
    if return_last_logits:
        return decode_ids, next_token_logits
    return decode_ids


class MonoT5:
    def __init__(self, model_path="castorini/monot5-base-msmarco", device=None):
        self.model = self.get_model(model_path, device=device)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            "t5-base",
        )
        self.token_false_id = self.tokenizer.get_vocab()["▁false"]
        self.token_true_id = self.tokenizer.get_vocab()["▁true"]
        self.device = next(self.model.parameters(), None).device

    @staticmethod
    def get_model(
        pretrained_model_name_or_path: str, *args, device: str = None, **kwargs
    ):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(device)
        return (
            transformers.AutoModelForSeq2SeqLM.from_pretrained(
                pretrained_model_name_or_path,
                *args,
                **kwargs,
            )
            .to(device)
            .eval()
        )

    def rerank(self, query, docs):
        d = self.rescore(query, docs)
        id_ = np.argsort([i["score"] for i in d])[::-1]
        return np.array(d)[id_]

    def rescore(self, query, docs):
        for b in batch(docs, 10):
            with torch.no_grad():
                text = [f'Query: {query} Document: {d["text"]} Relevant:' for d in b]
                model_inputs = self.tokenizer(
                    text,
                    max_length=512,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                )
                input_ids = model_inputs["input_ids"].to(self.device)
                attn_mask = model_inputs["attention_mask"].to(self.device)
                _, batch_scores = greedy_decode(
                    self.model,
                    input_ids=input_ids,
                    length=1,
                    attention_mask=attn_mask,
                    return_last_logits=True,
                )
                batch_scores = batch_scores[
                    :, [self.token_false_id, self.token_true_id]
                ]
                batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
                batch_log_probs = batch_scores[:, 1].tolist()
            for doc, score in zip(b, batch_log_probs):
                doc["score"] = score  # dont update, only used as Initial with query
        return docs
    
class Tool:
    def __init__(self, name, start_token, end_token):
        self.name = name
        self.start_token = start_token
        self.end_token = end_token

    def __call__(self, message, **kwargs):
        """
        This method allows calling the tool like a function with the message and optional keyword arguments.
        """
        tool_query = self.parse_last(message)
        docids, doctext, tool_answer = self.process(tool_query, **kwargs)
        return docids, doctext, message + tool_answer

    def process(self, parsed_message, **kwargs):
        """
        This method defines the core functionality of the tool and can be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement the process method")

    def parse(self, message):
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
            begin_loc = message.find(self.start_token, start_index)
            if begin_loc == -1:
                break
            end_loc = message.find(self.end_token, begin_loc + len(self.start_token))
            if end_loc == -1:
                break
            substring = message[begin_loc + len(self.start_token):end_loc]
            substrings.append(substring)
            start_index = end_loc + len(self.end_token)
        return substrings

    def parse_last(self, text):
        return self.parse(text)[-1]


class SearchTool(Tool):
    def __init__(self, name = "search", index = 'robust04' , start_token="[boq]", end_token="[eoq]" ):
        super().__init__(name=name, start_token=start_token, end_token=end_token)
        self.docs_ids = []
        self.searcher = LuceneSearcher.from_prebuilt_index(index)
        self.ranker = MonoT5(device='cuda')

    def search( self, query, k=3):
        docs = self.searcher.search(query, k=100)
        retrieved_docid = [i.docid for i in docs]
        docs_text = [eval(self.searcher.doc(docid).raw()) for j, docid  in enumerate(retrieved_docid)]
        ranked_doc = self.ranker.rerank(query, docs_text)[:k]
        docids = [i['docid'] for i in ranked_doc]
        docs_text = [self.searcher.doc(docid).raw() for j, docid  in enumerate(docids)]
        return docids, docs_text

    def process(self, query, **kwargs):
        docids, docs_text = self.search(query , **kwargs)
        return docids, docs_text , f"\n[DOCS] {docs_text} [/DOCS]\n"



