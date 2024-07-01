from typing import Any
from pyserini.search.lucene import LuceneSearcher

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
import json
import pickle
import time
import glob
from pathlib import Path
import copy

import numpy as np
import torch
import transformers

import src.index
import src.contriever
import src.utils
import src.slurm
import src.data
from src.evaluation import calculate_matches
import src.normalize_text

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["HTTP_PROXY"] = "http://hacienda:3128"
os.environ["HTTPS_PROXY"] = "http://hacienda:3128"


class Retriever:
    def __init__(self, args, model=None, tokenizer=None):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer

    def embed_queries(self, args, queries):
        embeddings, batch_question = [], []
        with torch.no_grad():
            for k, q in enumerate(queries):
                if args.lowercase:
                    q = q.lower()
                if args.normalize_text:
                    q = src.normalize_text.normalize(q)
                batch_question.append(q)

                if (
                    len(batch_question) == args.per_gpu_batch_size
                    or k == len(queries) - 1
                ):

                    encoded_batch = self.tokenizer.batch_encode_plus(
                        batch_question,
                        return_tensors="pt",
                        max_length=args.question_maxlength,
                        padding=True,
                        truncation=True,
                    )
                    encoded_batch = {k: v.cuda() for k, v in encoded_batch.items()}
                    output = self.model(**encoded_batch)
                    embeddings.append(output.cpu())

                    batch_question = []

        embeddings = torch.cat(embeddings, dim=0)
        print(f"Questions embeddings shape: {embeddings.size()}")

        return embeddings.numpy()

    def embed_queries_demo(self, queries):
        embeddings, batch_question = [], []
        with torch.no_grad():
            for k, q in enumerate(queries):
                batch_question.append(q)

                if len(batch_question) == 16 or k == len(queries) - 1:

                    encoded_batch = self.tokenizer.batch_encode_plus(
                        batch_question,
                        return_tensors="pt",
                        max_length=200,
                        padding=True,
                        truncation=True,
                    )
                    encoded_batch = {k: v.cuda() for k, v in encoded_batch.items()}
                    output = self.model(**encoded_batch)
                    embeddings.append(output.cpu())

                    batch_question = []

        embeddings = torch.cat(embeddings, dim=0)
        print(f"Questions embeddings shape: {embeddings.size()}")

        return embeddings.numpy()

    def index_encoded_data(self, index, embedding_files, indexing_batch_size):
        allids = []
        allembeddings = np.array([])
        for i, file_path in enumerate(embedding_files):
            print(f"Loading file {file_path}")
            with open(file_path, "rb") as fin:
                ids, embeddings = pickle.load(fin)

            allembeddings = (
                np.vstack((allembeddings, embeddings))
                if allembeddings.size
                else embeddings
            )
            allids.extend(ids)
            while allembeddings.shape[0] > indexing_batch_size:
                allembeddings, allids = self.add_embeddings(
                    index, allembeddings, allids, indexing_batch_size
                )

        while allembeddings.shape[0] > 0:
            allembeddings, allids = self.add_embeddings(
                index, allembeddings, allids, indexing_batch_size
            )

        print("Data indexing completed.")

    def add_embeddings(self, index, embeddings, ids, indexing_batch_size):
        end_idx = min(indexing_batch_size, embeddings.shape[0])
        ids_toadd = ids[:end_idx]
        embeddings_toadd = embeddings[:end_idx]
        ids = ids[end_idx:]
        embeddings = embeddings[end_idx:]
        index.index_data(ids_toadd, embeddings_toadd)
        return embeddings, ids

    def add_passages(self, passages, top_passages_and_scores):
        # add passages to original data
        docs = [passages[doc_id] for doc_id in top_passages_and_scores[0][0]]
        return docs

    def setup_retriever(self):
        print(f"Loading model from: {self.args.model_name_or_path}")
        self.model, self.tokenizer, _ = src.contriever.load_retriever(
            self.args.model_name_or_path
        )
        self.model.eval()
        self.model = self.model.cuda()
        if not self.args.no_fp16:
            self.model = self.model.half()

        self.index = src.index.Indexer(
            self.args.projection_size, self.args.n_subquantizers, self.args.n_bits
        )

        # index all passages
        input_paths = glob.glob(self.args.passages_embeddings)
        input_paths = sorted(input_paths)
        embeddings_dir = os.path.dirname(input_paths[0])
        index_path = os.path.join(embeddings_dir, "index.faiss")
        if self.args.save_or_load_index and os.path.exists(index_path):
            self.index.deserialize_from(embeddings_dir)
        else:
            print(f"Indexing passages from files {input_paths}")
            start_time_indexing = time.time()
            self.index_encoded_data(
                self.index, input_paths, self.args.indexing_batch_size
            )
            print(f"Indexing time: {time.time()-start_time_indexing:.1f} s.")
            if self.args.save_or_load_index:
                self.index.serialize(embeddings_dir)

        # load passages
        print("loading passages")
        self.passages = src.data.load_passages(self.args.passages)
        self.passage_id_map = {x["id"]: x for x in self.passages}
        print("passages have been loaded")

    def search_document(self, query, top_n=10):
        questions_embedding = self.embed_queries(self.args, [query])

        # get top k results
        start_time_retrieval = time.time()
        top_ids_and_scores = self.index.search_knn(
            questions_embedding, self.args.n_docs
        )
        print(f"Search time: {time.time()-start_time_retrieval:.1f} s.")

        return self.add_passages(self.passage_id_map, top_ids_and_scores)[:top_n]

    def search_document_demo(self, query, n_docs=10):
        questions_embedding = self.embed_queries_demo([query])

        # get top k results
        start_time_retrieval = time.time()
        top_ids_and_scores = self.index.search_knn(questions_embedding, n_docs)
        print(f"Search time: {time.time()-start_time_retrieval:.1f} s.")

        return self.add_passages(self.passage_id_map, top_ids_and_scores)[:n_docs]

    def setup_retriever_demo(
        self,
        model_name_or_path,
        passages,
        passages_embeddings,
        n_docs=5,
        save_or_load_index=False,
    ):
        print(f"Loading model from: {model_name_or_path}")
        self.model, self.tokenizer, _ = src.contriever.load_retriever(
            model_name_or_path
        )
        self.model.eval()
        self.model = self.model.cuda()

        self.index = src.index.Indexer(768, 0, 8)

        # index all passages
        input_paths = glob.glob(passages_embeddings)
        input_paths = sorted(input_paths)
        embeddings_dir = os.path.dirname(input_paths[0])
        index_path = os.path.join(embeddings_dir, "index.faiss")
        if save_or_load_index and os.path.exists(index_path):
            self.index.deserialize_from(embeddings_dir)
        else:
            print(f"Indexing passages from files {input_paths}")
            start_time_indexing = time.time()
            self.index_encoded_data(self.index, input_paths, 1000000)
            print(f"Indexing time: {time.time()-start_time_indexing:.1f} s.")

        # load passages
        print("loading passages")
        self.passages = src.data.load_passages(passages)
        self.passage_id_map = {x["id"]: x for x in self.passages}
        print("passages have been loaded")


def add_hasanswer(data, hasanswer):
    # add hasanswer to data
    for i, ex in enumerate(data):
        for k, d in enumerate(ex["ctxs"]):
            d["hasanswer"] = hasanswer[i][k]


def load_data(data_path):
    if data_path.endswith(".json"):
        with open(data_path, "r") as fin:
            data = json.load(fin)
    elif data_path.endswith(".jsonl"):
        data = []
        with open(data_path, "r") as fin:
            for k, example in enumerate(fin):
                example = json.loads(example)
                data.append(example)
    return data


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
        doctext, scores, tool_answer = self.process(tool_query, **kwargs)
        return doctext, scores, message + tool_answer

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
            substring = message[begin_loc + len(self.start_token) : end_loc]
            substrings.append(substring)
            start_index = end_loc + len(self.end_token)
        return substrings

    def parse_last(self, text):
        return self.parse(text)[-1]


class SearchToolALCE(Tool):
    def __init__(
        self,
        name="search",
        index="robust04",
        start_token="[boq]",
        end_token="[eoq]",
        args=None,
    ):
        super().__init__(name=name, start_token=start_token, end_token=end_token)
        self.docs_ids = []
        self.args = args
        self.retriever = Retriever(args)
        self.retriever.setup_retriever()

    def search(self, query, k=3):

        ranked_doc = self.retriever.search_document(query, k)
        scores = [i["score"] for i in ranked_doc]
        docs = []
        for doc in ranked_doc:
            if "id" in doc.keys():
                added_docid = {"docid": doc["id"]}
                doc = {**added_docid, **doc}
                del doc["id"]
            else:
                print("keys", doc.keys())
            docs.append(doc)
        return docs, scores

    def process(self, query, **kwargs):
        docs_text, scores = self.search(query, **kwargs)
        return copy.deepcopy(docs_text), scores, f"\n[DOCS] {docs_text} [/DOCS]\n"
