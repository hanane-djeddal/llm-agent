from typing import Any
from pyserini.search.lucene import LuceneSearcher

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
        tool_answer = self.process(tool_query, **kwargs)
        return message + tool_answer

    def process(self, parsed_message, **kwargs):
        """
        This method defines the core functionality of the tool and can be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement the process method")

    def parse(self, message, begin_token, end_token):
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

    def search( self, query, k=1):
        docs = self.searcher.search(query, k=100)
        return docs
    def parse(self, message):
        """
        Parses the message to extract the search query between start and end tokens.
        """
        start = message.find(self.start_token) + len(self.start_token)
        end = message.find(self.end_token, start)
        if start == -1 or end == -1:
            return None
        return message[start:end].strip()

    def process(self, query, **kwargs):
        #docs = self.search(query , **kwargs)
        #docs_string = self._docs_to_text(docs)
        return 'Documents: example'# docs_string

    def _docs_to_text(self, docs):
        docs_txt = [f'docid: {i+len(self.docs_ids)} text: {docs[i]}'for i in range(len(docs))]
        for i in docs:
            self.docs_ids.update({'docid' : {len(self.docs_ids)}, 'text': i.text })
        return 'Documents: '+' '.join(docs_txt)


