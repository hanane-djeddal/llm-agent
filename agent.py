
from transformers import pipeline

class Agent:
    def __init__(self, model_name, tools):
        self.pipe = pipeline(name  = model_name ,  device_map="auto")
        self.tools = tools
        
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


    def _update_kwargs(self, genkwargs):
        list_end_gen = []
        for tool in self.tools:
            list_end_gen.append(tool.end_token)
        return genkwargs.update({'eos_token': list_end_gen})
    

    def generate(self, message, **kwargs):
        self.kwargs = self._update_kwargs(self.tools)
        while True:
            output = self.pipe( message , **kwargs)[0]['generated_text']
            print(output)
            tool_id = self.detect_tool(output)
            if tool_id is not None:
                output = self.tools[tool_id](output)
            else:
                break
        return output
