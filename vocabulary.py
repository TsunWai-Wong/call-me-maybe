from typing import List
import json
from llm_sdk import Small_LLM_Model


class Vocabulary:
    def __int__(self, vocab_file):
        self.vocab_file = vocab_file

    def get_vocab(self) -> List[str]:

        try:
            with open(self.vocab_file) as file:
                data = json.loads(file)
                print(data)

        except FileNotFoundError:
            raise Exception("Config file is not found")
        except PermissionError:
            raise Exception("Config file cannot be opened"
                            "due to permission error")


if __name__ == "__main__":
    model = Small_LLM_Model()
    path = model.get_path_to_vocab_file()
    vocab = Vocabulary(path)
