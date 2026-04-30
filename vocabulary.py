from typing import List, Dict, Set
import json
from llm_sdk import Small_LLM_Model


class Vocabulary:
    def __init__(self, model):
        self.vocab_path = model.get_path_to_vocab_file()
        self.vocabs = self._get_all_vocabs(self.vocab_path)


    def _get_all_vocabs(self, vocab_path) -> List[str]:
        """
        get vocab of the model
        turn json to python dictionary
        """

        try:
            with open(vocab_path) as file:
                data = json.load(file)
            return data
        except FileNotFoundError:
            raise Exception("Vocab file is not found")
        except PermissionError:
            raise Exception("Vocab file cannot be opened"
                            "due to permission error")

    # get from regular expression
    
    def search_for_vocab(self, targets: List[str]) -> Set[int] | None:
        valid_tokens = set()
        for vocab, token_id in self.vocabs.items():
            if vocab in targets:
                valid_tokens.add(token_id)
        return valid_tokens

    def get_valid_tokens_match_str(self, patterns: List[str], generated_text: str) -> Dict[str, int]:
        """
        return all valid potential next tokens which can match the string pattern
        """
        valid_tokens = set()
        for vocab, token_id in self.vocabs.items():
            prefix = generated_text + vocab
            if any(pattern.startswith(prefix) for pattern in patterns):
                valid_tokens.add(token_id)
        return valid_tokens

    def get_valid_tokens_match_token(self, token_sequences: List[List[int]], generated_tokens: List[int]) -> Dict[str, int]:
        """
        return all valid potential next tokens which can match the token id list pattern
        """
        valid_tokens = set()
        for _, token_id in self.vocabs.items():
            prefix = generated_tokens + [token_id]
            if any(sequence[:len(prefix)] == prefix for sequence in token_sequences):
                valid_tokens.add(token_id)
        return valid_tokens


if __name__ == "__main__":
    model = Small_LLM_Model()
    print(model.get_path_to_vocab_file())
    vocab = Vocabulary(model)

    pattern1 = "Hello, how are you today?"
    pattern2 = "Hello, how old are you now?"
    encoded1 = model.encode(pattern1).tolist()[0]
    encoded2 = model.encode(pattern2).tolist()[0]
    generated_tokens = [9707, 11, 1246]
    print([encoded1, encoded2])
    result = vocab.get_valid_tokens_match_token([encoded1, encoded2], generated_tokens)
    print(result)
    print(vocab.search_for_vocab([".", "]", "}"]))

    print(model.decode([16485, 5746, 25, 5168, 2891, 32964, 11, 5168, 1889]))
    #print(model.decode([2610, 1410, 990, 279, 2701, 5746, 25, 5168, 2891]))
    # print(model.decode([5097, 510, 1302, 3744, 3712, 41861, 382, 13314, 510]))