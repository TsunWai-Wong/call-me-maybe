from typing import List, Dict, Set
import json
import regex
from llm_sdk import Small_LLM_Model


class Vocabulary:
    """
    Load and index a model's token vocabulary for constrained decoding.

    Attributes:
        model: The language model instance.
        vocab_path (str): Path to the vocabulary JSON file.
        vocabs (dict): Mapping of token string to token ID.
        number_vocabs (dict): Subset of vocabs containing numeric characters.
        number_regex: Compiled regex matching valid JSON numbers.
        string_content_tokens (set[int]): Token IDs safe inside a JSON string
            (no quote or newline characters).
        string_closer_tokens (set[int]): Token IDs whose decoded form contains
            a quote character and can close a JSON string.
        exact_quote_tokens (set[int]): Token IDs that decode to exactly '"'.
    """

    def __init__(self, model: Small_LLM_Model) -> None:
        """
        Initialize Vocabulary and pre-compute token sets.

        Args:
            model: The language model used for encoding and decoding.
        """
        self.model = model
        self.vocab_path = model.get_path_to_vocab_file()
        self.vocabs = self._get_all_vocabs(self.vocab_path)
        self.number_vocabs = {
            k: v
            for k, v in self.vocabs.items()
            if k in [char for char in "0123456789+-.Ee"]
        }
        self.number_regex = regex.compile(
            r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?"
        )
        # self.str_regex = regex.compile(r'"?(?:[^"\\]|\\.)*"?')
        self.string_content_tokens = set()
        self.string_closer_tokens = set()
        self.exact_quote_tokens = set()
        for _, token_id in self.vocabs.items():
            decoded = self.model.decode([token_id])
            if decoded == '"':
                self.exact_quote_tokens.add(token_id)
            if '"' in decoded:
                self.string_closer_tokens.add(token_id)
            elif '\n' not in decoded and '\r' not in decoded:
                self.string_content_tokens.add(token_id)

    def _get_all_vocabs(self, vocab_path: str) -> Dict[str, int]:
        """
        Load the vocabulary JSON file and return it as a dictionary.

        Args:
            vocab_path (str): Path to the vocabulary JSON file.

        Returns:
            dict: Mapping of token string to token ID.
        """

        try:
            with open(vocab_path) as file:
                data: Dict[str, int] = json.load(file)
            return data
        except FileNotFoundError:
            raise Exception("Error: Vocab file is not found")
        except PermissionError:
            raise Exception("Error: Vocab file cannot be opened"
                            "due to permission error")

    def search_for_vocab(self, targets: List[str]) -> Set[int]:
        """Return token IDs whose vocabulary string is in targets.

        Args:
            targets (List[str]): Token strings to look up.

        Returns:
            Set[int]: Matching token IDs.
        """
        valid_tokens = set()
        for vocab, token_id in self.vocabs.items():
            if vocab in targets:
                valid_tokens.add(token_id)
        return valid_tokens

    def get_valid_tokens_number(
        self,
        reg_exp: regex.Pattern[str],
        generated_tokens: List[int],
    ) -> Set[int]:
        """
        Return number token IDs that extend generated_tokens and match reg_exp.

        Only tokens in number_vocabs (digits and numeric punctuation) are
        checked, keeping the search space small.

        Args:
            reg_exp: Compiled regex supporting partial matching.
            generated_tokens (List[int]): Token IDs generated so far.

        Returns:
            Set[int]: Token IDs that keep the decoded prefix valid.
        """
        valid_tokens = set()
        for _, token_id in self.number_vocabs.items():
            prefix = generated_tokens + [token_id]
            prefix_str = self.model.decode(prefix)
            match = reg_exp.fullmatch(prefix_str, partial=True)
            if match or (match and match.partial):
                valid_tokens.add(token_id)
        return valid_tokens

    def get_valid_tokens_sequences(
        self,
        token_sequences: List[List[int]],
        generated_tokens: List[int],
    ) -> Set[int]:
        """
        Return token IDs that extend generated_tokens toward a valid sequence.

        Args:
            token_sequences (List[List[int]]): Allowed complete token
            sequences.
            generated_tokens (List[int]): Token IDs generated so far.

        Returns:
            Dict[str, int]: Token IDs whose addition keeps the prefix on track
                for at least one sequence in token_sequences.
        """
        valid_tokens = set()
        for _, token_id in self.vocabs.items():
            prefix = generated_tokens + [token_id]
            if any(
                sequence[:len(prefix)] == prefix
                for sequence in token_sequences
            ):
                valid_tokens.add(token_id)
        return valid_tokens
