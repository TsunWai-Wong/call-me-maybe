from abc import ABC, abstractmethod
from typing import List, Optional, Set
from vocabulary import Vocabulary


class State(ABC):
    def __init__(self, model, next_state: "State"):
        self.model = model
        self.vocabs = Vocabulary(model)
        self.next_state = next_state

    @abstractmethod
    def get_valid_tokens(self, generated_text: str | List[int]) -> Set[int]:
        pass

    @abstractmethod
    def update_state(self, generated_tokens: List[int]) -> Optional["State"]:
        pass


class TerminationState(State):
    def get_valid_tokens(self, generated_text: str | List[int]) -> Set[int]:
        return {}

    def update_state(self, generated_tokens: List[int]) -> Optional["State"]:
        return None


class StringGenerationState(State):
    def __init__(self, model, next_state: State, delimiters: List[str]):
        super().__init__(model, next_state)
        self.quote_tokens = self.vocabs.exact_quote_tokens
        self.started = False
        self.has_open_quote = False
        self.generated_tokens = []

    def get_valid_tokens(self, generated_text: List[int]) -> Set[int]:
        if not self.started:
            self.started = True
            return self.quote_tokens
        if generated_text:
            self.generated_tokens.append(generated_text[-1])
        if len(self.generated_tokens) >= 20:
            return self.vocabs.string_closer_tokens  # force a closer token
        # return a union of both sets of vocabs
        return (
            self.vocabs.string_content_tokens
            | self.vocabs.string_closer_tokens
        )

    def update_state(self, generated_tokens: List[int]):
        last_token = generated_tokens[-1]
        if not self.has_open_quote:
            self.has_open_quote = True
            return None
        last_str = self.model.decode([last_token])
        if '"' in last_str:
            return LiteralState(self.model, self.next_state, '"')
        return None


class NumberGenerationState(State):
    def __init__(self, model, next_state, delimiters: List[str]):
        super().__init__(model, next_state)
        self.delimiters = self.vocabs.search_for_vocab(delimiters)
        self.started = False
        self.generated_tokens = []

    def get_valid_tokens(self, generated_text: List[int]) -> Set[int]:
        """
        by string matching in the Vocabulary class
        """
        if self.started and generated_text:
            self.generated_tokens.append(generated_text[-1])
        valid_tokens = set()
        valid_tokens = self.vocabs.get_valid_tokens_match_number_re(
            self.vocabs.math_regex,
            self.generated_tokens,
        )
        valid_tokens.update(self.delimiters)
        return valid_tokens

    def update_state(self, generated_tokens: List[int]):
        """
        Check whether the current state has ended.
        Transit when delimiter (e.g. ,}]) is reached or when one
        function name is matched.
        """
        self.started = True
        if generated_tokens and generated_tokens[-1] in self.delimiters:
            return self.next_state
        if len(self.generated_tokens) >= 10:
            return self.next_state


class LiteralState(State):
    def __init__(self, model, next_state, text):
        """"""
        super().__init__(model, next_state)
        self.text = text

    def get_valid_tokens(self, generated_text: List[int]) -> Set[int]:
        """"""
        return {}

    def update_state(self, generated_tokens: List[int]):
        """"""
        return self.next_state


class SelectionState(State):
    """
    receive a list of string as accepted options
    keep record of what have been generated
    return valid tokens
    """
    def __init__(
        self,
        model,
        next_state,
        allowed_sequences: List[int],
        delimiters: List[str],
    ) -> None:
        super().__init__(model, next_state)
        self.allowed_sequences = allowed_sequences
        self.delimiters = self.vocabs.search_for_vocab(delimiters)

    def get_valid_tokens(self, generated_text: List[int]) -> Set[int]:
        valid_tokens = set()
        valid_tokens = self.vocabs.get_valid_tokens_match_token(
            self.allowed_sequences,
            generated_text,
        )
        valid_tokens.update(self.delimiters)
        return valid_tokens

    def update_state(self, generated_tokens: List[int]):
        """
        Check whether the current state has ended.
        Transit when delimiter (e.g. whitespace) is reached or when one
        function name is matched.
        """
        if generated_tokens and generated_tokens[-1] in self.delimiters:
            return self.next_state
