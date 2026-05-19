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
    def update_state(self, generated_text: str) -> Optional["State"]:
        pass


class TerminationState(State):
    def get_valid_tokens(self, generated_text: str | List[int]) -> Set[int]:
        return {}

    def update_state(self, generated_text: str) -> Optional["State"]:
        return None


class StringGenerationState(State):
    def __init__(self, model, next_state: State, delimiters: List[str]):
        super().__init__(model, next_state)
        self.delimiters = self.vocabs.search_for_vocab(delimiters)
        self.started = False

    def get_valid_tokens(self, generated_text: List[int]) -> Set[int]:
        if len(generated_text):
            self.started = True
        if not self.started:
            return self.vocabs.search_for_vocab("\"")
        else:
            all_vocabs = self.vocabs.vocabs
            return {id for _, id in all_vocabs.items() if "\n" not in _}

    # or called update_state
    def update_state(self, next_token: int):
        """
        Check whether the current state has ended
        transit when delimiter (e.g. \") is reached? or when one function name is matched
        """
        if next_token in self.delimiters and self.started:
            return self.next_state


class NumberGenerationState(State):
    def __init__(self, model, next_state, delimiters: List[str]):
        super().__init__(model, next_state)
        self.delimiters = self.vocabs.search_for_vocab(delimiters)

    def get_valid_tokens(self, generated_text: List[int]) -> Set[int]:
        """
        by string matching in the Vocabulary class
        """
        valid_tokens = set()
        valid_tokens = self.vocabs.get_valid_tokens_match_number_re(self.vocabs.math_regex, generated_text)
        valid_tokens.update(self.delimiters)
        print(f"valid_tokens: {valid_tokens}")
        return valid_tokens

    def update_state(self, next_token: int):
        """
        Check whether the current state has ended
        transit when delimiter (e.g. ,}]) is reached? or when one function name is matched
        """
        if next_token in self.delimiters:
            return self.next_state


class LiteralState(State):
    def __init__(self, model, next_state, text):
        """"""
        super().__init__(model, next_state)
        self.text = text

    def get_valid_tokens(self) -> Set[int]:
        """"""
        return {}

    def update_state(self):
        """"""
        return self.next_state


class SelectionState(State):
    """
    receive a list of string as accepted options
    keep record of what have been generated
    return valid tokens
    """
    def __init__(self, model, next_state, allowed_sequences: List[int], delimiters: List[str]):
        super().__init__(model, next_state)
        self.allowed_sequences = allowed_sequences
        self.delimiters = self.vocabs.search_for_vocab(delimiters)

    def get_valid_tokens(self, generated_text: List[int]) -> Set[int]:
        valid_tokens = set()
        valid_tokens = self.vocabs.get_valid_tokens_match_token(self.allowed_sequences, generated_text)
        valid_tokens.update(self.delimiters)
        return valid_tokens

    def update_state(self, next_token: int):
        """
        Check whether the current state has ended
        transit when delimiter (e.g. whitespace) is reached? or when one function name is matched 
        """
        if next_token in self.delimiters:
            return self.next_state
