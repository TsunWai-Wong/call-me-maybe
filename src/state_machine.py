from abc import ABC, abstractmethod
from typing import List, Set, Optional
from llm_sdk import Small_LLM_Model
from src.vocabulary import Vocabulary


class State(ABC):
    """
    Abstract base for a constrained-decoding state machine node.

    Attributes:
        model: The language model instance.
        vocabs (Vocabulary): Vocabulary helper for token lookup.
        next_state (State | None): State to transition to after this one.
    """

    def __init__(self, model: Small_LLM_Model, next_state: Optional["State"]):
        """
        Initialize State with a model and the successor state.

        Args:
            model: The language model instance.
            next_state (State | None): State to transition to on completion.
        """
        self.model = model
        self.vocabs = Vocabulary(model)
        self.next_state = next_state

    @abstractmethod
    def get_valid_tokens(self, generated_tokens: List[int]) -> Set[int]:
        """
        Return the set of token IDs allowed as the next token.

        Args:
            generated_tokens (List[int]): Text or token IDs generated
                so far.

        Returns:
            Set[int]: Allowed next token IDs.
        """
        ...

    @abstractmethod
    def update_state(self, generated_tokens: List[int]) -> Optional["State"]:
        """
        Determine whether to transition and return the next state.

        Args:
            generated_tokens (List[int]): Full token sequence so far.

        Returns:
            State | None: Next state if a transition occurred, else None.
        """
        ...


class TerminationState(State):
    """
    Signal the end of string generation
    """

    def get_valid_tokens(self, generated_tokens: List[int]) -> Set[int]:
        """
        Return an empty set to halt generation.

        Args:
            generated_tokens (List[int]): Unused.

        Returns:
            Set[int]: Always empty.
        """
        return set()

    def update_state(self, generated_tokens: List[int]) -> State | None:
        """
        Return None to indicate no further transitions.

        Args:
            generated_tokens (List[int]): Unused.

        Returns:
            None: Always None.
        """
        return None


class StringGenerationState(State):
    """
    Generate a quoted JSON string value token by token.

    Attributes:
        quote_tokens (set[int]): Token IDs that decode to exactly '"'.
        started (bool): Whether the opening quote has been emitted.
        has_open_quote (bool): Whether the opening quote token was accepted
            and the string has not been closed.
        generated_tokens (List[int]): Tokens produced inside the string.
    """

    def __init__(self,
                 model: Small_LLM_Model,
                 next_state: Optional[State]):
        """
        Initialize StringGenerationState.

        Args:
            model: The language model instance.
            next_state (State): State to enter after the string closes.
        """
        super().__init__(model, next_state)
        self.quote_tokens = self.vocabs.exact_quote_tokens
        self.started = False
        self.has_open_quote = False
        self.generated_tokens: List[int] = []

    def get_valid_tokens(self, generated_tokens: List[int]) -> Set[int]:
        """
        Return tokens valid at the current position inside the string.

        Forces a closing token after 20 content tokens to bound length.

        Args:
            generated_tokens (List[int]): Token IDs generated so far.

        Returns:
            Set[int]: Allowed next token IDs.
        """
        if not self.started:
            self.started = True
            return self.quote_tokens
        if generated_tokens:
            self.generated_tokens.append(generated_tokens[-1])
        if len(self.generated_tokens) >= 20:
            return self.vocabs.string_closer_tokens  # force a closer token
        # return a union of both sets of vocabs
        return (
            self.vocabs.string_content_tokens
            | self.vocabs.string_closer_tokens
        )

    def update_state(self, generated_tokens: List[int]) -> State | None:
        """
        Transition to next state when a closing quote token is seen.

        Args:
            generated_tokens (List[int]): Full token sequence so far.

        Returns:
            LiteralState | None: A closing-quote literal state on close,
                else None.
        """
        last_token = generated_tokens[-1]
        if not self.has_open_quote:
            self.has_open_quote = True
            return None
        last_str = self.model.decode([last_token])
        if '"' in last_str:
            return LiteralState(self.model, self.next_state, '"')
        return None


class NumberGenerationState(State):
    """
    Generate a JSON number value, terminating on a delimiter token.

    Attributes:
        delimiters (set[int]): Token IDs that end the number (e.g. '}', ']').
        started (bool): Whether the first token has been accepted.
        generated_tokens (List[int]): Tokens produced so far for the number.
    """

    def __init__(self, model: Small_LLM_Model,
                 next_state: Optional[State],
                 delimiters: List[str]):
        """
        Initialize NumberGenerationState.

        Args:
            model: The language model instance.
            next_state (State): State to enter after the number ends.
            delimiters (List[str]): String tokens that terminate the number.
        """
        super().__init__(model, next_state)
        self.delimiters = self.vocabs.search_for_vocab(delimiters)
        self.started = False
        self.generated_tokens: List[int] = []

    def get_valid_tokens(self, generated_tokens: List[int]) -> Set[int]:
        """
        Return tokens that extend the current number or are delimiters.

        Args:
            generated_tokens (List[int]): Token IDs generated so far.

        Returns:
            Set[int]: Allowed next token IDs.
        """
        if self.started and generated_tokens:
            self.generated_tokens.append(generated_tokens[-1])
        valid_tokens: Set[int] = set()
        valid_tokens = self.vocabs.get_valid_tokens_number(
            self.vocabs.number_regex,
            self.generated_tokens,
        )
        valid_tokens.update(self.delimiters)
        return valid_tokens

    def update_state(self, generated_tokens: List[int]) -> State | None:
        """
        Transition to next state on delimiter or max-length reached.

        Args:
            generated_tokens (List[int]): Full token sequence so far.

        Returns:
            State | None: next_state on termination, else None.
        """
        self.started = True
        if generated_tokens and generated_tokens[-1] in self.delimiters:
            return self.next_state
        if len(self.generated_tokens) >= 10:
            return self.next_state
        return None


class LiteralState(State):
    """
    Emit a fixed literal string without sampling.

    Attributes:
        text (str): The literal text to emit.
    """

    def __init__(self,
                 model: Small_LLM_Model,
                 next_state: Optional[State],
                 text: str):
        """
        Initialize LiteralState with the text to emit.

        Args:
            model: The language model instance.
            next_state (State | None): State to enter after the literal.
            text (str): Literal string to append to the output.
        """
        super().__init__(model, next_state)
        self.text = text

    def get_valid_tokens(self, generated_tokens: List[int]) -> Set[int]:
        """
        Return an empty set; literal tokens are appended directly.

        Args:
            generated_tokens (List[int]): Unused.

        Returns:
            Set[int]: Always empty.
        """
        return set()

    def update_state(self, generated_tokens: List[int]) -> State | None:
        """
        Immediately transition to the next state.

        Args:
            generated_tokens (List[int]): Unused.

        Returns:
            State | None: Always next_state.
        """
        return self.next_state


class SelectionState(State):
    """
    Constrain generation to one of several allowed token sequences.

    Attributes:
        allowed_sequences (List[List[int]]): Permitted complete token
            sequences (e.g. one sequence per valid function name).
        delimiters (set[int]): Token IDs that terminate selection
            (e.g. ']', '}').
    """

    def __init__(
        self,
        model: Small_LLM_Model,
        next_state: Optional[State],
        allowed_sequences: List[List[int]],
        delimiters: List[str],
    ) -> None:
        """
        Initialize SelectionState.

        Args:
            model: The language model instance.
            next_state (State | None): State to enter after selection ends.
            allowed_sequences (List[List[int]]): Permitted token sequences.
            delimiters (List[str]): String tokens that end the selection.
        """
        super().__init__(model, next_state)
        self.allowed_sequences = allowed_sequences
        self.delimiters = self.vocabs.search_for_vocab(delimiters)

    def get_valid_tokens(self, generated_tokens: List[int]) -> Set[int]:
        """Return tokens that keep the prefix on track for an allowed sequence.

        Args:
            generated_tokens (List[int]): Token IDs generated so far.

        Returns:
            Set[int]: Tokens consistent with at least one allowed sequence,
                plus delimiters.
        """
        valid_tokens: Set[int] = set()
        valid_tokens = self.vocabs.get_valid_tokens_sequences(
            self.allowed_sequences,
            generated_tokens,
        )
        valid_tokens.update(self.delimiters)
        return valid_tokens

    def update_state(self, generated_tokens: List[int]) -> State | None:
        """Transition to next state when a delimiter token is seen.

        Args:
            generated_tokens (List[int]): Full token sequence so far.

        Returns:
            State | None: next_state on delimiter, else None.
        """
        if generated_tokens and generated_tokens[-1] in self.delimiters:
            return self.next_state
        return None
