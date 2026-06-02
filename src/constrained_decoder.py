from typing import List, Set
import numpy as np
from numpy.typing import NDArray
from src.state_machine import State, LiteralState, TerminationState
from llm_sdk import Small_LLM_Model


class ConstrainedDecoder:
    """
    Drive constrained token-by-token generation using a state machine.

    Attributes:
        model: The language model used for logits and decoding.
    """

    def __init__(self, model: Small_LLM_Model) -> None:
        """
        Initialize ConstrainedDecoder with a language model.

        Args:
            model: The language model instance.
        """
        self.model = model

    @staticmethod
    def _softmax(x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute numerically stable softmax probabilities.

        Args:
            x: Array of raw logit values.

        Returns:
            numpy.ndarray: Probability distribution over x.
        """
        shifted_x: NDArray[np.float64] = x - np.max(x)   # stability
        exp_x: NDArray[np.float64] = np.exp(shifted_x)
        probs: NDArray[np.float64] = exp_x / np.sum(exp_x)
        return probs

    def _get_next_token(
        self,
        prompt: List[int],
        valid_tokens: Set[int],
    ) -> int:
        """
        Sample the next token from a restricted set of valid tokens.

        Args:
            prompt (List[int]): Token IDs of the current context.
            valid_tokens (Set[int]): Allowed next token IDs.

        Returns:
            int: The sampled next token ID.
        """
        # logits of all vocabs
        logits_list = self.model.get_logits_from_input_ids(prompt)

        # sort the valid token ids set and turn it into a list
        valid_token_ids = sorted(valid_tokens)

        # get a list of logits correspondent to the order of the valid
        # token id list
        logits_filtered = np.array(
            [logits_list[i] for i in valid_token_ids]
        )

        temperature = 0.000001
        logits_filtered = logits_filtered / temperature

        # turn the array of logits to a array of probability
        probs = self._softmax(logits_filtered)

        # Sample an index, then map back to the real token ID
        sampled_index = np.random.choice(len(probs), p=probs)
        return valid_token_ids[sampled_index]

    def generate(self, state: State, prompt: str, max_tokens: int) -> str:
        """
        Generate a constrained output string driven by a state machine.

        Iterates up to max_tokens steps. At each step the current state
        supplies the allowed tokens.

        Args:
            state (State): Initial state of the state machine.
            prompt (str): System/context prompt to condition generation.
            max_tokens (int): Maximum number of tokens to generate.

        Returns:
            str: Decoded output string.
        """
        generated_tokens: List[int] = []
        sys_prompt_tokens = self.model.encode(prompt).tolist()[0]

        for _ in range(max_tokens):
            if isinstance(state, LiteralState):
                tokens = self.model.encode(state.text)
                generated_tokens += tokens.tolist()[0]
                next_state = state.update_state(generated_tokens)
                if next_state:
                    state = next_state
            elif isinstance(state, TerminationState):
                return self.model.decode(generated_tokens)
            else:
                valid_tokens = state.get_valid_tokens(generated_tokens)
                prompt_tokens = sys_prompt_tokens + generated_tokens
                next_token = self._get_next_token(prompt_tokens, valid_tokens)
                next_state = state.update_state(
                    generated_tokens + [next_token]
                )
                if next_state:
                    state = next_state
                else:
                    generated_tokens.append(next_token)

        return self.model.decode(generated_tokens)
