from typing import List, Set
import numpy as np
from state_machine import LiteralState, TerminationState


class ConstrainedDecoder:
    def __init__(self, model) -> None:
        self.model = model

    @staticmethod
    def softmax(x):
        x = x - np.max(x)   # stability trick
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x)

    def _get_next_token(self, prompt: List[int], valid_tokens: Set[int]) -> int:
        """
        receive a list of valid tokens
        sample a next token
        return a next token
        """
        # logits of all vocabs
        logits_list = self.model.get_logits_from_input_ids(prompt)

        # sort the valid token ids set and turn it into a list
        valid_token_ids = sorted(valid_tokens)

        # get a list of logits correspondent to the order of the valid token id list
        logits_filtered = np.array([logits_list[i] for i in valid_token_ids])

        temperature = 0.000001
        logits_filtered = logits_filtered / temperature

        # turn the list of logits to a list of probability
        probs = self.softmax(logits_filtered)

        # Sample an index, then map back to the real token ID
        sampled_index = np.random.choice(len(probs), p=probs)
        return valid_token_ids[sampled_index]

    def generate(self, state, prompt: str, max_tokens: int) -> str:
        """

        if state == literal, return the original text
        if state == termination, return the accumlated text
        else,
        - keep getting the next token
        - get the updated state
        - keep getting the next token according to the updated state
        """
        # get_valid_tokens

        generated_tokens = []
        sys_prompt_tokens = self.model.encode(prompt).tolist()[0]

        for _ in range(max_tokens):
            if isinstance(state, LiteralState):
                tokens = self.model.encode(state.text)
                generated_tokens += tokens.tolist()[0]
                next_state = state.update_state(tokens)
                state = next_state
            elif isinstance(state, TerminationState):
                return self.model.decode(generated_tokens)
            else:
                valid_tokens = state.get_valid_tokens(generated_tokens)
                prompt = sys_prompt_tokens + generated_tokens
                next_token = self._get_next_token(prompt, valid_tokens)
                next_state = state.update_state(generated_tokens + [next_token])
                if next_state:
                    state = next_state
                else:
                    generated_tokens.append(next_token)

        # return the tokens anyway
        return self.model.decode(generated_tokens)
