from pydantic import BaseModel
from typing import List
from input_loader import InputLoader
from constrained_decoder import ConstrainedDecoder
from state_machine import State, LiteralState, SelectionState, StringGenerationState, NumberGenerationState, TerminationState


class Output(BaseModel):
    prompt: str
    name: str
    parameters: List


class OutputGenerator:

    def __init__(self, model, input: InputLoader):
        self.model = model
        self.decoder = ConstrainedDecoder(model)
        self.functions = input.functions

    def _generate_name(self, prompt: str):
        """
        get list of valid function names
        use constrained decoder to generate a function name
        get valid tokens from the state (state will use the filter in vocabulory)

        """
        sys_prompt = f"""
<|im_start|>system
You are an assistant to choose the correct function name to perform a task.

Rules [Very Important!]:
- Must choose the functions provided below. Do not invent a new function name.
- Only output one function name. Do not include any other texts (e.g. explanations)
- After writing the function name, you must stop by writing a foostoop delimiter (.)
- Do not think

Available functions and their descriptions:
{'\n'.join(f"Function: {f.name} ({f.description})" for f in self.functions)}<|im_end|>

<|im_start|>user
Task: {prompt} <|im_end|>
<|im_start|>assistant
"""
        allowed_functions = [self.model.encode(f.name).tolist()[0] for f in self.functions]
        # print(f"allowed function tokens: {allowed_functions}")
        next_state = TerminationState(self.model, None)
        # footstop can not be used
        state = SelectionState(self.model, next_state, allowed_functions, ["]", "}"])
        return self.decoder.generate(state, sys_prompt, 10)

    def _generate_parameters(self, prompt: str):
        """
        get list of required parameters

        """


    def generate_output(self, prompt: str):
        """
        copy the prompt to output's prompt
        use _generate_name
        use _generate_parameters
        """
        return self._generate_name(prompt)