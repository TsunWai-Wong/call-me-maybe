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
- Only output one function name. Do not include any other texts (e.g. explanations, reasoning)
- After writing the function name, you must stop by writing a delimiter (])

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

    def _generate_parameters_number(self, prompt: str):
        """
        get list of required parameters

        """
        sys_prompt = """
        <|im_start|>system
You are an assistant to generate parameters for a function call according to an user's prompt.
Rule:
- After generating the exact number, you must generate a ] or } symbol to indicate the end of the value
- Do not output a . symbol when you want to produce a integer number

Function selected:
{'name': 'fn_add_numbers', 'description': 'Add two numbers together and return their sum.', 'parameters': {'a': {'type': 'number'}, 'b': {'type': 'number'}}<|im_end|>
<|im_start|>user
User prompt: What is the sum of 265 and 345?<|im_end|>
<|im_start|>assistant
{'a': 
"""
        next_state = TerminationState(self.model, None)
        state = NumberGenerationState(self.model, next_state, ["]", "}", " ", "!"])
        result = self.decoder.generate(state, sys_prompt, 20)
        print(f"Result: {result}")
        return result
    
    def _generate_parameters(self, prompt: str):
        """
        get list of required parameters

        """
        sys_prompt = """
        <|im_start|>system
You are an assistant to extract parameters for a function call according to an user's prompt.

For example:
- Prompt: "Greet shrek"
- Output: "parameters": {"s": "shrek"}

Rule:
- Output directly. Do not include the thinking process.

Function selected:
{
    "name": "fn_reverse_string",
    "description": "Reverse a string and return the reversed result.",
    "parameters": {
      "s": {
        "type": "string"
      }
    }
    }
  }<|im_end|>
<|im_start|>user
Extract the parameters in the JSON object for this prompt: Reverse the string 'hello'<|im_end|>
<|im_start|>assistant
"parameters": {"s": "
"""
        next_state = TerminationState(self.model, None)
        state = StringGenerationState(self.model, next_state, ["'", "\""])
        result = self.decoder.generate(state, sys_prompt, 20)
        print(f"Result: {result}")
        return result

    def generate_output(self, prompt: str):
        """
        copy the prompt to output's prompt
        use _generate_name
        use _generate_parameters
        """
        return self._generate_name(prompt)
