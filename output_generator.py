import json
from pydantic import BaseModel
from typing import List
from input_loader import InputLoader
from constrained_decoder import ConstrainedDecoder
from state_machine import (
    LiteralState,
    SelectionState,
    StringGenerationState,
    NumberGenerationState,
    TerminationState,
)


class Output(BaseModel):
    prompt: str
    name: str
    parameters: List


class OutputGenerator:

    def __init__(self, model, input: InputLoader) -> None:
        self.model = model
        self.decoder = ConstrainedDecoder(model)
        self.functions = input.functions

    def _generate_name(self, prompt: str) -> str:
        """
        get list of valid function names
        use constrained decoder to generate a function name
        get valid tokens from the state
        (state will use the filter in vocabulary)

        """
        functions_list = "\n".join(
            f"Function: {f.name} ({f.description})"
            for f in self.functions
        )
        sys_prompt = f"""
<|im_start|>system
You are an assistant to choose the correct function name to perform a task.

Rules [Very Important!]:
- Must choose the functions provided below. Do not invent a new function name.
- Only output one function name.
- Do not include any other texts (e.g. explanations, reasoning)
- After writing the function name, you must stop by writing a delimiter (])

Available functions and their descriptions:
{functions_list}<|im_end|>

<|im_start|>user
Task: {prompt} <|im_end|>
<|im_start|>assistant
"""
        allowed_functions = [
            self.model.encode(f.name).tolist()[0]
            for f in self.functions
        ]
        # print(f"allowed function tokens: {allowed_functions}")
        next_state = TerminationState(self.model, None)
        # footstop can not be used
        state = SelectionState(
            self.model,
            next_state,
            allowed_functions,
            ["]", "}"]
        )
        return self.decoder.generate(state, sys_prompt, 10)

    def _generate_parameters(self, prompt: str, function: str) -> str:
        """
        get list of required parameters

        """
        function_selected = next(
            (f for f in self.functions if f.name == function),
            None,
        )
        params_info = ", ".join(
            [
                f"'{param_name}' ({param_info['type']})"
                for param_name, param_info in
                function_selected.parameters.items()
            ]
        )

        sys_prompt = (
            "<|im_start|>system\n"
            "Extract the specific parameters for the function"
            f" '{function_selected.name}'.\n"
            f"You must find these parameters: {params_info}\n"
            "CRITICAL: Do NOT execute the command."
            "Do NOT calculate or reverse anything."
            "ONLY extract the exact literal values from the text.\n"
            "For string parameters, Put a \" symbol to indicate the end\n"
            "of a string.\n"
            "Preserve the EXACT case from the input.\n"
            "Example of output: {{\"s\": \"I am a string\"}}"
            "<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        print(sys_prompt)

        end_state = TerminationState(self.model, None)
        prev_state = LiteralState(self.model, end_state, "}")

        if function_selected:
            param_count = len(function_selected.parameters)
            for param_name, param_info in reversed(
                function_selected.parameters.items()
            ):
                if param_info['type'] == "string":
                    prev_state = StringGenerationState(
                        self.model,
                        prev_state,
                        ["!", "\""],
                    )
                elif param_info['type'] == "number":
                    prev_state = NumberGenerationState(
                        self.model,
                        prev_state,
                        ["]", "}", " ", "!"],
                    )
                prev_state = LiteralState(
                    self.model,
                    prev_state,
                    f"\"{param_name}\": ",
                )
                if param_count > 1:
                    prev_state = LiteralState(
                        self.model,
                        prev_state,
                        ", ",
                    )
                param_count -= 1

        start_state = LiteralState(self.model, prev_state, "{")
        result = self.decoder.generate(start_state, sys_prompt, 100)
        return result

    def generate_output(self, prompt: str):
        """
        copy the prompt to output's prompt
        use _generate_name
        use _generate_parameters
        """
        function_name = self._generate_name(prompt)
        function_parameters = self._generate_parameters(prompt, function_name)
        return {
            "prompt": prompt,
            "name": function_name if function_name else None,
            "parameters": (json.loads(function_parameters)
                           if function_parameters else None)
        }
