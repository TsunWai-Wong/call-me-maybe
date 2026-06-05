import json
from pydantic import BaseModel
from typing import List, Dict, Self
from src.input_loader import InputLoader
from llm_sdk import Small_LLM_Model
from src.constrained_decoder import ConstrainedDecoder
from src.state_machine import (
    State,
    LiteralState,
    SelectionState,
    StringGenerationState,
    NumberGenerationState,
    TerminationState,
)


class Output(BaseModel):
    """
    Structured result for a single function-call generation.

    Attributes:
        prompt (str): The original user prompt.
        name (str): The predicted function name.
        parameters (List): The extracted parameter values.
    """
    prompt: str
    name: str
    parameters: List[object]


class OutputGenerator:
    """
    Generate structured function-call outputs using constrained decoding.

    Attributes:
        model: The language model instance.
        decoder (ConstrainedDecoder): Decoder driving constrained generation.
        functions (List[Function]): Available function definitions.
    """

    def __init__(
        self: Self, model: Small_LLM_Model, input: InputLoader
    ) -> None:
        """
        Initialize OutputGenerator with a model and loaded input.

        Args:
            model: The language model instance.
            input (InputLoader): Loaded input containing function definitions.
        """
        self.model = model
        self.decoder = ConstrainedDecoder(model)
        self.functions = input.functions
        self.no_function_sentinel = "fn_none"

    def _generate_name(self: Self, prompt: str) -> str:
        """
        Generate a valid function name for the given prompt.
        Uses a SelectionState to constrain decoding to known function names.
        Returns NO_FUNCTION_SENTINEL if no function fits the task.

        Args:
            prompt (str): The user task description.

        Returns:
            str: The predicted function name, or NO_FUNCTION_SENTINEL.
        """
        functions_list = "\n".join(
            f"Function: {f.name} ({f.description})"
            for f in self.functions
        )
        functions_list += (
            f"\nFunction: {self.no_function_sentinel}"
            " (No suitable function found for this task)"
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
        sentinel_tokens = self.model.encode(
            self.no_function_sentinel
        ).tolist()[0]
        allowed_functions = [
            self.model.encode(f.name).tolist()[0]
            for f in self.functions
        ] + [sentinel_tokens]
        next_state = TerminationState(self.model, None)
        state = SelectionState(
            self.model,
            next_state,
            allowed_functions,
            ["]", "}"]
        )
        return self.decoder.generate(state, sys_prompt, 10)

    def _generate_parameters(self: Self, prompt: str, function: str) -> str:
        """
        Generate a JSON parameter string for the selected function.

        Builds a state machine chain matching the parameter schema of
        the named function, then drives constrained decoding to extract
        parameter values from the prompt.

        Args:
            prompt (str): The user task description.
            function (str): The function name whose parameters to extract.

        Returns:
            str: A JSON string of extracted parameter values.
        """
        function_selected = next(
            (f for f in self.functions if f.name == function),
            None,
        )
        function_name = function_selected.name if function_selected else "None"
        params_info = ""
        if function_selected is not None and function_selected.parameters:
            params_info = ", ".join(
                [
                    f"'{param_name}' ({param_info['type'].replace("integer",
                                                                  "number")})"
                    for param_name, param_info in
                    function_selected.parameters.items()
                ]
            )
        sys_prompt = (
            "<|im_start|>system\n"
            "Extract the specific parameters for the function"
            f" '{function_name}'.\n"
            f"You must find these parameters: {params_info}\n"
            "CRITICAL: Do NOT execute the command."
            "Do NOT calculate or reverse anything."
            "ONLY extract the exact literal values from the text.\n"
            "For string parameters, Put a \" symbol to indicate the end\n"
            "of a string.\n"
            "Preserve the EXACT case from the input.\n"
            "Example of output: {{\"s\": \"I am a string\", \"n\": 123.0}}\n"
            "<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        end_state = TerminationState(self.model, None)
        prev_state: State = LiteralState(self.model, end_state, "}")

        if function_selected:
            # when the function does not require a parameter
            if function_selected.parameters is None:
                return "{}"
            param_count = len(function_selected.parameters)
            for param_name, param_info in reversed(
                function_selected.parameters.items()
            ):
                if param_info['type'] == "string":
                    prev_state = StringGenerationState(
                        self.model,
                        prev_state
                    )
                elif param_info['type'] in {"number", "integer"}:
                    prev_state = NumberGenerationState(
                        self.model,
                        prev_state,
                        ["]", "}", " ", "!"],
                        param_info['type']
                    )
                else:
                    prev_state = LiteralState(
                        self.model,
                        prev_state,
                        "null"
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

    def generate_output(self: Self, prompt: str) -> Dict[str, object]:
        """
        Generate a function-call result dict for a single prompt.

        Args:
            prompt (str): The user task description.

        Returns:
            dict: Keys are 'prompt', 'name', and 'parameters'.
        """
        function_name = self._generate_name(prompt)
        if function_name == self.no_function_sentinel:
            return {"prompt": prompt, "name": None, "parameters": None}
        function_parameters = self._generate_parameters(prompt, function_name)
        return {
            "prompt": prompt,
            "name": function_name if function_name else None,
            "parameters": (json.loads(function_parameters)
                           if function_parameters else None)
        }
