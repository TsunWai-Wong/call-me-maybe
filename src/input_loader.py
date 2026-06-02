import json
from typing import List, Dict
from pydantic import BaseModel, Field, ValidationError


class Prompt(BaseModel):
    """
    Represent an user prompt loaded from JSON.

    Attributes:
        content (str): The user task description.
    """
    content: str = Field(min_length=5, max_length=1000)


class FunctionDefinition(BaseModel):
    """
    Represent a callable function definition loaded from JSON.

    Attributes:
        name (str): Function identifier.
        description (str): Human-readable description of what it does.
        parameters (Dict[str, Dict[str, str]]): Parameter name to
            metadata mapping (e.g. type information).
        returns (Dict[str, str]): Return type metadata.
    """
    name: str
    description: str
    parameters: Dict[str, Dict[str, str]]
    returns: Dict[str, str]

    # validator can be added here


class InputLoader:
    """
    Load prompts and function definitions from JSON files.

    Attributes:
        prompts_path (str): Path to the prompts JSON file.
        functions_path (str): Path to the function definitions JSON file.
        prompts (List[str]): Loaded prompt strings.
        functions (List[Function]): Loaded function definitions.
    """

    def __init__(self, prompts_path: str, functions_path: str):
        """Initialize InputLoader with paths to input files.

        Args:
            prompts_path (str): Path to the prompts JSON file.
            functions_path (str): Path to the function definitions JSON file.
        """
        self.prompts_path = prompts_path
        self.functions_path = functions_path
        self.prompts: List[Prompt] = []
        self.functions: List[FunctionDefinition] = []

    def _read_prompts(self) -> List[Prompt]:
        """
        Read and validate prompts from the prompts JSON file.

        Returns:
            List[Prompt]: Loaded and validated Prompt objects.
        """
        try:
            with open(self.prompts_path) as file:
                data = json.load(file)
            errors = []
            prompts = []
            for i, item in enumerate(data):
                try:
                    prompts.append(Prompt(content=item["prompt"]))
                except KeyError:
                    errors.append(f"index {i}: missing key 'prompt'")
                except ValidationError as e:
                    issues = [err["msg"] for err in e.errors()]
                    errors.append(f"index {i}: {', '.join(issues)}")
            if errors:
                raise ValueError(
                    f"Error: Invalid prompt(s): {'; '.join(errors)}"
                )
            self.prompts = prompts
            return self.prompts
        except FileNotFoundError:
            raise Exception("Error: Input file is not found")

    def _read_func_definition(self) -> List[FunctionDefinition]:
        """
        Read and store function definitions from the functions JSON file.

        Returns:
            List[Function]: Loaded function definition objects.
        """
        try:
            with open(self.functions_path) as file:
                data = json.load(file)
            errors = []
            functions = []
            for i, item in enumerate(data):
                try:
                    functions.append(FunctionDefinition(**item))
                except ValidationError as e:
                    missing = [err["loc"][0] for err in e.errors()]
                    errors.append(
                        f"index {i}: missing key(s) "
                        f"{', '.join(str(k) for k in missing)}"
                    )
            if errors:
                raise ValueError(
                    f"Error: Invalid function definition(s): "
                    f"{'; '.join(errors)}"
                )
            self.functions = functions
            return self.functions
        except FileNotFoundError:
            raise Exception("Error: Functions definition file is not found")

    def load(self) -> None:
        """
        Load prompts and function definitions, raising on file errors.
        """
        try:
            self._read_prompts()
            self._read_func_definition()
        except PermissionError:
            raise Exception(
                "Error: Input file(s) cannot be opened due to permission error"
            )
        except json.JSONDecodeError as e:
            raise Exception(f"Error: Invalid JSON - {e}")
