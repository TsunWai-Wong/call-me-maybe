import json
from typing import List, Dict
from pydantic import BaseModel


class Function(BaseModel):
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
        self.prompts: List[str] = []
        self.functions: List[Function] = []

    def _read_prompts(self) -> List[str]:
        """
        Read and store prompt strings from the prompts JSON file.

        Returns:
            List[str]: Loaded prompt strings.
        """
        with open(self.prompts_path) as file:
            data = json.load(file)
        self.prompts = [
            item["prompt"]
            for item in data
            if isinstance(item.get("prompt"), str)
        ]
        return self.prompts

    def _read_func_definition(self) -> List[Function]:
        """
        Read and store function definitions from the functions JSON file.

        Returns:
            List[Function]: Loaded function definition objects.
        """
        with open(self.functions_path) as file:
            data = json.load(file)
        self.functions = [Function(**item) for item in data]
        return self.functions

    def load(self) -> None:
        """
        Load prompts and function definitions, raising on file errors.
        """
        try:
            self._read_prompts()
            self._read_func_definition()
        except FileNotFoundError:
            raise Exception("Error: file is not found")
        except PermissionError:
            raise Exception(
                "Error: file cannot be opened due to permission error"
            )
        except json.JSONDecodeError as e:
            raise Exception(f"Error: Invalid JSON - {e}")
