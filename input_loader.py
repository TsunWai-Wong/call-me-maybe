import json
from typing import List, Dict
from pydantic import BaseModel


class Function(BaseModel):
    """"""
    name: str
    description: str
    parameters: Dict[str, Dict[str, str]]
    returns: Dict[str, str]

    # validator can be added here


class InputLoader:
    """
    """
    def __init__(self, prompts_path: str, functions_path: str):
        """
        """
        self.prompts_path = prompts_path
        self.functions_path = functions_path
        self.prompts: List[str] = []
        self.functions: List[Function] = []

    def read_prompts(self) -> List[str]:
        """
        """
        with open(self.prompts_path) as file:
            data = json.load(file)
        self.prompts = [item["prompt"] for item in data if isinstance(item.get("prompt"), str)]
        return self.prompts

    def read_func_definition(self) -> List[Function]:
        with open(self.functions_path) as file:
            data = json.load(file)
        print(data)
        self.functions = [Function(**item) for item in data]
        return self.functions

    def load(self) -> None:
        try:
            self.read_prompts()
            self.read_func_definition()
        except FileNotFoundError:
            raise Exception("Error: file is not found")
        except PermissionError:
            raise Exception("Error: file cannot be opened due to permission error")
        except json.JSONDecodeError as e:
            raise Exception(f"Invalid JSON: {e}")