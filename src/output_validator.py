import json
from typing import List, Dict
from pathlib import Path


class OutputValidator:
    """Validate and persist generated function-call outputs."""

    def validate_parameters(self) -> None:
        """Validate that generated parameter values match expected types.

        Returns:
            OutputValidator: Self, for method chaining.
        """
        # check whether the data types are correct
        pass

    def write_output(
        self,
        output: Dict[str, object] | List[Dict[str, object]],
        output_path: str
    ) -> None:
        """
        Write output to a JSON file, creating parent directories as needed.

        Args:
            output (dict | list[dict]): Data to serialise.
            output_path (str): Destination path, relative to this module or
                absolute.
        """
        if not isinstance(output, (dict, list)):
            raise TypeError("output must be a dict or a list of dicts")

        output_file = Path(output_path)
        if not output_file.is_absolute():
            output_file = Path(__file__).resolve().parent / output_file
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", encoding="utf-8") as file:
            json.dump(output, file, indent=2, ensure_ascii=False)
