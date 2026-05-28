import json
from pathlib import Path


class OutputValidator:

    def validate_parameters(self):
        # check whether the data types are correct
        return self

    def write_output(
        self,
        output: dict | list[dict],
        output_path: str = "data/output/function_calling_results.json",
    ) -> None:
        """Write the generated output dictionary or array to a JSON file."""
        if not isinstance(output, (dict, list)):
            raise TypeError("output must be a dict or a list of dicts")

        output_file = Path(output_path)
        if not output_file.is_absolute():
            output_file = Path(__file__).resolve().parent / output_file
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", encoding="utf-8") as file:
            json.dump(output, file, indent=2, ensure_ascii=False)
