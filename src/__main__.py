import argparse
import logging
from pathlib import Path
from src.input_loader import InputLoader
from src.output_generator import OutputGenerator
from src.output_writer import OutputWriter
from llm_sdk import Small_LLM_Model

logging.getLogger("huggingface_hub").setLevel(logging.ERROR)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for input and output file paths.

    Returns:
        argparse.Namespace: Parsed arguments with attributes:
            - functions_definition (str): Path to function definitions JSON.
            - input (str): Path to input prompts JSON.
            - output (str): Path to write generated outputs JSON.
    """
    parser = argparse.ArgumentParser(description="Function Call Generation")
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calls.json",
    )
    args, unknown = parser.parse_known_args()
    if unknown:
        raise ValueError(f"Error: Invalid flag(s): {', '.join(unknown)}")
    return args


def main() -> None:
    """
    Load inputs, generate function-call outputs, and write results.
    """
    try:
        args = parse_args()
        output_path = str(Path(args.output).resolve())

        input = InputLoader(args.input, args.functions_definition)
        input.load()
        prompts = input.prompts

        try:
            model = Small_LLM_Model()
        except Exception:
            print("Error: language model cannot be initialised")
            return

        results = []
        generator = OutputGenerator(model, input)
        for prompt in prompts:
            results.append(generator.generate_output(prompt.content))

        validator = OutputWriter()
        validator.write_output(results, output_path)

    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
