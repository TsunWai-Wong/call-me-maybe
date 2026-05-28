from input_loader import InputLoader
from output_generator import OutputGenerator
from output_validator import OutputValidator
from llm_sdk import Small_LLM_Model


PROMPTS_PATH = "./data/input/function_calling_tests.json"
FUNCTIONS_PATH = "data/input/functions_definition.json"


def main():
    try:
        model = Small_LLM_Model()

        input = InputLoader(PROMPTS_PATH, FUNCTIONS_PATH)
        input.load()
        prompts = input.prompts

        results = []
        generator = OutputGenerator(model, input)
        for prompt in prompts:
            results.append(generator.generate_output(prompt))

        validator = OutputValidator()
        validator.write_output(results)

    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
