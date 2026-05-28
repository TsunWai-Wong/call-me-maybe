from input_loader import InputLoader
from output_generator import OutputGenerator
from llm_sdk import Small_LLM_Model


PROMPTS_PATH = "./data/input/function_calling_tests.json"
FUNCTIONS_PATH = "data/input/functions_definition.json"


def main():
    # try:
    model = Small_LLM_Model()

    input = InputLoader(PROMPTS_PATH, FUNCTIONS_PATH)
    input.load()
    prompts = input.prompts

    generator = OutputGenerator(model, input)
    i = 0
    for prompt in prompts:
        if i > 7:
            function_name = generator._generate_name(prompt)
            print(f"function name: {function_name}")
            generator._generate_parameters(prompt, function_name)
        i += 1
    # except Exception as e:
    #     print(e)


if __name__ == "__main__":
    main()
