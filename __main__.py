from input_loader import InputLoader

PROMPTS_PATH = "./data/input/function_calling_tests.json"
FUNCTIONS_PATH = "data/input/functions_definition.json"

def main():
    try:
        loader = InputLoader(PROMPTS_PATH, FUNCTIONS_PATH)
        loader.load()
        print(loader.functions)

    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
