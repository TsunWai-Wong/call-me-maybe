from llm_sdk import Small_LLM_Model
import numpy as np
import json


TEMPERATURE = 0.8


def softmax(x):
    x = x - np.max(x)   # stability trick
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


def main() -> None:
    model = Small_LLM_Model()

    path = model.get_path_to_vocab_file()
    with open(path) as json_file:w
        vocab_dict = json.load(json_file)

    prompt = """
<|im_start|>system
Choose the exact function name.
Only answer the function name.

Functions:
- get_weather: returns weather info
- book_flight: books a flight between cities
- cancel_flight: cancels an existing booking<|im_end|>
<|im_start|>user
Book me a flight from Berlin to Paris tomorrow<|im_end|>
<|im_start|>assistant
"""

    input_ids = model.encode(prompt)
    input_ids = input_ids.tolist()[0]

    eos_id = []

    for token, idx in vocab_dict.items():
        if token in ["<|im_end|>", "</s>", "<eos>", "eos"]:
            eos_id.append(idx)

    print(eos_id)

    for _ in range(100):
        logits_list = model.get_logits_from_input_ids(input_ids)
        # Convert list into a numpy array
        logits_np = np.array(logits_list)
        logits_np = logits_np / TEMPERATURE
        probs_np = softmax(logits_np)
        next_token_id = np.random.choice(len(probs_np), p=probs_np)
        input_ids.append(next_token_id)
        if next_token_id == 84399 or next_token_id == 128247:
            break

    decoded_str = model.decode(input_ids)
    print(decoded_str)


if __name__ == "__main__":
    main()
