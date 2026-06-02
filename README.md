*This project has been created as part of the 42 curriculum by tswong.*

# Call Me Maybe - Function Calling in LLMs

## Introduction

### What is this project about?

**Call Me Maybe** is a function calling system that translates natural language prompts into structured, machine-executable function calls. Instead of a Large Language Model (LLM) returning text answers, this project uses the LLM to intelligently select which function to call and generate a valid JSON output.

For example:
- **Input:** "What is the sum of 40 and 2?"
- **Traditional LLM Output:** "The sum of 40 and 2 is 42."
- **Function Calling Output:** `{"function": "fn_add_numbers", "arguments": {"a": 40, "b": 2}}`

### Why is this project meaningful?

**Demonstrates constrained decoding:** This project solves that problem using **constrained decoding** to guide token generation and guarantee 100% schema-compliant output. It shows how even small language models (0.6B parameters) can achieve highrreliability through structural guidance

---

## Description

### Language Model

This project use **Qwen/Qwen3-0.6B**, a small language model with only 600 million parameters. 

**Why this is challenging:**
- Limited capacity to understand complex schemas
- Tendency to hallucinate or produce incomplete outputs
- Cannot be relied upon for format compliance through prompting alone


### What is Constrained Decoding and How Does It Help?

**Constrained decoding** is a token-level filtering technique that modifies the language model's output before token selection. It ensures the model never produces invalid JSON or schema-violating content:

1. **The model generates logits** (probability scores) for all possible next tokens
2. **We identify valid tokens** based on:
   - Current JSON structure (must be syntactically valid)
   - Schema requirements (types must match function definitions)
   - State context (what should come next?)
3. **We set invalid token logits to negative infinity** (probability ≈ 0)
4. **We sample only from valid tokens**, ensuring perfect compliance

### Inputs

#### Input File: `function_calling_tests.json`
Array of natural language prompts:
```json
[
  {
    "prompt": "What is the sum of 2 and 3?"
  },
  {
    "prompt": "Reverse the string 'hello'"
  },
  ...
]
```

#### Input File: `functions_definition.json`
Schema describing available functions:
```json
[
  {
    "name": "fn_add_numbers",
    "description": "Add two numbers together",
    "parameters": {
      "a": { "type": "number" },
      "b": { "type": "number" }
    },
    "returns": { "type": "number" }
  },
  ...
]
```

### Output

#### Output File: `data/output/function_calling_results.json`
Array of function calls with exact format:
```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
  },
  ...
]
```

---

## Instructions

### Installation

1. **Clone the repository and navigate to the project directory**

2. **Install dependencies using `uv`**
   ```bash
   uv sync
   ```
   
   This command will:
   - Create a virtual environment
   - Install `numpy` and `pydantic` (as required by the project)
   - Install the `llm_sdk` package (local, included in the repo)
   - Set up all other dependencies from `pyproject.toml`

### Running the Program

**Basic usage (default paths):**
```bash
uv run python -m src
```
Reads from `data/input/` and writes to `data/output/`

**Custom input/output paths:**
```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

**Debug mode:**
```bash
make debug
```

**Clean temporary files:**
```bash
make clean
```

---

## Resources

- **Qwen Model Documentation:** https://qwen.readthedocs.io/en/latest/getting_started/concepts.html
- **JSON Specification:** https://www.json.org/json-en.html
- **3Blue1Brown - Neural Networks:** https://www.youtube.com/watch?v=LPZh9BOjkQs&list=PLZHQObOWTQDM4E-dwvbnQTiyKDO-y9T2t

AI was used for assistance for the following tasks
1. Claude: Debugging
2. VS Code Copilot: Documentation 
3. ChatGPT: Key concepts clarification

---

## Algorithm Explanation

### Core Architecture

The system consists of four main components working together:

1. **`vocabulary.py`** - Vocabulary filtering and token management
2. **`state_machine.py`** - State tracking and transitions
3. **`constrained_decoder.py`** - Token-level constraint enforcement
4. **`output_generator.py`** - Main orchestrator coordinating the pipeline

### Processing Workflow

When given a natural language prompt, the system generates a JSON object with three components:

#### Phase 1: Function Name Selection
```
Prompt: "What is the sum of 2 and 3?"
         ↓
Use LLM to select function name
Constrain tokens to valid function names only
         ↓
Output: "fn_add_numbers"
```

#### Phase 2: Parameter Extraction
For each parameter, apply type-specific constraints:

**For STRING parameters:**
- Generate characters until delimiter (e.g., `"`) or max tokens reached
- Constraint: Allow printable characters + escaping rules

**For NUMBER parameters:**
- Generate digits matching number regex: `^-?\d+(\.\d+)?$`
- Constraint: Only allow digits, optional minus sign, optional decimal point
- Stops when complete number is detected

#### Phase 3: JSON Assembly
Combine all components into a single JSON object:
```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": {"a": 2.0, "b": 3.0}
}
```

---

## Design Decisions

### 1. Greedy Decoding 
- Set a low temperature for the language model and always pick highest probability token

### 2. Literal Short-Circuiting
- Directly insert deterministic strings (e.g., JSON punctuation, parameter names) without LLM generation to improve performance

### 3. Regular Expression-Based Number Validation
- Use regex patterns to validate number format.

---

## Performance Analysis

### Accuracy

| Component | Accuracy | Notes |
|-----------|----------|-------|
| Function Name Selection | **100%** | Constrained to valid function names only |
| Literal Parameter Extraction | **100%** |  extraction with delimiter termination |
| Parameter Translation | **>=90%** | e.g. Regex-based parameters |
| JSON Validity | **100%** | Constrained decoding guarantees valid structure |

### Speed

- **Per-prompt generation:** 15-20 seconds on reference hardware
- **Reference hardware:** 16.0 GiB RAM, 13th Gen Intel Core i7-13700 (×24 cores), AMD Radeon RX 6500

---

## Challenges Faced

### Challenge 1: Stopping the Language Model
**Problem:** How to make the LLM stop generating tokens?

**Solution:** 
- Don't rely on EOS token; instead, stop based on state machine state
- When state = DONE or max tokens reached, stop generation

---

### Challenge 2: Validating Numbers in JSON
**Problem:** Numbers in JSON must match specific format

**Solution:**
- Implemented regex-based validation: `^-?\d+(\.\d+)?$`
- Tokens are constrained to only those matching valid number prefixes

---

### Challenge 3: Function Name Generation Produces Nonsense
**Problem:** When generating function names, the LLM sometimes select the incorrect functions

**Solution:**
- Use special system message tokens (`<|im_start|>` and `<|im_end|>`) to frame the prompt

---

### Challenge 4: Runaway String and Number Generation
**Problem:** 
- Numbers: generating tokens like `25.000000000000000` (infinite zeros)
- Strings: generating tokens like `aAeEiIuUoOeOeOeOeOeOeO` (repetitive noise)

**Solution:**
- **Max token limit:** Set maximum tokens for each parameter generation
- **Delimiter enforcement:** Strings end at closing quote `"`, numbers at space or `}`
- **Prompt engineering:** Include clearer and concise instructions in system prompts
- **One-shot examples:** Show model exactly what format to produce

---

## Testing Strategy

### Input Validation Testing

#### Missing/Malformed Files:
- Input files don't exist
- Files not readable (permission denied)
- Wrong file type (not JSON)
- Invalid JSON syntax → error message with line number
- Missing required keys → clear error indicating which keys
- Wrong value types → type validation errors
- Empty values → validation failure with explanation

#### Edge Cases:
- Empty string prompt: e.g. `"prompt": ""`
- Numeric prompt: `"prompt": 123`
- Null prompt: `"prompt": null`
- Very long prompt: >1000 characters
- Special characters: Unicode, emojis, control characters

### Processing Testing

#### Semantic Edge Cases:
- Prompt doesn't require function calling (ambiguous interpretation)
- Prompt requires function not in definitions
- LLM fails to initialize (model not found)
- Unsupported parameter types
- Functions with no parameters
- Functions with many parameters (5+)

### Output Testing

#### File & Format Validation:
- Output file created successfully
- Invalid output filename
- Invalid output directory
- Output is valid JSON (can be parsed)
- All required keys present (prompt, name, parameters)
- No extra keys in output
- Types match schema definitions

---

## Example Usage

### Example 1: Basic Function Calling

**Input file:** `data/input/function_calling_tests.json`
```json
[
  {"prompt": "What is the sum of 40 and 2?"},
  {"prompt": "Reverse the string 'hello'"},
  {"prompt": "Greet Alice"}
]
```

**Functions file:** `data/input/functions_definition.json`
```json
[
  {
    "name": "fn_add_numbers",
    "description": "Add two numbers",
    "parameters": {"a": {"type": "number"}, "b": {"type": "number"}},
    "returns": {"type": "number"}
  },
  {
    "name": "fn_reverse_string",
    "description": "Reverse a string",
    "parameters": {"s": {"type": "string"}},
    "returns": {"type": "string"}
  },
  {
    "name": "fn_greet",
    "description": "Generate a greeting",
    "parameters": {"name": {"type": "string"}},
    "returns": {"type": "string"}
  }
]
```

**Run command:**
```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/results.json
```

**Output file:** `data/output/results.json`
```json
[
  {
    "prompt": "What is the sum of 40 and 2?",
    "name": "fn_add_numbers",
    "parameters": {"a": 40.0, "b": 2.0}
  },
  {
    "prompt": "Reverse the string 'hello'",
    "name": "fn_reverse_string",
    "parameters": {"s": "hello"}
  },
  {
    "prompt": "Greet Alice",
    "name": "fn_greet",
    "parameters": {"name": "Alice"}
  }
]
```