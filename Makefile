# Variables
PYTHON = uv run python3  # Run python with uv 
MAIN = -m src  # Run src module as a program
SRC = src/  # Source code directory

all: install  # when you run make, it will run install by default

# Install or update the project dependencies
install:
	@echo "Installing dependencies using uv..."
	uv sync

run: install
	@echo "Running the program..."
	$(PYTHON) $(MAIN) $(ARGS) 

debug: install
	@echo "Starting debug mode..."
	$(PYTHON) -m pdb $(MAIN) $(ARGS)

lint:
	@echo "Running standard linting..."
	uv run flake8 $(SRC)
	uv run mypy --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs $(SRC)

lint-strict:
	@echo "Running strict linting..."
	uv run flake8 $(SRC)
	uv run mypy --strict $(SRC)

clean:
	@echo "Cleaning up..."
	rm -rf .mypy_cache \
	       .pytest_cache \
	       data/output
	find . -type d -name "__pycache__" -exec rm -rf {} +

.PHONY: all install run debug lint lint-strict clean profile