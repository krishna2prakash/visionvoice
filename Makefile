.PHONY: install install-all test lint demo serve docker clean

install:        ## Install core + dev (no heavy extras; tests run on mock)
	pip install -e ".[dev]"

install-all:    ## Install everything (vision + voice + web + backends)
	pip install -e ".[all,dev]"

test:           ## Run the test suite (mock backend)
	pytest

lint:           ## Lint with ruff
	ruff check .

demo:           ## Offline scripted demo — no camera/keys/GPU
	visionvoice demo

serve:          ## Launch the FastAPI web demo
	visionvoice serve

docker:         ## Build + run the web demo in Docker
	docker compose up --build

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache captures
	find . -type d -name __pycache__ -exec rm -rf {} +
