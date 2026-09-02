VENV := .venv
PY   := $(VENV)/bin/python

.PHONY: help venv test app docs theme clean

help:
	@echo "make venv   - create .venv and install the package with all extras"
	@echo "make test   - run the test suite"
	@echo "make app    - launch the Streamlit workbench"
	@echo "make docs   - render the Quarto tutorial to HTML"
	@echo "make theme  - regenerate .streamlit/config.toml from smithlib/style.py"
	@echo "make clean  - remove build and render artefacts"

venv:
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -e ".[app,docs,dev]"

test:
	$(PY) -m pytest -q

# Run from the project root so Streamlit finds .streamlit/config.toml.
app:
	$(VENV)/bin/streamlit run app/app.py

# The app chrome is themed from a config file Streamlit reads at start-up, so
# it is generated from the palettes rather than hand-copied.
theme:
	$(PY) scripts/gen_streamlit_theme.py

# Quarto reuses a persistent Jupyter kernel between renders; restarting it
# avoids picking up state from an earlier run.
docs:
	cd docs && QUARTO_PYTHON=$(CURDIR)/$(PY) quarto render smith-chart.qmd \
		--to html --execute-daemon-restart

clean:
	rm -rf docs/smith-chart.html docs/smith-chart_files docs/figures docs/.jupyter_cache
	rm -rf .pytest_cache **/__pycache__ *.egg-info build dist
