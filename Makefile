.PHONY: run-streamlit-app generate-requirements

run-streamlit-app:
	uv run streamlit run app/main.py

generate-requirements:
	uv pip compile pyproject.toml -o requirements.txt