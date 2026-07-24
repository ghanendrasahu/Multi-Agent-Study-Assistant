# Multi-Agent Study Assistant — Project Commands

.PHONY: install run-ui run-cli test clean

install:
	pip install -r requirements.txt

run-ui:
	PYTHONPATH=src streamlit run src/study_agent/app.py

run-cli:
	PYTHONPATH=src python src/study_agent/main.py

test:
	python -c "import sys; sys.path.insert(0, 'src'); from study_agent.agents.state import AgentState; print('All imports OK')"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf data/chroma_db/*
