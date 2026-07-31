.PHONY: install ingest run dashboard eval eval-rag search ask clean

install:
	pip install -r requirements.txt

ingest:
	python ingest.py

run:
	streamlit run app.py

dashboard:
	streamlit run dashboard.py

eval:
	python eval.py

eval-rag:
	python eval_rag.py

search:
	python search.py "$(q)"

ask:
	python ask.py "$(q)"

clean:
	python -c "import os; [os.remove(f) for f in ['data/documents.json', 'data/feedback.jsonl', 'data/conversations.jsonl'] if os.path.exists(f)]"
