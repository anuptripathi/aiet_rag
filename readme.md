## change env

```bash
.venv\Scripts\activate
```

```bash
# -m tells Python → “this is a package module”
# Now rag is recognized as a package
# Make sure you have:
# rag/__init__.py
# rag/ingest/__init__.py
# Even if empty:
# touch rag/__init__.py
# touch rag/ingest/__init__.py

python -m rag.ingest.chunker
python -m rag/ingest/indexer.py

# see count in db.
python .\test\data-verify.py

# last few data in db.
#simple
python .\test\data-tail.py -n 5
#with collection name
python .\test\data-tail.py --collection 3gpp_docs -n 5
#with batch (faster if collection is large)
python .\test\data-tail.py -n 5 --batch 1000
```
