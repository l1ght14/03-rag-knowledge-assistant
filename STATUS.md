# STATUS: this project is incomplete

**Read this before the code.** The pipeline works and is tested. The project does not
yet, and the difference matters if you are reviewing this for a portfolio.

## What is done

| | State |
|---|---|
| Chunking with overlap | done, tested |
| BM25 implemented from scratch in numpy | done, tested |
| Pluggable `Embedder` seam | done, tested |
| Hybrid retrieval + reciprocal rank fusion | done, tested |
| Abstention gate | done, tested |
| 20-question eval harness (`evaluate.py`) | done, tested |
| `RAGSystem` end-to-end wiring | done, tested |

92 tests pass in 0.5 seconds. That number is misleading and here is why.

## What is NOT done, and it is the part that matters

1. **No real corpus.** The tests run against 3 short in-test documents. There is no
   `corpus/` directory. Everything is unvalidated at realistic scale.
2. **No real embeddings.** `HashEmbedder` produces deterministic random vectors —
   identical text gets an identical vector, unrelated text gets an orthogonal one. It
   has *no semantic signal at all*. It exists so the tests can run offline and
   instantly. `tests/conftest.py` has a TF-IDF `BagOfWordsEmbedder` that does carry
   lexical signal, but that is still not an embedding model.
3. **No LLM.** `EchoAnswerer` is a stub. It abstains correctly and populates
   `cited_sources`, but the "answer" is a placeholder string.
4. **No 20-question scorecard.** `evaluate.py` can produce it; nobody has run it
   against a real corpus, so the retriever recall and abstention rates this project
   should be reporting **do not exist yet**.
5. **No `src/ask.py` CLI.**

## The abstention default is a known compromise

`DEFAULT_MIN_SIMILARITY = 0.15`. It was `0.0` originally, which meant the system always
returned something — the exact failure mode this project exists to fix. It is now
non-zero, which means it refuses too often with a hash embedder. That is the correct
direction to fail (garbage in, refuse to answer), and it disappears once real
embeddings are wired in. **Recalibrate it per corpus from the eval scorecard.**

## To finish it

1. Pick a domain. Expense/travel policy is easiest because you can hand-write ground
   truth from it.
2. Put 20–50 pages of documents in `corpus/`.
3. Replace `HashEmbedder` — `pip install sentence-transformers`, then
   `SentenceTransformer("all-MiniLM-L6-v2")` behind the existing `Embedder` seam.
4. Wire a real LLM into `answer.py`. Never commit the key; use `.env`.
5. **Write 20 questions with their answers, plus 5 you know are unanswerable.**
   Run the eval. Report retriever recall@5, answer accuracy (grade them yourself —
   reading 20 answers takes five minutes and an LLM grader is not a substitute), and
   5/5 abstention on the unanswerable.

Step 5 is what turns this from a demo into a portfolio piece.

## What this repo is honest about

The BM25 implementation, the RRF fusion, and the abstention gate are real and would
survive review. The tests are genuine — an earlier draft shipped abstention disabled
by default and a test caught it. But a repository that presents 92 passing tests as
evidence of a working RAG system is overclaiming, and this file exists to prevent that.