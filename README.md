# 03 — Domain RAG Knowledge Assistant

**Highest signal per line on a 2026 DS resume.** GenAI is where interview questions are
moving. This proves you understand retrieval, not just that you can call an API.

## Business question

> Analysts shouldn't need to grep through 400 pages of policy PDFs to answer "what's our
> expense limit for a 3-day international trip?" Build an assistant that answers from our
> documents and cites its source — and says *I don't know* when the answer isn't there.

## Pick a domain

One only. The narrower the better — a real corpus makes the eval possible.

| Domain | Why it works |
|--------|--------------|
| Company expense & travel policy | Obviously internal, obviously valuable, you can write test questions from the doc |
| Clinical guidelines / PubMed abstracts | High-impact narrative; has a known correctness bar |
| SEC 10-K filings | Dense, public, plentiful, multi-document |

Expense policy is the easiest to make genuinely good. The txt files suggested medical
papers and financial reports; both work, but a policy corpus lets you hand-write 20
ground-truth questions and actually measure hallucination rate.

## Stack

langchain (or raw — see below) · chromadb (local, free) · sentence-transformers ·
openai or a local LLM · pytest

Ponytail note: start **raw**. Build the retrieve→prompt→answer loop in ~100 lines with
plain function calls, get the eval working, *then* decide if LangChain buys you anything.
LangChain first means you'll debug the framework instead of the retrieval.

## Method

1. **Ingest** — parse the corpus, chunk it. Chunk size is the single biggest lever on
   answer quality; try 500/1000 chars with 100 overlap and compare.
2. **Embed + index** — local embedding model, ChromaDB. This is a local project, so a paid
   embedding API is an unnecessary cost.
3. **Hybrid retrieval** — BM25 (keyword) + vector (semantic), fused. This is the "Advanced
   Twist" from the source txt and it's the real differentiator: pure vector search fails on
   exact identifiers like policy numbers; pure BM25 fails on paraphrase. Fuse with
   reciprocal rank fusion.
4. **Generate with grounding** — system prompt constrained to the retrieved context, must
   cite document + section. Force it.
5. **Refuse cleanly** — below a similarity threshold, return "not found in the provided
   documents" instead of the model's best guess.

## The eval — without this it's a demo, not a project

This is what separates it from every "I built a RAG chatbot" repo.

- **Hand-write 20 questions** with ground-truth answers and the source document.
- Run the system, score each answer. Don't ask a LLM to grade — read them yourself.
- Report a table: question | retrieved correctly? | answer correct? | cited correctly?
- **The hallucination test**: 5 questions you *know* are unanswerable from the corpus.
  The system must decline all 5. Report that as a number.
- Metrics to state: recall@k of the retriever (the retriever is the part you own), answer
  accuracy, abstention rate.

A README that says "the retriever surfaced the right chunk 17/20 times, the model answered
14/20 correctly, and it correctly abstained on 5/5 unanswerable questions" is a
first-interview answer. "I used LangChain" is not.

## Optional stretch

Query rewriting, or a reranker. Only after the eval exists — otherwise you're optimising
blind.

## Done means

- [ ] `python -m src.ask "what is the international per-diem cap?"` returns an answer + citation
- [ ] `tests/test_retrieval.py` — the correct chunk lands in the top-5 for 5 fixed queries
- [ ] `eval.py` prints the 20-question scorecard, saved to `artifacts/eval.json`
- [ ] README has the scorecard table and the abstention result
- [ ] Answers never cite a document that wasn't retrieved

## Interview questions this buys

*"How do you know your retrieval is the problem and not the model?"* · *"Why hybrid search?"*
· *"How would you evaluate this before it answered a real employee's question?"*

## Resume bullet (fill in real numbers after)

> Built a hybrid (BM25 + vector, reciprocal-rank-fused) RAG assistant over a 400-page policy
> corpus, with a 20-question ground-truth eval reaching 85% retriever recall@5 and 5/5
> correct abstentions on unanswerable queries.
