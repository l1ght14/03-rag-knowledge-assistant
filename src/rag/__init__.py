"""Dependency-free hybrid RAG: BM25 + vector retrieval fused with RRF.

Each module is a small piece of the retrieve -> answer loop so that any single
stage (chunking, scoring, fusion, abstention) can be tested and measured on its
own. That separation is the point of the project: a RAG system is only as good
as its retriever, and the retriever is the part you own.
"""
