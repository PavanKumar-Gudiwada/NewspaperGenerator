---
title: RAG News Article Generator
emoji: 📰
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.42.0
app_file: src/app/app.py
pinned: false
hf_oauth: true
hf_oauth_scopes:
- inference-api
license: mit
short_description: RAG system to generate news articles on provided documents
fullWidth: true
models:
  - google/embeddinggemma-300m # Example model used for embeddings
  - mistralai/Mistral-7B-Instruct-v0.3 # Example LLM
tags:
  - rag
  - llm
  - gradio
  - news-generation
  - documents-qa
---

# GenAI project: Newspaper Article Generator
RAG for newspaper article generation
