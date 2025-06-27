# Cleaning-Merging

This project aims to classify keywords into sub-categories using a pretrained Transformer model and then match those keywords to the most appropriate program titles based on semantic similarity (cosine similarity of embeddings).

Strengths
- Scalable, batch processing for inference.
- GPU-compatible.
- Modularized with reusable components (ModelLoader, KeywordMatcher).
- Effective filtering using sub-category to reduce computation and increase accuracy.

