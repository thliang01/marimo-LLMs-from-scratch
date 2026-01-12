# Chapter 3: Coding Attention Mechanisms

This chapter covers attention mechanisms, which are the core engine of Large Language Models (LLMs).

## Files in this Directory

### [ch03.ipynb](ch03.ipynb)
- **Source**: Original Jupyter notebook from [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) repository by Sebastian Raschka
- **Language**: English
- **Format**: Jupyter Notebook (.ipynb)
- **Description**: The original implementation covering all aspects of attention mechanisms for LLMs

### [marimo_ch03_zh_tw.py](marimo_ch03_zh_tw.py)
- **Source**: Converted and translated version
- **Language**: Traditional Chinese (繁體中文)
- **Format**: Marimo notebook (.py)
- **Description**: Interactive marimo notebook version with Traditional Chinese translations
- **How to run**:
  ```bash
  marimo edit marimo_ch03_zh_tw.py
  ```

### [marimo_ch03.py](marimo_ch03.py)
- **Language**: English
- **Format**: Marimo notebook (.py)
- **Description**: English version in marimo format
- **How to run**:
  ```bash
  marimo edit marimo_ch03.py
  ```

## Chapter Content Overview

This chapter implements the following attention mechanisms:

1. **Simple Self-Attention (3.3.1)**: A basic self-attention mechanism without trainable weights for illustration purposes

2. **Self-Attention with Trainable Weights (3.4)**: Implementation of scaled dot-product attention with Query, Key, and Value weight matrices

3. **Causal Attention (3.5)**: Self-attention with causal masking to prevent the model from accessing future tokens
   - Implements causal masking using attention masks
   - Adds dropout for regularization

4. **Multi-Head Attention (3.6)**: Extends single-head attention to multiple attention heads
   - `MultiHeadAttentionWrapper`: Stacks multiple single-head attention modules
   - `MultiHeadAttention`: Efficient implementation with weight splitting

## Key Concepts

- **Attention Scores vs Attention Weights**: Unnormalized scores vs normalized weights (sum to 1)
- **Query, Key, Value (Q, K, V)**: Three trainable projection matrices for computing attention
- **Scaled Dot-Product**: Scaling attention scores by √d_k for training stability
- **Causal Masking**: Masking future tokens to maintain autoregressive property
- **Multi-Head Attention**: Running multiple attention mechanisms in parallel to capture different aspects of relationships

## Requirements

```bash
pip install torch>=2.4.0
pip install marimo>=0.19.2  # For marimo notebooks
```

## Related Chapters

- **Chapter 2**: Data preparation and tokenization (prerequisite)
- **Chapter 4**: Implementing the GPT architecture (uses attention mechanisms from this chapter)

## References

- Book: [Build a Large Language Model From Scratch](http://mng.bz/orYv) by Sebastian Raschka
- Original Repository: [https://github.com/rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)
