# marimo-LLMs-from-scratch

Implement a ChatGPT-like LLM in PyTorch from scratch, step by step with marimo

This repository provides interactive [marimo](https://marimo.io) notebook implementations of the "Build a Large Language Model From Scratch" book by Sebastian Raschka, with Traditional Chinese (繁體中文) translations.

## About

- **Original Source**: [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) by Sebastian Raschka
- **Book**: [Build a Large Language Model From Scratch](http://mng.bz/orYv)
- **Format**: Interactive marimo notebooks (`.py` files that can be run as notebooks)
- **Languages**: English and Traditional Chinese (繁體中文)

## Repository Structure

```text
marimo-LLMs-from-scratch/
├── ch3/                           # Chapter 3: Coding Attention Mechanisms
│   ├── ch03.ipynb                # Original Jupyter notebook (English)
│   ├── marimo_ch03.py            # Marimo notebook (English)
│   ├── marimo_ch03_zh_tw.py      # Marimo notebook (繁體中文)
│   └── README.md                 # Chapter 3 documentation
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Getting Started

### Prerequisites

- Python 3.12 or higher
- pip package manager

### Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/marimo-LLMs-from-scratch.git
cd marimo-LLMs-from-scratch
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

### Running Marimo Notebooks

Marimo notebooks are interactive Python notebooks that run as both scripts and notebooks.

To open a notebook in edit mode:

```bash
marimo edit ch3/marimo_ch03.py           # English version
marimo edit ch3/marimo_ch03_zh_tw.py     # Traditional Chinese version
```

To run a notebook as an app:

```bash
marimo run ch3/marimo_ch03.py
```

## Chapters

### Chapter 3: Coding Attention Mechanisms

Covers the implementation of attention mechanisms, the core engine of LLMs:

- Simple self-attention without trainable weights
- Scaled dot-product attention with Q, K, V matrices
- Causal attention with masking
- Multi-head attention

📁 [See ch3/README.md for detailed documentation](ch3/README.md)

## Why Marimo?

[Marimo](https://marimo.io) is a next-generation Python notebook that offers several advantages:

- ✅ **Reactive**: Cells automatically update when dependencies change
- ✅ **Reproducible**: No hidden state, deterministic execution order
- ✅ **Git-friendly**: Notebooks are stored as `.py` files
- ✅ **Interactive**: Rich UI elements and real-time feedback
- ✅ **Executable**: Can be run as both notebooks and Python scripts

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

This project follows the license of the original [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) repository.

## References

- Original Repository: [https://github.com/rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)
- Book: [Build a Large Language Model From Scratch](http://mng.bz/orYv) by Sebastian Raschka
- Marimo: [https://marimo.io](https://marimo.io)

## Acknowledgments

Special thanks to Sebastian Raschka for creating the original LLMs-from-scratch materials and book.
