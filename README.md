<div align="center">

<img src="images/marimo-read-llmfromscratch-banner.jpeg" alt="marimo LLMs from scratch banner" width="100%">

# Marimo LLMs From Scratch
### Implement a ChatGPT-like LLM in PyTorch from scratch, step by step with marimo.

<!-- Badges -->
[![Marimo](https://img.shields.io/badge/Made%20with-Marimo-brightgreen?style=flat&logo=marimo&logoColor=white)](https://marimo.io)
[![Python](https://img.shields.io/badge/Python-3.12%2B-blue?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Language](https://img.shields.io/badge/Language-English%20%7C%20%E7%B9%81%E9%AB%94%E4%B8%AD%E6%96%87-orange)](README.md)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)

</div>

---

## 📖 About The Project

This repository provides interactive **[marimo](https://marimo.io)** notebook implementations of the best-selling book **"Build a Large Language Model From Scratch"** by *Sebastian Raschka*.

We focus on providing a seamless experience for readers of the **Traditional Chinese Edition (讓 AI 好好說話！從頭打造 LLM 實戰秘笈)**, featuring:
*   **Dual Language Support:** Code comments and explanations in both English and Traditional Chinese.
*   **Interactive Learning:** Visualizing Attention mechanisms and Transformers using Marimo's UI.

**Original Source:** [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) by Sebastian Raschka

---

## 🚀 Getting Started

### Prerequisites
*   Python 3.12+
*   pip

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/thliang01/marimo-LLMs-from-scratch.git
    cd marimo-LLMs-from-scratch
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### 🐱 Twinkle's Tip: Running Marimo
> **"Wait! Don't use `python filename.py`!"**
>
> Marimo files look like standard Python scripts, but to see the magic (graphs, sliders, and interactivity), you need to run them with the marimo editor.

**To open a notebook in edit mode:**
```bash
marimo edit ch3/marimo_ch03.py           # English version
marimo edit ch3/marimo_ch03_zh_tw.py     # Traditional Chinese version

```

**To run a notebook as an app:**
```bash
marimo run ch3/marimo_ch03.py
```

---

## 📂 Repository Structure

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

---

## 📚 Chapters

### Chapter 3: Coding Attention Mechanisms

Covers the implementation of attention mechanisms, the core engine of LLMs:

* Simple self-attention without trainable weights
* Scaled dot-product attention with Q, K, V matrices
* Causal attention with masking
* Multi-head attention

📁 [See ch3/README.md for detailed documentation](ch3/README.md)

---

## 💡 Why Marimo?

[Marimo](https://marimo.io) is a next-generation Python notebook that offers several advantages:

* ✅ **Reactive**: Cells automatically update when dependencies change
* ✅ **Reproducible**: No hidden state, deterministic execution order
* ✅ **Git-friendly**: Notebooks are stored as `.py` files
* ✅ **Interactive**: Rich UI elements and real-time feedback
* ✅ **Executable**: Can be run as both notebooks and Python scripts

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

---

## 📄 License

This project follows the license of the original [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) repository.

---

## 📖 References

* Original Repository: [https://github.com/rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)
* Book: [Build a Large Language Model From Scratch](http://mng.bz/orYv) by Sebastian Raschka
* Marimo: [https://marimo.io](https://marimo.io)

---

## 🙏 Acknowledgments

Special thanks to Sebastian Raschka for creating the original LLMs-from-scratch materials and book.
