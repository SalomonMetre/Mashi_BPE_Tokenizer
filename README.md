# 🧩 Mashi BPE Tokenizer

A custom **Byte Pair Encoding (BPE)** implementation specifically designed for the [**Mashi language**] (https://en.wikipedia.org/wiki/Shi_language). This project aims to reduce the "tokenization tax" paid by low-resource languages by learning subword units that align with the language's actual morphology.

## 🚀 Key Features

* **Custom BPE Trainer:** Optimized for unique chunk counting to speed up training on large corpora.
* **Fertility Analysis:** Built-in metrics to compare how many tokens are produced per word compared to industry giants.
* **Comparative Visualizer:** A Streamlit-based web interface to see side-by-side tokenization results against **GPT-4o** and **Llama 3**.
* **Strict Boundary Logic:** Respects whitespace and escape sequences to ensure clean morphological learning.

## 📁 Project Structure

```text
.
├── data/                       # Raw text corpora (Mashi/French)
├── src/
│   ├── TokenizerTrainer.py    # Training logic & optimization
│   ├── Tokenizer.py           # Core encoding/decoding engine
│   ├── train_tokenizer.py     # Entry point for training
│   ├── test_tokenizer.py      # Basic CLI testing
│   ├── compare_tokenizers.py  # Script for cross-model metrics
│   ├── comparative_visualizer.py # Streamlit UI for fertility analysis
│   └── tokenizer_files/       # Exported vocab.txt and merges.txt
└── pyproject.toml             # UV environment configuration

```

## 🛠️ Installation

This project uses `uv` for extremely fast and reproducible Python environments.

1. **Clone the repository:**
```bash
git clone https://github.com/SalomonMetre/Mashi_BPE_Tokenizer.git
cd Mashi_BPE_Tokenizer.git

```


2. **Sync the environment:**
```bash
uv sync

```



## 📖 Usage

### 1. Training the Tokenizer

Place your corpus in `data/` and run the trainer. This will produce `vocab.txt` and `merges.txt` in the `tokenizer_files/` directory.

```bash
uv run src/train_tokenizer.py

```

### 2. Running Comparative Analysis

To see how your custom tokenizer stacks up against GPT-4o and Llama 3 in the terminal:

```bash
uv run src/compare_tokenizers.py

```

### 3. Web Visualization (Streamlit)

Launch the interactive dashboard to visualize **Fertility Rates** and subword segmentation with high-contrast color mapping.

```bash
uv run streamlit run src/comparative_visualizer.py

```

## 📊 Evaluation Metrics: Fertility

The primary metric of this project is **Fertility**.