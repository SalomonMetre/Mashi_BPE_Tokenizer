Here is your updated README with the Streamlit link integrated at the top for maximum visibility. I have also slightly polished the layout to ensure it looks professional for your thesis repository.

---

# 🧩 Language-Agnostic BPE Pipeline & Fertility Analysis

A robust, language-independent **Byte Pair Encoding (BPE)** implementation designed to optimize subword segmentation for any natural language. While the framework is universal, this repository focuses on the **[Mashi language](https://en.wikipedia.org/wiki/Shi_language)** as a primary case study to demonstrate how specialized training can reduce the "tokenization tax" in low-resource Bantu contexts.

### 🔗 Live Demo

**[Explore the Comparative Visualizer here](https://custom-mashi-bpe-tokenizer.streamlit.app/)**

---

## 🚀 Key Features

* **Universal BPE Trainer:** A language-agnostic pipeline optimized with unique chunk counting to handle any UTF-8 text corpus efficiently.
* **Fertility Analysis Framework:** Built-in metrics to calculate **Fertility** (tokens per word), providing a standardized way to measure tokenization efficiency across different models.
* **Cross-Model Comparison:** A Streamlit-based interface for side-by-side visualization of your custom results against industry standards like **GPT-4o** and **Llama 3**.
* **Morphological Preservation:** Employs strict boundary logic to ensure subword units respect linguistic structures better than "one-size-fits-all" multilingual models.

## 📁 Project Structure

```text
.
├── data/                       # Text corpora (Mashi case study included)
├── src/
│   ├── TokenizerTrainer.py    # Language-agnostic training logic
│   ├── Tokenizer.py           # Core encoding/decoding engine
│   ├── train_tokenizer.py     # Entry point for training
│   ├── compare_tokenizers.py  # Script for cross-model metrics
│   ├── comparative_visualizer.py # Web UI for fertility analysis (Streamlit)
│   └── tokenizer_files/       # Exported vocab and merges
└── pyproject.toml             # UV environment configuration

```

## 🛠️ Installation

This project uses `uv` for fast, reproducible Python environments.

1. **Clone the repository:**

```bash
git clone https://github.com/SalomonMetre/Custom_BPE_Tokenizer.git
cd Custom_BPE_Tokenizer

```

2. **Sync the environment:**

```bash
uv sync

```

## 📖 Usage

### 1. Training the Tokenizer

The trainer accepts any UTF-8 text file. Simply point it to your corpus in `data/` and run:

```bash
uv run src/train_tokenizer.py

```

### 2. Running Comparative Analysis

Evaluate the efficiency of your custom-trained model against **GPT-4o** and **Llama 3** on any test set via the CLI:

```bash
uv run src/compare_tokenizers.py

```

### 3. Web Visualization (Streamlit)

Visualize subword segmentation with high-contrast color mapping and real-time **Fertility Rate** calculations:

```bash
uv run streamlit run src/comparative_visualizer.py

```

## 📊 Evaluation Metrics: Fertility

The primary metric used to evaluate model performance is **Fertility**.