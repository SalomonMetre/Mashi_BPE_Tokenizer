import os
from Tokenizer import Tokenizer

def test_on_list(tok, sentences):
    print("\n--- Testing on List of Sentences ---")
    for s in sentences:
        ids = tok.encode(s)
        visualized = tok.decode(ids, visualize=True)
        print(f"Input:  {s.strip()}")
        print(f"Tokens: {visualized}")
        print("-" * 20)

def test_on_file(tok, file_path):
    if not os.path.exists(file_path):
        print(f"Test file {file_path} not found.")
        return

    print(f"\n--- Testing on File: {file_path} ---")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                ids = tok.encode(line)
                print(tok.decode(ids, visualize=True))

if __name__ == "__main__":
    # Load the trained model
    VOCAB_FILE = "tokenizer_files/vocab.txt"
    MERGES_FILE = "tokenizer_files/merges.txt"
    
    if not (os.path.exists(VOCAB_FILE) and os.path.exists(MERGES_FILE)):
        print("Error: Model files not found. Run train_tokenizer.py first.")
    else:
        tok = Tokenizer(VOCAB_FILE, MERGES_FILE)

        # # 1. Test a hardcoded list (Good for specific Tshiluba/French cases)
        # sample_sentences = [
        #     "Moyo wa bantu.",
        #     "L'école est belle.",
        #     "Bionso bidi bimpe!"
        # ]
        # test_on_list(tok, sample_sentences)

        # 2. Test an external file (e.g., a validation set)
        test_on_file(tok, "../data/mashi_text_test_small.txt")