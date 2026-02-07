from TokenizerTrainer import TokenizerTrainer

def main():
    # Configuration
    INPUT_CORPUS = "../data/mashi_text.txt"
    VOCAB_FILE = "tokenizer_files/vocab.txt"
    MERGES_FILE = "tokenizer_files/merges.txt"
    VOCAB_LIMIT = 32_000  # Adjust as needed for your thesis

    print(f"--- Starting Training on {INPUT_CORPUS} ---")
    
    trainer = TokenizerTrainer(voc_limit=VOCAB_LIMIT)
    
    try:
        trainer.train(
            input_path=INPUT_CORPUS, 
            vocab_path=VOCAB_FILE, 
            merges_path=MERGES_FILE
        )
        print("--- Training Successful ---")
        print(f"Files generated: {VOCAB_FILE}, {MERGES_FILE}")
        
    except FileNotFoundError:
        print(f"Error: {INPUT_CORPUS} not found. Please ensure your training text is ready.")

if __name__ == "__main__":
    main()