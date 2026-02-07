import os
from Tokenizer import Tokenizer as MyTokenizer
from transformers import AutoTokenizer
import tiktoken

def compare_tokenizers(input_file, my_vocab, my_merges):
    # 1. Initialize your custom tokenizer
    my_tok = MyTokenizer(my_vocab, my_merges)

    # 2. Initialize Industry Tokenizers
    print("Loading industry tokenizers...")
    # GPT-4o / GPT-4
    gpt_tok = tiktoken.get_encoding("o200k_base") 
    # Llama 3 (requires access, or use a base version)
    llama_tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
    # Mistral
    mistral_tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

    # 3. Read test sentences
    if not os.path.exists(input_file):
        print(f"File {input_file} not found.")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip()]

    print(f"\n{'Tokenizer':<20} | {'Token Count':<12} | {'Visualized Result'}")
    print("-" * 80)

    for s in sentences:
        print(f"\nSentence: \"{s}\"")
        
        # --- Custom BPE ---
        my_ids = my_tok.encode(s)
        my_vis = my_tok.decode(my_ids, visualize=True)
        print(f"{'Mine (Custom)':<20} | {len(my_ids):<12} | {my_vis}")

        # --- GPT-4o ---
        gpt_ids = gpt_tok.encode(s)
        gpt_vis = "|".join([gpt_tok.decode([i]) for i in gpt_ids])
        print(f"{'GPT-4o (tiktoken)':<20} | {len(gpt_ids):<12} | {gpt_vis}")

        # --- Llama 3 ---
        llama_ids = llama_tok.encode(s, add_special_tokens=False)
        llama_vis = "|".join([llama_tok.decode([i]) for i in llama_ids])
        print(f"{'Llama 3':<20} | {len(llama_ids):<12} | {llama_vis}")

        # --- Mistral ---
        mistral_ids = mistral_tok.encode(s, add_special_tokens=False)
        mistral_vis = "|".join([mistral_tok.decode([i]) for i in mistral_ids])
        print(f"{'Mistral':<20} | {len(mistral_ids):<12} | {mistral_vis}")

if __name__ == "__main__":
    # Ensure your trained files exist
    VOCAB = "tokenizer_files/vocab.txt"
    MERGES = "tokenizer_files/merges.txt"
    TEST_DATA = "../data/mashi_text_test_small.txt" # Create this with a few lines of Mashi/French
    
    if os.path.exists(VOCAB) and os.path.exists(MERGES):
        compare_tokenizers(TEST_DATA, VOCAB, MERGES)
    else:
        print("Please train your tokenizer first!")