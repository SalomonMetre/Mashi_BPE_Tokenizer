import os
import unicodedata
import regex as re
from collections import Counter

class TokenizerTrainer:
    def __init__(self, voc_limit=32000):
        self.voc_limit = voc_limit

    def _is_valid_utf_8(self, data):
        try:
            data.decode('utf-8')
            return True
        except UnicodeDecodeError:
            return False

    def _preprocess(self, text_str):
        text = unicodedata.normalize('NFC', text_str).lower()
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        
        pattern = r"\p{L}+|\p{N}+|[^\s\p{L}\p{N}]|\s"
        chunks = re.findall(pattern, text)
        
        # Optimization: Store unique chunks and their counts
        # This reduces the number of merges we perform significantly
        raw_chunks = [tuple(bytes([b]) for b in c.encode('utf-8')) for c in chunks]
        return Counter(raw_chunks)

    def train(self, input_path, vocab_path, merges_path):
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Cannot find {input_path}")

        for path in [vocab_path, merges_path]:
            dir_name = os.path.dirname(path)
            if dir_name: 
                os.makedirs(dir_name, exist_ok=True)

        print(f"Reading {input_path}...")
        with open(input_path, 'r', encoding='utf-8') as f:
            text_str = f.read()

        print("Preprocessing and counting unique chunks...")
        # chunk_counts is { (b'm', b'o', b'y', b'o'): 500, ... }
        chunk_counts = self._preprocess(text_str)
        
        vocabulary = [bytes([b]) for b in range(256)]
        merges = []

        print(f"Starting BPE training (Limit: {self.voc_limit})...")

        while len(vocabulary) < self.voc_limit:
            pair_counts = Counter()
            
            # Optimization: Only iterate over UNIQUE chunks
            for chunk, freq in chunk_counts.items():
                if len(chunk) < 2: 
                    continue
                for i in range(len(chunk) - 1):
                    pair_counts[chunk[i], chunk[i+1]] += freq
            
            if not pair_counts: 
                break
                
            (pair_parts, count) = pair_counts.most_common(1)[0]
            most_freq_pair = pair_parts[0] + pair_parts[1]
            
            merges.append(pair_parts)
            vocabulary.append(most_freq_pair)
            
            # Progress reporting
            current_size = len(vocabulary)
            if current_size % 100 == 0 or current_size == self.voc_limit:
                readable = most_freq_pair.decode('utf-8', 'ignore') or most_freq_pair.hex()
                print(f"Vocab Size: {current_size}/{self.voc_limit} | Merging: '{readable}' ({count})")

            # Optimization: Update the unique chunks list
            new_chunk_counts = {}
            for chunk, freq in chunk_counts.items():
                new_chunk = []
                i = 0
                while i < len(chunk):
                    if i < len(chunk) - 1 and (chunk[i], chunk[i+1]) == pair_parts:
                        new_chunk.append(most_freq_pair)
                        i += 2
                    else:
                        new_chunk.append(chunk[i])
                        i += 1
                new_chunk_counts[tuple(new_chunk)] = freq
            chunk_counts = new_chunk_counts

        print("Filtering vocabulary for valid UTF-8...")
        filtered_vocab = [t for t in vocabulary if len(t) == 1 or self._is_valid_utf_8(t)]

        with open(vocab_path, "wb") as f:
            for token in filtered_vocab: 
                f.write(token + b"\n")

        with open(merges_path, "w", encoding="utf-8") as f:
            for p1, p2 in merges: 
                f.write(f"{p1.hex()} {p2.hex()}\n")
        
        print(f"\nTraining Complete! Final valid vocabulary: {len(filtered_vocab)}")