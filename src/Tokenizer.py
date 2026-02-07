import unicodedata
import regex as re

class Tokenizer:
    def __init__(self, vocab_path, merges_path):
        # Load Vocabulary
        with open(vocab_path, "rb") as f:
            self.vocabulary = [line.strip(b"\n") for line in f.readlines()]
        
        # Load Merges (convert hex back to bytes)
        self.merges = []
        with open(merges_path, "r", encoding="utf-8") as f:
            for line in f:
                h1, h2 = line.strip().split()
                self.merges.append((bytes.fromhex(h1), bytes.fromhex(h2)))
        
        self.vocab_lookup = {token: i for i, token in enumerate(self.vocabulary)}

    def _preprocess(self, text_str):
        text = unicodedata.normalize('NFC', text_str).lower()
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        pattern = r"\p{L}+|\p{N}+|[^\s\p{L}\p{N}]|\s"
        chunks = re.findall(pattern, text)
        return [[bytes([b]) for b in chunk.encode('utf-8')] for chunk in chunks]

    def encode(self, text_str):
        chunks = self._preprocess(text_str)
        for pair in self.merges:
            for j in range(len(chunks)):
                new_chunk = []
                i = 0
                chunk = chunks[j]
                while i < len(chunk):
                    if i < len(chunk) - 1 and (chunk[i], chunk[i+1]) == pair:
                        new_chunk.append(chunk[i] + chunk[i+1])
                        i += 2
                    else:
                        new_chunk.append(chunk[i])
                        i += 1
                chunks[j] = new_chunk
        
        ids = []
        for chunk in chunks:
            for token in chunk:
                if token in self.vocab_lookup:
                    ids.append(self.vocab_lookup[token])
                else:
                    for b in token:
                        ids.append(self.vocab_lookup.get(bytes([b])))
        return [i for i in ids if i is not None]

    def decode(self, ids, visualize=False):
        tokens = [self.vocabulary[idx].decode('utf-8', errors='replace') for idx in ids]
        return "|".join(tokens) if visualize else "".join(tokens)