import os
import urllib.request

class SAFTDictionary:
    def __init__(self, data_dir=None):
        if data_dir is None:
            # Default to the data directory at the root of the project
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self.data_dir = os.path.join(base_dir, "data")
        else:
            self.data_dir = data_dir
            
        os.makedirs(self.data_dir, exist_ok=True)
        self.dict_path = os.path.join(self.data_dir, "vnedict.txt")
        self.vnedict_url = "http://www.denisowski.org/Vietnamese/vnedict.txt"
        self.dictionary = {}
        
    def load_dictionary(self):
        """Loads VNEDICT from local file."""
        if not os.path.exists(self.dict_path):
            print(f"    [WARN] Dictionary file not found at {self.dict_path}")
            return False
            
        print("    Loading VNEDICT into memory...")
        try:
            with open(self.dict_path, 'r', encoding='utf-8') as f:
                for line in f:
                    # Format: Vietnamese : English ; English
                    if ':' in line and not line.startswith('#'):
                        parts = line.split(':', 1)
                        vi_word = parts[0].strip()
                        en_gloss = parts[1].split(';')[0].strip() # Take the first meaning
                        if vi_word and en_gloss:
                            self.dictionary[vi_word.lower()] = en_gloss
            print(f"    ✓ Loaded {len(self.dictionary)} dictionary entries.")
            return True
        except Exception as e:
            print(f"    [WARN] Failed to parse VNEDICT: {e}")
            return False

    def enrich_linear_amr(self, linear_amr):
        """Enriches BFS linear AMR with English glosses."""
        if not self.dictionary:
            return linear_amr
            
        tokens = linear_amr.split()
        enriched_tokens = []
        
        for tok in tokens:
            # Skip relations (start with :) and special tokens (start with <)
            if tok.startswith(':') or tok.startswith('<'):
                enriched_tokens.append(tok)
            else:
                # Remove quotes if any
                clean_tok = tok.strip('"').lower()
                # Lookup in dictionary
                if clean_tok in self.dictionary:
                    gloss = self.dictionary[clean_tok]
                    # Keep English gloss short (first word)
                    short_gloss = gloss.split()[0].strip(',.()[]')
                    # Format: concept[gloss]
                    enriched_tokens.append(f"{tok}[{short_gloss}]")
                else:
                    enriched_tokens.append(tok)
                    
        return ' '.join(enriched_tokens)
