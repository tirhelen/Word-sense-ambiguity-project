import os
import re
import math

def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
    
def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

def load_corpus(corpus_dir):
    files = [os.path.join(corpus_dir, f) for f in os.listdir(corpus_dir)]
    documents = []
    for path in files:
        text = read_file(path)
        documents.append(tokenize(text))
    return documents

def pick_seeds(corpus, word1, word2, k=5):
    seeds = []
    combined_word = word1 + word2

    # Sense 0 -> came from word1, Sense 1 -> came from word 2
    sense_map = {
        f"ORIGINAL={word1}": 0,
        f"ORIGINAL={word2}": 1}

    for document in corpus:
        tokens = document
        for i, token in enumerate(tokens):

            if token == combined_word:
                # Context window of size k to the left/right of the target
                left = tokens[max(0, i-k): i]
                right = tokens[i+1: i+k+1]
                # Creating an instance of the window
                instance = (left, token, right)
                # Checking if word1 or word2 appears in the context
                for marker, sense_id in sense_map.items():
                    if marker in left or marker in right:
                        seeds.append((instance, sense_id))
                        break
    return seeds


if __name__ == "__main__":
    corpus_dir = "./Corpus-spell-AP88"
    corpus = load_corpus(corpus_dir)
    print(pick_seeds(corpus,"car","speech"))
    