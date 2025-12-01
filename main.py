import os
import re
import math
from collections import Counter, defaultdict

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

if __name__ == "__main__":
    corpus_dir = "./Corpus-spell-AP88"
    corpus = load_corpus(corpus_dir)
    