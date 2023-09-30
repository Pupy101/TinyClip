from typing import List

from nltk.corpus import wordnet


def get_synonyms(word: str) -> List[str]:
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            word = lemma.name()
            if "_" in word:
                word = " ".join(word.split("_"))
            synonyms.append(word)
    return list(set(synonyms))
