from typing import List

from nltk.corpus import wordnet


def get_synonyms(word: str) -> List[str]:
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name())
    return list(set(synonyms))
