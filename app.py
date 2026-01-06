# WORDNET
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
# EMBEDDINGS 
from gensim.models import KeyedVectors
# DATAMUSE
import datamuse
# DIRECT EMBEDDING COMPARISON
from sentence_transformers import SentenceTransformer
import numpy as np
# SENTANCE TRANSORMERS
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import lemminflect
# FLASK
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
# TIME
import time
import re

app = Flask(__name__)
CORS(app)

start_time = time.time()
api = datamuse.Datamuse()
STOPWORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


## FASTTEXT MODEL ##########
# -- setup --
# ft_model = KeyedVectors.load_word2vec_format(model_path, binary=False)
# ----
ft_model = KeyedVectors.load("ft_model_pruned.kv", mmap='r')
# ----
### For Pruning code see pruning.py


################### 
# HELPERS FOR clean_fasttext_results()

def get_dominant_pos(word):
    """
    Determine the dominant WordNet POS for a word

    returns 'v' for verb, 'n' for noun, 's' for adjective
    """
    synsets = wn.synsets(word)
    if not synsets:
        return None

    pos_counts = {}
    for syn in synsets:
        pos = syn.pos()
        pos_counts[pos] = pos_counts.get(pos, 0) + 1

    return max(pos_counts, key=pos_counts.get)

def has_pos(word, target_pos):
    """
    Check if a word appears in WordNet with a given POS
    """
    return any(syn.pos() == target_pos for syn in wn.synsets(word))

def get_pos(word):
    """
    Returns a set of possible WordNet POS tags for a word in isolation.
    returns: {'v'} or {'n', 'v'}
    """
    tag = pos_tag([word])[0][1]

    if tag.startswith("N"):
        return {wn.NOUN}
    if tag.startswith("V"):
        return {wn.VERB}
    if tag.startswith("J"):
        return {wn.ADJ}
    if tag.startswith("R"):
        return {wn.ADV}

    return set()

def lemmatize(word, pos_set):
    """
    Lemmatize a word under all possible POS.
    Returns a set of lemmas.
    """
    return {
        lemmatizer.lemmatize(word, pos)
        for pos in pos_set
    }

def get_pos_and_lemmas(word):
    pos_set = get_pos(word)
    lemmas = lemmatize(word, pos_set)
    return pos_set, lemmas

def is_morphological_variant(candidate, query):
    """
    Filter trivial morphology like:
    walk → walking, walked, walks
    """
    return (
        candidate.startswith(query)
        or query.startswith(candidate)
    )

def clean_fasttext_results(query, topn=50, final_k=10):
    """
    Retrieve and clean FastText similarity results for a query word.
    
    :query: target word
    :topn: number of FastText neighbors to retrieve
    :final_k: number of clean results to return
    """
    query = query.lower()
    target_pos = get_dominant_pos(query)

    raw_results = ft_model.most_similar(query, topn=topn)

    cleaned = []
    for word, score in raw_results:
        word = word.lower()

        # stopwords & short junk
        if word in STOPWORDS or len(word) <= 2:
            continue

        # morphology filter
        if is_morphological_variant(word, query):
            continue

        # POS filter (only if WordNet knows the query POS)
        if target_pos and not has_pos(word, target_pos):
            continue

        cleaned.append((word, score))

        if len(cleaned) == final_k:
            break

    return cleaned

################### 
# HELPERS FOR rerank_smart()

def semantic_compatible(candidate, query_pos_set, query_lemma):
    """
    returns whether or not the candidate can grammatically replace said word 
    """
    cand_pos_set, cand_lemmas = get_pos_and_lemmas(candidate)


    return (
        bool(query_pos_set & cand_pos_set) and
        query_lemma not in cand_lemmas
    )

def rerank_smart(query, candidates):
    reranked = []

    q_pos_set, q_lemmas = get_pos_and_lemmas(query)
    # print(q_pos_set, q_lemmas)
    for cand in candidates:
        if semantic_compatible(cand, q_pos_set, q_lemmas):
            reranked.append(cand)

    return reranked

##############################
# SYNONYM BANK HELPERS

def faststext_synonyms(word: str, k: int, topn=100):
    results = clean_fasttext_results(word, topn=topn, final_k=k)
    final = []

    # if no score: 
    for w, s in results:
        final.append(w)
    # # # #

    if final:
        return final
    return results

def datamuse_synonyms(word, max_results=10):
    """
    Returns a list of semantically related words,
    sorted by relevance (highest score first).
    """
    results = api.words(ml=word, max=max_results)

    # Extract just the words, already ranked by Datamuse
    return [item["word"] for item in results]

def get_all_forms(word):
    """
    Get all inflected forms or target word using lemminflect.
    """
    
    forms = set()
    forms.add(word.lower())
    
    # Try different POS tags
    pos_tags = ['NOUN', 'VERB', 'ADJ', 'ADV']
    
    for pos in pos_tags:
        inflections = lemminflect.getAllInflections(word, upos=pos)
        for form_list in inflections.values():
            forms.update(form_list)
    
    return forms

##############################
# SYNONYM BANK 
def synonym_bank(word:str, k=20, topn=100):
    """
    Takes the target word and returns a list of filtered synonyms using 
    fastText and Datamuse for synonym banks

    Args:
    word: target word 
    k: number of clean results to return for fastText
    topn: numbger of fastText neighbors to retrieve

    returns a list of candidate words
    """
    # check if word exists
    if word not in ft_model:
        return 'word not found'
    
    first_syn_set = set()

    # fasttext synonym bank
    fastext_bank = faststext_synonyms(word, k, topn)
    # datamust synonym bank
    datamuse_bank = datamuse_synonyms(word, 10)

    first_syn_set |= set(fastext_bank + datamuse_bank) 
    syn_list = list(first_syn_set)

    lemmas = get_all_forms(lemmatize('walking', get_pos('walking')).pop())

    for word in syn_list:
        if word in lemmas:
            syn_list.remove(word)

    final_list = syn_list

    for word in final_list:
        if " " in word:
            final_list.remove(word)

    return final_list

def rank_synonyms(original_sentence, target_word, synonyms, top_n=None, threshold=None):
    """
    Rank synonyms by contextual fit using embeddings.
    
    Args:
    original_sentence: Original sentence with target word
    target_word: Word to replace
    synonyms: List of potential synonyms
    top_n: Return only top N results (optional)
    threshold: Return only results above this score (optional)
    
    returns List of (synonym, similarity_score) tuples, sorted by score
    """
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Use word boundary replacement
    def replace_word(sentence, old_word, new_word):
        pattern = r'\b' + re.escape(old_word) + r'\b'
        # Adding count=1 only replaces the first occurrence
        return re.sub(pattern, new_word, sentence, count=1, flags=re.IGNORECASE)

    # Batch encode


    candidates = []
    for syn in synonyms:
        candidates.append(replace_word(original_sentence, target_word, syn))

    # candidates = [replace_word(original_sentence, target_word, syn) for syn in synonyms]
    original_embedding = model.encode(original_sentence)
    candidate_embeddings = model.encode(candidates)
    
    # Calculate similarities
    similarities = cosine_similarity(
        original_embedding.reshape(1, -1),
        candidate_embeddings
    )[0]
    
    # Rank
    ranked = sorted(zip(synonyms, similarities), key=lambda x: x[1], reverse=True)
    
    # Filter
    if threshold is not None:
        ranked = [(syn, sim) for syn, sim in ranked if sim >= threshold]
    
    if top_n is not None:
        ranked = ranked[:top_n]
    
    return ranked

##############################
# FINAL FUNCTIONS
def rank_syn_usage(sentance, target_word, top_n=None, threshold=None):
    """
    Generates top synonyms for a word given the context of a sentence

    Ranks the results based on replacing the target word with a number of synonyms,
    measuring the embedding, and comparing that score with the original and then against other
    candidates. Returns based on scores from high to low.

    Args:
    sentence: string sentence or input text
    target_word: word to create nynonyms from 
    top_n: top number of words to return
    threshold: score theshold in which to return words
    
    returns a list of sets containing the top candidate and respective score
    """
    if target_word not in ft_model:
        return []
    
    synonyms = synonym_bank(target_word)
    if not synonyms:
        return []

    results = rank_synonyms(
        sentance,
        target_word,
        synonym_bank(target_word),
        top_n,
        threshold
    )

    results = results[:10]
    return results

# EX USAGE
# test = rank_syn_usage(
#     "we are walking to the bank",
#     "walking")
# print(test)


def syn_ranked_similarity(syn_list, origin_word):
    """
    Compares the results from rank_syn_usage, top synonyms in context, with the original target word
    
    Comparing embeddings isn't always perfect, so this function reranks the top results
    based on their fastText similarity score with the original word (without context.
    Results that are more than one word are ignored and automatically put at the end.
    
    Args:
    syn_list: list of sets, result of rank_syn_usage
    origin_word: target word

    Returns an ordered list of sets containing the candidate word and their score. 
    """

    first, end = [], []
    for cand, score in syn_list:
        try:
            sim_score = ft_model.similarity(cand, origin_word)
            first.append((cand, sim_score))
        except:
            end.append((cand, 0.0))
    sorted_first = sorted(first, key=lambda x: x[1], reverse=True)

    final = sorted_first + end 
    return  final

    

##########
# FLASK
 
# @app.route("/get-synonyms", methods=["POST"])
# def rank_usage():
#     data = request.get_json()

#     sentence = data.get("sentence", "").strip()
#     target_word = data.get("word", "").strip()
#     top_n = data.get("top_n")
#     threshold = data.get("threshold")

#     if not sentence or not target_word:
#         return jsonify([])

#     results = rank_syn_usage(
#         sentence,
#         target_word,
#         top_n=top_n,
#         threshold=threshold
#     )

#     results_alt = syn_ranked_similarity(results, target_word)

#     # Convert tuples to JSON-safe format
#     response = {
#         "context_rank": [
#             {"word": syn, "score": float(score)}
#             for syn, score in results
#         ],
#         "similarity_rank": [
#             {"word": syn, "score": float(score)}
#             for syn, score in results_alt
#         ]
#     }

#     return jsonify(response)

# if __name__ == "__main__":
#     app.run(debug=True)


##########

# TESTING TIMING
# end_time = time.time()
# print(end_time - start_time)

##########
