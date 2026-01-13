import os
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


# flask one time NLTK download block
NLTK_DATA_DIR = os.path.join(os.path.dirname(__file__), "nltk_data")
nltk.data.path.append(NLTK_DATA_DIR)


# start_time = time.time()
api = datamuse.Datamuse()
STOPWORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

## FASTTEXT MODEL ##########
# -- setup --
# ft_model = KeyedVectors.load_word2vec_format(model_path, binary=False)
# ----
#print
# print('loading ft_model')
# ft_model = KeyedVectors.load("ft_model_pruned_85k.kv", mmap='r')
# print('ft_model loaded')

ft_model = None
ft_vocab = None

def get_model():
    global ft_model, ft_vocab
    if ft_model is None:
        ft_model = KeyedVectors.load(
            "ft_model_pruned_85k.kv",
            mmap="r"
        )
        ft_vocab = set(ft_model.key_to_index.keys())
    return ft_model
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

def clean_fasttext_results(query, topn=25, final_k=10):
    """
    Retrieve and clean FastText similarity results for a query word.
    
    :query: target word
    :topn: number of FastText neighbors to retrieve
    :final_k: number of clean results to return
    """
    print('[clean_fasttext_results] start', flush=True)

    query = query.lower()
    target_pos = get_dominant_pos(query)

    print('[clean_fasttext_results] attempt to find most similar words with ft_model.most_similar()', flush=True)
    model = get_model()
    raw_results = model.most_similar(query, topn=topn)

    #print
    print('[clean_fasttext_results] get raw results from ft_model', flush=True)

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
    for cand in candidates:
        if semantic_compatible(cand, q_pos_set, q_lemmas):
            reranked.append(cand)

    return reranked

##############################
# SYNONYM BANK HELPERS

def faststext_synonyms(word: str, k: int, topn=100):
    print('[faststext_synonyms] starting func -> clean_fasttext_results', flush=True)
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
    print('[synonym_bank] start')
    
    first_syn_set = set()

    # fasttext synonym bank
    fastext_bank = faststext_synonyms(word, k, topn)
    #print
    print('[synonym_bank] generated fasttext syn bank in synonym_bank',flush=True)
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

    #print
    print('[synonym_bank] finish synonym_bank',flush=True)

    return final_list

sent_model = None

def get_sentence_model():
    global sent_model
    if sent_model is None:
        sent_model = SentenceTransformer("all-MiniLM-L6-v2")
    return sent_model

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
    #print
    print('[rank_synonyms] starting rank_synonyms',flush=True)
    # model = SentenceTransformer('all-MiniLM-L6-v2')
    model = get_sentence_model()
    print('[rank_synonyms] loaded model from rank_synonyms',flush=True)
    
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
    
    #print
    print('[rank_synonyms] getting all candidates and embedding them w sentence transformer',flush=True)

    # Calculate similarities
    similarities = cosine_similarity(
        original_embedding.reshape(1, -1),
        candidate_embeddings
    )[0]
    
    #print
    print('[rank_synonyms] calculating cosine similarities',flush=True)

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
    #print
    print('[rank_syn_usage] starting rank_syn_usage',flush=True)
    if target_word not in ft_vocab:
        ## CHANGE  to ft_vocab from ft_model
        return []
    
    #print
    print('[rank_syn_usage] check if word in model',flush=True)
    
    synonyms = synonym_bank(target_word)
    if not synonyms:
        return []

    print('[rank_syn_usage] post synonym bank',flush=True)


    results = rank_synonyms(
        sentance,
        target_word,
        synonym_bank(target_word),
        top_n,
        threshold
    )

    #print
    print('[rank_synonyms] finish rank_synonyms',flush=True)

    results = results[:10]
    return results

# EX USAGE
# print(rank_syn_usage(
#     "I deposited money at the bank",
#     "bank"
#     ))

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
 
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get-synonyms", methods=["POST"])
def rank_usage():
    print('[rank_usage] loading ft_model and ft_vocab',flush=True)
    model = get_model()
    #print
    print("HIT /get-synonyms", flush=True)
    data = request.get_json(force=True)

    sentence = data.get("sentence", "").strip()
    target_word = data.get("word", "").strip()
    top_n = data.get("top_n", 10)
    threshold = data.get("threshold", None)

    if not sentence or not target_word:
        return jsonify([])

    #################
    print("ABOUT TO CALL rank_syn_usage", flush=True)
    try:
        results = rank_syn_usage(
            sentence,
            target_word,
            top_n=top_n,
            threshold=threshold
        )
        print("RETURNED FROM rank_syn_usage", flush=True)
    except Exception as e:
        print("EXCEPTION CALLING rank_syn_usage:", repr(e), flush=True)
        raise
    ################

    # results = rank_syn_usage(
    #     sentence,
    #     target_word,
    #     top_n=top_n,
    #     threshold=threshold
    # )

    #print
    print("AFTER rank_syn_usage", flush=True)

    results_alt = syn_ranked_similarity(results, target_word)

    # Convert tuples to JSON-safe format
    response = {
        "context_rank": [
            {"word": syn, "score": float(score)}
            for syn, score in results
        ],
        "similarity_rank": [
            {"word": syn, "score": float(score)}
            for syn, score in results_alt
        ]
    }

    return jsonify(response)

# if __name__ == "__main__":
#     app.run(debug=True)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

##########

# TESTING TIMING
# end_time = time.time()
# print(end_time - start_time)

##########
