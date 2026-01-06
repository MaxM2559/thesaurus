### File that stores pruning code for smaller FastText model 


### 1.17 GB -> 117 mb



## FASTTEXT MODEL ##########
# -- setup --
# ft_model = KeyedVectors.load_word2vec_format(model_path, binary=False)
# ----


### PRUNING
# ft = KeyedVectors.load("ft_model_fast.kv")
# TOP_K = 100_000   # start here

# def is_clean(word):
#     return (
#         word.isalpha() and
#         word.islower() and
#         len(word) > 2
#     )

# filtered_words = [
#     w for w in ft.index_to_key
#     if is_clean(w)
# ][:TOP_K]

# pruned = KeyedVectors(
#     vector_size=ft.vector_size
# )

# pruned.add_vectors(
#     filtered_words,
#     ft[filtered_words]
# )

# pruned.save("ft_model_pruned.kv")