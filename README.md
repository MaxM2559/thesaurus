## About
This project is a thesaurus that uses the context of the surrounding sentence/text to help in finding the best fitting synonyms for the target word. For example, the word "bank" could mean edge of a river or financial institution. Normal thesaurus tools would give synonyms for both definition, but this tool uses the context of the sentence such as "I deposited money into the bank" to determine the correct meaning. This is done mostly through contextual embeddings with Meta's open source fastText text models.

This project was created as an exploration of using "dumb" tools in response to my own increasing use of AI in my writing. However, for finding synonyms, I find myself often using websites like Dictionary.com because these sites are faster and provide a more bare bones tool that forces people like me to more involved in choosing which word fits best. This project is a test of a more improved version of an online thesaurus. This trend in dumber tech can be seen in the rise with "dumb" smartphones and disposable cameras.

---

## Demo

The demo video contains two examples of the program eprofrms well, decent, and poorly, in that order. the examples are: 

### GOOD
He made a strong point during the debate.
Can you point me in the direction of the school?

The judge will sentence the defendant tomorrow.
The grammar teacher corrected the sentence.

### OK
He opened a new window on his laptop.
A cold breeze came through the open window.

I deposited my paycheck at the bank.
The fisherman sat quietly on the bank of the river.

### BAD
She filed a motion with the court.
The kids played basketball on the court.

The program crashed unexpectedly.
The car crashed into a guardrail.

---

## Methods
The general structure for this tool is gathering synonyms, filtering, and then measuring how each candidate synonym fits in the sentence with embeddings, all in Python. The synonyms are gathered only using fastText text classification library and the datamuse API. I have tried using Wordnet and other Python tools (like AyDictionary and GloVe) but through testing I have foudn fastText and datamuse to be the best combination for results and speed. Suggestions for potential better alternatives are listed in the improvements section. Also, the filtering is not heavy since fastText and datamuse already draw on words based on lemmas.

The final step involves embedding the original sentence to a vector (with fastText), and then replacing the original word in the sentence with each candidate synonym, getting the candidate embedding, and scoring the two vectors with cosine similarity. The top 10 highest scoring candidate synonyms are then shown to the user.

This process is not perfect, so there is an option for an extra final step of directly comparing the final 10 words to the original word (without context) with fastText and scoring based on similarity. This option is shown as the "Word Simlarity" button. This process still involves the context of the input text for the selection of the candidate words.

---

## Improvements
I spent a long while trying to get this project hosted on Render. I tried pruning the FastText model and various methods of uploading vocabulary, but it seems I can't skate around the general 502 error. 

This tool is far from perfect. It fails in regards to ambiguous words, such as an input sentence "big can of worms" would not provide the correct synonyms for "can" in this context. Also, because of the way embedding works, "the quick brown fox" would provide "slow" as a high scoring synonym for "quick" (the Word Similarity option is a band-aid fix for this).

The more direct improvement would be using a larger language model. I used the model trained on the "1 million word vectors trained on Wikipedia 2017, UMBC webbase corpus and statmt.org news dataset (16B tokens)" pruned to only keep the top 100,000 words. Larger models fastText models exist, which would improve the gathering of synonyms as well as the sentence embedding.

I believe the best way to get synonyms would be to directly scrape of existing sites like Dictionary.com. Libraries for this existed before, but no longer function from websites constantly being updated.
