import math
NO_SMOOTHING = 0
ADD_ONE = 1
ADD_K = 2

NUM_OF_DIFFERENT_CHARS = 44
  
def language_model_prob(language_model, word, n=4, smoothing=ADD_K):

    word_counts1 = language_model[0]
    word_counts2 = language_model[1]
    
    word = "S" + word + "E"
    
    word_probability = 0

    for i in range(0, len(word) - (n - 1)):
        cur_ngram_minus_one = word[i: i + n - 1]
        cur_ngram = word[i: i + n]

        if smoothing == NO_SMOOTHING:
            if cur_ngram_minus_one in word_counts1 and cur_ngram in word_counts2:
                word_probability += math.log2(word_counts2[cur_ngram] / word_counts1[cur_ngram_minus_one])
            else:
                word_probability = 0
                break

        elif smoothing == ADD_ONE or smoothing == ADD_K:
            if cur_ngram_minus_one not in word_counts1:
                cur_ngram_minus_one_count = 0
            else:
                cur_ngram_minus_one_count = word_counts1[cur_ngram_minus_one]
            if cur_ngram not in word_counts2:
                cur_ngram_count = 0
            else:
                cur_ngram_count = word_counts2[cur_ngram]

            if smoothing == ADD_ONE:
                word_probability += math.log2((cur_ngram_count + 1) / (cur_ngram_minus_one_count + NUM_OF_DIFFERENT_CHARS ** (n - 1)))  # add-one formula
            else:  # smoothing == ADD_K
                word_probability += math.log2((cur_ngram_count + 1 / (NUM_OF_DIFFERENT_CHARS ** (n - 1))) / (cur_ngram_minus_one_count + 1))  # add-one formula
                

    if word_probability != 0:
        word_probability = 2 ** (word_probability / (len(word)))  # len(word)-2: the length of the original word before adding start and end chars.

    return word_probability
