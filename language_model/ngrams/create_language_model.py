
# data_path: path of data text file in which in every row there is one word.
# n: 2 for bigram, 3 for trigram etc.
# returns a dictionary.
import json
import math

# smoothing types:
from os import listdir
from os.path import isfile

NO_SMOOTHING = 0
ADD_ONE = 1
ADD_K = 2  # K = 1 / V (where V is the vocab. size)
KENESER_NEY = 3

VOCAB_SIZE = 44  # num of different hebrew chars (including start and end of sentence)

def count_ngrams(data_path, out_folder, n):

    ngram_counts = {}
    with open(data_path, 'r', encoding='utf8') as f:
        for word in f:
            word = word[:-1]  # remove \n
            word = "S" + word + "E"  # add start of sentence and end of sentence to
            for i in range(0, len(word) - (n-1)):
                ngram = word[i: i + n]
                if ngram not in ngram_counts:
                    ngram_counts[ngram] = 0
                ngram_counts[ngram] += 1

    with open(out_folder + data_path.split('\\')[1].replace('.txt', '_' + str(n) + "gram.json"), 'w', encoding='utf8') as outfile:
        json.dump(ngram_counts, outfile)
    return ngram_counts


def create_ngram_count_dicts():

    unigram_counts = count_ngrams("data\\with_nikud_data_new_nbn_unique_train.txt", "language model dicts\\", 1)
    bigram_counts = count_ngrams("data\\with_nikud_data_new_nbn_unique_train.txt", "language model dicts\\",2)
    trigram_counts = count_ngrams("data\\with_nikud_data_new_nbn_unique_train.txt", "language model dicts\\", 3)
    quadgram_counts = count_ngrams("data\\with_nikud_data_new_nbn_unique_train.txt", "language model dicts\\", 4)
    cinqgram_counts = count_ngrams("data\\with_nikud_data_new_nbn_unique_train.txt", "language model dicts\\", 5)
    sixgram_counts = count_ngrams("data\\with_nikud_data_new_nbn_unique_train.txt", "language model dicts\\", 6)

    unigram_counts_cleaned = count_ngrams("data\\with_nikud_data_cleaned5_new_nbn_unique_train.txt", "language model dicts\\", 1)
    bigram_counts_cleaned = count_ngrams("data\\with_nikud_data_cleaned5_new_nbn_unique_train.txt", "language model dicts\\", 2)
    trigram_counts_cleaned = count_ngrams("data\\with_nikud_data_cleaned5_new_nbn_unique_train.txt", "language model dicts\\", 3)
    quadgram_counts_cleaned = count_ngrams("data\\with_nikud_data_cleaned5_new_nbn_unique_train.txt", "language model dicts\\", 4)
    cinqgram_counts_cleaned = count_ngrams("data\\with_nikud_data_cleaned5_new_nbn_unique_train.txt", "language model dicts\\", 5)
    sixqgram_counts_cleaned = count_ngrams("data\\with_nikud_data_cleaned5_new_nbn_unique_train.txt", "language model dicts\\", 6)

    unigram_counts = count_ngrams("data\\with_nikud_data_new_nbn_unique_train_val.txt", "language model dicts\\", 1)
    bigram_counts = count_ngrams("data\\with_nikud_data_new_nbn_unique_train_val.txt", "language model dicts\\", 2)
    trigram_counts = count_ngrams("data\\with_nikud_data_new_nbn_unique_train_val.txt", "language model dicts\\", 3)
    quadgram_counts = count_ngrams("data\\with_nikud_data_new_nbn_unique_train_val.txt", "language model dicts\\", 4)
    cinqgram_counts = count_ngrams("data\\with_nikud_data_new_nbn_unique_train_val.txt", "language model dicts\\", 5)
    sixgram_counts = count_ngrams("data\\with_nikud_data_new_nbn_unique_train_val.txt", "language model dicts\\", 6)

    unigram_counts_cleaned = count_ngrams("data\\with_nikud_data_cleaned5_new_nbn_unique_train_val.txt", "language model dicts\\", 1)
    bigram_counts_cleaned = count_ngrams("data\\with_nikud_data_cleaned5_new_nbn_unique_train_val.txt", "language model dicts\\", 2)
    trigram_counts_cleaned = count_ngrams("data\\with_nikud_data_cleaned5_new_nbn_unique_train_val.txt", "language model dicts\\", 3)
    quadgram_counts_cleaned = count_ngrams("data\\with_nikud_data_cleaned5_new_nbn_unique_train_val.txt", "language model dicts\\", 4)
    cinqgram_counts_cleaned = count_ngrams("data\\with_nikud_data_cleaned5_new_nbn_unique_train_val.txt", "language model dicts\\", 5)
    sixqgram_counts_cleaned = count_ngrams("data\\with_nikud_data_cleaned5_new_nbn_unique_train_val.txt", "language model dicts\\", 6)


def merge_files(train, val, new_path):
    with open(new_path, 'w', encoding='utf8') as outfile:
        with open(train, 'r', encoding='utf8') as f:
            for line in f:
                outfile.write(line)
        with open(val, 'r', encoding='utf8') as f:
            for line in f:
                outfile.write(line)


# input: file with two columns - the first contains the word, the second contains its probability according
# to the language model.
def compute_perplexity_file(file_path):

    log_sum = 0
    num_of_words = 0

    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            spltd = line.split(',')
            probability = float(spltd[1].strip())
            if probability != 0:
                log_prob = math.log2(probability)
                log_sum += log_prob
                num_of_words += 1

    l = log_sum / num_of_words
    preplexity = 2 ** ((-1)*l)

    return preplexity


# compute perplexity for every file in given folder
def compute_perplexity_folder(folder_path):
    for file_name in listdir(folder_path):
        if (file_name.endswith('.csv')):
            print(file_name)
            print(compute_perplexity_file(folder_path + "\\" + file_name))
            print()


# word counts path1: n-1 gram counts
# word counts path2: ngram counts
# data path: data to compute probability for every word (line) in it.
# probabilities path: path to write the probabilities for every word in the data.
def compute_ngram_lm_prob(word_counts_path1, word_counts_path2, data_path, probabilities_path, n, smoothing=NO_SMOOTHING):

    with open(word_counts_path1, 'r', encoding='utf8') as f:
        word_counts1 = json.load(f)

    with open(word_counts_path2, 'r', encoding='utf8') as f:
        word_counts2 = json.load(f)

    with open(data_path, 'r', encoding='utf8') as f, open(probabilities_path, 'w', encoding='utf8') as outfile:
        for word in f:
            word = word[:-1]
            word = "S" + word + "E"
            word_probability = 0
            for i in range(0, len(word) - (n-1)):
                cur_ngram_minus_one = word[i: i + n - 1]
                cur_ngram = word[i: i + n]
                if smoothing == NO_SMOOTHING:
                    if cur_ngram_minus_one in word_counts1 and cur_ngram in word_counts2:
                        word_probability += math.log2(word_counts2[cur_ngram]/word_counts1[cur_ngram_minus_one])
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
                        word_probability += math.log2((cur_ngram_count + 1)/(cur_ngram_minus_one_count + VOCAB_SIZE**(n-1)))  # add-one formula
                    else: # smoothing == ADD_K
                        # word_probability += math.log2((cur_ngram_count + 1/(VOCAB_SIZE ** (n - 1))) / (cur_ngram_minus_one_count + 1))  # add-one formula
                        word_probability += math.log2((cur_ngram_count + 1/(VOCAB_SIZE ** n)) / (cur_ngram_minus_one_count + 1/n))  # add-one formula


            if word_probability != 0:
                word_probability = 2 ** (word_probability/(len(word)))  # len(word)-2: the length of the original word before adding start and end chars.

            outfile.write(word[1:-1] + ", " + str(word_probability) + "\n")


def compute_all_ngram_lm_probs(train_base, test_path, out_base, method=NO_SMOOTHING):

    method_str = "" if method == NO_SMOOTHING else "_add1" if method == ADD_ONE else "_addK2"

    for i in range(2, 7):
        compute_ngram_lm_prob(train_base + "_" + str(i-1) + "gram.json", train_base + "_" + str(i) + "gram.json",
                              test_path, out_base + "_" + str(i) + "_gram" + method_str + ".csv", i, method)



if __name__ == '__main__':

    # train: train + val. test: test
    # compute_ngram_lm_prob("language model dicts\\with_nikud_data_cleaned5_train_val_new_nbn_5gram.json",
    #                                    "language model dicts\\with_nikud_data_cleaned5_train_val_new_nbn_6gram.json",
    #                                    "data\\with_nikud_data_cleaned5_test_new_nbn.txt",
    #                                    "probability files\\train_val5_6gram_addK.csv", 6, ADD_K)
    #
    # compute_ngram_lm_prob("language model dicts\\with_nikud_data_train_val_new_nbn_5gram.json",
    #                                    "language model dicts\\with_nikud_data_train_val_new_nbn_6gram.json",
    #                                    "data\\with_nikud_data_test_new_nbn.txt",
    #                                    "probability files\\train_val_6gram_addK.csv", 6, ADD_K)

    # train: train. test: val
    # compute_ngram_lm_prob("language model dicts\\with_nikud_data_cleaned5_train_new_nbn_5gram.json",
    #                                    "language model dicts\\with_nikud_data_cleaned5_train_new_nbn_6gram.json",
    #                                    "data\\with_nikud_data_cleaned5_val_new_nbn.txt",
    #                                    "probability files\\train5_6gram_addK.csv", 6, ADD_K)
    #
    # compute_ngram_lm_prob("language model dicts\\with_nikud_data_train_new_nbn_5gram.json",
    #                                    "language model dicts\\with_nikud_data_train_new_nbn_6gram.json",
    #                                    "data\\with_nikud_data_val_new_nbn.txt",
    #                                    "probability files\\train_6gram_addK.csv", 6, ADD_K)

    # create_ngram_count_dicts()
    #
    # compute_all_ngram_lm_probs("language model dicts\\with_nikud_data_cleaned5_new_nbn_unique_train",
    #                            "data\\with_nikud_data_cleaned5_new_nbn_unique_val.txt", "probability files\\train5_unique", method=ADD_K)

    compute_perplexity_folder("probability files")


