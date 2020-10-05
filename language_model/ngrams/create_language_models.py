import json
import math
import ast
import operator

NO_SMOOTHING = 0
ADD_ONE = 1
ADD_K = 2

NUM_OF_DIFFERENT_CHARS = 44


def ngrams_in_word_heb(word, n):
    ngrams = []
    for i in range(0, len(word) - (n - 1)):
        cur_ngram = tuple(word[i: i + n])
        ngrams += [cur_ngram]

    return ngrams


def language_model_prob(word, n, counts_file1_path, counts_file2_path, smoothing=NO_SMOOTHING):

    with open(counts_file1_path, 'r', encoding='utf8') as f:
        word_counts1 = json.load(f)

    with open(counts_file2_path, 'r', encoding='utf8') as f:
        word_counts2 = json.load(f)

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
                word_probability += math.log2((cur_ngram_count + 1) / (
                            cur_ngram_minus_one_count + NUM_OF_DIFFERENT_CHARS ** (n - 1)))  # add-one formula
            else:  # smoothing == ADD_K
                word_probability += math.log2((cur_ngram_count + 1 / (NUM_OF_DIFFERENT_CHARS ** (n - 1))) / (cur_ngram_minus_one_count + 1))  # add-one formula

    if word_probability != 0:
        word_probability = 2 ** (word_probability / (len(word)))  # len(word)-2: the length of the original word before adding start and end chars.

    return word_probability


def language_model_prob_bigram(word):
    counts_file1_path = "language model dicts\\with_nikud_data_cleaned5_new_nbn_unique_train_val_1gram.json"
    counts_file2_path = "language model dicts\\with_nikud_data_cleaned5_new_nbn_unique_train_val_2gram.json"
    return language_model_prob(word, 2, counts_file1_path, counts_file2_path, smoothing=NO_SMOOTHING)


def language_model_prob_trigram(word):
    counts_file1_path = "language model dicts\\with_nikud_data_cleaned5_new_nbn_unique_train_val_2gram.json"
    counts_file2_path = "language model dicts\\with_nikud_data_cleaned5_new_nbn_unique_train_val_3gram.json"
    return language_model_prob(word, 3, counts_file1_path, counts_file2_path, smoothing=NO_SMOOTHING)


def language_model_prob_4gram(word):
    counts_file1_path = "language model dicts\\with_nikud_data_cleaned5_new_nbn_unique_train_val_3gram.json"
    counts_file2_path = "language model dicts\\with_nikud_data_cleaned5_new_nbn_unique_train_val_4gram.json"
    return language_model_prob(word, 4, counts_file1_path, counts_file2_path, smoothing=NO_SMOOTHING)


def language_model_prob_5gram(word):
    counts_file1_path = "language model dicts\\with_nikud_data_cleaned5_new_nbn_unique_train_val_4gram.json"
    counts_file2_path = "language model dicts\\with_nikud_data_cleaned5_new_nbn_unique_train_val_5gram.json"
    return language_model_prob(word, 5, counts_file1_path, counts_file2_path, smoothing=NO_SMOOTHING)


def language_model_prob_6gram(word):
    counts_file1_path = "language model dicts\\with_nikud_data_cleaned5_new_nbn_unique_train_val_5gram.json"
    counts_file2_path = "language model dicts\\with_nikud_data_cleaned5_new_nbn_unique_train_val_6gram.json"
    return language_model_prob(word, 6, counts_file1_path, counts_file2_path, smoothing=NO_SMOOTHING)


def language_model_prob_bigram_add1(word):
    counts_file1_path = "language model dicts\\with_nikud_data_cleaned5_new_nbn_unique_train_val_1gram.json"
    counts_file2_path = "language model dicts\\with_nikud_data_cleaned5_new_nbn_unique_train_val_2gram.json"
    return language_model_prob(word, 2, counts_file1_path, counts_file2_path, smoothing=ADD_ONE)


def language_model_prob_trigram_add1(word):
    counts_file1_path = "language model dicts\\with_nikud_data_cleaned5_new_nbn_unique_train_val_2gram.json"
    counts_file2_path = "language model dicts\\with_nikud_data_cleaned5_new_nbn_unique_train_val_3gram.json"
    return language_model_prob(word, 3, counts_file1_path, counts_file2_path, smoothing=ADD_ONE)


def language_model_prob_4gram_add1(word):
    counts_file1_path = "language model dicts\\with_nikud_data_cleaned5_new_nbn_unique_train_val_3gram.json"
    counts_file2_path = "language model dicts\\with_nikud_data_cleaned5_new_nbn_unique_train_val_4gram.json"
    return language_model_prob(word, 4, counts_file1_path, counts_file2_path, smoothing=ADD_ONE)


def language_model_prob_5gram_add1(word):
    counts_file1_path = "language model dicts\\with_nikud_data_cleaned5_new_nbn_unique_train_val_4gram.json"
    counts_file2_path = "language model dicts\\with_nikud_data_cleaned5_new_nbn_unique_train_val_5gram.json"
    return language_model_prob(word, 5, counts_file1_path, counts_file2_path, smoothing=ADD_ONE)


def language_model_prob_6gram_add1(word):
    counts_file1_path = "language model dicts\\with_nikud_data_cleaned5_new_nbn_unique_train_val_5gram.json"
    counts_file2_path = "language model dicts\\with_nikud_data_cleaned5_new_nbn_unique_train_val_6gram.json"
    return language_model_prob(word, 6, counts_file1_path, counts_file2_path, smoothing=ADD_ONE)


def language_model_prob_bigram_addK(word):
    counts_file1_path = "language model dicts\\with_nikud_data_cleaned5_new_nbn_unique_train_val_1gram.json"
    counts_file2_path = "language model dicts\\with_nikud_data_cleaned5_new_nbn_unique_train_val_2gram.json"
    return language_model_prob(word, 2, counts_file1_path, counts_file2_path, smoothing=ADD_K)


def language_model_prob_trigram_addK(word):
    counts_file1_path = "language model dicts\\with_nikud_data_cleaned5_new_nbn_unique_train_val_2gram.json"
    counts_file2_path = "language model dicts\\with_nikud_data_cleaned5_new_nbn_unique_train_val_3gram.json"
    return language_model_prob(word, 3, counts_file1_path, counts_file2_path, smoothing=ADD_K)


def language_model_prob_4gram_addK_train_only(word):
    counts_file1_path = "language model dicts\\with_nikud_data_cleaned5_new_nbn_unique_train_3gram.json"
    counts_file2_path = "language model dicts\\with_nikud_data_cleaned5_new_nbn_unique_train_4gram.json"
    return language_model_prob(word, 4, counts_file1_path, counts_file2_path, smoothing=ADD_K)


def language_model_prob_4gram_addK(word):
    counts_file1_path = "language model dicts\\with_nikud_data_cleaned5_new_nbn_unique_train_val_3gram.json"
    counts_file2_path = "language model dicts\\with_nikud_data_cleaned5_new_nbn_unique_train_val_4gram.json"
    return language_model_prob(word, 4, counts_file1_path, counts_file2_path, smoothing=ADD_K)


def language_model_prob_5gram_addK(word):
    counts_file1_path = "language model dicts\\with_nikud_data_cleaned5_new_nbn_unique_train_val_4gram.json"
    counts_file2_path = "language model dicts\\with_nikud_data_cleaned5_new_nbn_unique_train_val_5gram.json"
    return language_model_prob(word, 5, counts_file1_path, counts_file2_path, smoothing=ADD_K)


def language_model_prob_6gram_addK(word):
    counts_file1_path = "language model dicts\\with_nikud_data_cleaned5_new_nbn_unique_train_val_5gram.json"
    counts_file2_path = "language model dicts\\with_nikud_data_cleaned5_new_nbn_unique_train_val_6gram.json"
    return language_model_prob(word, 6, counts_file1_path, counts_file2_path, smoothing=ADD_K)


def language_model_prob_6gram_addK_train_only(word):
    counts_file1_path = "language model dicts\\with_nikud_data_cleaned5_new_nbn_unique_train_5gram.json"
    counts_file2_path = "language model dicts\\with_nikud_data_cleaned5_new_nbn_unique_train_6gram.json"
    return language_model_prob(word, 6, counts_file1_path, counts_file2_path, smoothing=ADD_K)


def langauge_model_good_turing_probs(word, n, probs1_dict_path, probs2_dict_path):

    with open(probs1_dict_path, "r", encoding='utf8') as f:
        probs1 = json.load(f)

    with open(probs2_dict_path, "r", encoding='utf8') as f:
        probs2 = json.load(f)

    word = "S" + word + "E"
    ngrams_in_word = ngrams_in_word_heb(word, n)

    word_probability = 0
    for ngram in ngrams_in_word:
        p2 = probs2["UNKNOWN"] if str(ngram) not in probs2 else probs2[str(ngram)]
        if n > 2:
            p1 = probs1["UNKNOWN"] if str(str((ngram[0], ngram[1]))) not in probs1 else probs1[str((ngram[0], ngram[1]))]
            word_probability += math.log2(p2 / p1)
        else:  # n == 2
            p1 = probs1["UNKNOWN"] if str((ngram[0],)) not in probs1 else probs1[str((ngram[0],))]
            word_probability += math.log2(p2 / p1)

    if word_probability != 0:
        word_probability = 2 ** (word_probability / (len(word)))

    return word_probability


def langauge_model_good_turing_bigram(word):
    probs1_dict_path = "language model dicts\\good_turing_training_only_1grams.json"
    probs2_dict_path = "language model dicts\\good_turing_training_only5_2grams.json"
    return langauge_model_good_turing_probs(word, 2, probs1_dict_path, probs2_dict_path)


def langauge_model_good_turing_trigram(word):
    probs1_dict_path = "language model dicts\\good_turing_training_only5_2grams.json"
    probs2_dict_path = "language model dicts\\good_turing_training_only5_3grams.json"
    return langauge_model_good_turing_probs(word, 3, probs1_dict_path, probs2_dict_path)


def langauge_model_good_turing_4gram(word):
    probs1_dict_path = "language model dicts\\good_turing_training_only5_3grams.json"
    probs2_dict_path = "language model dicts\\good_turing_training_only5_4grams.json"
    return langauge_model_good_turing_probs(word, 4, probs1_dict_path, probs2_dict_path)


def langauge_model_good_turing_5gram(word):
    probs1_dict_path = "language model dicts\\good_turing_4grams.json"
    probs2_dict_path = "language model dicts\\good_turing_5grams.json"
    return langauge_model_good_turing_probs(word, 5, probs1_dict_path, probs2_dict_path)


def langauge_model_good_turing_6gram(word):
    probs1_dict_path = "language model dicts\\good_turing_5grams.json"
    probs2_dict_path = "language model dicts\\good_turing_6grams.json"
    return langauge_model_good_turing_probs(word, 6, probs1_dict_path, probs2_dict_path)


if __name__ == '__main__':

    new_words = []
    with open("conspiracy.txt", 'r', encoding='utf8') as f:
        for line in f:
            words_in_line = ast.literal_eval(line)
            new_words += words_in_line

    probs_list = [0.1274358332157135, 0.19896048307418823, 0.22165332734584808, 0.2160610854625702, 0.15577422082424164, 0.2505057752132416, 0.22417007386684418, 0.2439037710428238, 0.1347336620092392, 0.1634588986635208, 0.18735745549201965, 0.24020928144454956, 0.10428937524557114, 0.16415974497795105, 0.22612857818603516, 0.24150630831718445, 0.09999341517686844, 0.2203160971403122, 0.15070119500160217, 0.17780713737010956, 0.08765581250190735, 0.10335046797990799, 0.16002577543258667, 0.09179438650608063, 0.08011222630739212, 0.1482027918100357, 0.13701608777046204, 0.19415654242038727, 0.06534101814031601, 0.2011863887310028, 0.1084870845079422, 0.2398720532655716, 0.09869162738323212, 0.22770781815052032, 0.14007200300693512, 0.25607922673225403, 0.10781432688236237, 0.1970767229795456, 0.15871958434581757, 0.17517076432704926, 0.19406680762767792, 0.19043667614459991, 0.1478937417268753, 0.23223549127578735, 0.19406680762767792, 0.19043667614459991, 0.1478937417268753, 0.23223549127578735, 0.15577422082424164, 0.2505057752132416, 0.22417007386684418, 0.2439037710428238, 0.12041326612234116, 0.18676449358463287, 0.13479310274124146, 0.2104727029800415, 0.16188077628612518, 0.21156661212444305, 0.10837635397911072, 0.10831857472658157, 0.11286308616399765, 0.19323231279850006, 0.14923623204231262, 0.19196604192256927, 0.08775684982538223, 0.14845460653305054, 0.09780757129192352, 0.18567411601543427, 0.11404549330472946, 0.1631377637386322, 0.13612554967403412, 0.1566648930311203, 0.08705144375562668, 0.14855992794036865, 0.13932113349437714, 0.166173055768013]
    probs_list = [round(prob, 5) for prob in probs_list]
    words_probs = zip(new_words, probs_list)

    sorted = sorted(words_probs, key=operator.itemgetter(1), reverse=True)

    for item in sorted:
        print(item[0] + "\t" + str(item[1]))

    # word1 = "×¦Ö°×”×•Ö¼×¤Ö´×™×¤Ö´×¢"
    # word2 = "×’×•Ö¹×œÖ´×™×¦Ö´×™×Ÿ"
    # word3 = "×žÖ°×‘Ö»×™Ö¸×©×�"
    # word4 = "×žÖ·×©Ö°×‚×›Ö¼×•Ö¹×¨Ö¶×ª"

    # print(word1)
    # print(round(language_model_prob_bigram(word1), 5))
    # print(round(language_model_prob_bigram_add1(word1), 5))
    # print(round(language_model_prob_bigram_addK(word1), 5))
    # print(round(language_model_prob_trigram(word1), 5))
    # print(round(language_model_prob_trigram_add1(word1), 5))
    # print(round(language_model_prob_trigram_addK(word1), 5))
    # print(round(language_model_prob_4gram(word1), 5))
    # print(round(language_model_prob_4gram_add1(word1), 5))
    # print(round(language_model_prob_4gram_addK(word1), 5))
    # print(round(language_model_prob_5gram(word1), 5))
    # print(round(language_model_prob_5gram_add1(word1), 5))
    # print(round(language_model_prob_5gram_addK(word1), 5))
    # print(round(language_model_prob_6gram(word1), 5))
    # print(round(language_model_prob_6gram_add1(word1), 5))
    # print(round(language_model_prob_6gram_addK(word1), 5))
    # print(round(langauge_model_good_turing_bigram(word1), 5))
    # print(round(langauge_model_good_turing_trigram(word1), 5))
    # print(round(langauge_model_good_turing_4gram(word1), 5))
    # print()



