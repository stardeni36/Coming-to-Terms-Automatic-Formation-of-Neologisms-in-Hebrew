import json
import operator
import re
import random

HEB_NIKUD_START = 1456
HEB_NIKUD_END = 1479
HEB_LETTERS_START = 1488
HEB_LETTERS_END = 1514

# nikud specific:
SHVA = 1456
HATAF_SEGOL = 1457
HATAF_PATAH = 1458
HATAF_KAMATZ = 1459
HIRIQ = 1460
TZERE = 1461
SEGOL = 1462
PATAH = 1463
KAMATZ = 1464
HOLAM = 1465
KUBUTZ = 1467
DAGESH = 1468
SHIN_YEMANIT = 1473

BEGED_KEFET = "בגדכפת"
GRONIYOT = "אהחער"
BUMAP = "בומפ"

ords_to_remove = [8220, 8221]
MIN_NIKUD_ALEFBET_RATIO = 0.3

HEH_FACTOR = 5
VAV_FACTOR = 2

TRAIN_RATIO = 0.8
VALIDATION_RATIO = 0.1
TEST_RATIO = 0.1

def is_nikud(letter):

    if ord(letter)>= HEB_NIKUD_START and ord(letter)<=HEB_NIKUD_END:
        return True
    return False


def is_heb_character(letter):

    if ord(letter)>=HEB_LETTERS_START and ord(letter)<=HEB_LETTERS_END:
        return True
    return False


def basic_clean(text):

    nikud_chars = 0
    alefbet_chars = 0

    cleaned = ""
    # clean text from non hebrew characters:
    for letter in text:
        if letter == ' ' or letter=='-':
            cleaned += letter
        elif is_nikud(letter):
            cleaned += letter
            nikud_chars += 1
        elif is_heb_character(letter):
            cleaned += letter
            alefbet_chars += 1
        else:
            if not (letter == '"' or ord(letter) in ords_to_remove):
                cleaned += " "

    # clean unnecessary spaces:
    cleaned = re.sub(' +', ' ', cleaned)

    # clean one character words:
    cleaned = ' '.join([x for x in cleaned.split() if len(x)>1])

    nikud_alefbet_ratio = 0
    if alefbet_chars != 0:
        nikud_alefbet_ratio = nikud_chars/alefbet_chars

    if nikud_alefbet_ratio > MIN_NIKUD_ALEFBET_RATIO:
        return cleaned, True  # we consider the text to be with nikud
    return cleaned, False  # we consider the text to be without nikud

    return cleaned


def combine_dictionaries_and_clean_text():

    general_no_nikud_path = "no_nikud_data.txt"
    general_with_nikud_path = "with_nikud_data.txt"

    dictionaries = ["heb_texts1", "heb_texts2", "heb_texts3", "heb_texts4", "heb_texts5", "heb_texts6"]

    with open(general_no_nikud_path, 'w', encoding='utf8') as nnf, open(general_with_nikud_path, 'w', encoding='utf8') as wnf:
        for dict_name in dictionaries:
            with open(dict_name + ".json", 'r', encoding='utf8') as f:
                print(dict_name)
                heb_texts = json.load(f)
                for text_key in heb_texts:
                    text = heb_texts[text_key][0]
                    cleaned_text, with_nikud = basic_clean(text)
                    if with_nikud:
                        for token in cleaned_text.split():
                            wnf.write(token + "\n")
                    else:
                        for token in cleaned_text.split():
                            nnf.write(token + "\n")


def with_nikud_data_numeric_info(data_path):
    line_count = 0
    char_count = 0
    vocab = {}

    with open(data_path, 'r', encoding='utf8') as f:
        for line in f:
            line_count += 1
            char_count += len(line) - 1  # we do not count '\n'
            for c in line:
                vocab[c] = 0

    print("word count: ", str(line_count))
    print("char count: ", str(char_count))
    print("vocab size: ", str(len(vocab)))


def divide_data_to_train_val_test(data_path):

    dict = {}
    with open(data_path, 'r', encoding='utf8') as f:
        idx = 0
        for line in f:
            dict[idx] = line
            idx += 1


    dict_size = len(dict)
    indices_list = list(range(dict_size))
    random.shuffle(indices_list)
    random.shuffle(indices_list)

    train_indices = indices_list[:int(TRAIN_RATIO*dict_size)]
    val_indices = indices_list[int(TRAIN_RATIO*dict_size): int(TRAIN_RATIO*dict_size) + int(VALIDATION_RATIO*dict_size)]
    test_indices = indices_list[int(TRAIN_RATIO*dict_size) + int(VALIDATION_RATIO*dict_size):]

    with open(data_path.replace(".txt", "_train.txt"), 'w', encoding='utf8') as f:
        for idx in train_indices:
            f.write(dict[idx])

    with open(data_path.replace(".txt", "_val.txt"), 'w', encoding='utf8') as f:
        for idx in val_indices:
            f.write(dict[idx])

    with open(data_path.replace(".txt", "_test.txt"), 'w', encoding='utf8') as f:
        for idx in test_indices:
            f.write(dict[idx])

    print(len(train_indices))
    print(len(val_indices))
    print(len(test_indices))


# if there are two nikud chars in a row sort them based on their unicode..
def arrange_nikud(word):
    new_word = ""
    nikud_seq = []
    for letter in word:
        if is_nikud(letter):
            nikud_seq += [ord(letter)]
        else:
            if len(nikud_seq) != 0:
                nikud_seq = sorted(nikud_seq)
                nikud = "".join([chr(nikud_ord) for nikud_ord in nikud_seq])
                nikud_seq = []
                new_word += nikud
            new_word += letter

    if len(nikud_seq) != 0:
        nikud_seq = sorted(nikud_seq)
        nikud = "".join([chr(nikud_ord) for nikud_ord in nikud_seq])
        new_word += nikud

    if word != new_word:
        return new_word

    return new_word


def create_word_counts_dict(data_path, dict_path):

    word_counts = {}

    with open(data_path, 'r', encoding='utf8') as f:
        for word in f:
            word = word[:-1]
            word = arrange_nikud(word)
            if word not in word_counts:
                word_counts[word] = 0
            word_counts[word] += 1

    with open(dict_path, 'w', encoding='utf8') as outfile:
        json.dump(word_counts, outfile)

    return word_counts #(אותך אוהב אני)


def clean_mem_shimush(word, word_counts, to_print=False):

    # first case: mem, hiriq and dagesh hazak on the first letter.
    if len(word) > 4 and word[1] == chr(HIRIQ) and (word[3] == chr(DAGESH) or word[4] == chr(DAGESH)):

        inx = word.find(chr(DAGESH))
        if word[2] in BEGED_KEFET:
            filtered_word = word[2:]
        else:
            filtered_word = word[2:inx] + word[inx+1:]
        if filtered_word in word_counts:
            if word_counts[word] < word_counts[filtered_word]:
                return filtered_word

    # second case: mem, tzere and ot gronit without dagesh
    elif len(word) > 2 and word[1] == chr(TZERE) and word[2] in GRONIYOT:
        filtered_word = word[2:]
        if filtered_word in word_counts:
            if word_counts[word] < word_counts[filtered_word]:
                return filtered_word

    return word


def clean_shin_shimush(word, word_counts):

    if len(word) > 5 and word[1] == chr(SEGOL) and word[2] == chr(SHIN_YEMANIT):

        filtered_word = word

        # first case: shin, shin_yemanit, segol, dagesh
        if word[4] == chr(DAGESH) or word[5] == chr(DAGESH):

            inx = word.find(chr(DAGESH))
            if word[3] in BEGED_KEFET:
                filtered_word = word[3:]
            else:
                filtered_word = word[3:inx] + word[inx+1:]

        # second case: shin, shin yemanit, segol and ot gronit without dagesh
        if word[3] in GRONIYOT:
            filtered_word = word[3:]

        if filtered_word in word_counts:
            if word_counts[word] < word_counts[filtered_word]:
                return filtered_word

    return word


def clean_heh_shimush(word, word_counts):

    # first case: heh, patah, dagesh
    if len(word) > 4 and word[1] == chr(PATAH) and (word[3] == chr(DAGESH) or word[4] == chr(DAGESH)):
        inx = word.find(chr(DAGESH))
        if word[2] in BEGED_KEFET:
            filtered_word = word[2:]
        elif inx == 3:
            filtered_word = word[2] + word[4:]
        else:  # inx == 4
            filtered_word = word[2:4] + word[5:]

        if filtered_word in word_counts:
            if word_counts[word] < HEH_FACTOR*word_counts[filtered_word]:  # TODO: magic number
                return filtered_word

    # second case: heh, patah/kamatz/segol, ot gronit, without dagesh
    elif len(word) > 2 and word[2] in GRONIYOT:
        if (word[1] == chr(KAMATZ)) or (word[1] == chr(PATAH)) or (word[1] == chr(SEGOL)):
            filtered_word = word[2:]
            if filtered_word in word_counts:
                if word_counts[word] < HEH_FACTOR*word_counts[filtered_word]:  # TODO: magic number
                    return filtered_word

    return word


def filter_vav_lamed(word):

    if word[2] in BEGED_KEFET:
        if is_nikud(word[3]):
            filtered = word[2:4] + chr(DAGESH) + word[4:]
        else:
            filtered = word[2] + chr(DAGESH) + word[3:]
    else:
        filtered = word[2:]

    return filtered


def filter_kaf_bet(word):

    if word[3] in BEGED_KEFET:
        if is_nikud(word[4]):
            filtered = word[3:5] + chr(DAGESH) + word[5:]
        else:
            filtered = word[3] + chr(DAGESH) + word[4:]
    else:
        filtered = word[3:]

    return filtered


def clean_vav_shimush(word, word_counts):

    if len(word) < 4:
        return word

    filtered = word

    # first case: vav with shva
    if word[1] == chr(SHVA):
        filtered = filter_vav_lamed(word)

    # second case: vav, shuruq, shva/bumap
    elif word[1] == chr(DAGESH) and (word[3] == chr(SHVA) or word[2] in BUMAP):
        filtered = filter_vav_lamed(word)

    # third case: vav nikud is as the hataf nikud on the first letter
    elif (word[1] == chr(PATAH) and word[3] == chr(HATAF_PATAH)) \
            or (word[1] == chr(KAMATZ) and word[3] == chr(HATAF_KAMATZ)) \
            or (word[1] == chr(SEGOL) and word[3] == chr(HATAF_SEGOL)):
        filtered = filter_vav_lamed(word)

    # forth case: vav with kamatz
    elif word[1] == chr(KAMATZ):
        filtered = filter_vav_lamed(word)

    # fifth case: vav, hiriq, yud without nikud
    elif word[1] == chr(HIRIQ) and word[2] == "י" and is_heb_character(word[3]):
        filtered = word[2] + chr(SHVA) + word[3:]

    if filtered in word_counts:
        if word_counts[word] < word_counts[filtered]:
            return filtered

    return word


def clean_kaf_lamed_bet_shimush(word, word_counts):

    if len(word) < 5:
        return word

    filtered = word

    # first case: kaf/lamed/bet, shva, dagesh
    if word[1] == chr(SHVA):
        if word[0] == "ל":
            filtered = filter_vav_lamed(word)
        else:
            filtered = filter_kaf_bet(word)

    # second case: (kaf/bet or lamed), hiriq, shva
    elif word[0] in "כב" and word[1] == chr(HIRIQ) and word[4] == chr(SHVA):
        filtered = filter_kaf_bet(word)
    elif word[0] == "ל" and word[1] == chr(HIRIQ) and word[3] == chr(SHVA):
        filtered = filter_vav_lamed(word)

    # third case: (kaf/bet or lamed) nikud is as the hataf nikud on the first letter
    elif (word[0] in "כב" and word[1] == chr(PATAH) and word[4] == chr(HATAF_PATAH)) \
            or (word[1] == chr(KAMATZ) and word[4] == chr(HATAF_KAMATZ)) \
            or (word[1] == chr(SEGOL) and word[4] == chr(HATAF_SEGOL)):
        filtered = filter_kaf_bet(word)
    elif (word[0] == "ל" and word[1] == chr(PATAH) and word[3] == chr(HATAF_PATAH)) \
            or (word[1] == chr(KAMATZ) and word[3] == chr(HATAF_KAMATZ)) \
            or (word[1] == chr(SEGOL) and word[3] == chr(HATAF_SEGOL)):
        filtered = filter_vav_lamed(word)

    # forth case: (kaf/bet or lamed), hiriq, yud without nikud
    elif word[0] in "כב" and word[1] == chr(HIRIQ) and word[3] == "י" and is_heb_character(word[4]):
        filtered = word[3] + chr(SHVA) + word[4:]
    elif word[0] == "ל" and word[1] == chr(HIRIQ) and word[2] == "י" and is_heb_character(word[3]):
        filtered = word[2] + chr(SHVA) + word[3:]

    # fifth case: kaf/lamed/bet with heh haydia:
    elif word[0] == "ל" and word[1] == chr(PATAH) and (word[3] == chr(DAGESH) or word[4] == chr(DAGESH)):
        inx = word.find(chr(DAGESH))
        if word[2] in BEGED_KEFET:
            filtered = word[2:]
        else:
            filtered = word[2:inx] + word[inx+1:]
    elif word[0] in "כב" and word[1] == chr(PATAH) and (word[3] == chr(DAGESH) or word[4] == chr(DAGESH) or (len(word)>5 and word[5] == chr(DAGESH))):
        inx = word[3:].find(chr(DAGESH)) + 3
        if word[3] in BEGED_KEFET:
            filtered = word[3:]
        else:
            filtered = word[3:inx] + word[inx+1:]
    else:
        if word[0] == "ל" and word[2] in GRONIYOT:
            filtered = word[2:]
        elif word[0] in "כב" and word[3] in GRONIYOT:
            filtered = word[3:]

    if filtered in word_counts:
        if word_counts[word] < word_counts[filtered]:  # TODO: magic number
            return filtered

    return word


def clean_simush_letters(data_path, cleaned_data_path):

    word_counts = create_word_counts_dict(data_path, "word_counts_" + data_path + ".json")
    print("num of different words: " + str(len(word_counts)))

    with open(data_path, 'r', encoding='utf8') as f, open(cleaned_data_path, 'w', encoding='utf8') as outfile:

        for word in f:

            word = word[:-1]
            word = arrange_nikud(word)

            if word[0] == "מ":
                word = clean_mem_shimush(word, word_counts)

            elif word[0] == "ש":
                word = clean_shin_shimush(word, word_counts)

            elif word[0] == "ה":
                word = clean_heh_shimush(word, word_counts)

            elif word[0] == "ו":
                word = clean_vav_shimush(word, word_counts)

            elif word[0] in "כלב":
                word = clean_kaf_lamed_bet_shimush(word, word_counts)

            outfile.write(word + "\n")


if __name__ == '__main__':

    # divide_data_to_train_val_test("with_nikud_data_cleaned5.txt")

    char_count = 0
    word_count = 0
    with open("with_nikud_data.txt", 'r', encoding='utf8') as f:
        for word in f:
            word = word[:-1]
            word_count += 1
            char_count += len(word)

    print(word_count)
    print(char_count)
    print(char_count/word_count)








