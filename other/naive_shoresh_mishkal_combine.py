import re


HEB_NIKUD_START = 1456
HEB_NIKUD_END = 1479
HEB_LETTERS_START = 1488
HEB_LETTERS_END = 1514
# ENG_LETTERS_START = 65
# ENG_LETTERS_END = 122

ahvi_letters = "אהוי"
final_letters = list('םןךףץ') 
regular_letters = list('מנכפצ')  

DEBUG = True


# returns true iff the given word is with nikud:
def word_with_nikud(word):
    for letter in word:
        if is_nikud_char(letter):
            return True
    return False

# returns true if the given character is a nikud character.
def is_nikud_char(letter):
    if (ord(letter)<HEB_LETTERS_START) and (ord(letter)>=HEB_NIKUD_START):
        return True
    return False

def is_letter_char(letter):
    if (ord(letter)>=HEB_LETTERS_START and ord(letter)<=HEB_LETTERS_END):
        return True
    return False
      
def swap_nikud(word, i, j):
    return ''.join((word[:i], word[j], word[i+1:j], word[i], word[j+1:]))

# returns a list of words with different ways to add a specific nikud
def get_nikud_options(word):
    options = [word]
    for i in range(1,len(word)):
        if (is_nikud_char(word[i-1]) and is_nikud_char(word[i])):
            options_len = len(options)
            for j in range(options_len):
                new_word = swap_nikud(options[j], i-1, i)
                options += [new_word]
    return options

# cleans a given word from "nikud":
def clean_word_from_nikud(word):
    cleaned = ""
    for letter in word:
        if (ord(letter)>=HEB_LETTERS_START) or (letter==' ' or letter=='-'):
            cleaned += letter
    return cleaned


def clean_word_from_garbage(word):
    new_word = ""
    for c in word:
        if (is_nikud_char(c) or is_letter_char(c) or c.isspace()):
            new_word += c
    return new_word


def shoresh_mishkal_combine(shoresh, mishkal):
    shoresh = final_letter_to_regular_letter(clean_word_from_nikud(shoresh))
    word = mishkal
    if (len(shoresh)>3): # we cannot handle 4 letters shorashim right now
        if (DEBUG):
            print (shoresh + " : can't handle 4 letters shoresh yet")
        word = ""
    elif (shoresh[1]=='ו' or shoresh[1]=='י'): # we cannot handle ain-vav, ain-yodgzarot yet.
        if (DEBUG):
            print (shoresh + ' : problematic shoresh (ע"ו או ע"י)')
        word = "" 
    elif (shoresh[2]=='ה' or shoresh[2]=='י'): # we cannot handle lamed-heh, lamed-yod gzarot yet.
        if (DEBUG):
            print (shoresh + ' : problematic shoresh (ל"ה או ל"י)')
        word = ""
    else:
        
        idx_k = word.find('ק')
        idx_t = word.find('ט')
        idx_l = word.find('ל')
        word = word[:idx_k] + shoresh[0] + word[idx_k+1:idx_t] + shoresh[1] + word[idx_t+1:idx_l] + shoresh[2] + word[idx_l+1:]
        
    word = change_to_final_letter(word)
    return word

# if there are two nikud chars in a row sort them based on their unicode..
def arrange_nikud(word):
    new_word = ""
    nikud_seq = []
    for letter in word:
        if is_nikud_char(letter):
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

def change_to_final_letter(word):
    if (len(word)>0):
        final_letters_dict = {r:f for f,r in zip(final_letters,regular_letters)}
        if word[-1] in regular_letters:
            word = word[:-1] + final_letters_dict[word[-1]]
    return word
    
def final_letter_to_regular_letter(shoresh):
    final_letters_dict = {f:r for f,r in zip(final_letters,regular_letters)}
    for letter in shoresh:
        if letter in final_letters:
            shoresh = shoresh.replace(letter,final_letters_dict[letter])
    return shoresh


# stav's addition here

def shoresh_mishkal_combine_with_spacials(shoresh, mishkal, return_shoresh_letter_places=False):
    """
    combine shoresh and mishkal without excluding special gzarot and 4 letter shorashim
    the resulting form is not necessarily accurate, but will be used later to create
    the correct version of the word according to gzarot rules
    :param shoresh: shoresh letters for word
    :param mishkal: mishkal of word
    :return: mishkal where the ktl letters are replaced with shoresh letters
    """
    shoresh = final_letter_to_regular_letter(clean_word_from_nikud(shoresh))
    word = mishkal
    shoresh_places = [0]*len(word) # holds shoresh letter places. added.
    
    if len(shoresh) > 4:  # we cannot handle 5 letters shorashim right now
        if DEBUG:
            print(shoresh + " : can't handle 5 letters shoresh yet")
        word = ""

    elif mishkal == 'קַלְקַל':  # special treatment of this mishkal
        idx_k = [m.start() for m in re.finditer('ק', word)]
        idx_l = [m.start() for m in re.finditer('ל', word)]
        for idx in idx_k:
            word = word[:idx] + shoresh[0] + word[idx+1:]
            shoresh_places[idx] = 1 # added.
        for idx in idx_l:
            word = word[:idx] + shoresh[1] + word[idx + 1:]
            shoresh_places[idx] = 2 # added.

    elif len(shoresh) == 4:  # 4 letter shorashim
        idx_k = [m.start() for m in re.finditer('ק', word)]
        idx_l = [m.start() for m in re.finditer('ל', word)]
        idx_t = [m.start() for m in re.finditer('ט', word)]

        for idx in idx_k:
            word = word[:idx] + shoresh[0] + word[idx+1:]
            shoresh_places[idx] = 1 # added
        for idx in idx_l:
            word = word[:idx] + shoresh[3] + word[idx+1:]
            shoresh_places[idx] = 4 # added
        # replace tet with shoresh[1] + shva + shoresh[2]
        for idx in idx_t:
            word = word[:idx] + shoresh[1] + chr(1456) + shoresh[2] + word[idx+1:]
            shoresh_places[idx] = 2 # added
            shoresh_places[idx+2] = 3 # added

    else:  # the regular case of 3 letter shorashim (including gzarot)
        idx_k = [m.start() for m in re.finditer('ק', word)]
        idx_t = [m.start() for m in re.finditer('ט', word)]
        idx_l = [m.start() for m in re.finditer('ל', word)]

        for idx in idx_k:
            word = word[:idx] + shoresh[0] + word[idx+1:]
            shoresh_places[idx] = 1 # added
        for idx in idx_t:
            word = word[:idx] + shoresh[1] + word[idx+1:]
            shoresh_places[idx] = 2 # added
        for idx in idx_l:
            word = word[:idx] + shoresh[2] + word[idx+1:]
            shoresh_places[idx] = 3 # added

    word = change_to_final_letter(word)
    if (return_shoresh_letter_places):
        return word, shoresh_places
    return word

# print(shoresh_mishkal_combine_with_spacials('עגב', 'קְטָלִים'))