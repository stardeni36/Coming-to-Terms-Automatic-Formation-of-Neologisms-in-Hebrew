
import operator
import re
from other.naive_shoresh_mishkal_combine import get_nikud_options,\
    clean_word_from_nikud

DEBUG = False


def clean_mishni(line):
    sentence = re.sub(r'\{\{(.*?)\}\}', '', line) 
    sentence = re.sub(r'\((.*?)\)', '', sentence) # TODO: make sure this line does not make troubles
    sentence = sentence.replace("'",'')
    return sentence.strip()


def translate_words_eng_heb(translator, words):
    
    translations = {}
    trans_errors = 0 # TODO: added
    
    for i,w in enumerate(words):
#         print(w)
        try:
            trans = translator[w]
#             print(trans)
            if len(trans) == 0:
                trans_errors += 1 # TODO: added
            for t in trans:
                nikud_exist = False
                for nikud_opt in get_nikud_options(t):
                    if nikud_opt in translations:
                        nikud_exist = True
                if not nikud_exist:
                    translations[t] = i
                    
#             translations += trans
            if (DEBUG):
                print (w + " --- " + str(trans))
        except:
            trans_errors += 1 # TODO: added
            if (DEBUG):
                print (w + " has no translation!")
    
    translations = sorted(translations.items(), key=operator.itemgetter(1), reverse=False)
    translations = [item[0] for item in translations]
    translations = [clean_mishni(t) for t in translations]
#     translations = list(set([clean_mishni(t) for t in translations]))
    
#     return translations
    return translations, trans_errors


def get_shorashim(shorashim_dict, translations):
    
    word_shorashim = {}
    shoresh_error = 0 # TODO: added
    i = 1 
    
    for t in translations:
        for nikud_opt in get_nikud_options(t):
            if nikud_opt in shorashim_dict:  # TODO: was: "try"
                shoresh = clean_word_from_nikud(shorashim_dict[nikud_opt])
                shoresh = shoresh.split()[0]
                if len(shoresh) == 0: # TODO: added
                    shoresh_error += 1     
                if shoresh not in word_shorashim:
                    word_shorashim[shoresh] = i
                    i += 1
            else:  # TODO: was "except"
                shoresh_error += 1 # TODO: added
    
    if (DEBUG):
        print (list(set(word_shorashim)))
#     return list(set(word_shorashim))
    word_shorashim = [item[0] for item in sorted(word_shorashim.items(), key=operator.itemgetter(1))]
    
    return word_shorashim, shoresh_error


def get_translation_shoresh_pairs(shorashim_dict, translations):
    
    ts_pairs = []
    
    for t in translations:
        for nikud_opt in get_nikud_options(t):
            if nikud_opt in shorashim_dict:
                ts_pairs += [(t, shorashim_dict[nikud_opt])]
    
    return ts_pairs
                
            