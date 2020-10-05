from nltk.corpus import wordnet as wn
import json
import os
from _operator import itemgetter
import operator
from collections import Counter


from other.naive_shoresh_mishkal_combine import shoresh_mishkal_combine_with_spacials,\
    get_nikud_options
from other.gzarot_tagging import shoresh_mishkal_to_gzarot_vecs,\
    get_gzarot_vectors_str
from other.utils import translate_words_eng_heb, get_translation_shoresh_pairs


dirname = os.path.dirname(os.path.abspath(__file__))

DEBUG = False
CLOSEST_SISTER_TERMS_K = 40  # was 100

def get_hypernyms(synsets):
    """
    :param synsets: list of wordnet synsets
    :return: all the hypernyms derived from the synsets
    """
    all_hypernyms = []
    for syn in synsets:
        hypernyms = syn.hypernyms()
        all_hypernyms.extend(hypernyms)
    return all_hypernyms


def get_hyponyms(synsets):
    """
    :param synsets: a list of wordnet synsets
    :return: all the hyponyms derived from the synsets
    """
    all_hyponyms = []
    for syn in synsets:
        hypos = syn.hyponyms()
        all_hyponyms.extend(hypos)
    return all_hyponyms


def get_sister_terms(word, depth):
    """
    move up and then down the wordnet tree
    to get sister terms
    :param word: the word we want to get sister terms for
    :param depth: how many times we travel up/down the tree
    :return: sister terms (list)
    """
    original_synsets = wn.synsets(word, pos='n')  # saved to be used later
#     if (len(original_synsets) == 0):
#         return None
    # extract synsets and move up and down the tree
    synsets = wn.synsets(word, pos='n')
    for i in range(depth):
        synsets = get_hypernyms(synsets)
    for i in range(depth):
        synsets = get_hyponyms(synsets)
    # filter out the original synsets
    new_synsets = list(set(synsets).difference(set(original_synsets)))
    # get only the words themselves out of the synsets
    sister_terms = [i.name().split('.')[0] for i in new_synsets]
    return sister_terms


# Addition to control the minimal number of sisters
def get_k_closest_sister_terms(word, k):
    """
    get at least k sister terms of word
    :param word: the word we want to get sister terms for
    :param k: minimal number of sister terms
    :return: sisters : sister terms (len>k)
    """
    depth = 1
    sisters = {}
    while len(sisters.keys()) < k:
        cur_sister_term = get_sister_terms(word, depth)
#         if cur_sister_term == None:
#             return sisters
        for term in cur_sister_term:
            if term not in sisters:
                sisters[term] = depth
        depth += 1
    
    sisters = sorted(sisters.items(), key=operator.itemgetter(1), reverse=False)
    sisters = [item[0] for item in sisters][:k]
    return sisters


def create_eng_word_to_mishkal_dictionary():
    
    # load eng-heb wiktionary translator: 
    with open(dirname + '\\translations\\jsons\\hewiktionary_eng_heb_dict_cleaned.json', 'r') as f:
        translator_eng_heb = json.load(f)
    
    print("total number of english translated words in hewiktionary: " + str(len(translator_eng_heb)))
    
    # load heb word -> mishkal dictionary:
    with open(dirname + "\\mishkalim\\jsons\\word_mishkal.json", 'r') as f:
        heb_word_mishakl = json.load(f)
    
    print("total number of hebrew words with mishkal in hewiktionary: " + str(len(heb_word_mishakl)))
    
    eng_word_to_mishkal = {}
      
    for eng_word in translator_eng_heb:
        translations = translator_eng_heb[eng_word]
        mishkalim = []
        good_translations = []
        for translation in translations:
            if translation in heb_word_mishakl:
                mishkalim += [heb_word_mishakl[translation]]
                good_translations += [translation] # "good translation" = a hebrew translation that has a mishkal in hewiktionary 
        if len(mishkalim) > 0:
            eng_word_to_mishkal[eng_word] = {}
            eng_word_to_mishkal[eng_word]['translations'] = good_translations
            eng_word_to_mishkal[eng_word]['mishkalim'] = mishkalim
    
    with open(dirname + "\\mishkalim\\jsons\\eng_word_mishkal.json", 'w', encoding='utf8') as outfile:
        json.dump(eng_word_to_mishkal, outfile)
    
    print("total number of english words that has mishkal in hebrew: " + str(len(eng_word_to_mishkal)))
    

def get_mishkal_old(word, shoresh, all_mishkalim_list, seq2seq_model):
    
    for mishkal in all_mishkalim_list:

        try: 
            combined = shoresh_mishkal_combine_with_spacials(shoresh, mishkal)
            gzarot_vectors = shoresh_mishkal_to_gzarot_vecs(shoresh, mishkal)
            gzarot_vec_str = get_gzarot_vectors_str(gzarot_vectors)
            new_word = seq2seq_model.fix_combine(combined, gzarot_vec_str)  # TODO: undo comment.
            if new_word in get_nikud_options(word):
                return mishkal
        except:
            print("EXCEPTION in get_mishkal")

    return ""


def get_mishkal(word, shoresh, mishkalim_dict):
    
    for nikud_option in get_nikud_options(word):
        if nikud_option in mishkalim_dict:
            return mishkalim_dict[nikud_option]["mishkal"]
    
    return ""
 
    
# this method gets a list of terms and returns a list of mishkalim (for the terms that have "mishkal"s).
# eng_word_mishkal: a dictionary. key: english word, value: list of mishkalim for this word. 
def get_terms_mishkalim(eng_word_mishkal, terms, translator=None, shorashim_dict=None, mishkalim_dict=None, to_print=False): #all_mishkalim_list=None, seq2seq_model=None):
    
    mishkalim = [eng_word_mishkal[term]['mishkalim'] for term in terms if term in eng_word_mishkal]
    mishkalim = [item for sublist in mishkalim for item in sublist] # faltten list

#     for term in terms:
#         if term in eng_word_mishkal:
#             print(term)
#             print(eng_word_mishkal[term]['translations'])

    translations = [eng_word_mishkal[term]['translations'] for term in terms if term in eng_word_mishkal]
    translations = [item for sublist in translations for item in sublist] # flatten list
    
#     if (to_print):
#     print("sister terms translations: ", translations)
    
    problematic_terms_mishkalim = []
    
    if mishkalim_dict:
        problematic_terms = [term for term in terms if term not in eng_word_mishkal] # terms without mishkal in our eng_word_mishkal dictionary
        problematic_terms_translations, trans_errors = translate_words_eng_heb(translator, problematic_terms)
        problematic_terms_pairs = list(set(get_translation_shoresh_pairs(shorashim_dict, problematic_terms_translations))) # returns list of pairs: [(word, shoresh),...]
#         print(len(problematic_terms_translations))
#         print(len(problematic_terms_pairs))
        
        problematic_terms_mishkalim = [(item[0], get_mishkal(item[0], item[1], mishkalim_dict)) for item in problematic_terms_pairs]
        
        if (to_print):
            print("lost shorashim: ", problematic_terms_pairs)
            print("lost mishkalim: ", problematic_terms_mishkalim)
            
        problematic_terms_mishkalim = [item[1] for item in problematic_terms_mishkalim if item[1]!=""]
    
    
    if DEBUG: 
        print (terms)
        print (translations)
        print (mishkalim)
    
    if (to_print):
        print("mishkalim pre: ", mishkalim) 
    mishkalim += problematic_terms_mishkalim
    if (to_print):
        print("mishkalim post: ", mishkalim)
    
    return mishkalim, translations
#     mishkalim_flat_list = [item for sublist in mishkalim for item in sublist]
#     return mishkalim_flat_list

# eng_word_to_mishkalim: dictionary (eng_word -> heb_mishkal). word: word to search mishkalim for.
def get_relevant_mishkalim_sister_terms(eng_word_to_mishkalim, word, translator=None, shorashim_dict=None, mishkalim_dict=None, all_mishkalim_dict=None, to_print=False): #all_mishkalim_list=None, seq2seq_model=None):
    
    word_synsets = wn.synsets(word, pos='n')
    if(len(word_synsets) == 0):
        if(to_print):
            print("no synsets")
        if(len(word.split()) > 1):
            word = '_'.join(word.split())
            word_synsets = wn.synsets(word, pos='n')
             
    sister_terms = []
    mishkalim = []
     
    if (len(word_synsets) > 0):
        sister_terms = get_k_closest_sister_terms(word, CLOSEST_SISTER_TERMS_K)
        mishkalim = get_terms_mishkalim(eng_word_to_mishkalim, sister_terms, translator, shorashim_dict, mishkalim_dict, to_print)[0] #all_mishkalim_list, seq2seq_model)
        mishkalim = [(mishkal, len(all_mishkalim_dict[mishkal])) for mishkal in mishkalim]
        mishkalim = [item[0] for item in sorted(mishkalim, key=operator.itemgetter(1), reverse=True)]  # TODO: reverse = True
        mishkalim = [item[0] for item in sorted(Counter(mishkalim).items(), key=operator.itemgetter(1), reverse=True)] #  removed: if item[1]>1 (save mishkalim that appear at least once)
        if (to_print):
            print("sister terms: ", sister_terms) 
            
    return sister_terms, mishkalim

          
if __name__ == '__main__':
    
#     create_eng_word_to_mishkal_dictionary()
    
    with open(dirname + "\\mishkalim\\jsons\\eng_word_mishkal.json", 'r') as f:
        eng_word_to_mishkalim = json.load(f)
    
    with open(dirname + "\\translations\\jsons\\mega_translation_dict_eng_heb_no_wikipedia_cleaned.json", 'r') as f:
        eng_heb_trans = json.load(f)
        
#     for word in eng_word_mishkal.items():
#         print(word)
    
    print("allergy")
    word = 'allergy'
    sister_terms = get_sister_terms('allergy', depth=1)
    print(sister_terms)
    sister_terms = get_sister_terms('allergy', depth=2)
    print(sister_terms)
    sister_terms = get_sister_terms('allergy', depth=3)
    print(sister_terms)
    sister_terms = get_sister_terms('allergy', depth=4)
    print(sister_terms)
    sister_terms = get_sister_terms('allergy', depth=5)
    print(sister_terms)
    
    print("")
    k_closest_sister_terms = get_k_closest_sister_terms(word, 10)
    print("final:")
    print(k_closest_sister_terms)
    
#     print("flu")
#     sister_terms = get_sister_terms('flu', depth=2) 
#     mishkalim, translations = get_terms_mishkalim(eng_word_to_mishkalim, sister_terms)
#     print(len(translations))
#     have_trans_general = [term for term in sister_terms if term in eng_heb_trans]
#     trans_general = [(term, eng_heb_trans[term]) for term in sister_terms if term in eng_heb_trans]
#     print(have_trans_general)
#     print(len(have_trans_general))
#     print(trans_general)
#     print()
#      
#     print("carpenter")
#     sister_terms = get_sister_terms('carpenter', depth=2) 
#     mishkalim, translations = get_terms_mishkalim(eng_word_to_mishkalim, sister_terms)
#     print(len(translations))
#     have_trans_general = [term for term in sister_terms if term in eng_heb_trans]
#     trans_general = [(term, eng_heb_trans[term]) for term in sister_terms if term in eng_heb_trans]
#     print(have_trans_general)
#     print(trans_general)
#     print()
#      
#     print("allergy")
#     sister_terms = get_sister_terms('allergy', depth=2) 
#     mishkalim, translations = get_terms_mishkalim(eng_word_to_mishkalim, sister_terms)
#     print(len(translations))
#     have_trans_general = [term for term in sister_terms if term in eng_heb_trans]
#     trans_general = [(term, eng_heb_trans[term]) for term in sister_terms if term in eng_heb_trans]
#     print(have_trans_general)
#     print(trans_general)
#     print()

    # print examples
#     print('FLU:')
#     [print(i) for i in get_sister_terms('flu', depth=1)]
#     print('')
#     print('CARPENTER:')
#     [print(i) for i in get_sister_terms('carpenter', depth=2)]
#     print('')
#     print('ALLERGY:')
#     [print(i) for i in get_sister_terms('allergy', depth=5)]
#     print('')
