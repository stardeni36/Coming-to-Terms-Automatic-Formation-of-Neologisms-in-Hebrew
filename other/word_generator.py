from gensim.models.keyedvectors import KeyedVectors
import os
import pandas
import re
import json
from tqdm import tqdm
import operator
from ufal.udpipe import Model 
from other.run_udpipe import is_optional_suggestion  
from other.naive_shoresh_mishkal_combine import clean_word_from_nikud, shoresh_mishkal_combine_with_spacials 
from other.seq2seq_shoresh_mishkal_concat import ShoreshMishkalCombineFixer
from combining_shoresh_mishkal_learning.rnn.rnn_for_combine_padded_batches import \
    EncoderDecoder, EncoderRNN, AttnDecoderRNN  # should be imported for it to work
from other.gzarot_tagging import shoresh_mishkal_to_gzarot_vecs,\
    get_gzarot_vectors_str
from other.utils import get_shorashim, translate_words_eng_heb
from other.wordnet_sister_terms import get_relevant_mishkalim_sister_terms
from other.edit_distance import string_preprocessing
import random  
from language_model.language_model import language_model_prob

sound_like_heb = 0
doesnt_sound_like_heb = 0

DEBUG = False
CHECK_SMIXUT = True
IMPORTANT_WORDS_NUM = 10
NUM_OF_MISHKALIM_PER_CATEGORY = 20
MAX_WORD_LEN = 15
MAX_FORCE_COMBINE_LEN = 10
MIN_FREQ = 0.05
MAX_NUM_OF_MISHKALIM = 4
CUTOFF = 0.10

TFIDF_ABLATION = False
SEQ2SEQ_ABLATION = False
LM_ABLATION = False

OPPOSITE_COMBINATION = True
COMBINATIONS_WORD_LIMIT = 20
MAX_NUM_OF_COMBINATIONS = 50

dirname = os.path.dirname(os.path.abspath(__file__))
ahvi_letters = "אהוי"
final_letters = list('םןךףץ') 
regular_letters = list('מנכפצ')  

    
def get_all_new_academy_words(filename):
    path = dirname + "\\" + filename
    df = pandas.read_csv(path, encoding='utf-8')
    word_list = (list(set(df['אנגלית'])))
    word_list = [(re.sub(r'\[(.*?)\]', "", word)).lower() for word in word_list if isinstance(word,str)]
    word_list = [word.replace(';',',') for word in word_list]
    new_word_list = []
    for word in word_list:
        new_word_list += [word.strip() for word in word.split(',')]
    return new_word_list
      
    
def final_letter_to_regular_letter(shoresh):
    final_letters_dict = {f:r for f,r in zip(final_letters,regular_letters)}
    for letter in shoresh:
        if letter in final_letters:
            shoresh = shoresh.replace(letter,final_letters_dict[letter])
    return shoresh


def translate_eng_heb_with_shoresh_and_remove_nikud(translator, shorashim_dict, words_score):
    translations = []
    for i in range(len(words_score)):
        try:
            trans = translator[words_score[i][0]]
            trans = [t for t in trans if t in shorashim_dict] # we assume that non-hebrew words do not have shoresh.
            trans = list(set([clean_word_from_nikud(t) for t in trans if len(t.split())==1]))
            if len(trans) > 0:
                translations += [(trans, words_score[i][1])]
            if (DEBUG):
                print (words_score[i][0] + " --- " + str(trans))
        except:
            if (DEBUG):
                print (words_score[i][0] + " has no translation!")
        
    return translations


def get_important_words_in_def(important_words_in_def, word, to_print=False, file=None):
 
    # get the 'important' words in the definition of word. 
    important_words_score = []
    important_words = []
     
    if word in important_words_in_def:
        important_words_score = important_words_in_def[word]
        for item in important_words_in_def[word][:IMPORTANT_WORDS_NUM]:
            important_words += [item[0]]
     
    if (to_print):
        print ("important word tf-idf score: " + str(important_words_score)) 
        print("important words in def: " + str(important_words))
    if (file):
        file.write("important words in def: " + str(important_words) + "\n")
        file.write("important word tf-idf score: " + str(important_words_score) + "\n")  
     
    return important_words, important_words_score


# rank the suggestions first by their shoresh. For every shoresh: pick the most probable word in the shoresh+mishkal suggestions (based on the lm model) 
# that do not sound like a word that is already in hebrew.
# for every shoresh return at most one suggestion.  
def rank_suggestions_shoresh_lm(word_suggestions, sound_like_heb_dict, language_model):
     
    final_suggestions = []
     
    for sublist in word_suggestions:
        cur_suggestions = []
        if LM_ABLATION:
            final_suggestions += random.sample(sublist, 1)
        else:
            for suggestion in sublist:
                if not (string_preprocessing(suggestion) in sound_like_heb_dict):
                    global doesnt_sound_like_heb
                    doesnt_sound_like_heb += 1
                    prob = round(language_model_prob(language_model, suggestion), 5)
                    cur_suggestions += [(suggestion, prob)]
                else:  # TODO: remove
#                     print(suggestion, " sounds like hebrew")
                    global sound_like_heb
                    sound_like_heb += 1
             
            cur_suggestions = [item for item in cur_suggestions if item[1]>CUTOFF]
             
            if cur_suggestions:
                sugg_sorted = sorted(cur_suggestions, key=operator.itemgetter(1), reverse=True)
                final_suggestions += [sugg_sorted[0][0]]
                if len(cur_suggestions) > 1:
                    final_suggestions += [sugg_sorted[1][0]]
                 
     
    return final_suggestions
        

def generate_word_shoresh_based(important_words, translator, sound_like_heb_dict, shorashim_dict, mishkalim_dict, word, eng_word_to_mishkalim, seq2seq, to_print=False, file=None, all_mishkalim_dict=None, all_mishkalim_list=None, seq2seq_model=None, language_model=None):
    
    word_suggestions = [] 
    
    important_words_score = []
    
    if (to_print):
        print("important words in def: " + str(important_words))
    if (file):
        file.write("important words in def: " + str(important_words) + "\n")
         
    # translate these 'important' words to hebrew (save only translation with nikud) 
    translations = translate_words_eng_heb(translator, important_words)[0]
    
    # get the important words shorashim:
    shorashim = get_shorashim(shorashim_dict, translations)[0]
    
    if (to_print):
        print ("translations: " + str(translations))
        print ("shorashim: " + str(shorashim))
    if (file):
        file.write("translations: " + str(translations) + "\n")
        file.write("shorashim: " + str(shorashim) + "\n")
        
    # get relevant mishkalim based on "sister terms":
    most_relevant_mishkalim_sister_terms = get_relevant_mishkalim_sister_terms(eng_word_to_mishkalim, word, translator, shorashim_dict, mishkalim_dict, all_mishkalim_dict, to_print)[1] #all_mishkalim_list, seq2seq)

    if (to_print):
        print("most relevant mishkalim: ", most_relevant_mishkalim_sister_terms)

    most_relevant_mishkalim_sister_terms = most_relevant_mishkalim_sister_terms[:min(MAX_NUM_OF_MISHKALIM, len(most_relevant_mishkalim_sister_terms))]
    
    # combine shorashim and mishkalim together to a list of new words:
    for s in shorashim:
        cur_suggestions = []
        for m in most_relevant_mishkalim_sister_terms:
            new_word = "" 
            if (seq2seq_model):
                combine_res = shoresh_mishkal_combine_with_spacials(s, m)
                if (combine_res):
                    gzarot_vectors = shoresh_mishkal_to_gzarot_vecs(s, m)
                    gzarot_vec_str = get_gzarot_vectors_str(gzarot_vectors)
                    new_word = seq2seq.fix_combine(combine_res, gzarot_vec_str)
            else:
                new_word = shoresh_mishkal_combine_with_spacials(s, m).strip()
            if (len(new_word) > 0):
                cur_suggestions += [new_word]
        if (len(cur_suggestions) > 0):
            word_suggestions += [cur_suggestions] 
    
    if (to_print):
        print ("shoresh + mishkal suggestions: ")
        [print(word_suggestion) for word_suggestion in word_suggestions]
        
    ranked_suggestions_shoresh_lm = rank_suggestions_shoresh_lm(word_suggestions, sound_like_heb_dict, language_model)
    
    if (to_print):
        print("ranked suggestions shoresh lm: ")
        print(ranked_suggestions_shoresh_lm)     

    word_suggestions = ranked_suggestions_shoresh_lm  
    return word_suggestions 


def get_group_synonyms(word_group, synonyms):
    syn_group = []
    for w in word_group:
        if w in synonyms:
            syn_group += [s for s in synonyms[w] if len(s.split())==1]
    return list(set(word_group + syn_group))  


def get_all_legit_combinations(udpipe_model, group_a, group_b, check_smixut):
      
    suggestion_score = group_a[1] + group_b[1]
      
    bad_suggestions = [] 
    legit_suggestions = []
    for a in group_a[0]:
        for b in group_b[0]:
            suggestion = a + " " + b        
            if (is_optional_suggestion(udpipe_model, suggestion, check_smixut)):
                legit_suggestions += [(suggestion, suggestion_score)]
            else: 
                bad_suggestions += [(suggestion, suggestion_score)] 
      
    return legit_suggestions, len(bad_suggestions)


def get_all_optional_combinations(udpipe_model, syn_groups, check_smixut, to_print, file=None):
    optional_suggestions = []
    num_of_bad_suggestions = 0
    num_of_groups = len(syn_groups)
    for i in range(num_of_groups):
        for j in range(num_of_groups):
            if j!=i:
                good_suggestions, num_of_bad = get_all_legit_combinations(udpipe_model, syn_groups[i], syn_groups[j],check_smixut)
                optional_suggestions += good_suggestions
                num_of_bad_suggestions += num_of_bad
     
    if (to_print):
        print ('num of suggestios before filtering: ' + str(len(optional_suggestions) + num_of_bad))
        print ('after filtering: ' + str(len(optional_suggestions)))
    if (file):
        file.write('num of suggestios before filtering: ' + str(len(optional_suggestions) + num_of_bad) 
                   + ", after filtering: " + str(len(optional_suggestions)) + "\n")
         
    return optional_suggestions


def generate_words_combination(udpipe_model, important_words_score, translator_eng_heb, shorashim_dict, synonyms, word, to_print=False, file=None):
     
    if (to_print):
        print ("--- generate word combination suggestions: ---")
        print ("word: " + word)
     
    # translate these 'important' words to hebrew, remove nikud and remove translations with more than one word:
    translations = translate_eng_heb_with_shoresh_and_remove_nikud(translator_eng_heb, shorashim_dict, important_words_score) 
 
    translations = translations if len(translations)<COMBINATIONS_WORD_LIMIT else translations[:COMBINATIONS_WORD_LIMIT]
         
    syn_groups = translations
    syn_groups = [(get_group_synonyms(l[0], synonyms), l[1]) for l in translations]  # TODO: synonyms
 
    if (to_print):
        print ("important words in definition: " + str(important_words_score))
        print ("translations: " + str(translations)) # TODO: remove
    if (file):
        file.write("translations: " + str(translations) + "\n")
     
    legit_combinations = get_all_optional_combinations(udpipe_model, syn_groups, check_smixut=CHECK_SMIXUT, to_print=to_print, file=file)
    legit_combinations = sorted(legit_combinations, key=operator.itemgetter(1), reverse=True)
    legit_combinations_no_score = [item[0] for item in legit_combinations]
    legit_no_dup = []
    for item in legit_combinations_no_score:
        if item not in legit_no_dup:
            if (OPPOSITE_COMBINATION):
                legit_no_dup += [item]
            else:
                if (item.split()[1] + " " +item.split()[0]) not in legit_no_dup:
                    legit_no_dup += [item]
             
    if (to_print):
        print ("suggestions with scores: " + str(legit_combinations))
        print ("suggestions (filter no shoresh translations): " + str(legit_combinations_no_score))
        print ("")
     
    return legit_no_dup[:MAX_NUM_OF_COMBINATIONS] 


def delete_last_ahvi_letter(word):
    if (word[-1] in ahvi_letters):
        return word[:-1], True
    return word, False

def delete_first_ahvi_letter(word):
    if (word[0] in ahvi_letters):
        return word[1:], True
    return word, False

def is_overlap_almost_equal(part1, part2):
    if (part1 == part2):
        return True
    part1 = part1.replace('כ','ח')
    part1 = part1.replace('ט','ת')
    part2 = part2.replace('כ','ח')
    part2 = part2.replace('ט','ת')
    if (part1 == part2):
        return True
    return False

def combine_overlap_words(word1, word2):
    word1 = final_letter_to_regular_letter(word1)
    for i in range(3,0,-1): 
        if (is_overlap_almost_equal(word1[-i:],word2[:i])):
            res = word1[:-i]+word2
            if (len(res)<MAX_WORD_LEN+1):
                return res
    return None

def combine_words(word1, word2):
     
    # combine two words if the end of one word is identical to the start of the second
    combined = combine_overlap_words(word1, word2)
    if (combined!=None):
        return combined
     
    # try to combine two words after deleting ahvi letters from the end of the first and the end of the second
    word1, last_deleted = delete_last_ahvi_letter(word1)
    word2, first_deleted = delete_first_ahvi_letter(word2)
     
    if (last_deleted or first_deleted):
        combined = combine_overlap_words(word1, word2)
        if (combined!=None):
            return combined
         
        combined = final_letter_to_regular_letter(word1) + word2
        if (len(combined)<MAX_FORCE_COMBINE_LEN+1):
            return combined
         
    return None

# gets a combination of two words in hebrew and returns a combination of the words into one word if it is possible
def combine_parts(combination, to_print=False, file=None):
    spltd = combination.split()
    cp = combine_words(spltd[0].strip(), spltd[1].strip())
    if (to_print):
        print (spltd[0] + " + " + spltd[1] + " --> " + str(cp))
    if (file):
        file.write(spltd[0] + " + " + spltd[1] + " --> " + str(cp) + "\n")
    return cp


def generate_word_basis_combine(legit_combinations, to_print=False, file=None):
    if (to_print):
        print ("--- generate word basis combine (הלחם בסיסים) suggestions: ---")
        print ("word combinations suggestions: " + str(legit_combinations))
     
    basis_combined = [combine_parts(s,to_print=False,file=file) for s in legit_combinations if combine_parts(s)!=None]
     
    if (to_print):
        print ("suggestions: " + str(basis_combined))
        print ("")
     
    return basis_combined 
