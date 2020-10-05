import json
import os
from ufal.udpipe import Model #, Pipeline, ProcessingError
from other.seq2seq_shoresh_mishkal_concat import ShoreshMishkalCombineFixer
from combining_shoresh_mishkal_learning.rnn.rnn_for_combine_padded_batches import \
    EncoderDecoder, EncoderRNN, AttnDecoderRNN  # should be imported for it to work
from other.word_generator import get_important_words_in_def,\
    generate_word_shoresh_based, generate_words_combination,\
    generate_word_basis_combine

dirname = os.path.dirname(os.path.abspath(__file__))


class Eliezer:
    """ Automatic Hebrew Neologism """

    def __init__(self):
        
        # load dictionary for important words in every definition based on tf-idf:
        with open(dirname + '\\data\\important words\\definitions_merged_all_12_important_without_urban2.json','r') as f:
            self.important_words_in_def = json.load(f)

        # load  translator based on wiktionary + prolog + wordnet (only with nikud)
        with open(dirname + '\\data\\translation\\mega_translation_dict_eng_heb_no_wikipedia_nikud_only_cleaned.json', 'r') as f:
            self.all_translator_eng_heb_nikud_only = json.load(f)  # translator based on wiktionary + wordnet (only with nikud)

        # load words that sound like words in hebrew (with nikud):
        with open(dirname + "\\data\\translation\\mega_translation_dict_heb_eng_no_wikipedia_cleaned_sound_like.json", 'r', encoding='utf8') as f:
            self.sound_like_heb_word_dict = json.load(f)

        # load shorashim dictionary
        with open(dirname + "\\data\\shorashim\\united_shorashim.json", 'r') as f:
            self.united_shorashim_dict = json.load(f)  # based on wiktionary + even-shoshan
        
        # load heb_word -> shoresh, mishkal dictionary:
        with open(dirname + "\\data\\shorashim\\united_shoresh_mishkal.json") as f:
            self.united_shoresh_mishkal_dict = json.load(f)
           
        # load hebrew synonyms dictionary:
        with open(dirname + "\\data\\synonyms\\all_synonyms.json", 'r') as f:
            self.synonyms = json.load(f)
    
        # load english word to hebrew word mishkalim dictionary:
        with open(dirname + "\\data\\mishkalim\\eng_word_mishkal.json", 'r') as f:
            self.eng_word_to_mishkalim = json.load(f)
    
        # load all mishkalim list:
        with open(dirname + "\\data\\mishkalim\\mishkalim.json", 'r', encoding='utf8') as f:
            self.all_mishkalim_dict = json.load(f)
            
        # load udpipe hebrew model:
        self.udpipe_model = Model.load(dirname + '\\udpipe-ud-2.3-181115\\udpipe-ud-2.3-181115\\hebrew-htb-ud-2.3-181115.udpipe')
        
        # load seq2seq (shoresh mishkal combine) concat model:
        path_of_model = 'combining_shoresh_mishkal_learning/rnn/model_bi_concat_winning_train_and_val.pt'
        directory_of_jsons_letters = 'combining_shoresh_mishkal_learning/rnn/'
        self.seq2seq = ShoreshMishkalCombineFixer(path_of_model, directory_of_jsons_letters)
        
        # language model dictionaries:
        with open(dirname + "\\data\\language model dictionaries\\with_nikud_data_cleaned5_new_nbn_unique_train_val_3gram.json", 'r', encoding='utf8') as f:
            word_counts_3grams = json.load(f)
    
        with open(dirname + "\\data\\language model dictionaries\\with_nikud_data_cleaned5_new_nbn_unique_train_val_4gram.json", 'r', encoding='utf8') as f:
            word_counts_4grams = json.load(f)
    
        self.language_model_4gram = [word_counts_3grams, word_counts_4grams]

    def suggest(self, word):
        """
        generate Hebrew word suggestions for a foreign word
        :param word: foreign word
        :return: list of 3 sub-lists: root-pattern suggestions, compound suggestions and portmanteaus suggestions
        """
        return [self.suggest_shoresh_based(word)] + self.suggest_compounds_and_portmanteaus(word)

    def suggest_shoresh_based(self, word):
        """
        get suggestions based on root-pattern combination
        :param word: foreign word
        :return: list of suggestions
        """
        important_words, important_words_score = get_important_words_in_def(self.important_words_in_def, word, to_print=False)
        word_suggestions = generate_word_shoresh_based(important_words, self.all_translator_eng_heb_nikud_only, self.sound_like_heb_word_dict, self.united_shorashim_dict, self.united_shoresh_mishkal_dict, word, self.eng_word_to_mishkalim, self.seq2seq, all_mishkalim_dict=self.all_mishkalim_dict, seq2seq_model=self.seq2seq, language_model=self.language_model_4gram, to_print=False) # TODO: deleted (3.8) - all_mishkalim_list=all_mishkalim_lst,
        return word_suggestions

    def suggest_compounds_and_portmanteaus(self, word):
        """
        get suggestions based on compounds and portmanteaus
        :param word: foreign word
        :return: list 2 sub-lists: compound suggestions, portmanteaus suggestions
        """
        # get the 'important' words in the definition of the given word:
        important_words, important_words_score = get_important_words_in_def(self.important_words_in_def, word, to_print=False)
        compounds = generate_words_combination(self.udpipe_model, important_words_score, self.all_translator_eng_heb_nikud_only, self.united_shorashim_dict, self.synonyms, word, to_print=False)
        portmanteaus = generate_word_basis_combine(compounds, to_print=False)
        return [compounds, portmanteaus]


if __name__ == '__main__':
    e = Eliezer()
    print(e.suggest('palette'))
