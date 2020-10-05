import torch
import json
from combining_shoresh_mishkal_learning.rnn.rnn_for_combine_padded_batches import \
    EncoderDecoder, EncoderRNN, AttnDecoderRNN  # should be imported for it to work


class ShoreshMishkalCombineFixer:
    """
    fixing the naive shoresh mishkal combine function
    using a trained seq2seq encoder decoder model (details in training code)
    """
    def __init__(self, model_path, dicts_directory):
        """
        :param model_path: path to trained model file, saved with torch.save
        :param dicts_directory: directory with cat_to_letter and letter_to_cat json files
        """
        # args
        self.max_length = 31  # 20 + 11 gzarot
        self.device = torch.device("cpu")  # TODO: allow gpu

        # load model
        self.model = torch.load(model_path)

        # load dictionaries of letters to numbers and numbers to letters
        with open(dicts_directory + 'cat_to_letter_rnn2.json', 'r') as f1:
            cat_to_letter = json.load(f1)
        self.cat_to_letter = {int(i): j for i, j in cat_to_letter.items()}
        with open(dicts_directory + 'letter_to_cat_rnn2.json', 'r') as f1:
            letter_to_cat = json.load(f1)
        self.letter_to_cat = letter_to_cat

    def word_2_input_tensor(self, word, gzarot):
        """
        convert word to input tensor
        :param word: input word
        :param gzarot: the word's gzarot based on their shoresh
        :return: input tensor for the model
        """
        # convert letters to numbers using the letter_to_cat dictionary
        # add start of sequence token (SOS) in the beginning
        char_inds = [self.letter_to_cat["SOS"]] + [self.letter_to_cat[i] for i in word if i in self.letter_to_cat] # else 27 is kamatz
        char_inds.append(self.letter_to_cat["EOS"])  # add end of sequence token (EOS) in the end
        gz_vectors = [''] * self.max_length

        # if there are gzarot - add their corresponding numbers
        # to the beginning of the tensor (before SOS token)
        if gzarot:
            gz = gzarot.split(' 2 ')
            gz_vectors = []
            for item in gz:
                vector = item.split()
                vector = torch.tensor([int(x) for x in vector], dtype=torch.float)
                gz_vectors.append(vector)
            for i in range(self.max_length - len(gz_vectors)):
                gz_vectors.append(torch.tensor([self.letter_to_cat["PAD"]] * 12, dtype=torch.float))

        # add padding until maximal sequence length
        char_inds = char_inds + ([self.letter_to_cat["PAD"]] * (self.max_length - len(char_inds)))
        # convert to tensor
        tensor_output = torch.tensor(char_inds, dtype=torch.long, device=self.device).view(-1, 1)
        return tensor_output, gz_vectors

    def clean_final_word(self, word):
        # the word returned should not have the start and end of sequence tokens
        word = word.replace('SOS', '').replace('EOS', '')
        return word

    def fix_combine(self, word, gzarot):
        """
        use model to fix the combined result
        :param word: word to fix (input)
        :param gzarot: gzarot of word
        :return: decoded word (input fixed)
        """
        # transform word to model input
        input_word, gz_vectors = self.word_2_input_tensor(word, gzarot)

        self.model.eval()  # set model to evaluation
        decoded_word = ''  # output word
        with torch.no_grad():
            changed_dim = input_word.view(1, self.max_length, 1)
            # get model output
            outputs = self.model(changed_dim, None, 'eval', gz_vectors)  # no target (None)
            for di in outputs:
                topv, topi = di.data.topk(1)
                decoded_word += self.cat_to_letter[topi.item()]
                if self.cat_to_letter[topi.item()] == 'EOS':  # stop if end of sequence token is reached
                    decoded_word = self.clean_final_word(decoded_word)
                    return decoded_word

        decoded_word = self.clean_final_word(decoded_word)
        return decoded_word

if __name__ == '__main__':
    # usage
    path_of_model = 'combining_shoresh_mishkal_learning/rnn/model_bi_concat_winning_train_and_val.pt'
    directory_of_jsons_letters = 'combining_shoresh_mishkal_learning/rnn/'

    seq2seq = ShoreshMishkalCombineFixer(path_of_model, directory_of_jsons_letters)
    
    word ='מְשֻנָּן'
    gzarot = '0 1 0 0 0 0 0 0 0 0 0 0 2 1 0 0 0 0 0 0 0 0 0 0 0 2 0 1 0 0 0 0 0 0 0 0 0 0 2 1 0 0 0 0 0 0 0 0 0 0 0 2 0 1 0 0 0 0 0 0 0 0 0 0 2 1 0 0 0 0 0 0 0 0 0 0 0 2 0 0 0 0 0 0 0 0 0 0 0 0 2 0 0 0 0 0 0 0 0 0 0 0 0'
    print(seq2seq.fix_combine(word, gzarot))

