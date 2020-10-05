import torch
import json
from torch import nn, optim
from torch.nn import functional as F
import random
import pandas as pd
import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch.utils.data as data
import numpy as np

# global vars
PAD_TOKEN = 0  # padding token
SOS_token = 1  # start of sequence token
EOS_token = 2  # end of sequence token


# dataset #


class Words_Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data_path, hp_dict):
        self.data = pd.read_csv(data_path)
        self.num_of_examples = len(self.data)
        self.source_words = self.data['combine_result']
        self.target_words = self.data['word']
        self.gzarot = self.data['gzarot']
        self.shoresh = self.data['shoresh']
        self.hp_dict = hp_dict
        # get index dictionaries
        with open('letter_to_cat_rnn2.json', 'r') as f:
            self.letter_to_cat = json.load(f)
        with open('cat_to_letter_rnn2.json', 'r') as f:
            self.cat_to_letter = json.load(f)

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        src_seq = self.source_words.iloc[index]
        trg_seq = self.target_words.iloc[index]
        # TODO: get gizra by index
        gzarot = self.gzarot.iloc[index]
        shorashim = self.shoresh.iloc[index]
        # src_seq = self.get_word_tensor(src_seq, gzarot, shorashim)  # TODO: use gzarot
        src_seq = self.get_word_tensor(src_seq, None, shorashim)  # TODO: use gzarot
        trg_seq = self.get_word_tensor(trg_seq)
        return src_seq, trg_seq

    def __len__(self):
        return self.num_of_examples

    def get_word_tensor(self, word, gzarot=None, shorashim=None):  # TODO: gzarot=None
        """
        use dictionary: letters -> integers to get tensor with integers
        :param word: input word to turn into tensor
        :return: tensor with integers corresponding to the word letters,
        with SOS, EOS and PAD tokens.
        """
        char_inds = [SOS_token] + [self.letter_to_cat[i] for i in word]
        char_inds.append(EOS_token)  # add end of sequence token
        # TODO: if gzarot then concat gzarot numbers at the beginning
        if shorashim:
            sh = shorashim.split()
            sh_tokens = [self.letter_to_cat[i] for i in sh]
            char_inds = sh_tokens + char_inds
        if gzarot:
            gz = gzarot.split()
            gz_tokens = [self.letter_to_cat[i] for i in gz]
            char_inds = gz_tokens + char_inds
        char_inds = char_inds + ([PAD_TOKEN] * (self.hp_dict['max_length'] - len(char_inds)))
        return torch.tensor(char_inds, dtype=torch.long, device=self.hp_dict['device']).view(-1, 1)


# network architecture #

class EncoderDecoder(nn.Module):
    """
    Encoder-Decoder architecture
    """

    def __init__(self, encoder, decoder, hp_dict):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.hp_dict = hp_dict
        self.input_output_size = self.hp_dict['input_output_size']
        self.max_length = self.hp_dict['max_length']

    def forward(self, input_tensor, target_tensor, phase):

        if phase == 'train':
            teacher_forcing_ratio = self.hp_dict['teacher_forcing']
        else:
            teacher_forcing_ratio = 0.0

        curr_batch_size = input_tensor.shape[0]

        encoder_hidden = self.encoder.initHidden(curr_batch_size)

        # addition
        input_tensor = input_tensor.transpose(0, 1)
        if target_tensor is None:
            pass
        else:
            target_tensor = target_tensor.transpose(0, 1)

        network_outputs = torch.zeros([self.max_length, curr_batch_size, self.input_output_size])

        encoder_outputs = torch.zeros(self.max_length, curr_batch_size, self.encoder.hidden_size,
                                     device=self.hp_dict['device'])

        # encoder - loop over input letters
        for ei in range(self.max_length):
            encoder_output, encoder_hidden = self.encoder(
                input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0]

        decoder_input = torch.tensor([SOS_token] * curr_batch_size, device=self.hp_dict['device'])

        decoder_hidden = encoder_hidden

        # teacher forcing
        use_teacher_forcing = False
        if teacher_forcing_ratio != 0:
            if random.random() < teacher_forcing_ratio:
                use_teacher_forcing = True

        # decoder - loop over output letters

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(self.max_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                # decoder_output, decoder_hidden = self.decoder(
                #     decoder_input, decoder_hidden)
                network_outputs[di] = decoder_output
                decoder_input = target_tensor[di]  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(self.max_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                # decoder_output, decoder_hidden = self.decoder(
                #     decoder_input, decoder_hidden)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(1).detach()  # detach from history as input
                network_outputs[di] = decoder_output
                if min(decoder_input == EOS_token) == 1:
                    break
        return network_outputs


class EncoderRNN(nn.Module):
    """
    encoder part of the network
    embbedding + gru
    """
    def __init__(self, hp_dict):
        super(EncoderRNN, self).__init__()
        self.device = hp_dict['device']
        self.input_size = hp_dict['input_output_size']
        self.hidden_size = hp_dict['hidden_size']

        self.embedding = nn.Embedding(self.input_size, self.hidden_size, padding_idx=PAD_TOKEN)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)

    def forward(self, input, hidden):
        curr_batch_size = input.shape[0]
        embedded = self.embedding(input).view(1, curr_batch_size, self.hidden_size)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=self.device)


class DecoderRNN(nn.Module):
    """
    decoder part of the network - option 1
    embedding + relu +  gru + linear + softmax
    """
    def __init__(self, hp_dict):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hp_dict['hidden_size']
        self.output_size = hp_dict['input_output_size']

        self.embedding = nn.Embedding(self.output_size, self.hidden_size, padding_idx=PAD_TOKEN)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        curr_batch_size = input.shape[0]
        output = self.embedding(input).view(1, curr_batch_size, self.hidden_size)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=self.hp_dict['device'])


class AttnDecoderRNN(nn.Module):
    """
    decoder part of the network - option 2
    embedding + dropout + attention + relu +  gru + linear + softmax
    """
    def __init__(self, hp_dict):
        super(AttnDecoderRNN, self).__init__()
        self.hp_dict = hp_dict
        self.hidden_size = self.hp_dict['hidden_size']
        self.output_size = self.hp_dict['input_output_size']
        self.dropout_p = self.hp_dict['dropout_p']
        self.max_length = self.hp_dict['max_length']
        self.batch_size = self.hp_dict['batch_size']

        self.embedding = nn.Embedding(self.output_size, self.hidden_size, padding_idx=PAD_TOKEN)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        curr_batch_size = input.shape[0]
        embedded = self.embedding(input).view(1, curr_batch_size, self.hidden_size)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1),
                                 encoder_outputs.transpose(0,1))

        output = torch.cat((embedded.squeeze(0), attn_applied.squeeze(1)), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=self.hp_dict['device'])


# training #

def train(input_tensor, target_tensor, model, optimizer, criterion):

    model.train()

    optimizer.zero_grad()

    target_length = target_tensor.size(1)

    outputs = model(input_tensor, target_tensor, 'train')

    loss = 0
    for di in range(target_length):
        loss += criterion(outputs[di], target_tensor.transpose(0,1)[di].squeeze(1))

    loss.backward()

    optimizer.step()

    return loss.item() / target_length


def eval_ex(input_tensor, target_tensor, model, criterion):
    outputs = model(input_tensor, target_tensor, 'eval')
    target_length = target_tensor.size(1)
    loss = 0
    for di in range(target_length):
        loss += criterion(outputs[di], target_tensor.transpose(0,1)[di].squeeze(1))
    return loss.item() / target_length


def training_epochs(model, train_loader, eval_loader, optimizer, criterion, num_epochs):
    # train and eval losses for plots
    train_losses_per_epoch = []
    eval_losses_per_epoch = []

    for epoch in range(num_epochs):
        loss = 0
        eval_loss = 0

        # train
        for batch_index, (input_tensor, target_tensor) in enumerate(train_loader):
            iter_loss = train(input_tensor, target_tensor, model, optimizer, criterion)
            loss += iter_loss
        train_losses_per_epoch.append(loss / len(train_loader))  # loss per epoch (avg)
        print(str(epoch) + ': training loss: ' + str(loss / len(train_loader)))

        # evaluation
        model.eval()
        for batch_index1, (input_tensor1, target_tensor1) in enumerate(eval_loader):
            iter_eval_loss = eval_ex(input_tensor1, target_tensor1, model, criterion)
            eval_loss += iter_eval_loss
        eval_losses_per_epoch.append(eval_loss / len(eval_loader))
        print('eval loss: ' + str(eval_loss / len(eval_loader)))

    # loss plots
    plt.plot(train_losses_per_epoch, label='train')
    plt.plot(eval_losses_per_epoch, label='eval')
    plt.legend()
    plt.show()

# def asMinutes(s):
#     m = math.floor(s / 60)
#     s -= m * 60
#     return '%dm %ds' % (m, s)
#
#
# def timeSince(since, percent):
#     now = time.time()
#     s = now - since
#     es = s / (percent)
#     rs = es - s
#     return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def showPlot(points, eval_points):
    # plt.figure()
    # fig, ax = plt.subplots()
    # # this locator puts ticks at regular intervals
    # loc = ticker.MultipleLocator(base=0.2)
    # ax.yaxis.set_major_locator(loc)
    plt.plot(points, label='train')
    plt.plot(eval_points, label='eval')
    plt.legend()
    plt.show()


def generate_word(model, input_word, cat_to_letter):
    """
    generate word given input_word
    :param model: trained model
    :param input_word: input
    :param cat_to_letter: dictionary {number: letter}
    :return: decoded word: generated word
    """
    model.eval()
    decoded_word = ''
    with torch.no_grad():
        changed_dim = input_word.view(1, 23, 1)  # 34
        outputs = model(changed_dim, None, 'eval')
        for di in outputs:
            topv, topi = di.data.topk(1)
            decoded_word += cat_to_letter[topi.item()]
            if cat_to_letter[topi.item()] == 'EOS':
                return decoded_word

    return decoded_word


def generate_words_for_eval(model, eval_dataset, results_filename):
    """
    use the model to generate words given all of the eval dataset words
    :param model: trained model
    :param eval_dataset: dataset object with evaluation data
    :param results_filename: path to write results in (.csv)
    """
    with open('cat_to_letter_rnn.json', 'r') as f1:
        cat_to_letter = json.load(f1)
    cat_to_letter = {int(i): j for i, j in cat_to_letter.items()}
    with open(results_filename, 'w') as f:
        f.write('input,predicted,target\n')
        for index, (input, target) in enumerate(eval_dataset):
            gen = generate_word(model, input, cat_to_letter)
            gen = gen.replace('EOS', '').replace('SOS', '')
            target_letters = ''.join([cat_to_letter[i] for i in target.squeeze().numpy()])
            target_letters = target_letters.replace('PAD', '').replace('SOS', '').replace('EOS', '')

            input = input.squeeze().numpy()   #input[11:]  # TODO
            start_ind = np.where(input==1)[0].item()
            input = input[start_ind:]
            input_letters = ''.join([cat_to_letter[i] for i in input])
            input_letters = input_letters.replace('PAD', '').replace('EOS', '').replace('SOS', '')
            f.write(input_letters + ',' + gen + ',' + target_letters + '\n')

    print('finished generating words for eval')


# def showAttention(input_sentence, output_words, attentions):
#     # Set up figure with colorbar
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     cax = ax.matshow(attentions.numpy(), cmap='bone')
#     fig.colorbar(cax)
#
#     # Set up axes
#     ax.set_xticklabels([''] + list(input_sentence) +
#                        ['<EOS>'], rotation=90)
#     ax.set_yticklabels([''] + output_words)
#
#     # Show label at every tick
#     ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
#     ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
#
#     plt.show()
#
#
# def evaluateAndShowAttention(input_sentence):
#     output_words, attentions = evaluate(
#         encoder1, attn_decoder1, input_sentence)
#     print('input =', input_sentence)
#     print('output =', ' '.join(output_words))
#     showAttention(input_sentence, output_words, attentions)


def sev(model, hp_dict):
    # save model in pt file
    name = 'models/epochs=' + str(hp_dict['num_epochs']) + 'bs=' + str(hp_dict['batch_size']) +\
           'hs=' + str(hp_dict['hidden_size']) + 'teacher_forcing=' + str(hp_dict['teacher_forcing']) + 'model.pt'
    torch.save(model, name)


def ev(it, hp_dict, cat_to_letter):
    # load saved model and generate a word using it
    name = 'models/epochs=' + str(hp_dict['num_epochs']) + 'bs=' + str(hp_dict['batch_size']) + \
           'hs=' + str(hp_dict['hidden_size']) + 'teacher_forcing=' + str(hp_dict['teacher_forcing']) + 'model.pt'
    model_read = torch.load(name)
    model_read.eval()
    print(generate_word(model_read, it, cat_to_letter))
    return model_read


def create_hp_dict():
    """
    save important parameters in a dictionary
    :return: hp_dict: dictionary with necessary params
    """
    data_dir = 'data/'
    hidden_size = 100
    input_output_size = 46  # 68
    device = torch.device("cpu")
    batch_size = 4
    learning_rate = 5e-4
    report_every = 100
    dropout_p = 0.1
    num_epochs = 10
    seed = 0
    max_length = 23  # 34
    teacher_forcing = 0.8  # 0.8

    hp_dict = {'data_dir': data_dir, 'hidden_size': hidden_size, 'input_output_size': input_output_size, 'device': device,
               'batch_size': batch_size, 'learning_rate': learning_rate, 'report_every': report_every,
               'dropout_p': dropout_p, 'num_epochs': num_epochs, 'seed': seed, 'max_length': max_length,
               'teacher_forcing': teacher_forcing}

    return hp_dict


def main():

    # hyper parameters and other parameters
    hp_dict = create_hp_dict()

    # reproducibility
    torch.manual_seed(hp_dict['seed'])
    np.random.seed(hp_dict['seed'])
    random.seed(hp_dict['seed'])

    # datasets and loaders
    train_set = Words_Dataset(hp_dict['data_dir'] + 'train_new2.csv', hp_dict)
    eval_set = Words_Dataset(hp_dict['data_dir'] + 'val_new2.csv', hp_dict)
    train_loader = data.DataLoader(train_set, batch_size=hp_dict['batch_size'], shuffle=True)
    eval_loader = data.DataLoader(eval_set, batch_size=hp_dict['batch_size'])

    # model
    encoder1 = EncoderRNN(hp_dict).to(hp_dict['device'])
    attn_decoder1 = AttnDecoderRNN(hp_dict).to(hp_dict['device'])
    # attn_decoder1 = DecoderRNN(hp_dict).to(hp_dict['device'])
    model = EncoderDecoder(encoder1, attn_decoder1, hp_dict)

    optimizer = optim.Adam(model.parameters(), lr=hp_dict['learning_rate'])
    criterion = nn.NLLLoss(ignore_index=PAD_TOKEN)

    # training
    training_epochs(model, train_loader, eval_loader, optimizer, criterion, hp_dict['num_epochs'])

    # word generation - mashot - just to get a sense of performance
    word = 'מִשְוָט'
    with open('letter_to_cat_rnn.json', 'r') as f:
        letter_to_cat = json.load(f)
    with open('cat_to_letter_rnn.json', 'r') as f:
        cat_to_letter = json.load(f)
    cat_to_letter = {int(i): j for i, j in cat_to_letter.items()}

    char_inds = [SOS_token] + [letter_to_cat[i] for i in word]
    char_inds.append(EOS_token)  # add end of sequence token
    char_inds = char_inds + ([PAD_TOKEN] * (hp_dict['max_length'] - len(char_inds)))
    input_tensor = torch.tensor(char_inds, dtype=torch.long, device=hp_dict['device']).view(-1, 1)

    print(generate_word(model, input_tensor, cat_to_letter))

    # save model in models directory
    sev(model, hp_dict)

    # use saved model to generate mashot - supposed to have same result (sanity check)
    ev(input_tensor, hp_dict, cat_to_letter)

    # generating all eval words for further analysis - accuracy etc. (eval_metrics.py)
    generate_words_for_eval(model, eval_set, 'results_eval_teacher_h=' + str(hp_dict['teacher_forcing']) + '.csv')

    # evaluateRandomly(encoder1, attn_decoder1, pairs, n=100)
    #
    # evaluateAndShowAttention('כִיווֹר')
    # evaluateAndShowAttention('מְתַפֵּף')
    # evaluateAndShowAttention('מִשְוָט')
    # evaluateAndShowAttention('וַתְרָן')


if __name__ == '__main__':
    main()
