import torch
import json
import os
from torch import nn, optim
from torch.nn import functional as F
import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch.utils.data as data
import numpy as np
import argparse
import sys

# global vars
PAD_TOKEN = 0  # padding token
SOS_token = 1  # start of sequence token
EOS_token = 2  # end of sequence token


# dataset #


class Words_Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data_path, args):
        self.data = pd.read_csv(data_path)
        self.num_of_examples = len(self.data)
        self.source_words = self.data['combine_result']
        self.target_words = self.data['word']
        self.is_gzarot = args.is_gzarot
        self.is_gzarot_concat = args.is_gzarot_concat

        if self.is_gzarot:
            self.gzarot_prefix = self.data['gzarot_prefix']
        elif self.is_gzarot_concat:
            self.gzarot_concat = self.data['gzarot_concat']
        self.device = args.device
        self.max_length = args.max_length
        # get index dictionaries
        with open('letter_to_cat_rnn2.json', 'r') as f:
            self.letter_to_cat = json.load(f)
        with open('cat_to_letter_rnn2.json', 'r') as f:
            self.cat_to_letter = json.load(f)

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        src_seq = self.source_words.iloc[index]
        trg_seq = self.target_words.iloc[index]
        gzarot = None
        if self.is_gzarot:
            gzarot = self.gzarot_prefix.iloc[index]
        elif self.is_gzarot_concat:
            gzarot = self.gzarot_concat.iloc[index]
        src_seq, gz_vectors = self.get_word_tensor(src_seq, gzarot=gzarot)
        trg_seq = self.get_word_tensor(trg_seq)[0]
        return src_seq, trg_seq, gz_vectors

    def __len__(self):
        return self.num_of_examples

    def get_word_tensor(self, word, gzarot=None):
        """
        use dictionary: letters -> integers to get tensor with integers
        :param word: input word to turn into tensor
        :param gzarot: if exists - specifies gzarot of word
        :return: tensor with integers corresponding to the word letters,
        with SOS, EOS and PAD tokens.
        """
        char_inds = [SOS_token] + [self.letter_to_cat[i] for i in word]
        char_inds.append(EOS_token)  # add end of sequence token
        gz_vectors = [''] * self.max_length
        if gzarot:
            if self.is_gzarot:
                gz = gzarot.split()
                gz_tokens = [self.letter_to_cat[i] for i in gz]
                char_inds = gz_tokens + char_inds
            elif self.is_gzarot_concat:
                gz = gzarot.split(' 2 ')
                gz_vectors = []
                for item in gz:
                    vector = item.split()
                    vector = torch.tensor([int(x) for x in vector], dtype=torch.float)
                    gz_vectors.append(vector)
                for i in range(self.max_length - len(gz_vectors)):
                    gz_vectors.append(torch.tensor([PAD_TOKEN] * 12, dtype=torch.float))

        char_inds = char_inds + ([PAD_TOKEN] * (self.max_length - len(char_inds)))
        tensor_output = torch.tensor(char_inds, dtype=torch.long, device=self.device).view(-1, 1)
        return tensor_output, gz_vectors


# network architecture #

class EncoderDecoder(nn.Module):
    """
    Encoder-Decoder architecture
    """

    def __init__(self, encoder, decoder, args):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.input_output_size = args.input_output_size
        self.max_length = args.max_length
        self.teacher_forcing = args.teacher_forcing
        self.device = args.device
        self.is_attn = args.is_attn
        self.is_bi = args.is_bi

    def forward(self, input_tensor, target_tensor, phase, gz_vectors):

        if phase == 'train':
            teacher_forcing_ratio = self.teacher_forcing
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
                                     device=self.device)

        # encoder - loop over input letters
        for ei in range(self.max_length):
            if isinstance(gz_vectors[0], tuple) or isinstance(gz_vectors[0], str):
                encoder_output, encoder_hidden = self.encoder(
                    input_tensor[ei], encoder_hidden)
            else:
                encoder_output, encoder_hidden = self.encoder(
                    input_tensor[ei], encoder_hidden, gz_vectors[ei])

            encoder_outputs[ei] = encoder_output[0]

        decoder_input = torch.tensor([SOS_token] * curr_batch_size, device=self.device)

        decoder_hidden = encoder_hidden
        if self.is_bi:
            decoder_hidden = encoder_hidden[0, :, :] + encoder_hidden[1, :, :]
            decoder_hidden = decoder_hidden.view(1, curr_batch_size, -1)

        # teacher forcing
        use_teacher_forcing = False
        if teacher_forcing_ratio != 0:
            if random.random() < teacher_forcing_ratio:
                use_teacher_forcing = True

        # decoder - loop over output letters

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(self.max_length):
                if self.is_attn:
                    decoder_output, decoder_hidden, decoder_attention = self.decoder(
                        decoder_input, decoder_hidden, encoder_outputs)
                else:
                    decoder_output, decoder_hidden = self.decoder(
                        decoder_input, decoder_hidden)
                network_outputs[di] = decoder_output
                decoder_input = target_tensor[di]  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(self.max_length):
                if self.is_attn:
                    decoder_output, decoder_hidden, decoder_attention = self.decoder(
                        decoder_input, decoder_hidden, encoder_outputs)
                else:
                    decoder_output, decoder_hidden = self.decoder(
                        decoder_input, decoder_hidden)
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
    def __init__(self, args):
        super(EncoderRNN, self).__init__()
        self.device = args.device
        self.input_size = args.input_output_size
        self.hidden_size = args.hidden_size
        self.is_bi = args.is_bi
        self.not_embedded = args.not_embedded
        self.is_gzarot_concat = args.is_gzarot_concat

        if not self.not_embedded:
            if self.is_gzarot_concat:
                self.embedding = nn.Embedding(self.input_size, self.hidden_size-12, padding_idx=PAD_TOKEN)
            else:
                self.embedding = nn.Embedding(self.input_size, self.hidden_size, padding_idx=PAD_TOKEN)
        # alternative to embedding - one hot (in forward function)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, bidirectional=self.is_bi)

    def forward(self, input, hidden, gz_vector=None):
        curr_batch_size = input.shape[0]
        # one hot
        if self.not_embedded:
            one_hot = torch.zeros([1, curr_batch_size, self.input_size])
            one_hot[0, np.arange(curr_batch_size), input.squeeze()] = 1
            output = one_hot
        else:
            if gz_vector is not None:
                embedded = self.embedding(input).view(1, curr_batch_size, self.hidden_size - 12)
            else:
                embedded = self.embedding(input).view(1, curr_batch_size, self.hidden_size)
            output = embedded

        if gz_vector is not None:  # if gz_vector:
            output = torch.cat([output, gz_vector.view(1, curr_batch_size, 12)], dim=2)

        output, hidden = self.gru(output, hidden)
        if self.is_bi:
            output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]
        return output, hidden

    def initHidden(self, batch_size):
        if self.is_bi:
            return torch.zeros(2, batch_size, self.hidden_size, device=self.device)
        return torch.zeros(1, batch_size, self.hidden_size, device=self.device)


class DecoderRNN(nn.Module):
    """
    decoder part of the network - option 1
    embedding + relu +  gru + linear + softmax
    """
    def __init__(self, args):
        super(DecoderRNN, self).__init__()
        self.hidden_size = args.hidden_size
        self.output_size = args.input_output_size
        self.device = args.device

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
        return torch.zeros(1, batch_size, self.hidden_size, device=self.device)


class AttnDecoderRNN(nn.Module):
    """
    decoder part of the network - option 2
    embedding + dropout + attention + relu +  gru + linear + softmax
    """
    def __init__(self, args):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = args.hidden_size
        self.output_size = args.input_output_size
        self.dropout_p = args.dropout_p
        self.max_length = args.max_length
        self.batch_size = args.batch_size
        self.device = args.device

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
        return torch.zeros(1, batch_size, self.hidden_size, device=self.device)


# training #

def train(input_tensor, target_tensor, model, optimizer, criterion, gz_vectors):

    model.train()

    optimizer.zero_grad()

    target_length = target_tensor.size(1)

    outputs = model(input_tensor, target_tensor, 'train', gz_vectors)

    loss = 0
    for di in range(target_length):
        loss += criterion(outputs[di], target_tensor.transpose(0,1)[di].squeeze(1))

    loss.backward()

    optimizer.step()

    return loss.item() / target_length


def eval_ex(input_tensor, target_tensor, model, criterion, gz_vectors):
    outputs = model(input_tensor, target_tensor, 'eval', gz_vectors)
    target_length = target_tensor.size(1)
    loss = 0
    for di in range(target_length):
        loss += criterion(outputs[di], target_tensor.transpose(0,1)[di].squeeze(1))
    return loss.item() / target_length


def training_epochs(model, train_loader, eval_loader, optimizer, criterion, num_epochs, name_of_savedir):
    # train and eval losses for plots
    train_losses_per_epoch = []
    eval_losses_per_epoch = []

    for epoch in range(num_epochs):
        loss = 0
        eval_loss = 0

        # train
        for batch_index, (input_tensor, target_tensor, gz_vectors) in enumerate(train_loader):
            iter_loss = train(input_tensor, target_tensor, model, optimizer, criterion, gz_vectors)
            loss += iter_loss
        train_losses_per_epoch.append(loss / len(train_loader))  # loss per epoch (avg)
        print(str(epoch) + ': training loss: ' + str(loss / len(train_loader)))

        # evaluation
        model.eval()
        for batch_index1, (input_tensor1, target_tensor1, gz_vectors1) in enumerate(eval_loader):
            iter_eval_loss = eval_ex(input_tensor1, target_tensor1, model, criterion, gz_vectors1)
            eval_loss += iter_eval_loss
        eval_losses_per_epoch.append(eval_loss / len(eval_loader))
        print('eval loss: ' + str(eval_loss / len(eval_loader)))

    # loss plots
    plt.plot(train_losses_per_epoch, label='train')
    plt.plot(eval_losses_per_epoch, label='eval')
    plt.legend()
    plt.title('last eval loss: ' + str(eval_loss / len(eval_loader)))
    plt.savefig(name_of_savedir + '/losses.png', dpi=600)
    # plt.show()
    return (loss / len(train_loader)), (eval_loss / len(eval_loader))


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


def generate_word(model, input_word, cat_to_letter, args, gz_vectors):
    """
    generate word given input_word
    :param model: trained model
    :param input_word: input
    :param cat_to_letter: dictionary {number: letter}
    :param args: has max length
    :return: decoded word: generated word
    """
    model.eval()
    decoded_word = ''
    with torch.no_grad():
        changed_dim = input_word.view(1, args.max_length, 1)  # 20
        outputs = model(changed_dim, None, 'eval', gz_vectors)
        for di in outputs:
            topv, topi = di.data.topk(1)
            decoded_word += cat_to_letter[topi.item()]
            if cat_to_letter[topi.item()] == 'EOS':
                return decoded_word

    return decoded_word


def generate_words_for_eval(model, eval_dataset, results_filename, args):
    """
    use the model to generate words given all of the eval dataset words
    :param model: trained model
    :param eval_dataset: dataset object with evaluation data
    :param results_filename: path to write results in (.csv)
    """
    with open('cat_to_letter_rnn2.json', 'r') as f1:
        cat_to_letter = json.load(f1)
    cat_to_letter = {int(i): j for i, j in cat_to_letter.items()}
    with open(results_filename, 'w') as f:
        f.write('input,predicted,target\n')
        for index, (input, target, gz_vectors) in enumerate(eval_dataset):
            gen = generate_word(model, input, cat_to_letter, args, gz_vectors)
            gen = gen.replace('EOS', '').replace('SOS', '')
            target_letters = ''.join([cat_to_letter[i] for i in target.squeeze().numpy()])
            target_letters = target_letters.replace('PAD', '').replace('SOS', '').replace('EOS', '')
            input = input.squeeze().numpy()
            start_ind = np.where(input == 1)[0].item()
            input = input[start_ind:]
            input_letters = ''.join([cat_to_letter[i] for i in input])
            input_letters = input_letters.replace('PAD', '').replace('EOS', '').replace('SOS', '')
            f.write(input_letters + ',' + gen + ',' + target_letters + '\n')

    print('finished generating words for eval')


def compute_acc(generated_words_file):
    df = pd.read_csv(generated_words_file)
    acc = sum(df['predicted'] == df['target']) / len(df)
    return acc


def sev(model, name_of_savedir):
    # save model in pt file
    torch.save(model, name_of_savedir + '/model.pt')


def ev(it, name_of_savedir, cat_to_letter, args):
    model_read = torch.load(name_of_savedir + '/model.pt')
    model_read.eval()
    print(generate_word(model_read, it, cat_to_letter, args))
    return model_read


def add_bool_arg(parser, name, default=False):
    parser.add_argument('--' + name, dest=name, action='store_true')
    parser.add_argument('--no_' + name, dest=name, action='store_false')
    parser.set_defaults(**{name: default})


def parseArgs():
    parser = argparse.ArgumentParser()
    # could be played with and changed
    parser.add_argument('--hidden_size', type=int, default=100)  # 100
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--dropout_p', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=2)  # 10
    parser.add_argument('--teacher_forcing', type=float, default=0.8)

    add_bool_arg(parser, 'is_attn', default=True)
    add_bool_arg(parser, 'is_bi', default=True)
    add_bool_arg(parser, 'is_gzarot', default=True)
    add_bool_arg(parser, 'not_embedded', default=False)
    add_bool_arg(parser, 'is_gzarot_concat', default=False)

    # shouldn't change
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default=torch.device("cpu"))
    parser.add_argument('--max_length', type=int, default=20)
    parser.add_argument('--input_output_size', type=int, default=46)
    parser.add_argument('--data_dir', default='data/')
    parser.add_argument('--save_mega_dir', default='final_experiments/')

    return parser.parse_args()


def main():

    # hyper parameters and other parameters
    args = parseArgs()

    assert args.is_gzarot_concat != args.is_gzarot

    if args.is_attn:
        attn_val_for_name = '_attn'
    else:
        attn_val_for_name = ''
    if args.is_bi:
        bi_val_for_name = '_bi'
    else:
        bi_val_for_name = ''
    if args.is_gzarot:
        gzarot_val_for_name = '_gzarot'
        # change parameters to fit the gzarot case
        args.max_length = args.max_length + 11
        args.input_output_size = args.input_output_size + 22
    else:
        gzarot_val_for_name = ''
    if args.not_embedded:
        embedded_val_for_name = '_not_embedded'
        args.hidden_size = args.input_output_size
    else:
        embedded_val_for_name = ''

    if args.is_gzarot_concat:
        gzarot_concat_val_for_name = '_gzarot_concat'
        args.hidden_size += 12
    else:
        gzarot_concat_val_for_name = ''


    vals_for_name = attn_val_for_name + bi_val_for_name + gzarot_val_for_name +\
                    embedded_val_for_name + gzarot_concat_val_for_name
    args.save_mega_dir = args.save_mega_dir + vals_for_name[1:] + '/'

    name_of_savedir = str(args.save_mega_dir) + 'hd=' + str(args.hidden_size) + '_bs=' + str(args.batch_size)\
                      + '_lr=' + str(args.learning_rate) + '_do=' + str(args.dropout_p)\
                      + '_epochs=' + str(args.num_epochs) + '_teacher=' \
                      + str(args.teacher_forcing)

    # create directory - if exists exit program
    if os.path.isdir(name_of_savedir):
        sys.exit('directory already exists! exiting')
    else:
        os.makedirs(name_of_savedir)

    # reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # datasets and loaders
    train_set = Words_Dataset(args.data_dir + 'train_val_new5.csv', args)
    eval_set = Words_Dataset(args.data_dir + 'test_new5.csv', args)
    train_loader = data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    eval_loader = data.DataLoader(eval_set, batch_size=args.batch_size)

    # model
    encoder1 = EncoderRNN(args).to(args.device)
    if args.is_attn:
        attn_decoder1 = AttnDecoderRNN(args).to(args.device)
    else:
        attn_decoder1 = DecoderRNN(args).to(args.device)
    model = EncoderDecoder(encoder1, attn_decoder1, args)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.NLLLoss(ignore_index=PAD_TOKEN)

    # training
    t_loss, e_loss = training_epochs(model, train_loader, eval_loader, optimizer, criterion, args.num_epochs, name_of_savedir)

    # save model in models directory
    sev(model, name_of_savedir)

    # generating all eval words for further analysis - accuracy etc. (eval_metrics.py)
    generate_words_for_eval(model, eval_set, name_of_savedir + '/results_eval.csv', args)
    acc = compute_acc(name_of_savedir + '/results_eval.csv')

    # add result to csv file
    # open file

    filename_csv = args.save_mega_dir + 'all_stats' + vals_for_name + '.csv'
    with open(filename_csv, 'a') as outfile:
        line = str(args.learning_rate) + ',' + str(args.hidden_size) + ',' + \
               str(args.batch_size) + ',' + str(args.teacher_forcing) + ',' + str(args.num_epochs) + ',' +\
               str(t_loss) + ',' + str(e_loss) + ',' + str(acc) + '\n'
        outfile.write(line)

if __name__ == '__main__':
    main()
