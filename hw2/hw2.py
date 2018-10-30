# First lets improve libraries that we are going to be used in this lab session
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from collections import Counter
import pickle as pkl
import random
import pdb
import csv
import matplotlib.pyplot as plt
import sys
plt.switch_backend('agg')
random.seed(134)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
# create the dictionary and all train tokens
VOCAB_SIZE = 50000
EMBED_SIZE = 300
# load data
snli_val_data_tokens = pkl.load(open("snli_val_data_tokens.p", "rb"))
snli_train_data_tokens = pkl.load(open("snli_train_data_tokens.p", "rb"))
all_train_tokens = pkl.load(open("all_train_tokens.p", "rb"))
embedding_dict = pkl.load(open("embedding_dict.p", "rb"))
# save index 0 for unk and 1 for pad
PAD_IDX = 0
UNK_IDX = 1

def build_vocab(all_tokens):
    # Returns:
    # id2token: list of tokens, where id2token[i] returns token that corresponds to token i
    # token2id: dictionary where keys represent tokens and corresponding values represent indices
    id2token = list(all_tokens)
    token2id = dict(zip(all_tokens, range(2,2+len(all_tokens))))
    id2token = ['<pad>', '<unk>']  + id2token
    token2id['<pad>'] = PAD_IDX
    token2id['<unk>'] = UNK_IDX
    return token2id, id2token

# convert token to id in the dataset
def token2index_dataset(tokens_data, token2id):
    prem_indices_data = []
    hyp_indices_data = []
    target_indices_data = []
    for tokens in tokens_data:
#         print(tokens[0])
#         print(tokens[1])
        prem_index_list = [token2id[token] if token in token2id else UNK_IDX for token in tokens[0]]
        hyp_index_list = [token2id[token] if token in token2id else UNK_IDX for token in tokens[1]]
        prem_indices_data.append(prem_index_list)
        hyp_indices_data.append(hyp_index_list)
        target_indices_data.append(tokens[2])
    return prem_indices_data, hyp_indices_data, target_indices_data
token2id, id2token = build_vocab(all_train_tokens)
val_prem_data_indices, val_hyp_data_indices, val_target_data_indices = token2index_dataset(snli_val_data_tokens, token2id)
train_prem_data_indices, train_hyp_data_indices, train_target_data_indices = token2index_dataset(snli_train_data_tokens, token2id)

MAX_SENTENCE_LENGTH = 25


# encode data loader
class EncodeDataset(Dataset):
    """
    Class that represents a train/validation/test dataset that's readable for PyTorch
    Note that this class inherits torch.utils.data.Dataset
    """

    def __init__(self, prem_data_list, hyp_data_list, target_list):
        """
        @param data_list: list of newsgroup tokens
        @param target_list: list of newsgroup targets

        """
        self.prem_data_list = prem_data_list
        self.hyp_data_list = hyp_data_list
        self.target_list = target_list

    def __len__(self):
        return len(self.prem_data_list)

    def __getitem__(self, key):
        """
        Triggered when you call dataset[i]
        """
        prem_token_idx = self.prem_data_list[key][:MAX_SENTENCE_LENGTH]
        hyp_token_idx = self.hyp_data_list[key][:MAX_SENTENCE_LENGTH]
        label = self.target_list[key]
        return [prem_token_idx, hyp_token_idx, label]


def encode_collate_func(batch):
    """
    Customized function for DataLoader that dynamically pads the batch so that all
    data have the same length
    """
    prem_data_list = []
    hyp_data_list = []
    label_list = []
    length_list = []
    # print("collate batch: ", batch[0][0])
    # batch[0][0] = batch[0][0][:MAX_SENTENCE_LENGTH]
    for datum in batch:
        label_list.append(datum[2])
    # padding
    for datum in batch:
        prem_padded_vec = np.pad(np.array(datum[0]),
                                 pad_width=((0, MAX_SENTENCE_LENGTH - len(datum[0]))),
                                 mode="constant", constant_values=0)
        hyp_padded_vec = np.pad(np.array(datum[1]),
                                pad_width=((0, MAX_SENTENCE_LENGTH - len(datum[1]))),
                                mode="constant", constant_values=0)
        prem_data_list.append(prem_padded_vec)
        hyp_data_list.append(hyp_padded_vec)
    return [torch.from_numpy((np.array(prem_data_list))), torch.from_numpy(np.array(hyp_data_list)),
            torch.LongTensor(label_list)]

BATCH_SIZE = 64
train_dataset = EncodeDataset(train_prem_data_indices, train_hyp_data_indices, train_target_data_indices)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=BATCH_SIZE,
                                           collate_fn=encode_collate_func,
                                           shuffle=True)

val_dataset = EncodeDataset(val_prem_data_indices, val_hyp_data_indices, val_target_data_indices)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                           batch_size=BATCH_SIZE,
                                           collate_fn=encode_collate_func,
                                           shuffle=True)
# CNN Encoder
class CNN(nn.Module):
    def __init__(self, emb_size, hidden_size, vocab_size, kernel_size, hid_dim, is_concat, is_dropout):

        super(CNN, self).__init__()

        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=PAD_IDX)
        self.embedding.weight.data.copy_(torch.from_numpy(np.array(embedding_dict).copy()))
        self.conv1 = nn.Conv1d(emb_size, hidden_size, kernel_size, padding=1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size, padding=1)
        self.linear1 = nn.Linear(hidden_size*2, hid_dim)
        self.linear2 = nn.Linear(hid_dim, 3)
        self.is_concat = is_concat
        self.is_dropout = is_dropout
        if self.is_dropout == True:
            self.dropout = nn.Dropout(0.5)
    def encode(self, x):
        batch_size, seq_len = x.size()
        embed = self.embedding(x)
        m = (x == 1) | (x == 0)
        embed = m * embed + (1-m) * embed.clone().detch()
        hidden = self.conv1(embed.transpose(1,2)).transpose(1,2)
        hidden = F.relu(hidden.contiguous().view(-1, hidden.size(-1))).view(batch_size, seq_len, hidden.size(-1))
        hidden = self.conv2(hidden.transpose(1,2)).transpose(1,2)
        hidden = F.relu(hidden.contiguous().view(-1, hidden.size(-1))).view(batch_size, seq_len, hidden.size(-1))
        hidden = torch.max(hidden, 1)[0]
        return hidden
    def forward(self, prem, hyp):
        batch_size, seq_len = prem.size()
        # encode premise
        prem_code = self.encode(prem)
        # encode hypothesis
        hyp_code = self.encode(hyp)
        # concat or multiply
        if self.is_concat:
            code = torch.cat((prem_code,hyp_code), dim=1)
        else:
            code = prem_code * hyp_code
        code = self.linear1(code)
        if self.is_dropout:
            code = self.dropout(code)
        code = self.linear2(code)
        return code

# RNN Encoder
class RNN(nn.Module):
    def __init__(self, emb_size, hidden_size, vocab_size, hid_dim, is_concat, is_dropout):

        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=PAD_IDX)
        self.embedding.weight.data.copy_(torch.from_numpy(np.array(embedding_dict).copy()))
        self.bi_gru = nn.GRU(emb_size, hidden_size, num_layers=1, batch_first=True,bidirectional=True)
        self.linear1 = nn.Linear(4 * hidden_size, hid_dim)
        self.linear2 = nn.Linear(hid_dim, 3)
        self.is_concat = is_concat
        self.is_dropout = is_dropout
        if self.is_dropout == True:
            self.dropout = nn.Dropout(0.5)
    def encode(self, x):
        lengths = MAX_SENTENCE_LENGTH - x.eq(0).long().sum(1).squeeze()
        _, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        lengths = lengths[idx_sort]
        x = x.index_select(0, idx_sort)
        embed = self.embedding(x)
        m = (x == 1) | (x == 0)
        embed = m * embed + (1-m) * embed.clone().detch()
        embed = torch.nn.utils.rnn.pack_padded_sequence(embed, lengths.cpu().numpy(), batch_first=True)
        output, hidden = self.bi_gru(embed)
        hidden = torch.cat((hidden[0],hidden[1]), dim = 1)
        hidden = hidden.index_select(0, idx_unsort)
        return hidden
    def forward(self, prem, hyp):
        batch_size, seq_len = prem.size()
        # encode premise
        prem_code = self.encode(prem)
        # encode hypothesis
        hyp_code = self.encode(hyp)
        # concat or multiply
        if self.is_concat:
            code = torch.cat((prem_code,hyp_code), dim=1)
        else:
            code = prem_code * hyp_code
        code = self.linear1(code)
        if self.is_dropout:
            code = self.dropout(code)
        code = self.linear2(code)
        return code
# Function for testing the model
def test_model(loader, model):
    """
    Help function that tests the model's performance on a dataset
    @param: loader - data loader for the dataset to test against
    """
    correct = 0
    total = 0
    model.eval()
    for prem_data, hyp_data, labels in loader:
        prem_data_batch, hyp_data_batch, label_batch = prem_data.to(device), hyp_data.to(device),labels.to(device)
        outputs = F.softmax(model(prem_data_batch, hyp_data_batch), dim=1)
        predicted = outputs.max(1, keepdim=True)[1]
        total += labels.size(0)
        correct += predicted.eq(label_batch.view_as(predicted)).sum().item()
    return (100 * correct / total)


def plot_func(train_accs, val_accs, filename):
    f = plt.figure()
    plt.plot(train_accs, label='train');
    plt.plot(val_accs, label='val');
    plt.title(filename);
    plt.legend()

    f.savefig(filename + ".pdf", bbox_inches='tight')
    plt.show()
# Training
# Train and valid function
# hidden_size hidden size in cnn/rnn; hid_dim hidden dimension in fully connected network
# encoder: 'cnn', 'rnn'
def train_valid(encoder='cnn', hidden_size=200, hid_dim=200, is_concat=True, kernel_size=3, lr=0.01, is_wd=True,
                is_dropout=True):
    print('encoder: ', encoder)
    print('hidden_size : ', hidden_size)
    print('hid_dim: ', hid_dim)
    print('is_concat? : ', is_concat)
    print('kernel_size: ', kernel_size)
    print('initial learning_rate: ', lr)
    print('weight decay? : ', is_wd)
    print('dropout? : ', is_dropout)
    sys.stdout.flush()
    filename = '_'.join([encoder, str(hidden_size),
                         str(hid_dim), str(is_concat), str(kernel_size), str(is_wd), str(is_dropout)]);
    if encoder == 'cnn':
        model = CNN(EMBED_SIZE, hidden_size, VOCAB_SIZE + 2, kernel_size, hid_dim, is_concat, is_dropout)
    else:
        model = RNN(EMBED_SIZE, hidden_size, VOCAB_SIZE + 2, hid_dim, is_concat, is_dropout)
    model = model.to(device)
    learning_rate = lr
    num_epochs = 10  # number epoch to train
    criterion = torch.nn.CrossEntropyLoss()
    if is_wd:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_accs = []
    val_accs = []
    for epoch in range(num_epochs):
        optimizer.defaults['lr'] = learning_rate/(epoch+1)
        for i, (prem_data, hyp_data, labels) in enumerate(train_loader):
            sys.stdout.flush()
            model.train()
            prem_data_batch, hyp_data_batch, label_batch = prem_data.to(device), hyp_data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(prem_data_batch, hyp_data_batch)
            loss = criterion(outputs, label_batch)
            loss.backward()
            optimizer.step()
            # validate every 10 iterations
            if i > 0 and i % 10 == 0:
                # validate
                val_acc = test_model(val_loader, model)
                train_acc = test_model(train_loader, model)
                val_accs.append(val_acc)
                train_accs.append(train_acc)
                print('Epoch: [{}/{}], Step: [{}/{}], Validation Acc: {}'.format(
                    epoch + 1, num_epochs, i + 1, len(train_loader), val_acc))
                print('Epoch: [{}/{}], Step: [{}/{}], Train Acc: {}'.format(
                    epoch + 1, num_epochs, i + 1, len(train_loader), train_acc))
                sys.stdout.flush()
    plot_func(train_accs, val_accs, filename)

    print ("After training for {} epochs".format(num_epochs))
    print ("Train Acc {}".format(test_model(train_loader, model)))
    print ("Val Acc {}".format(test_model(val_loader, model)))
    sys.stdout.flush()
    return test_model(val_loader, model)

# Example Training
hidden_sizes = [100, 200, 300]
encoders = ['rnn','cnn']
is_wds = [True, False]
is_dropouts = [True, False]
is_concats = [True, False]
kernel_sizes = [3, 5]
val_acc2 = train_valid(encoder='rnn', hidden_size=200, hid_dim=200, is_concat=True, lr=0.001, is_wd=True,
                is_dropout=True, kernel_size=3)
best_acc = 0
lr = 0.01
for is_wd in is_wds:
    for is_dropout in is_dropouts:
        for is_concat in is_concats:
            for hidden_size in hidden_sizes:
                for encoder in encoders:
                    if encoder == 'rnn':
                        train_valid(encoder, hidden_size, hid_dim, is_concat, lr, is_wd, is_dropout)
                    else:
                        for kernel_size in kernel_sizes:
                            train_valid(encoder, hidden_size, hid_dim, is_concat, lr, is_wd, is_dropout, kernel_size)


print('CNN acc', val_acc1)
print('RNN acc', val_acc2)