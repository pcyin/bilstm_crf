import argparse
from collections import Counter, defaultdict
from itertools import chain

import numpy as np
import sys, os

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.optim
import torch.nn.functional as F
from data import scores


def load_dataset(data_file):
    data_set = []
    tokens, pos_tags, syn_tags, ner_labels = [], [], [], []
    for line in open(data_file):
        line = line.strip()
        if line:
            token, pos_tag, syn_tag, ner_label = line.split('\t')
            tokens.append(token)
            pos_tags.append(pos_tag)
            syn_tags.append(syn_tag)
            ner_labels.append(ner_label)
        else:
            if len(tokens) > 1:
                data_set.append((tokens, pos_tags, syn_tags, ner_labels))

            tokens, pos_tags, syn_tags, ner_labels = [], [], [], []

    return data_set


def batch_iter(examples, batch_size, shuffle=False):
    index_arr = np.arange(len(examples))
    if shuffle:
        np.random.shuffle(index_arr)

    batch_num = int(np.ceil(len(examples) / float(batch_size)))
    for batch_id in range(batch_num):
        batch_ids = index_arr[batch_size * batch_id: batch_size * (batch_id + 1)]
        batch_examples = [examples[i] for i in batch_ids]
        batch_examples = sorted(batch_examples, key=lambda e: -len(e[0]))

        yield batch_examples


def log_sum_exp(x):
    max_val = torch.max(x, dim=-1)[0]
    sum_val = torch.exp(x - max_val.expand(x.size(0))).sum().log() + max_val

    return sum_val


def word2id(sents, vocab):
    if type(sents[0]) == list:
        return [[vocab[w] for w in s] for s in sents]
    else:
        return [vocab[w] for w in sents]


def init_vocab(dataset, cutoff=1):
    all_tokens = sorted(chain.from_iterable(x[0] for x in dataset))
    token_freq = Counter(all_tokens)
    word_types = sorted(token_freq.keys())

    vocab = defaultdict(int)
    vocab['<unk>'] = 0
    vocab['<pad>'] = 1
    for word in word_types:
        if token_freq[word] >= cutoff:
            vocab[word] = len(vocab)

    return vocab


def input_transpose(sents, pad_token):
    """
    transform the input List[sequence] of size (batch_size, max_sent_len)
    into a list of size (max_sent_len, batch_size), with proper padding
    """
    max_len = max(len(s) for s in sents)
    batch_size = len(sents)

    sents_t = []
    masks = []
    for i in range(max_len):
        sents_t.append([sents[k][i] if len(sents[k]) > i else pad_token for k in range(batch_size)])
        masks.append([1 if len(sents[k]) > i else 0 for k in range(batch_size)])

    return sents_t, masks


def to_input_variable(sequences, vocab, cuda=False, training=True, append_boundary_sym=False):
    """
    given a list of sequences,
    return a tensor of shape (max_sent_len, batch_size)
    """
    if append_boundary_sym:
        sequences = [['<s>'] + seq + ['</s>'] for seq in sequences]

    word_ids = word2id(sequences, vocab)
    sents_t, masks = input_transpose(word_ids, vocab['<pad>'])

    sents_var = Variable(torch.LongTensor(sents_t), volatile=(not training), requires_grad=False)
    if cuda:
        sents_var = sents_var.cuda()

    return sents_var


class BiLSTM_CRF(nn.Module):
    def __init__(self, embed_size, hidden_size, dropout, src_vocab, tags, cuda):
        super(BiLSTM_CRF, self).__init__()

        self.tags = tags = ['<START>'] + tags + ['<STOP>']

        self.state2id = {tag: i for i, tag in enumerate(tags)}
        self.id2state = {i: tag for i, tag in enumerate(tags)}
        self.start_sym = '<START>'
        self.end_sym = '<STOP>'
        self.tag_num = len(tags)

        self.cuda = cuda
        self.src_vocab = src_vocab
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.src_embed = nn.Embedding(len(src_vocab), embed_size)
        nn.init.xavier_normal(self.src_embed.weight)

        self.src_lstm = nn.LSTM(embed_size, hidden_size, bidirectional=True)
        self.enc_to_state = nn.Linear(hidden_size * 2, hidden_size)
        self.tag_readout = nn.Linear(hidden_size, len(tags))

        self.transition = nn.Parameter(torch.Tensor(len(tags), len(tags)))
        nn.init.xavier_normal(self.transition.data)

        if dropout:
            self.dropout_layer = nn.Dropout(dropout)

    def forward_algo(self, src_sents, tag_feat_scores):
        # tag_feat_scores: (src_sent_len, batch_size, tag_num)
        max_step = max(len(sent) for sent in src_sents)
        alphas = []
        for batch_id in src_sents:
            alpha_t = Variable(tag_feat_scores.data.new(self.tag_num).fill_(-10000.))
            alpha_t[self.state2id[self.start_sym]] = 0.
            alphas.append(alpha_t)

        for t in range(0, max_step):
            for batch_id, src_sent in enumerate(src_sents):
                if t >= len(src_sent): continue

                alpha_t = []
                for state_t in self.tags:
                    state_id = self.state2id[state_t]
                    trans_score = self.transition[state_id]
                    # (tag_size, )
                    state_scores = alphas[batch_id] + trans_score + tag_feat_scores[batch_id, t, state_id].expand(self.tag_num)
                    if t == 0:
                        state_score_sum = state_scores[self.state2id[self.start_sym]]
                    else:
                        state_score_sum = log_sum_exp(state_scores)

                    alpha_t.append(state_score_sum)

                alpha_t = torch.cat(alpha_t).view(-1)
                alphas[batch_id] = alpha_t

        alphas = [alpha + self.transition[self.state2id[self.end_sym]] for alpha in alphas]
        partition_func = torch.cat([log_sum_exp(alpha) for alpha in alphas])

        return partition_func

    def forward(self, src_sent):
        tag_feat_scores = self.get_tag_feat_scores([src_sent]).squeeze(0)
        alpha = Variable(torch.FloatTensor(self.tag_num).fill_(-10000))
        if self.cuda: alpha = alpha.cuda()
        alpha[self.state2id[self.start_sym]] = 0.
        back_pointers = np.zeros((len(src_sent), self.tag_num), dtype='int32')
        back_pointers[0, :] = self.state2id[self.start_sym]

        for t in range(len(src_sent)):
            alpha_t = []
            for state_t in self.tags:
                state_id = self.state2id[state_t]
                trans_score = self.transition[state_id]

                state_scores = alpha + trans_score + tag_feat_scores[t, state_id].expand(self.tag_num)
                if t == 0:
                    best_prev_state_id = start_state_id = self.state2id[self.start_sym]
                    best_prev_state_score = state_scores[start_state_id]
                else:
                    best_prev_state_score, best_prev_state_id = torch.max(state_scores, dim=0)

                back_pointers[t][state_id] = best_prev_state_id
                alpha_t.append(best_prev_state_score)

            alpha = torch.cat(alpha_t)

        alpha = alpha + self.transition[self.state2id[self.end_sym]]
        best_prev_state_score, best_prev_state_id = torch.max(alpha, dim=-1)

        # back tracing
        best_states = [best_prev_state_id.data[0]]
        for t in reversed(range(len(src_sent))):
            best_state_tm1 = back_pointers[t][best_states[-1]]
            best_states.append(best_state_tm1)

        best_states = best_states[::-1][1:]
        best_states = list(map(lambda s: self.id2state[s], best_states))

        return best_states

    def get_tag_sequence_score(self, src_sents, tag_feat_scores, tgt_tags):
        scores = [0. for _ in src_sents]
        max_step = max(len(sent) for sent in src_sents)
        padded_tags = [[self.start_sym] + list(_tags) for _tags in tgt_tags]

        for t in range(max_step):
            for i, src_sent in enumerate(src_sents):
                if t < len(src_sent):
                    scores[i] = scores[i] + tag_feat_scores[i, t, self.state2id[padded_tags[i][t + 1]]] + \
                                    self.transition[self.state2id[padded_tags[i][t + 1]], self.state2id[padded_tags[i][t]]]

        scores = [score + self.transition[self.state2id[self.end_sym], self.state2id[padded_tags[i][-1]]]
                  for i, score in enumerate(scores)]

        return torch.cat(scores)

    def get_tag_feat_scores(self, src_sents):
        src_sents_var = to_input_variable(src_sents, vocab=self.src_vocab, cuda=self.cuda, training=self.training)
        src_token_embed = self.src_embed(src_sents_var)
        packed_src_token_embed = pack_padded_sequence(src_token_embed, [len(sent) for sent in src_sents])

        src_encodings, (last_state, last_cell) = self.src_lstm(packed_src_token_embed)
        src_encodings, _ = pad_packed_sequence(src_encodings)
        # src_encodings: (batch_size, tgt_query_len, hidden_size)
        src_encodings = src_encodings.permute(1, 0, 2)

        states = F.tanh(self.enc_to_state(src_encodings))
        if self.dropout:
            states = self.dropout_layer(states)

        tag_feat_scores = self.tag_readout(states)

        return tag_feat_scores

    def get_loss(self, src_sents, tgt_tags):
        tag_feat_scores = self.get_tag_feat_scores(src_sents)
        tgt_tag_score = self.get_tag_sequence_score(src_sents, tag_feat_scores, tgt_tags)
        partition_funcs = self.forward_algo(src_sents, tag_feat_scores)

        loss = partition_funcs - tgt_tag_score

        return loss

    def save(self, path):
        params = {
            'args': {'embed_size': self.embed_size,
                     'hidden_size': self.hidden_size,
                     'dropout': self.dropout,
                     'tags': self.tags},
            'src_vocab': self.src_vocab,
            'state_dict': self.state_dict()
        }

        torch.save(params, path)


def train(args):
    train_data = load_dataset(args.train_data)
    dev_data = load_dataset(args.dev_data)
    all_tags = sorted(set(chain.from_iterable(x[-1] for x in train_data)))

    print('Tags:', ', '.join(all_tags), file=sys.stderr)
    print('Training data: %d sentences' % len(train_data))
    src_vocab = init_vocab(train_data, cutoff=1)

    tagger = BiLSTM_CRF(embed_size=args.embed_size, hidden_size=args.hidden_size, dropout=args.dropout,
                        src_vocab=src_vocab, tags=all_tags, cuda=args.cuda)
    tagger.train()
    optimizer = torch.optim.Adam(tagger.parameters())
    epoch = 0
    iter_num = 0

    while True:
        epoch += 1
        total_loss = 0.
        for batch_examples in batch_iter(train_data, batch_size=args.batch_size, shuffle=True):
            iter_num += 1
            optimizer.zero_grad()

            src_sents = [e[0] for e in batch_examples]
            tgt_tags = [e[-1] for e in batch_examples]
            loss = tagger.get_loss(src_sents, tgt_tags)
            loss_val = loss.sum().data[0]
            total_loss += loss_val
            loss = loss.mean()

            loss.backward()
            print('[epoch %d iter %d] loss: %.4f' % (epoch, iter_num, loss.data[0]), file=sys.stderr)

            # clip gradient
            if args.clip_grad > 0.:
                grad_norm = torch.nn.utils.clip_grad_norm(tagger.parameters(), args.clip_grad)

            optimizer.step()

        print('[epoch %d] Train loss: %f' % (epoch, total_loss / len(train_data)), file=sys.stderr)
        print('begin evaluation...', file=sys.stderr)
        pred_file = os.path.join(args.save_dir, 'prediction.epoch%d.txt' % epoch)
        model_file = os.path.join(args.save_dir, 'model.epoch%d.bin' % epoch)

        eval_result = evaluate(tagger, dev_data, output_file=pred_file)
        print('[epoch %d] Dev acc: %f, prec: %f, recall %f, F1: %f' % (epoch, eval_result[0], eval_result[1], eval_result[2], eval_result[3]),
              file=sys.stderr)

        dev_loss = get_loss(tagger, dev_data)
        print('[epoch %d] Dev loss: %f' % (epoch, dev_loss), file=sys.stderr)

        tagger.save(model_file)


def decode(model, dataset):
    was_training = model.training
    model.eval()

    decode_results = []
    for example in dataset:
        src_sent = example[0]
        hyp_tags = model(src_sent)
        decode_results.append(hyp_tags)

    model.train(was_training)

    return decode_results


def get_loss(model, dataset, batch_size=32):
    was_training = model.training
    model.eval()

    total_loss = 0.
    for batch_examples in batch_iter(dataset, batch_size=batch_size, shuffle=False):
        src_sents = [e[0] for e in batch_examples]
        tgt_tags = [e[-1] for e in batch_examples]
        loss = model.get_loss(src_sents, tgt_tags)
        loss = loss.sum().data[0]
        total_loss += loss

    model.train(was_training)
    total_loss /= len(dataset)

    return total_loss


def evaluate(model, dataset, output_file='predictions.txt'):
    decode_results = decode(model, dataset)

    with open(output_file, 'w') as f:
        for example, hyp in zip(dataset, decode_results):
            tokens, pos_tags, syn_tags, ner_labels = example

            for token, pos_tag, syn_tag, ref_ner_label, hyp_ner_label in zip(tokens, pos_tags, syn_tags, ner_labels, hyp):
                f.write(' '.join([token, pos_tag, syn_tag, ref_ner_label, hyp_ner_label]) + '\n')

            f.write('\n')

    # os.system("data/conlleval < %s" % output_file)
    eval_result = scores.scores(output_file)

    return eval_result


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--seed', default=5783287, type=int, help='random seed')
    arg_parser.add_argument('--cuda', action='store_true', default=False, help='use gpu')
    arg_parser.add_argument('--clip_grad', default=5., type=float, help='clip gradients')
    arg_parser.add_argument('--batch_size', default=5, type=int, help='batch size')
    arg_parser.add_argument('--embed_size', default=256, type=int, help='embed size')
    arg_parser.add_argument('--hidden_size', default=256, type=int, help='hidden size')
    arg_parser.add_argument('--dropout', default=0., type=float, help='dropout')

    arg_parser.add_argument('--train_data', default='data/train.data', type=str, help='train_data')
    arg_parser.add_argument('--dev_data', default='data/dev.data', type=str, help='dev_data')
    arg_parser.add_argument('--save_dir', default='output', type=str, help='save dir')

    args = arg_parser.parse_args()
    os.system('mkdir -p %s' % args.save_dir)

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    np.random.seed(int(args.seed * 13 / 7))

    train(args)


