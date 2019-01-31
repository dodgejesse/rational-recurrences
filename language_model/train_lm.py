import argparse
import time
import math
import sys
import numpy as np
import torch.nn as nn
from torch.optim import SGD
from torch.autograd import Variable

sys.path.append("..")
import rrnn
from semiring import *
SOS, EOS = "<s>", "</s>"


def read_corpus(path, sos=None, eos="</s>", shuffle=False):
    data = [ ]
    if sos is not None:
        data = [sos]
    with open(path) as fin:
        lines = [line.split() + [ eos ] for line in fin]
    if shuffle:
        np.random.shuffle(lines)
    for line in lines:
        data += line
    return data


def create_batches(data_text, map_to_ids, batch_size, cuda=False):
    data_ids = map_to_ids(data_text)
    N = len(data_ids)
    L = ((N-1) // batch_size) * batch_size
    x = np.copy(data_ids[:L].reshape(batch_size,-1).T)
    y = np.copy(data_ids[1:L+1].reshape(batch_size,-1).T)
    x, y = torch.from_numpy(x), torch.from_numpy(y)
    x, y = x.contiguous(), y.contiguous()
    if cuda:
        x, y = x.cuda(), y.cuda()
    return x, y


class EmbeddingLayer(nn.Module):
    def __init__(self, n_d, words, sos=SOS, fix_emb=False):
        super(EmbeddingLayer, self).__init__()
        word2id, id2word = {}, {}
        if sos not in word2id:
            word2id[sos] = len(word2id)
            id2word[word2id[sos]] = sos
        for w in words:
            if w not in word2id:
                word2id[w] = len(word2id)
                id2word[word2id[w]] =w
        self.word2id, self.id2word = word2id, id2word
        self.n_V, self.n_d = len(word2id), n_d
        self.embedding = nn.Embedding(self.n_V, n_d)
        self.sosid = word2id[sos]

    def forward(self, x):
        return self.embedding(x)

    def map_to_ids(self, text):
        return np.asarray([self.word2id[x] for x in text],
                 dtype="int64")

    def map_to_tokens(self, ids):
        return [self.id2word[x] for x in ids.cpu().numpy()]


class Model(nn.Module):
    def __init__(self, words, args):
        super(Model, self).__init__()
        self.args = args

        self.n_d = args.d
        self.depth = args.depth
        self.drop = nn.Dropout(args.dropout)
        self.input_drop = nn.Dropout(args.input_dropout)
        self.output_drop = nn.Dropout(args.output_dropout)
        self.embedding_layer = EmbeddingLayer(self.n_d, words)
        self.n_V = self.embedding_layer.n_V
        self.num_mlp_layer = args.num_mlp_layer

        use_tanh, use_relu, use_selu = 0, 0, 0
        if args.activation == "tanh":
            use_tanh = 1
        elif args.activation == "relu":
            use_relu = 1
        elif args.activation == "selu":
            use_selu = 1
        else:
            assert args.activation == "none"

        if args.model == "lstm":
            self.rnn=nn.LSTM(
                self.n_d, self.n_d,
                self.depth,
                dropout = args.rnn_dropout
            )
        elif args.model == "rrnn":
            if args.semiring == "plus_times":
                self.semiring = PlusTimesSemiring
            elif args.semiring == "max_plus":
                self.semiring = MaxPlusSemiring
            else:
                assert False, "Semiring should either be plus_times or max_plus, not {}".format(args.semiring)
            self.rnn = rrnn.RRNN(
                self.semiring,
                self.n_d,
                args.hidden_size,
                self.depth,
                pattern=args.pattern,
                dropout=args.dropout,
                rnn_dropout=args.rnn_dropout,
                use_tanh=use_tanh,
                use_relu=use_relu,
                use_selu=use_selu,
                layer_norm=args.use_layer_norm,
                use_output_gate=args.use_output_gate
            )
        else:
            assert False
        if args.num_mlp_layer == 2:
            self.hidden = nn.Linear(self.n_d*self.bidir, self.n_d)
        elif args.num_mlp_layer == 1:
            pass
        else:
            assert False
        self.output_layer = nn.Linear(self.n_d, self.n_V)
        # tie weights
        self.output_layer.weight = self.embedding_layer.embedding.weight

        self.init_weights()
        if args.model != "lstm":
           self.rnn.init_weights()


    def init_input(self, batch_size):
        args = self.args
        sosid = self.embedding_layer.sosid
        init_input = torch.from_numpy(np.array(
            [sosid, sosid, sosid, sosid] * batch_size, dtype=np.int64).reshape(4, batch_size))

        if args.gpu:
            init_input = init_input.cuda()
        return Variable(init_input)


    def init_weights(self):
        val_range = (3.0/self.n_d)**0.5
        for p in self.parameters():
            if p.dim() > 1:  # matrix
                p.data.uniform_(-val_range, val_range)
            else:
                p.data.zero_()


    def forward(self, x, init):
        emb = self.input_drop(self.embedding_layer(x))
        output, hidden = self.rnn(emb, init)

        if self.num_mlp_layer == 2:
            output = self.drop(output)
            output = output.view(-1, output.size(2))
            output = self.hidden(output).tanh()
            output = self.output_drop(output)
        elif self.num_mlp_layer == 1:
            output = self.output_drop(output)
            output = output.view(-1, output.size(2))
        output = self.output_layer(output)
        return output, hidden


    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        if self.args.model == "lstm":
            return (Variable(weight.new(self.depth, batch_size, self.n_d).zero_()),
                    Variable(weight.new(self.depth, batch_size, self.n_d).zero_()))
        elif self.args.model == "rrnn":
            init_input = self.init_input(batch_size)
            emb = self.input_drop(self.embedding_layer(init_input))
            output, hidden = self.rnn(emb, None)
            return hidden
        else:
            assert False


    def compute_loss(self, emb, y):
        batch_size = 1
        hidden = self.init_hidden(batch_size)
        hidden = repackage_hidden(self.args, hidden)
        output, hidden = self.rnn(emb, hidden)
        output = output.view(-1, output.size(2))
        output = self.output_layer(output)

        criterion = nn.CrossEntropyLoss(size_average=False)
        loss = criterion(output, y) / emb.size(1)
        print (loss)
        loss.backward()
        return loss


    def print_pnorm(self):
        norms = [ "{:.0f}".format(x.norm().data[0]) for x in self.parameters() ]
        sys.stdout.write("\tp_norm: {}\n".format(
            norms
        ))


def repackage_hidden(args, hidden):
    if args.model == "lstm":
        return (Variable(hidden[0].data), Variable(hidden[1].data))
    elif args.model == "rrnn":
        if args.pattern == "bigram":
            return (Variable(hidden[0].data), Variable(hidden[1].data))
        elif args.pattern == "unigram":
            return Variable(hidden.data)
        else:
            for i, hss in enumerate(hidden):
                for j, hs in enumerate(hss):
                    for k, h in enumerate(hs):
                        hidden[i][j][k] = Variable(hidden[i][j][k].data)
            return hidden
    else:
        assert False


def train_model(model):
    args = model.args
    unchanged, best_dev = 0, 200

    unroll_size = args.unroll_size
    batch_size = args.batch_size
    criterion = nn.CrossEntropyLoss(size_average=False)

    trainer = SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    map_to_ids = model.embedding_layer.map_to_ids
    train = read_corpus(args.train, shuffle=False)
    train = create_batches(train, map_to_ids, args.batch_size, cuda=args.gpu)
    dev = read_corpus(args.dev)
    dev = create_batches(dev, map_to_ids, 1, cuda=args.gpu)
    test = read_corpus(args.test)
    test = create_batches(test, map_to_ids, 1, cuda=args.gpu)
    for epoch in range(args.max_epoch):

        start_time = time.time()

        N = (len(train[0]) - 1) // unroll_size + 1
        hidden = model.init_hidden(batch_size)
        total_loss, cur_loss = 0.0, 0.0
        for i in range(N):
            model.train()
            x = train[0][i*unroll_size:(i+1)*unroll_size]
            y = train[1][i*unroll_size:(i+1)*unroll_size].view(-1)

            x, y = Variable(x), Variable(y)
            model.zero_grad()
            output, hidden = model(x, hidden)
            hidden = repackage_hidden(args, hidden)
            assert x.size(1) == batch_size
            loss = criterion(output, y) / x.size(1)
            loss.backward()
            if math.isnan(loss.data[0]) or math.isinf(loss.data[0]):
                print ("nan/inf loss encoutered in training.")
                sys.exit(0)
                return
            total_loss += loss.data[0] / x.size(0)
            cur_loss += loss.data[0] / x.size(0)
            torch.nn.utils.clip_grad_norm(model.parameters(), args.clip_grad)
            for p in model.parameters():
                if not p.requires_grad:
                    continue
                if p.grad is not None:
                    if args.weight_decay > 0:
                        p.data.mul_(1.0 - args.weight_decay)
                    p.data.add_(-args.lr, p.grad.data)

            if (i + 1) % args.eval_ite == 0:
                dev_ppl = eval_model(model, dev)
                sys.stdout.write("| Epoch={} | ite={} | lr={:.4f} | train_ppl={:.2f} | dev_ppl={:.2f} |"
                                 "\n".format(
                    epoch,
                    i+1,
                    trainer.defaults["lr"],
                    np.exp(cur_loss / args.eval_ite),
                    dev_ppl
                ))
                model.print_pnorm()
                sys.stdout.flush()
                cur_loss = 0.0

                if dev_ppl < best_dev:
                    unchanged = 0
                    best_dev = dev_ppl
                    test_ppl = eval_model(model, test)
                    sys.stdout.write("\t[eval]  test_ppl={:.2f}\n".format(
                        test_ppl
                    ))
                    sys.stdout.flush()

        train_ppl = np.exp(total_loss/N)
        dev_ppl = eval_model(model, dev)

        sys.stdout.write("-" * 89 + "\n")
        sys.stdout.write("| End of epoch {} | lr={:.4f} | train_ppl={:.2f} | dev_ppl={:.2f} |"
                         "[{:.2f}m] |\n".format(
            epoch,
            trainer.defaults["lr"],
            train_ppl,
            dev_ppl,
            (time.time() - start_time) / 60.0
        ))
        sys.stdout.write("-" * 89 + "\n")
        model.print_pnorm()
        sys.stdout.flush()

        if dev_ppl < best_dev:
            unchanged = 0
            best_dev = dev_ppl
            start_time = time.time()
            test_ppl = eval_model(model, test)
            sys.stdout.write("\t[eval]  test_ppl={:.2f}\t[{:.2f}m]\n".format(
                test_ppl,
                (time.time() - start_time) / 60.0
            ))
            sys.stdout.flush()
        else:
            unchanged += 1
        if args.lr_decay_epoch > 0 and epoch >= args.lr_decay_epoch:
            args.lr *= args.lr_decay
        if unchanged >= args.patience:
            sys.stdout.write("Reached " + str(args.patience)
                             + " iterations without improving dev loss. Reducing learning rate.")
            args.lr /= 2
            unchanged = 0
        trainer.defaults["lr"] = args.lr
        sys.stdout.write("\n")
    return


def eval_model(model, valid):
    model.eval()
    args = model.args
    total_loss = 0.0
    unroll_size = model.args.unroll_size
    criterion = nn.CrossEntropyLoss(size_average=False)
    hidden = model.init_hidden(1)
    N = (len(valid[0])-1)//unroll_size + 1
    for i in range(N):
        x = valid[0][i*unroll_size:(i+1)*unroll_size]
        y = valid[1][i*unroll_size:(i+1)*unroll_size].view(-1)
        x, y = Variable(x, volatile=True), Variable(y)
        output, hidden = model(x, hidden)
        hidden = repackage_hidden(args, hidden)
        loss = criterion(output, y)
        if math.isnan(loss.data[0]) or math.isinf(loss.data[0]):
            print("nan/inf loss encoutered in dev.")
            sys.exit(0)
            return
        total_loss += loss.data[0]
    avg_loss = total_loss / valid[1].numel()
    ppl = np.exp(avg_loss)
    return ppl


def get_states_weights(model, args):
    embed_dim = model.emb_layer.n_d
    num_edges_in_wfsa = model.encoder.rnn_lst[0].cells[0].k
    num_wfsas = int(args.d_out)

    reshaped_weights = model.encoder.rnn_lst[0].cells[0].weight.view(embed_dim, num_wfsas, num_edges_in_wfsa)
    if len(model.encoder.rnn_lst) > 1:
        reshaped_second_layer_weights = model.encoder.rnn_lst[1].cells[0].weight.view(num_wfsas, num_wfsas,
                                                                                      num_edges_in_wfsa)
        reshaped_weights = torch.cat((reshaped_weights, reshaped_second_layer_weights), 0)
    elif len(model.encoder.rnn_lst) > 2:
        assert False, "This regularization is only implemented for 2-layer networks."

    # to stack the transition and self-loops, so e.g. states[...,0] contains the transition and self-loop weights

    states = torch.cat((reshaped_weights[..., 0:int(num_edges_in_wfsa / 2)],
                        reshaped_weights[..., int(num_edges_in_wfsa / 2):num_edges_in_wfsa]), 0)
    return states


# this computes the group lasso penalty term
def get_regularization_groups(model, args):
    if args.sparsity_type == "wfsa":
        embed_dim = model.emb_layer.n_d
        num_edges_in_wfsa = model.encoder.rnn_lst[0].k
        reshaped_weights = model.encoder.rnn_lst[0].weight.view(embed_dim, args.d_out, num_edges_in_wfsa)
        l2_norm = reshaped_weights.norm(2, dim=0).norm(2, dim=1)
        return l2_norm
    elif args.sparsity_type == 'edges':
        return model.encoder.rnn_lst[0].weight.norm(2, dim=0)
    elif args.sparsity_type == 'states':
        states = get_states_weights(model, args)
        return states.norm(2, dim=0)  # a num_wfsa by n-gram matrix
    elif args.sparsity_type == "rho_entropy":
        assert args.depth == 1, "rho_entropy regularization currently implemented for single layer networks"
        bidirectional = model.encoder.rnn_lst[0].cells[0].bidirectional
        assert not bidirectional, "bidirectional not implemented"

        num_edges_in_wfsa = model.encoder.rnn_lst[0].cells[0].k
        num_wfsas = int(args.d_out)
        bias_final = model.encoder.rnn_lst[0].cells[0].bias_final

        sm = nn.Softmax(dim=2)
        # the 1 in the line below is for non-bidirectional models, would be 2 for bidirectional
        rho = sm(bias_final.view(1, num_wfsas, int(num_edges_in_wfsa / 2)))
        entropy_to_sum = rho * rho.log() * -1
        entropy = entropy_to_sum.sum(dim=2)
        return entropy


def log_groups(model, args, logging_file, groups=None):
    if groups is not None:

        if args.sparsity_type == "rho_entropy":
            num_edges_in_wfsa = model.encoder.rnn_lst[0].cells[0].k
            num_wfsas = int(args.d_out)
            bias_final = model.encoder.rnn_lst[0].cells[0].bias_final

            sm = nn.Softmax(dim=2)
            # the 1 in the line below is for non-bidirectional models, would be 2 for bidirectional
            rho = sm(bias_final.view(1, num_wfsas, int(num_edges_in_wfsa / 2)))
            logging_file.write(str(rho))

        else:
            logging_file.write(str(groups))
    else:
        if args.sparsity_type == "wfsa":
            embed_dim = model.emb_layer.n_d
            num_edges_in_wfsa = model.encoder.rnn_lst[0].k
            reshaped_weights = model.encoder.rnn_lst[0].weight.view(embed_dim, args.d_out, num_edges_in_wfsa)
            l2_norm = reshaped_weights.norm(2, dim=0).norm(2, dim=1)
            logging_file.write(str(l2_norm))

        elif args.sparsity_type == 'edges':
            embed_dim = model.emb_layer.n_d
            num_edges_in_wfsa = model.encoder.rnn_lst[0].k
            reshaped_weights = model.encoder.rnn_lst[0].weight.view(embed_dim, args.d_out, num_edges_in_wfsa)
            logging_file.write(str(reshaped_weights.norm(2, dim=0)))
            # model.encoder.rnn_lst[0].weight.norm(2, dim=0)
        elif args.sparsity_type == 'states':
            assert False, "can implement this based on get_regularization_groups, but that keeps changing"
            logging_file.write(str(states.norm(2, dim=0)))  # a num_wfsa by n-gram matrix


def init_logging(args):
    dir_path = "/home/jessedd/projects/rational-recurrences/classification/logging/" + args.dataset + "/"
    filename = args.filename() + ".txt"

    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    if not os.path.exists(dir_path + args.filename_prefix):
        os.mkdir(dir_path + args.filename_prefix)

    torch.set_printoptions(threshold=5000)

    logging_file = open(dir_path + filename, "w")

    tmp = args.loaded_embedding
    args.loaded_embedding = True
    logging_file.write(str(args))
    args.loaded_embedding = tmp

    # print(args)
    print("saving in {}".format(args.dataset + args.filename()))
    return logging_file


def regularization_stop(args, model):
    if args.sparsity_type == "states" and args.prox_step:
        states = get_states_weights(model, args)
        if states.norm(2, dim=0).sum().data[0] == 0:
            return True
    else:
        return False


# following https://en.wikipedia.org/wiki/Proximal_gradient_methods_for_learning#Group_lasso
# w_g - args.reg_strength * (w_g / ||w_g||_2)
def prox_step(model, args):
    if args.sparsity_type == "states":

        states = get_states_weights(model, args)
        num_states = states.shape[2]

        embed_dim = model.emb_layer.n_d
        num_edges_in_wfsa = model.encoder.rnn_lst[0].cells[0].k
        num_wfsas = int(args.d_out)

        reshaped_weights = model.encoder.rnn_lst[0].cells[0].weight.view(embed_dim, num_wfsas, num_edges_in_wfsa)
        if len(model.encoder.rnn_lst) > 1:
            assert False, "This regularization is only implemented for 2-layer networks."

        first_weights = reshaped_weights[..., 0:int(num_edges_in_wfsa / 2)]
        second_weights = reshaped_weights[..., int(num_edges_in_wfsa / 2):num_edges_in_wfsa]

        for i in range(num_wfsas):
            for j in range(num_states):
                cur_group = states[:, i, j].data
                cur_first_weights = first_weights[:, i, j].data
                cur_second_weights = second_weights[:, i, j].data
                if cur_group.norm(2) < args.reg_strength:
                    # cur_group.add_(-cur_group)
                    cur_first_weights.add_(-cur_first_weights)
                    cur_second_weights.add_(-cur_second_weights)
                else:
                    # cur_group.add_(-args.reg_strength*cur_group/cur_group.norm(2))
                    cur_first_weights.add_(-args.reg_strength * cur_first_weights / cur_group.norm(2))
                    cur_second_weights.add_(-args.reg_strength * cur_second_weights / cur_group.norm(2))
    else:
        assert False, "haven't implemented anything else"


def main(args):
    torch.manual_seed(args.seed)
    train = read_corpus(args.train, shuffle=False)
    model = Model(train, args)
    if args.gpu:
        model.cuda()
    sys.stdout.write("vocab size: {}\n".format(
        model.embedding_layer.n_V
    ))
    num_params = sum(x.numel() for x in model.parameters() if x.requires_grad)
    # if args.model == "rrnn":
        # num_in = args.depth * (2 * args.d)
        # num_params = num_params - num_in
    
    sys.stdout.write("num of parameters: {}\n".format(num_params))
    sys.stdout.flush()
    model.print_pnorm()
    sys.stdout.write("\n")

    train_model(model)
    return


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0], conflict_handler="resolve")
    argparser.add_argument("--seed", type=int, default=31415)
    argparser.add_argument("--model", type=str, default="rrnn")
    argparser.add_argument("--semiring", type=str, default="plus_times")
    argparser.add_argument("--pattern", type=str, default="1-gram,2-gram,3-gram,4-gram")
    argparser.add_argument("--use_rho", type=str2bool, default=False)
    argparser.add_argument("--use_layer_norm", type=str2bool, default=False)
    argparser.add_argument("--use_output_gate", type=str2bool, default=True)
    argparser.add_argument("--activation", type=str, default="tanh")
    argparser.add_argument("--train", type=str, required=True, help="training file")
    argparser.add_argument("--dev", type=str, required=True, help="dev file")
    argparser.add_argument("--test", type=str, required=True, help="test file")
    argparser.add_argument("--batch_size", "--batch", type=int, default=32)
    argparser.add_argument("--unroll_size", type=int, default=35)
    argparser.add_argument("--max_epoch", type=int, default=300)
    argparser.add_argument("--d", type=int, default=32)
    argparser.add_argument("--hidden_size", type=str, default="8,8,8,8")
    argparser.add_argument("--input_dropout", type=float, default=0.0,
        help="dropout of word embeddings")
    argparser.add_argument("--output_dropout", type=float, default=0.0,
        help="dropout of softmax output")
    argparser.add_argument("--dropout", type=float, default=0.0,
        help="dropout intra RNN layers")
    argparser.add_argument("--rnn_dropout", type=float, default=0.0,
        help="dropout of RNN layers")
    argparser.add_argument("--depth", type=int, default=2)
    argparser.add_argument("--num_mlp_layer", type=int, default=1)
    argparser.add_argument("--lr", type=float, default=1.0)
    argparser.add_argument("--lr_decay", type=float, default=0.98)
    argparser.add_argument("--lr_decay_epoch", type=int, default=0)
    argparser.add_argument("--weight_decay", type=float, default=1e-6)
    argparser.add_argument("--clip_grad", type=float, default=5.)
    argparser.add_argument("--gpu", type=str2bool, default=False)
    argparser.add_argument("--eval_ite", type=int, default=100)
    argparser.add_argument("--patience", type=int, default=30)

    args = argparser.parse_args()
    print(args)
    sys.stdout.flush()
    main(args)
