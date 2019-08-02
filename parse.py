import functools

import dynet as dy
import numpy as np

import src.trees as trees

START = "<START>"
STOP = "<STOP>"
UNK = "<UNK>"
NULL = "0"

def augment(scores, oracle_index):
    assert isinstance(scores, dy.Expression)
    shape = scores.dim()[0]
    assert len(shape) == 1
    increment = np.ones(shape)
    increment[oracle_index] = 0
    return scores + dy.inputVector(increment)

class Feedforward(object):
    def __init__(self, model, input_dim, hidden_dims, output_dim):
        self.spec = locals()
        self.spec.pop("self")
        self.spec.pop("model")

        self.model = model.add_subcollection("Feedforward")

        self.weights = []
        self.biases = []
        dims = [input_dim] + hidden_dims + [output_dim]
        for prev_dim, next_dim in zip(dims, dims[1:]):
            self.weights.append(self.model.add_parameters((next_dim, prev_dim)))
            self.biases.append(self.model.add_parameters(next_dim))

    def param_collection(self):
        return self.model

    @classmethod
    def from_spec(cls, spec, model):
        return cls(model, **spec)

    def __call__(self, x):
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            weight = dy.parameter(weight)
            bias = dy.parameter(bias)
            x = dy.affine_transform([bias, weight, x])
            if i < len(self.weights) - 1:
                x = dy.rectify(x)
        return x

class crf(object):
    def __init__(self, model, n_tags):

        #self.id_to_tag = id_to_tag
        #self.tag_to_id = {tag: id for id, tag in id_to_tag.items()}
        self.n_tags = n_tags
        self.b_id = n_tags
        self.e_id = n_tags + 1

        self.transitions = model.add_lookup_parameters((self.n_tags+2,
                                                 self.n_tags+2),
                                                name="transitions")


    # def param_collection(self):
    #     return self.model
    #
    # @classmethod
    # def from_spec(cls, spec, model):
    #     return cls(model, **spec)
    def score_sentence(self, observations, tags):
        assert len(observations) == len(tags)
        score_seq = [0]
        score = dy.scalarInput(0)
        tags = [self.b_id] + tags
        for i, obs in enumerate(observations):
            # print self.b_id
            # print self.e_id
            # print obs.value()
            # print tags
            # print self.transitions
            # print self.transitions[tags[i+1]].value()
            score = score \
                    + dy.pick(self.transitions[tags[i + 1]], tags[i])\
                    + dy.pick(obs, tags[i + 1])
            score_seq.append(score.value())
        score = score + dy.pick(self.transitions[self.e_id], tags[-1])
        return score


    def viterbi_loss(self, observations, tags):
        observations = [dy.concatenate([obs, dy.inputVector([-1e10, -1e10])], d=0) for obs in
                        observations]
        viterbi_tags, viterbi_score = self.viterbi_decoding(observations)
        if len(tags) != 0:
            if viterbi_tags != tags:
                gold_score = self.score_sentence(observations, tags)
                return (viterbi_score - gold_score), viterbi_tags
            else:
                return dy.scalarInput(0), viterbi_tags
        else:
            return dy.zeros(1), viterbi_tags



    def neg_log_loss(self, observations, tags):
        observations = [dy.concatenate([obs, dy.inputVector([-1e10, -1e10])], d=0) for obs in observations]
        gold_score = self.score_sentence(observations, tags)
        forward_score = self.forward(observations)
        return forward_score - gold_score


    def forward(self, observations):
        def log_sum_exp(scores):
            npval = scores.npvalue()
            argmax_score = np.argmax(npval)
            max_score_expr = dy.pick(scores, argmax_score)
            max_score_expr_broadcast = dy.concatenate([max_score_expr] * (self.n_tags+2))
            return max_score_expr + dy.log(
                dy.sum_dims(dy.transpose(dy.exp(scores - max_score_expr_broadcast)), [1]))

        init_alphas = [-1e10] * (self.n_tags + 2)
        init_alphas[self.b_id] = 0
        for_expr = dy.inputVector(init_alphas)
        for idx, obs in enumerate(observations):
            # print "obs: ", obs.value()
            alphas_t = []
            for next_tag in range(self.n_tags+2):
                obs_broadcast = dy.concatenate([dy.pick(obs, next_tag)] * (self.n_tags + 2))
                # print "for_expr: ", for_expr.value()
                # print "transitions next_tag: ", self.transitions[next_tag].value()
                # print "obs_broadcast: ", obs_broadcast.value()

                next_tag_expr = for_expr + self.transitions[next_tag] + obs_broadcast
                alphas_t.append(log_sum_exp(next_tag_expr))
            for_expr = dy.concatenate(alphas_t)
        terminal_expr = for_expr + self.transitions[self.e_id]
        alpha = log_sum_exp(terminal_expr)
        return alpha


    def viterbi_decoding(self, observations):
        backpointers = []
        init_vvars = [-1e10] * (self.n_tags + 2)
        init_vvars[self.b_id] = 0  # <Start> has all the probability
        for_expr = dy.inputVector(init_vvars)
        trans_exprs = [self.transitions[idx] for idx in range(self.n_tags + 2)]
        for obs in observations:
            bptrs_t = []
            vvars_t = []
            for next_tag in range(self.n_tags + 2):
                next_tag_expr = for_expr + trans_exprs[next_tag]
                next_tag_arr = next_tag_expr.npvalue()
                best_tag_id = np.argmax(next_tag_arr)
                bptrs_t.append(best_tag_id)
                vvars_t.append(dy.pick(next_tag_expr, best_tag_id))
            for_expr = dy.concatenate(vvars_t) + obs
            backpointers.append(bptrs_t)
        # Perform final transition to terminal
        terminal_expr = for_expr + trans_exprs[self.e_id]
        terminal_arr = terminal_expr.npvalue()
        best_tag_id = np.argmax(terminal_arr)
        path_score = dy.pick(terminal_expr, best_tag_id)
        # Reverse over the backpointers to get the best path
        best_path = [best_tag_id]  # Start with the tag that was best for terminal
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        start = best_path.pop()  # Remove the start symbol
        best_path.reverse()
        assert start == self.b_id
        # Return best path and best path's score
        return best_path, path_score

    def __call__(self, x, tags):
        #x = [dy.concatenate([obs, dy.inputVector([-1e10, -1e10])], d=0) for obs in x]
        #best_path, path_score = self.viterbi_decoding(x)
        loss, viterbi_tags = self.viterbi_loss(x, tags)

        return viterbi_tags, loss


class TopDownParser(object):
    def __init__(
            self,
            model,
            tag_vocab,
            word_vocab,
            label_vocab,
            tag_embedding_dim,
            word_embedding_dim,
            lstm_layers,
            lstm_dim,
            label_hidden_dim,
            split_hidden_dim,
            dropout,
    ):
        self.spec = locals()
        self.spec.pop("self")
        self.spec.pop("model")

        self.model = model.add_subcollection("Parser")
        self.tag_vocab = tag_vocab
        self.word_vocab = word_vocab
        self.label_vocab = label_vocab
        self.lstm_dim = lstm_dim

        self.tag_embeddings = self.model.add_lookup_parameters(
            (tag_vocab.size, tag_embedding_dim))
        self.word_embeddings = self.model.add_lookup_parameters(
            (word_vocab.size, word_embedding_dim))

        self.lstm = dy.BiRNNBuilder(
            lstm_layers,
            tag_embedding_dim + word_embedding_dim,
            2 * lstm_dim,
            self.model,
            dy.VanillaLSTMBuilder)

        self.f_label = Feedforward(
            self.model, 2 * lstm_dim, [label_hidden_dim], label_vocab.size)
        self.f_split = Feedforward(
            self.model, 2 * lstm_dim, [split_hidden_dim], 1)

        self.dropout = dropout

    def param_collection(self):
        return self.model

    @classmethod
    def from_spec(cls, spec, model):
        return cls(model, **spec)

    def parse(self, sentence, gold=None, explore=True):
        is_train = gold is not None

        if is_train:
            self.lstm.set_dropout(self.dropout)
        else:
            self.lstm.disable_dropout()

        embeddings = []
        for tag, word in [(START, START)] + sentence + [(STOP, STOP)]:
            tag_embedding = self.tag_embeddings[self.tag_vocab.index(tag)]
            if word not in (START, STOP):
                count = self.word_vocab.count(word)
                if not count or (is_train and np.random.rand() < 1 / (1 + count)):
                    word = UNK
            word_embedding = self.word_embeddings[self.word_vocab.index(word)]
            embeddings.append(dy.concatenate([tag_embedding, word_embedding]))

        lstm_outputs = self.lstm.transduce(embeddings)

        @functools.lru_cache(maxsize=None)
        def get_span_encoding(left, right):
            forward = (
                lstm_outputs[right][:self.lstm_dim] -
                lstm_outputs[left][:self.lstm_dim])
            backward = (
                lstm_outputs[left + 1][self.lstm_dim:] -
                lstm_outputs[right + 1][self.lstm_dim:])
            return dy.concatenate([forward, backward])

        def helper(left, right):
            assert 0 <= left < right <= len(sentence)

            label_scores = self.f_label(get_span_encoding(left, right))

            if is_train:
                oracle_label = gold.oracle_label(left, right)
                oracle_label_index = self.label_vocab.index(oracle_label)
                label_scores = augment(label_scores, oracle_label_index)

            label_scores_np = label_scores.npvalue()
            argmax_label_index = int(
                label_scores_np.argmax() if right - left < len(sentence) else
                label_scores_np[1:].argmax() + 1)
            argmax_label = self.label_vocab.value(argmax_label_index)

            if is_train:
                label = argmax_label if explore else oracle_label
                label_loss = (
                    label_scores[argmax_label_index] -
                    label_scores[oracle_label_index]
                    if argmax_label != oracle_label else dy.zeros(1))
            else:
                label = argmax_label
                label_loss = label_scores[argmax_label_index]

            if right - left == 1:
                tag, word = sentence[left]
                tree = trees.LeafParseNode(left, tag, word)
                if label:
                    tree = trees.InternalParseNode(label, [tree])
                return [tree], label_loss

            left_encodings = []
            right_encodings = []
            for split in range(left + 1, right):
                left_encodings.append(get_span_encoding(left, split))
                right_encodings.append(get_span_encoding(split, right))
            left_scores = self.f_split(dy.concatenate_to_batch(left_encodings))
            right_scores = self.f_split(dy.concatenate_to_batch(right_encodings))
            split_scores = left_scores + right_scores
            split_scores = dy.reshape(split_scores, (len(left_encodings),))

            if is_train:
                oracle_splits = gold.oracle_splits(left, right)
                oracle_split = min(oracle_splits)
                oracle_split_index = oracle_split - (left + 1)
                split_scores = augment(split_scores, oracle_split_index)

            split_scores_np = split_scores.npvalue()
            argmax_split_index = int(split_scores_np.argmax())
            argmax_split = argmax_split_index + (left + 1)

            if is_train:
                split = argmax_split if explore else oracle_split
                split_loss = (
                    split_scores[argmax_split_index] -
                    split_scores[oracle_split_index]
                    if argmax_split != oracle_split else dy.zeros(1))
            else:
                split = argmax_split
                split_loss = split_scores[argmax_split_index]

            left_trees, left_loss = helper(left, split)
            right_trees, right_loss = helper(split, right)

            children = left_trees + right_trees
            if label:
                children = [trees.InternalParseNode(label, children)]

            return children, label_loss + split_loss + left_loss + right_loss

        children, loss = helper(0, len(sentence))
        assert len(children) == 1
        tree = children[0]
        if is_train and not explore:
            assert gold.convert().linearize() == tree.convert().linearize()
        return tree, loss
class BottomUpParser(object):
    def __init__(
            self,
            model,
            tag_vocab,
            word_vocab,
            label_vocab,
            tag_embedding_dim,
            word_embedding_dim,
            label_embedding_dim,
            lstm_layers,
            lstm_dim,
            label_hidden_dim,
            split_hidden_dim,
            dropout,
    ):
        self.spec = locals()
        self.spec.pop("self")
        self.spec.pop("model")

        self.model = model.add_subcollection("Parser")
        self.tag_vocab = tag_vocab
        self.word_vocab = word_vocab
        self.label_vocab = label_vocab
        self.lstm_dim = lstm_dim

        self.tag_embeddings = self.model.add_lookup_parameters(
            (tag_vocab.size, tag_embedding_dim))
        self.word_embeddings = self.model.add_lookup_parameters(
            (word_vocab.size, word_embedding_dim))
        self.label_embeddings = self.model.add_lookup_parameters(
            (label_vocab.size, label_embedding_dim))

        self.lstm = dy.BiRNNBuilder(
            lstm_layers,
            tag_embedding_dim + word_embedding_dim,
            2 * lstm_dim,
            self.model,
            dy.VanillaLSTMBuilder)

        self.lstm2 = dy.BiRNNBuilder(
            lstm_layers,
            label_embedding_dim,
            2 * lstm_dim,
            self.model,
            dy.VanillaLSTMBuilder)

        self.f_label = Feedforward(
            self.model, 2 * lstm_dim, [label_hidden_dim], label_vocab.size)
        self.f_connect = Feedforward(
            self.model, 2 * lstm_dim, [split_hidden_dim], 2)

        self.crf = crf(
             self.model, label_vocab.size)
        self.crf_connect = crf(
            self.model, 2)

        self.dropout = dropout

    def param_collection(self):
        return self.model

    @classmethod
    def from_spec(cls, spec, model):
        return cls(model, **spec)

    def parse(self, sentence, level_gold=None, connect_tree=None, gold=None, explore=True):
        is_train = gold is not None

        if is_train:
            self.lstm.set_dropout(self.dropout)
        else:
            self.lstm.disable_dropout()

        embeddings = []
        for tag, word in [(START, START)] + sentence + [(STOP, STOP)]:
            tag_embedding = self.tag_embeddings[self.tag_vocab.index(tag)]
            if word not in (START, STOP):
                count = self.word_vocab.count(word)
                if not count or (is_train and np.random.rand() < 1 / (1 + count)):
                    word = UNK
            word_embedding = self.word_embeddings[self.word_vocab.index(word)]
            embeddings.append(dy.concatenate([tag_embedding, word_embedding]))

        lstm_outputs = self.lstm.transduce(embeddings)

        @functools.lru_cache(maxsize=None)
        def get_span_encoding(left, right):
            forward = (
                lstm_outputs[right][:self.lstm_dim] -
                lstm_outputs[left][:self.lstm_dim])
            backward = (
                lstm_outputs[left + 1][self.lstm_dim:] -
                lstm_outputs[right + 1][self.lstm_dim:])
            return dy.concatenate([forward, backward])


        lstm_scores_labels = []
        lstm_scores_connect=[]

        for left in range(0, len(sentence)):
            right = left + 1
            lstm_scores_labels.append(self.f_label(get_span_encoding(left, right)))
            if right +1 < len(sentence):
                lstm_scores_connect.append(self.f_connect(get_span_encoding(left,right+1)))
            else:
                lstm_scores_connect.append(self.f_connect(get_span_encoding(left, right)))

        label_loss = 0
        first_level_label = []
        for i in range(len(lstm_scores_labels)):
            if is_train:
                oracle_label = level_gold[0][i]
                oracle_label_index = self.label_vocab.index(oracle_label)
                lstm_scores_labels[i] = augment(lstm_scores_labels[i], oracle_label_index)
                label_scores_np = lstm_scores_labels[i].npvalue()
                argmax_label_index = int(label_scores_np.argmax())
                argmax_label = self.label_vocab.value(argmax_label_index)
                label = argmax_label if explore else oracle_label
                label_loss = label_loss + (
                    lstm_scores_labels[i][argmax_label_index] -
                    lstm_scores_labels[i][oracle_label_index]
                    if argmax_label != oracle_label else dy.zeros(1))
                first_level_label.append(label)
            else:
                label_scores_np = lstm_scores_labels[i].npvalue()
                argmax_label_index = int(label_scores_np.argmax())
                argmax_label = self.label_vocab.value(argmax_label_index)
                label = argmax_label
                label_loss = lstm_scores_labels[i][argmax_label_index]
                first_level_label.append(label)
        connect_loss = 0
        first_level_connect = []
        for i in range(len(lstm_scores_connect)):
            if is_train:
                oracle_connect = connect_tree[0][i]
                lstm_scores_connect[i] = augment(lstm_scores_connect[i], oracle_connect)
                connect_scores_np = lstm_scores_connect[i].npvalue()
                argmax_connect = int(connect_scores_np.argmax())
                connect_loss = connect_loss + (
                    lstm_scores_connect[i][argmax_connect]-
                    lstm_scores_connect[i][oracle_connect]
                    if argmax_connect != oracle_connect else dy.zeros(1))
            else:
                connect_scores_np = lstm_scores_connect[i].npvalue()
                argmax_connect = int(connect_scores_np.argmax())
                connect = argmax_connect
                connect_loss = argmax_connect
                first_level_connect.append(connect)

        tree = []
        for le in range(len(sentence)):
            if first_level_label[le] != "0":
                first_str = first_level_label[le] + " (" + sentence[le][0]  + " " + sentence[le][1] + ")"
            else:
                first_str = sentence[le][0] + " " + sentence[le][1]
            tree.append(first_str)

        def Helper(next_labels, temp_connect, next_connect):
            lstm_scores_labels = []
            lstm_scores_connect = []
            span = []
            i=0
            while i < len(temp_connect):
                if temp_connect[i] ==0:
                    span.append([i, i])
                    i += 1
                else:
                    j = i
                    while i < len(temp_connect) and temp_connect[i] == 1:
                        i +=1
                    span.append([j, i])
                    i += 1

            for q in range(len(span)):
                lstm_scores_labels.append(self.f_label(get_span_encoding(span[q][0], span[q][1])))
                if q+1 < len(span):
                    lstm_scores_connect.append(self.f_connect(get_span_encoding(span[q][0], span[q+1][1])))
                else:
                    lstm_scores_connect.append(self.f_connect(get_span_encoding(span[q][0], span[q][1])))

            #以下为第二次修改内容
            loss_i_label = 0
            level_label = []
            for i in range(len(lstm_scores_labels)):
                if is_train:
                    oracle_label = next_labels[i]
                    oracle_label_index = self.label_vocab.index(oracle_label)
                    lstm_scores_labels[i] = augment(lstm_scores_labels[i], oracle_label_index)
                    label_scores_np = lstm_scores_labels[i].npvalue()
                    argmax_label_index = int(label_scores_np.argmax())
                    argmax_label = self.label_vocab.value(argmax_label_index)
                    label = argmax_label if explore else oracle_label
                    loss_i_label = loss_i_label + (
                        lstm_scores_labels[i][argmax_label_index] -
                        lstm_scores_labels[i][oracle_label_index]
                        if argmax_label != oracle_label else dy.zeros(1))
                    level_label.append(label)
                else:
                    label_scores_np = lstm_scores_labels[i].npvalue()
                    argmax_label_index = int(label_scores_np.argmax())
                    argmax_label = self.label_vocab.value(argmax_label_index)
                    label = argmax_label
                    loss_i_label = lstm_scores_labels[i][argmax_label_index]
                    level_label.append(label)
            loss_i_connect = 0
            level_connect = []
            for i in range(len(lstm_scores_connect)):
                if is_train:
                    oracle_connect = next_connect[i]
                    lstm_scores_connect[i] = augment(lstm_scores_connect[i], oracle_connect)
                    connect_scores_np = lstm_scores_connect[i].npvalue()
                    argmax_connect = int(connect_scores_np.argmax())
                    loss_i_connect = loss_i_connect + (
                        lstm_scores_connect[i][argmax_connect] -
                        lstm_scores_connect[i][oracle_connect]
                        if argmax_connect != oracle_connect else dy.zeros(1))
                else:
                    connect_scores_np = lstm_scores_connect[i].npvalue()
                    argmax_connect = int(connect_scores_np.argmax())
                    connect = argmax_label
                    loss_i_connect = lstm_scores_connect[i][argmax_connect]
                    level_connect.append(connect)
            #############################################################

            return level_label, loss_i_label, level_connect, loss_i_connect



        connect = first_level_connect[:]
        level = 0
        while (is_train and level+1 < len(connect_tree)) or (not is_train and len(connect) != 1) and (level <= len(sentence)):
            if is_train:
                temp_connect = connect_tree[level]
                temp_labels = level_gold[level+1]
                if len(temp_connect) == 1:
                    break
                next_connect = connect_tree[level+1]
            else:
                temp_connect = connect[:]
                temp_labels = []
                next_connect = []
            label, lossi_label, connect, lossi_connect = Helper(temp_labels, temp_connect, next_connect)
            label_loss = label_loss + lossi_label
            connect_loss = connect_loss + lossi_connect

            # 根据connect值与label值还原出数据集形式的parsetree
            j = 0
            l = 0
            new_tree = []
            while j < len(temp_connect):
                if is_train:
                    str = ""
                    if temp_connect[j] == 0:
                        if temp_labels[l] == "0":
                            str += tree[j]
                        else:
                            if tree[j][0] == " ":
                                str += temp_labels[l] + tree[j]
                            else:
                                str += temp_labels[l] + " (" + tree[j] +")"
                        j += 1
                        new_tree.append(str)
                        l += 1
                    else:
                        if temp_labels[l] != "0":
                            str += temp_labels[l]
                        while temp_connect[j] == 1:
                            if tree[j][0] == " ":
                                str += tree[j]
                            else:
                                str += " (" + tree[j] +")"
                            j += 1
                        if tree[j][0] == " ":
                            str += tree[j]
                        else:
                            str += " (" + tree[j] + ")"
                        j += 1
                        new_tree.append(str)
                        l += 1
                else:
                    str = ""
                    if temp_connect[j] == 0:
                        if j<len(tree):
                            if l< len(label) and label[l] == "0":
                                str += tree[j]
                            else:
                                if tree[j][0] == " ":
                                    str += label[l] + tree[j]
                                else:
                                    str += label[l] + " (" + tree[j] +")"
                            j += 1
                            new_tree.append(str)
                            l += 1
                        else:
                            break
                    else:
                        if l<len(label) and label[l] != "0":
                            str += label[l]
                        while j < len(temp_connect) and j<len(tree) and temp_connect[j] == 1:
                            if tree[j][0] == " ":
                                str += tree[j]
                            else:
                                str += " (" + tree[j] +")"
                            j += 1
                        if j<len(tree):
                            if tree[j][0] == " ":
                                str += tree[j]
                            else:
                                str += " (" + tree[j] + ")"
                            j += 1
                            new_tree.append(str)
                            l += 1
                        else:
                            break

            tree = new_tree[:]
            level +=1

        tree_str="("
        for part in tree:
            tree_str = tree_str + part
        tree_str = tree_str + ")"

        return tree_str,label_loss,connect_loss



class ChartParser(object):
    def __init__(
            self,
            model,
            tag_vocab,
            word_vocab,
            label_vocab,
            tag_embedding_dim,
            word_embedding_dim,
            lstm_layers,
            lstm_dim,
            label_hidden_dim,
            dropout,
    ):
        self.spec = locals()
        self.spec.pop("self")
        self.spec.pop("model")

        self.model = model.add_subcollection("Parser")
        self.tag_vocab = tag_vocab
        self.word_vocab = word_vocab
        self.label_vocab = label_vocab
        self.lstm_dim = lstm_dim

        self.tag_embeddings = self.model.add_lookup_parameters(
            (tag_vocab.size, tag_embedding_dim))
        self.word_embeddings = self.model.add_lookup_parameters(
            (word_vocab.size, word_embedding_dim))

        self.lstm = dy.BiRNNBuilder(
            lstm_layers,
            tag_embedding_dim + word_embedding_dim,
            2 * lstm_dim,
            self.model,
            dy.VanillaLSTMBuilder)

        self.f_label = Feedforward(
            self.model, 2 * lstm_dim, [label_hidden_dim], label_vocab.size - 1)

        self.dropout = dropout

    def param_collection(self):
        return self.model

    @classmethod
    def from_spec(cls, spec, model):
        return cls(model, **spec)

    def parse(self, sentence, gold=None):
        is_train = gold is not None

        if is_train:
            self.lstm.set_dropout(self.dropout)
        else:
            self.lstm.disable_dropout()

        embeddings = []
        for tag, word in [(START, START)] + sentence + [(STOP, STOP)]:
            tag_embedding = self.tag_embeddings[self.tag_vocab.index(tag)]
            if word not in (START, STOP):
                count = self.word_vocab.count(word)
                if not count or (is_train and np.random.rand() < 1 / (1 + count)):
                    word = UNK
            word_embedding = self.word_embeddings[self.word_vocab.index(word)]
            embeddings.append(dy.concatenate([tag_embedding, word_embedding]))

        lstm_outputs = self.lstm.transduce(embeddings)

        @functools.lru_cache(maxsize=None)
        def get_span_encoding(left, right):
            forward = (
                lstm_outputs[right][:self.lstm_dim] -
                lstm_outputs[left][:self.lstm_dim])
            backward = (
                lstm_outputs[left + 1][self.lstm_dim:] -
                lstm_outputs[right + 1][self.lstm_dim:])
            return dy.concatenate([forward, backward])

        @functools.lru_cache(maxsize=None)
        def get_label_scores(left, right):
            non_empty_label_scores = self.f_label(get_span_encoding(left, right))
            return dy.concatenate([dy.zeros(1), non_empty_label_scores])

        def helper(force_gold):
            if force_gold:
                assert is_train

            chart = {}

            for length in range(1, len(sentence) + 1):
                for left in range(0, len(sentence) + 1 - length):
                    right = left + length

                    label_scores = get_label_scores(left, right)

                    if is_train:
                        oracle_label = gold.oracle_label(left, right)
                        oracle_label_index = self.label_vocab.index(oracle_label)

                    if force_gold:
                        label = oracle_label
                        label_score = label_scores[oracle_label_index]
                    else:
                        if is_train:
                            label_scores = augment(label_scores, oracle_label_index)
                        label_scores_np = label_scores.npvalue()
                        argmax_label_index = int(
                            label_scores_np.argmax() if length < len(sentence) else
                            label_scores_np[1:].argmax() + 1)
                        argmax_label = self.label_vocab.value(argmax_label_index)
                        label = argmax_label
                        label_score = label_scores[argmax_label_index]

                    if length == 1:
                        tag, word = sentence[left]
                        tree = trees.LeafParseNode(left, tag, word)
                        if label:
                            tree = trees.InternalParseNode(label, [tree])
                        chart[left, right] = [tree], label_score
                        continue

                    if force_gold:
                        oracle_splits = gold.oracle_splits(left, right)
                        oracle_split = min(oracle_splits)
                        best_split = oracle_split
                    else:
                        best_split = max(
                            range(left + 1, right),
                            key=lambda split:
                                chart[left, split][1].value() +
                                chart[split, right][1].value())

                    left_trees, left_score = chart[left, best_split]
                    right_trees, right_score = chart[best_split, right]

                    children = left_trees + right_trees
                    if label:
                        children = [trees.InternalParseNode(label, children)]

                    chart[left, right] = (
                        children, label_score + left_score + right_score)

            children, score = chart[0, len(sentence)]
            assert len(children) == 1
            return children[0], score

        tree, score = helper(False)
        if is_train:
            oracle_tree, oracle_score = helper(True)
            assert oracle_tree.convert().linearize() == gold.convert().linearize()
            correct = tree.convert().linearize() == gold.convert().linearize()
            loss = dy.zeros(1) if correct else score - oracle_score
            return tree, loss
        else:
            return tree, score
