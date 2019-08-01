import argparse
import itertools
import os.path
import time
import re

import dynet as dy
import numpy as np

import src.evaluate as evaluate
import src.parse as parse
import src.trees as trees
import src.vocabulary as vocabulary

def format_elapsed(start_time):
    elapsed_time = int(time.time() - start_time)
    minutes, seconds = divmod(elapsed_time, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    elapsed_string = "{}h{:02}m{:02}s".format(hours, minutes, seconds)
    if days > 0:
        elapsed_string = "{}d{}".format(days, elapsed_string)
    return elapsed_string

def run_train(args):
    if args.numpy_seed is not None:
        print("Setting numpy random seed to {}...".format(args.numpy_seed))
        np.random.seed(args.numpy_seed)

    print("Loading training trees from {}...".format(args.train_path))
    train_treebank = trees.load_trees(args.train_path)
    print("Loaded {:,} training examples.".format(len(train_treebank)))

    print("Loading development trees from {}...".format(args.dev_path))
    dev_treebank = trees.load_trees(args.dev_path)
    print("Loaded {:,} development examples.".format(len(dev_treebank)))

    print("Processing trees for training...")
    train_parse = [tree.convert() for tree in train_treebank]

    print("Constructing vocabularies...")

    tag_vocab = vocabulary.Vocabulary()
    tag_vocab.index(parse.START)
    tag_vocab.index(parse.STOP)

    word_vocab = vocabulary.Vocabulary()
    word_vocab.index(parse.START)
    word_vocab.index(parse.STOP)
    word_vocab.index(parse.UNK)

    label_vocab = vocabulary.Vocabulary()
    label_vocab.index(parse.NULL)

    if args.parser_type != "bottom-up":
        label_vocab.index(())

    for tree in train_parse:
        nodes = [tree]
        while nodes:
            node = nodes.pop()
            if isinstance(node, trees.InternalParseNode):
                if args.parser_type == "bottom-up":
                    label_vocab.index(node.label[0])
                else:
                    label_vocab.index(node.label)
                nodes.extend(reversed(node.children))
            else:
                tag_vocab.index(node.tag)
                word_vocab.index(node.word)

    tag_vocab.freeze()
    word_vocab.freeze()
    label_vocab.freeze()

    def print_vocabulary(name, vocab):
        special = {parse.START, parse.STOP, parse.UNK}
        print("{} ({:,}): {}".format(
            name, vocab.size,
            sorted(value for value in vocab.values if value in special)
            +sorted(value for value in vocab.values if value not in special)))

    if args.print_vocabs:
        print_vocabulary("Tag", tag_vocab)
        print_vocabulary("Word", word_vocab)
        print_vocabulary("Label", label_vocab)

    print("Initializing model...")
    model = dy.ParameterCollection()
    if args.parser_type == "top-down":
        parser = parse.TopDownParser(
            model,
            tag_vocab,
            word_vocab,
            label_vocab,
            args.tag_embedding_dim,
            args.word_embedding_dim,
            args.lstm_layers,
            args.lstm_dim,
            args.label_hidden_dim,
            args.split_hidden_dim,
            args.dropout,
        )
    elif args.parser_type == "bottom-up":
        parser = parse.BottomUpParser(
            model,
            tag_vocab,
            word_vocab,
            label_vocab,
            args.tag_embedding_dim,
            args.word_embedding_dim,
            args.label_embedding_dim,
            args.lstm_layers,
            args.lstm_dim,
            args.label_hidden_dim,
            args.split_hidden_dim,
            args.dropout,
        )
    else:
        parser = parse.ChartParser(
            model,
            tag_vocab,
            word_vocab,
            label_vocab,
            args.tag_embedding_dim,
            args.word_embedding_dim,
            args.lstm_layers,
            args.lstm_dim,
            args.label_hidden_dim,
            args.dropout,
        )
    trainer = dy.AdamTrainer(model)

    total_processed = 0
    current_processed = 0
    check_every = len(train_parse) / args.checks_per_epoch
    best_dev_fscore = -np.inf
    best_dev_model_path = None

    start_time = time.time()

    def check_dev():
        nonlocal best_dev_fscore
        nonlocal best_dev_model_path

        dev_start_time = time.time()

        dev_predicted = []
        for tree in dev_treebank:
            dy.renew_cg()
            sentence = [(leaf.tag, leaf.word) for leaf in tree.leaves()]
            predicted, _, _ = parser.parse(sentence)
            if args.parser_type == "bottom-up":
                dev_predicted.append(predicted)
            else:
                dev_predicted.append(predicted.convert())

        dev_fscore = evaluate.evalb(args.evalb_dir, dev_treebank, dev_predicted, args.parser_type)

        print(
            "dev-fscore {} "
            "dev-elapsed {} "
            "total-elapsed {}".format(
                dev_fscore,
                format_elapsed(dev_start_time),
                format_elapsed(start_time),
            )
        )

        if dev_fscore.fscore > best_dev_fscore:
            if best_dev_model_path is not None:
                for ext in [".data", ".meta"]:
                    path = best_dev_model_path + ext
                    if os.path.exists(path):
                        print("Removing previous model file {}...".format(path))
                        os.remove(path)

            best_dev_fscore = dev_fscore.fscore
            best_dev_model_path = "{}_dev={:.2f}".format(
                args.model_path_base, dev_fscore.fscore)
            print("Saving new best model to {}...".format(best_dev_model_path))
            dy.save(best_dev_model_path, [parser])

    def convert_to_level_trees(words, tags, tree):
        len_sen = len(words)
        level_tree = []
        words_level = []
        star = []
        symbols = ['#', '$', "''", ',', '.', ':', '``', "-RRB-", "-LRB-", "CD", "IN", "TO"]
        tree = tree.replace("(", " ( ").replace(")", " ) ").split()

        def cal_level(word, num):
            level = 0
            word_arr = []
            word_counter = 0
            word_star = []
            for i in range(len(tree)):
                if tree[i] == "(":
                    level += 1
                elif tree[i] == ")":
                    if len(word_arr)==0:
                        print(words)
                    del word_arr[-1]
                    del word_star[-1]
                    level -= 1
                elif tree[i] == word and tree[i+1] != word and tree[i+1] != "(" and word_counter == num:
                    del word_arr[-1]
                    del word_star[-1]
                    break
                elif tree[i] not in words and tree[i] not in symbols:
                    word_arr.append(tree[i])
                    word_star.append(tree[0:i].count(tree[i]))
                elif tree[i] == "S" and tree[i+1] == "(":
                    word_arr.append(tree[i])
                    word_star.append(tree[0:i].count(tree[i]))
                elif tree[i] in symbols and tree[i+1] in words[:num+1]:
                    word_arr.append(tree[i])
                    word_star.append(tree[0:i].count(tree[i]))
                else:
                    word_counter += 1

            return level, word_arr, word_star

        max_level = 0
        for i in range(len_sen):
            tempword_level, tempword_arr, tempword_star = cal_level(words[i], i)
            if len(tempword_arr)==0:
                tempword_arr.append(level_tree[0][0])
            words_level.append(tempword_level)
            star.append(tempword_star)
            level_tree.append(tempword_arr)
            if tempword_level > max_level:
                max_level = tempword_level

        return level_tree, max_level, star

    def get_connect_tree(words, level_tree, max_level, star):
        connect_tree = []
        level_labels = []
        for level in range(max_level+1):
            level_content = []
            temp_level_label = []
            last_level_label = []
            if level == 0:
                for k in range(len(words)):
                    last_level_label.append("0")
            if level!=0:
                last_level_label = level_labels[level][:]


            # 去掉单个词组成的短语部分
            temp_maxlen = get_tree_level(level_tree)
            for j in range(len(level_tree)):
                if j < len(level_tree) - 1 and len(level_tree[j]) == temp_maxlen and len(
                        level_tree[j + 1]) < temp_maxlen and len(
                        level_tree[j - 1]) < temp_maxlen:
                    if len(last_level_label)!=0 and last_level_label[j] == "0" and "*" not in level_tree[j][-1]:
                        last_level_label[j] = level_tree[j][-1]
                    del level_tree[j][-1]
                    del star[j][-1]
                if j < len(level_tree) - 1 and len(level_tree[j]) == temp_maxlen and len(
                        level_tree[j + 1]) == temp_maxlen and \
                        level_tree[j + 1][-1] != level_tree[j][-1]:
                    if len(last_level_label)!=0 and last_level_label[j] == "0" and "*" not in level_tree[j][-1]:
                        last_level_label[j] = level_tree[j][-1]
                    del level_tree[j][-1]
                    del star[j][-1]
                if j < len(level_tree) - 1 and len(level_tree[j]) == temp_maxlen and \
                        len(level_tree[j + 1]) == temp_maxlen and \
                        level_tree[j + 1][-1] == level_tree[j][-1] and \
                        star[j][-1] != star[j + 1][-1]:
                    if len(last_level_label)!=0 and last_level_label[j] == "0" and "*" not in level_tree[j][-1]:
                        last_level_label[j] = level_tree[j][-1]
                    del level_tree[j][-1]
                    del star[j][-1]
                    if len(last_level_label)!=0 and last_level_label[j+1] == "0" and "*" not in level_tree[j+1][-1]:
                        last_level_label[j+1] = level_tree[j+1][-1]
                    del level_tree[j + 1][-1]
                    del star[j + 1][-1]
                if j == len(level_tree) - 1 and len(level_tree[j]) > len(level_tree[j - 1]):
                    if len(last_level_label)!=0 and last_level_label[j] == "0" and "*" not in level_tree[j][-1]:
                        last_level_label[j] = level_tree[j][-1]
                    del level_tree[j][-1]
                    del star[j][-1]

            i = 0
            while i+1 < len(level_tree):
                if i + 1 < len(level_tree) and level_tree[i][-1] == level_tree[i+1][-1] and \
                        len(level_tree[i]) == len(level_tree[i+1]) and \
                        star[i][-1] == star[i+1][-1]:
                    count = 1
                    while i + 1 < len(level_tree) and level_tree[i][-1] == level_tree[i+1][-1] and len(level_tree[i]) == len(level_tree[i+1]) and star[i][-1] == star[i+1][-1]:
                        level_content.append(1)
                        i = i+1
                        count +=1
                    for num in range(count-1):
                        del level_tree[i]
                        del star[i]
                        i=i-1
                    if i + 1 < len(level_tree) and len(level_tree[i+1]) > len(level_tree[i]) and level_tree[i][-1] == level_tree[i+1][len(level_tree[i])-1]:
                        temp_level_label.append("0")
                    elif i - 1 >= 0 and len(level_tree[i-1]) > len(level_tree[i]) and level_tree[i][-1] == level_tree[i-1][len(level_tree[i])-1]:
                        temp_level_label.append("0")
                    else:
                        temp_level_label.append(level_tree[i][-1])
                        level_tree[i][-1] = level_tree[i][-1] + "*"
                    if i != len(level_tree) - 1:
                        level_content.append(0)
                    i=i+1
                else:
                    level_content.append(0)
                    temp_level_label.append("0")
                    i=i+1
            level_content.append(0)
            if len(level_content)>=2 and level_content[-2] != 1:
                temp_level_label.append("0")
            if len(level_content) == 1:
                temp_level_label.append("0")
            if level == 0:
                level_labels.append(last_level_label)
            if level != 0:
                level_labels[level] = last_level_label[:]
            connect_tree.append(level_content)
            level_labels.append(temp_level_label)


        return connect_tree, level_labels

    def get_tree_level(level_tree):
        level = 0
        for word in level_tree:
            if len(word) > level:
               level = len(word)
        return level

    for epoch in itertools.count(start=1):
        if args.epochs is not None and epoch > args.epochs:
            break

        np.random.shuffle(train_parse)
        epoch_start_time = time.time()

        for start_index in range(0, len(train_parse), args.batch_size):
            dy.renew_cg()
            #batch_losses = []
            batch_losses_label = []
            batch_losses_connect = []
            for tree in train_parse[start_index:start_index + args.batch_size]:
                sentence = [(leaf.tag, leaf.word) for leaf in tree.leaves()]
                if args.parser_type == "top-down":
                    _, loss = parser.parse(sentence, tree, args.explore)
                if args.parser_type == "bottom-up":
                    words = [leaf.word for leaf in tree.leaves()]
                    tags = [leaf.tag for leaf in tree.leaves()]
                    tree = tree.convert().linearize()
                    level_tree, max_level, star = convert_to_level_trees(words, tags, tree)
                    connect_tree, level_labels = get_connect_tree(words, level_tree, max_level, star)
                    _, loss_label, loss_connect = parser.parse(sentence, level_labels, connect_tree, tree, args.explore)
                else:
                    _, loss = parser.parse(sentence, tree)
                #batch_losses.append(loss)
                batch_losses_label.append(loss_label)
                batch_losses_connect.append(loss_connect)
                total_processed += 1
                current_processed += 1

            # batch_loss = dy.average(batch_losses)
            # batch_loss_value = batch_loss.scalar_value()
            # batch_loss.backward()
            # trainer.update()

            batch_loss_label = dy.average(batch_losses_label)
            batch_loss_label_value = batch_loss_label.scalar_value()
            batch_loss_label.backward()
            trainer.update()

            batch_loss_connect = dy.average(batch_losses_connect)
            batch_loss_connect_value = batch_loss_connect.scalar_value()
            batch_loss_connect.backward()
            trainer.update()

            print(
                "epoch {:,} "
                "batch {:,}/{:,} "
                "processed {:,} "
                "batch-loss1 {:.4f} "
                "batch-loss2 {:.4f} "
                "epoch-elapsed {} "
                "total-elapsed {}".format(
                    epoch,
                    start_index // args.batch_size + 1,
                    int(np.ceil(len(train_parse) / args.batch_size)),
                    total_processed,
                    batch_loss_label_value,
                    batch_loss_connect_value,
                    format_elapsed(epoch_start_time),
                    format_elapsed(start_time),
                )
            )

            if current_processed >= check_every:
                current_processed -= check_every
                check_dev()

def run_test(args):
    print("Loading test trees from {}...".format(args.test_path))
    test_treebank = trees.load_trees(args.test_path)
    print("Loaded {:,} test examples.".format(len(test_treebank)))

    print("Loading model from {}...".format(args.model_path_base))
    model = dy.ParameterCollection()
    [parser] = dy.load(args.model_path_base, model)

    print("Parsing test sentences...")

    start_time = time.time()

    test_predicted = []
    for tree in test_treebank:
        dy.renew_cg()
        sentence = [(leaf.tag, leaf.word) for leaf in tree.leaves()]
        predicted, _ = parser.parse(sentence)
        test_predicted.append(predicted.convert())

    test_fscore = evaluate.evalb(args.evalb_dir, test_treebank, test_predicted, args.parser_type)

    print(
        "test-fscore {} "
        "test-elapsed {}".format(
            test_fscore,
            format_elapsed(start_time),
        )
    )

def main():
    # dynet_args = [
    #     "--dynet-mem",
    #     "--dynet-weight-decay",
    #     "--dynet-autobatch",
    #     "--dynet-gpus",
    #     "--dynet-gpu",
    #     "--dynet-devices",
    #     "--dynet-seed",
    # ]
    #
    # parser = argparse.ArgumentParser()
    # subparsers = parser.add_subparsers()
    #
    # subparser = subparsers.add_parser("train")
    # subparser.set_defaults(callback=run_train)
    # for arg in dynet_args:
    #     subparser.add_argument(arg)
    # subparser.add_argument("--numpy-seed", type=int)
    # subparser.add_argument("--parser-type", choices=["top-down", "chart"], required=True)
    # subparser.add_argument("--tag-embedding-dim", type=int, default=50)
    # subparser.add_argument("--word-embedding-dim", type=int, default=100)
    # subparser.add_argument("--lstm-layers", type=int, default=2)
    # subparser.add_argument("--lstm-dim", type=int, default=250)
    # subparser.add_argument("--label-hidden-dim", type=int, default=250)
    # subparser.add_argument("--split-hidden-dim", type=int, default=250)
    # subparser.add_argument("--dropout", type=float, default=0.4)
    # subparser.add_argument("--explore", action="store_true")
    # subparser.add_argument("--model-path-base", required=True)
    # subparser.add_argument("--evalb-dir", default="EVALB/")
    # subparser.add_argument("--train-path", default="data/02-21.10way.clean")
    # subparser.add_argument("--dev-path", default="data/22.auto.clean")
    # subparser.add_argument("--batch-size", type=int, default=10)
    # subparser.add_argument("--epochs", type=int)
    # subparser.add_argument("--checks-per-epoch", type=int, default=4)
    # subparser.add_argument("--print-vocabs", action="store_true")
    #
    # subparser = subparsers.add_parser("test")
    # subparser.set_defaults(callback=run_test)
    # for arg in dynet_args:
    #     subparser.add_argument(arg)
    # subparser.add_argument("--model-path-base", required=True)
    # subparser.add_argument("--evalb-dir", default="EVALB/")
    # subparser.add_argument("--test-path", default="data/23.auto.clean")
    #
    # args = parser.parse_args()
    # args.callback(args)
    class item:
        def __init__(self):
            self.numpy_seed = None
            self.parser_type = "bottom-up"
            self.tag_embedding_dim = 50
            self.word_embedding_dim = 100
            self.label_embedding_dim = 50
            self.lstm_layers = 2
            self.lstm_dim = 250
            self.label_hidden_dim = 250
            self.split_hidden_dim = 250
            self.dropout = 0.4
            self.explore = True
            self.evalb_dir = "E:/Anew/EVALB"
            self.train_path = "E:/Anew/data/02-21.10way.clean"
            self.dev_path = "E:/Anew/data/22.auto.clean"
            self.batch_size = 10
            self.epochs = 1
            self.checks_per_epoch = 100
            self.print_vocabs = True
    args = item()
    run_train(args)

    class item2:
        def __init__(self):
            self.evalb_dir = "E:/Anew/EVALB"
            self.test_path = "E:/Anew/data/23.auto.clean"
    args2 = item2()
    run_test(args2)

if __name__ == "__main__":
    main()
