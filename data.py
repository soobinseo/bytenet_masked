# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import re
import codecs

class IWSLT(object):

    def __init__(self, batch_size=32):

        self.max_len = 150
        self.de_train = './data/train.tags.de-en.de'
        self.en_train = './data/train.tags.de-en.en'

        self.de_test = './data/IWSLT16.TED.tst2014.de-en.de.xml'
        self.en_test = './data/IWSLT16.TED.tst2014.de-en.en.xml'
        self.batch_size = batch_size

        # Load data
        X, Y = self.load_train_data()

        data_len = len(X)
        # calc total batch count
        self.num_batch = len(X) // self.batch_size

        # Convert to tensor
        X = tf.convert_to_tensor(X, tf.int32)
        Y = tf.convert_to_tensor(Y, tf.int32)

        # Create Queues
        input_queues = tf.train.slice_input_producer([X, Y])

        # create batch queues
        self.source, self.target = tf.train.shuffle_batch(input_queues,
                                      num_threads=8,
                                      batch_size=self.batch_size,
                                      capacity=self.batch_size * 64,
                                      min_after_dequeue=self.batch_size * 32,
                                      allow_smaller_final_batch=False)


        print 'Train data loaded.(total data=%d, total batch=%d)' % (data_len, self.num_batch)

    def load_vocab(self):
        # Note that ␀, ␂, ␃, and ⁇  mean padding, SOS, EOS, and OOV respectively.
        vocab = u'''␀␂␃⁇ ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÄÅÇÉÖ×ÜßàáâãäçèéêëíïñóôöøúüýāćČēīœšūβкӒ0123456789!"#$%&''()*+,-./:;=?@[\]^_` ¡£¥©«­®°²³´»¼½¾ยรอ่‒–—‘’‚“”„‟‹›€™♪♫你葱送﻿，'''
        char2idx = {char: idx for idx, char in enumerate(vocab)}
        idx2char = {idx: char for idx, char in enumerate(vocab)}

        return char2idx, idx2char

    def create_data(self, source_sents, target_sents):
        char2idx, idx2char = self.load_vocab()
        self.voca_size = len(char2idx)
        # Index
        x_list, y_list, Sources, Targets = [], [], [], []
        for source_sent, target_sent in zip(source_sents, target_sents):
            x = [char2idx.get(char, 3) for char in source_sent + u"␃"]  # 3: OOV, ␃: End of Text
            y = [1] + [char2idx.get(char, 3) for char in target_sent + u"␃"]
            if max(len(x), len(y)) <= 150:
                x_list.append(np.array(x))
                y_list.append(np.array(y))
                Sources.append(source_sent)
                Targets.append(target_sent)

        # Pad
        X = np.zeros([len(x_list), self.max_len], np.int32)
        Y = np.zeros([len(y_list), self.max_len], np.int32)
        for i, (x, y) in enumerate(zip(x_list, y_list)):
            X[i] = np.lib.pad(x, [0, self.max_len - len(x)], 'constant', constant_values=(0, 0))
            Y[i] = np.lib.pad(y, [0, self.max_len - len(y)], 'constant', constant_values=(0, 0))

        print("X.shape =", X.shape)
        print("Y.shape =", Y.shape)

        return X, Y, Sources, Targets

    def create_eval_data(self, source_sents, target_sents):
        char2idx, idx2char = self.load_vocab()

        # Index
        x_list, y_list, Sources, Targets = [], [], [], []
        for source_sent, target_sent in zip(source_sents, target_sents):
            x = [char2idx.get(char, 3) for char in source_sent + u"␃"]  # 3: OOV, ␃: End of Text
            y = [char2idx.get(char, 3) for char in target_sent + u"␃"]
            if max(len(x), len(y)) <= self.max_len:
                x_list.append(np.array(x))
                y_list.append(np.array(y))
                Sources.append(source_sent)
                Targets.append(target_sent)

        # Pad
        X = np.zeros([len(x_list), self.max_len], np.int32)
        Y = np.zeros([len(y_list), self.max_len], np.int32)
        for i, (x, y) in enumerate(zip(x_list, y_list)):
            X[i] = np.lib.pad(x, [0, self.max_len - len(x)], 'constant', constant_values=(0, 0))
            Y[i] = np.lib.pad(y, [0, self.max_len - len(y)], 'constant', constant_values=(0, 0))

        print("X.shape =", X.shape)
        print("Y.shape =", Y.shape)

        return X, Y, Sources, Targets

    def load_train_data(self):
        de_sents = [line for line in codecs.open(self.de_train, 'r', 'utf-8').read().split("\n") if
                    line and line[0] != "<"]
        en_sents = [line for line in codecs.open(self.en_train, 'r', 'utf-8').read().split("\n") if
                    line and line[0] != "<"]

        X, Y, Sources, Targets = self.create_data(de_sents, en_sents)
        return X, Y

    def load_test_data(self):
        def _remove_tags(line):
            line = re.sub("<[^>]+>", "", line)
            return line.strip()

        de_sents = [_remove_tags(line) for line in codecs.open(self.de_test, 'r', 'utf-8').read().split("\n") if
                    line and line[:4] == "<seg"]
        en_sents = [_remove_tags(line) for line in codecs.open(self.en_test, 'r', 'utf-8').read().split("\n") if
                    line and line[:4] == "<seg"]

        X, Y, Sources, Targets = self.create_data(de_sents, en_sents)
        return X, Sources, Targets  # (1064, 150)

    def load_eval_data(self):
        def _remove_tags(line):
            line = re.sub("<[^>]+>", "", line)
            return line.strip()

        de_sents = [_remove_tags(line) for line in codecs.open(self.de_test, 'r', 'utf-8').read().split("\n") if
                    line and line[:4] == "<seg"]
        en_sents = [_remove_tags(line) for line in codecs.open(self.en_test, 'r', 'utf-8').read().split("\n") if
                    line and line[:4] == "<seg"]

        X, Y, Sources, Targets = self.create_data(de_sents, en_sents)
        return X, Y  # (1064, 150)

