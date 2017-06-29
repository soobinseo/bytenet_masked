# -*- coding: utf-8 -*-
import tensorflow as tf
from model import *
from data import IWSLT
from nltk.translate.bleu_score import corpus_bleu
import codecs

flags = tf.app.flags

# Set hyper parameters
flags.DEFINE_integer("batch_size", 10, "size of a batch")
flags.DEFINE_integer("embedding_size", 256, "the length of input embedding")
flags.DEFINE_integer("sequence_length", 150, "filter length of Conv layers")
flags.DEFINE_integer("num_epochs", 50, "dimension of LSTM hidden layer")
flags.DEFINE_integer("voca_size", 124, "# of unique grapheme")
flags.DEFINE_integer("num_batch", 100, "# of batch")
flags.DEFINE_integer("num_layers", 2, '# of layers')
flags.DEFINE_float("learning_rate", 0.001, "Learning rate")
flags.DEFINE_float("keep_prob", 1.0, "Learning rate")
flags.DEFINE_string("save_dir", "./assets/summaries", "save summary")
flags.DEFINE_boolean("is_training", False, "boolean training")
flags.DEFINE_boolean("is_masked", True, "apply masking")
flags.DEFINE_boolean("is_logit_masked", True, "apply logit masking")

conf = flags.FLAGS


def eval():
    # Set configuration
    batch_size = conf.batch_size
    data = IWSLT(batch_size=batch_size)
    conf.voca_size = data.voca_size

    # Build model
    with tf.variable_scope("NMT"):
        g = Bytenet(conf)
        tensors = g.build_graph("","")

    # Get prediction which is processed through the network
    label = tensors['prediction']

    # Start translate
    with tf.Session() as sess:

        # Restore the model checkpoint
        saver = tf.train.Saver(tf.trainable_variables())
        saver.restore(sess, tf.train.latest_checkpoint(conf.save_dir))
        print("Restored!")
        mname = open(conf.save_dir + '/checkpoint', 'r').read().split('"')[1]  # model name
        print(mname)

        # Load test data
        X, Sources, Targets = data.load_test_data()
        char2idx, idx2char = data.load_vocab()

        with codecs.open(conf.save_dir + "/" + mname, "w", "utf-8") as fout:
            list_of_refs, hypotheses = [], []
            for i in range(len(X) // batch_size):

                # Get mini-batches
                x = X[i * batch_size: (i + 1) * batch_size]  # mini-batch
                sources = Sources[i * batch_size: (i + 1) * batch_size]
                targets = Targets[i * batch_size: (i + 1) * batch_size]

                preds_prev = np.zeros((batch_size, data.max_len), np.int32)
                preds_prev[:, 0] = 1
                preds = np.zeros((batch_size, data.max_len), np.int32)
                for j in range(data.max_len):
                    # predict next character
                    outs = sess.run(label, {tensors['source']: x, tensors['target_ts']: preds_prev})
                    # update character sequence
                    if j < data.max_len - 1:
                        preds_prev[:, j + 1] = outs[:, j]
                    preds[:, j] = outs[:, j]

                # Write to file
                i = 0
                for source, target, pred in zip(sources, targets, preds):  # sentence-wise
                    got = "".join(idx2char[idx] for idx in pred).split(u"␃")[0]
                    if i % 50 == 0:
                        print target
                        print got
                        print '-' * 50
                    fout.write("- source: " + source + "\n")
                    fout.write("- expected: " + target + "\n")
                    fout.write("- got: " + got + "\n\n")
                    fout.flush()

                    # For bleu score
                    ref = target.split()
                    hypothesis = got.split()
                    if len(ref) > 3 and len(hypothesis) > 3:
                        list_of_refs.append([ref])
                        hypotheses.append(hypothesis)
                    i += 1


            # Get bleu score
            score = corpus_bleu(list_of_refs, hypotheses)
            print "Bleu Score = " + str(100 * score)


if __name__ == '__main__':
    eval()