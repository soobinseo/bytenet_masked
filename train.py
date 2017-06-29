from model import Bytenet
from data import IWSLT
import tensorflow as tf
import sys

# Set hyper parameters
flags = tf.app.flags
flags.DEFINE_integer("batch_size", 64, "size of a batch")
flags.DEFINE_integer("embedding_size", 256, "the length of input embedding")
flags.DEFINE_integer("sequence_length", 150, "the length of sequence")
flags.DEFINE_integer("num_epochs", 50, "dimension of LSTM hidden layer")
flags.DEFINE_integer("voca_size", 124, "# of unique characters")
flags.DEFINE_integer("num_batch", 100, "# of batch")
flags.DEFINE_integer("num_layers", 2, '# of layers')
flags.DEFINE_float("learning_rate", 0.001, "Learning rate")
flags.DEFINE_float("learning_rate_decay_factor", 0.99, "Learning rate")
flags.DEFINE_float("keep_prob", 0.75, "Learning rate")
flags.DEFINE_string("save_dir", "./assets/summaries", "save summary")
flags.DEFINE_boolean("is_training", True, "boolean training")
flags.DEFINE_boolean("is_masked", True, "apply masking")
flags.DEFINE_boolean("is_logit_masked", True, "apply logit masking")

conf = flags.FLAGS

def main(_):

    # Get data
    data = IWSLT(batch_size=conf.batch_size)
    source = data.source
    target = data.target

    # Set configuration
    conf.voca_size = data.voca_size
    conf.num_batch = data.num_batch
    conf.sequence_length = data.max_len

    # Build model and get tensors for training
    with tf.variable_scope("NMT"):
        model_tr = Bytenet(opts=conf)
        tensors = model_tr.build_graph(source, target)

    # Get loss from model
    loss_tr = tensors['loss']

    # Set initial learning rate and apply decaying
    learning_rate = tf.Variable(conf.learning_rate, trainable=False, dtype=tf.float32)
    decay_factor = tf.constant(conf.learning_rate_decay_factor, dtype=tf.float32)
    learning_rate_update = tf.assign(learning_rate, tf.multiply(learning_rate, decay_factor))

    # Make Optimizer
    optimzer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_step = optimzer.minimize(loss_tr)

    # Create supervisor and save model every 60 seconds
    sv = tf.train.Supervisor(logdir=conf.save_dir,
                             save_model_secs=60)

    # Start training
    with sv.managed_session() as sess:

        for epoch in xrange(1, conf.num_epochs):
            sys.stdout.flush()

            if sv.should_stop(): break

            # Update weights through batches
            for step in range(tensors['num_batch']):
                current_loss, _ = sess.run([loss_tr, train_step])
                current_learning_rate = sess.run(learning_rate)

                # Print log and decay learning rate every 50 steps
                if step % 50 == 0:
                    print "epoch %d, step %d: training loss: %.4f" % (epoch, step, current_loss)
                    print "learning rate %.6f" % current_learning_rate
                    print "Update learning rate..."
                    sess.run(learning_rate_update)

if __name__ == '__main__':
    tf.app.run()