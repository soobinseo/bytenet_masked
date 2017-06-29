from ops import *
import numpy as np

"""
By Soobin Seo
https://www.github.com/soobin3230/bytenet_masked
"""

class Bytenet(object):

    def __init__(self, opts):
        """Bytenet network model class

        Build bytenet graph with encoder and decoder

        Args:
            opts: Hyper parameters for graph

        """
        self.embedding_size = opts.embedding_size
        self.sequence_length = opts.sequence_length
        self.batch_size = opts.batch_size
        self.is_training = opts.is_training
        self.voca_size= opts.voca_size
        self.num_batch = opts.num_batch
        self.is_masked = opts.is_masked
        self.is_logit_masked = opts.is_logit_masked
        self.num_layers = opts.num_layers
        self.keep_prob = opts.keep_prob

        self.make_masks()

        with tf.variable_scope("embeddings"):
            self.w_source_embedding = tf.get_variable('w_source_embedding',
                                                      [self.voca_size, self.embedding_size],
                                                      initializer=tf.contrib.layers.xavier_initializer())

            self.w_target_embedding = tf.get_variable('w_target_embedding',
                                                      [self.voca_size, self.embedding_size],
                                                      initializer=tf.contrib.layers.xavier_initializer())

    def build_graph(self, source, target):
        """
            Args:
              source: A 2-D tensor. Source sequences for graph.
              target: A 2-D tensor. Target sequences for graph.

            Returns:
              Tensors that should be used on training and test.
        """

        if not self.is_training:
            source = tf.placeholder('int32', [self.batch_size, self.sequence_length], name='source')
            target = tf.placeholder('int32', [self.batch_size, self.sequence_length], name='target')

        target_out = tf.concat([target[:, 1:], tf.zeros((self.batch_size, 1), dtype=tf.int32)], axis=1)

        source_emb = tf.nn.embedding_lookup(self.w_source_embedding, source)
        target_emb = tf.nn.embedding_lookup(self.w_target_embedding, target)


        # GET MASKING SEQUENCE FROM TABLE
        self.source_mask = tf.nn.embedding_lookup(self.enc_input_mask, source)

        # MASKS FOR DECODERS
        self.target_mask = tf.nn.embedding_lookup(self.dec_input_mask, target)
        self.decoder_mask = tf.concat([self.source_mask, self.target_mask], axis=2)
        self.loss_mask = tf.nn.embedding_lookup(self.loss_mask, target)
        self.logit_mask = tf.nn.embedding_lookup(self.logit_mask, target)

        source_emb = tf.multiply(source_emb, self.source_mask)
        enc = self.encoder(source_emb)
        dec = self.decoder(target_emb, enc)

        loss = self.loss(dec, target_out)

        flat_logits = tf.reshape(dec, [-1, self.sequence_length, self.voca_size])
        pred = tf.argmax(flat_logits, 2)
        tf.summary.scalar('LOSS', loss)
        variables = tf.trainable_variables()

        tensors = {
            "source":source,
            "target":target_out,
            "target_ts":target,
            "loss":loss,
            "prediction":pred,
            "variables":variables,
            "num_batch":self.num_batch,
            "keep_prob":self.keep_prob
        }

        return tensors


    def encode_layer(self, input_, dilation, layer_no, first_layer=False):
        """
            Args:
              input_: A 3-D tensor.
              dilation: An integer. Dilation rate.
              layer_no: An integer. The number of layer for naming weights.
              first_layer: A boolean. Either first layer or not.

            Returns:
              A tensor of the same shape as `tensor`, which has been
              processed through encoder residual block.
        """

        out_dims = input_.get_shape()[-1].value / 2


        if not first_layer:
            relu1 = normalize_activate(input_,
                                       normalization_type='ln', name="first_enc{}".format(layer_no))
        else:
            relu1 = tf.nn.relu(input_, name='enc_relu1_layer{}'.format(layer_no))

        relu1 = tf.nn.dropout(relu1, keep_prob=self.keep_prob, name="dropout_enc_1{}".format(layer_no))

        conv1 = conv1d(input_=relu1,
                       output_channels=out_dims,
                       name='enc_conv1d_1_layer{}'.format(layer_no)
                       )

        relu2 = normalize_activate(conv1,
                                   normalization_type='ln', name='norm_2_layer{}'.format(layer_no))

        dilated_conv = atrous_conv1d(relu2, out_dims,
                                     rate=dilation,
                                     is_causal=False,
                                     name="enc_dilated_conv_layer{}".format(layer_no)
                                     )

        relu3 = normalize_activate(dilated_conv,
                                   normalization_type='ln', name='norm_3_layer{}'.format(layer_no))

        relu3 = tf.nn.dropout(relu3, keep_prob=self.keep_prob, name="dropout_enc_3{}".format(layer_no))

        conv2 = conv1d(input_=relu3,
                       output_channels=out_dims * 2,
                       name='enc_conv1d_2_layer{}'.format(layer_no)
                       )

        result = tf.add(input_, conv2, name="enc_result_{}".format(layer_no))

        return result


    def encoder(self, input_):
        """
            Args:
              input_: A 3-D tensor.

            Returns:
              A tensor of the same shape as `tensor`, which has been
              processed through encoder layers.
        """

        curr_input = input_
        for repeat in range(self.num_layers):
            for layer_no, dilation in enumerate([1,2,4,8,16]):
                layer_num = layer_no + repeat * 5
                layer_output = self.encode_layer(input_=curr_input,
                                                 dilation=dilation,
                                                 layer_no= layer_num,
                                                 first_layer=True if (layer_num==0 and dilation==1) else False)
                # APPLY MASKING TO EACH LAYERS
                if self.is_masked:
                    layer_output = tf.multiply(layer_output, self.source_mask)
                curr_input = layer_output

        processed_output = conv1d(input_=curr_input,
                                  output_channels=input_.get_shape()[-1].value,
                                  name='encoder_post_processing')

        # APPLY MASKING TO OUTPUT LAYER
        if self.is_masked:
            processed_output = tf.multiply(processed_output, self.source_mask, name='encoder_processed')

        return processed_output

    def decode_layer(self, input_, dilation, layer_no, first_layer=False):
        """
            Args:
              input_: A 3-D tensor.
              dilation: An integer. Dilation rate.
              layer_no: An integer. The number of layer for naming weights.
              first_layer: A boolean. Either first layer or not.

            Returns:
              A tensor of the same shape as `tensor`, which has been
              processed through decoder residual block.
        """

        out_dims = input_.get_shape()[-1].value / 2

        if not first_layer:
            relu1 = normalize_activate(input_,
                                       normalization_type='ln', name="first_dec{}".format(layer_no))
        else:
            relu1 = tf.nn.relu(input_, name='dec_relu1_layer{}'.format(layer_no))

        relu1 = tf.nn.dropout(relu1, keep_prob=self.keep_prob, name="dropout_dec_1{}".format(layer_no))

        conv1 = conv1d(input_=relu1,
                           output_channels=out_dims,
                           name='dec_conv1d_1_layer{}'.format(layer_no))


        relu2 = normalize_activate(conv1,
                                   normalization_type='ln', name='dec_norm_2_layer{}'.format(layer_no))

        dilated_conv = atrous_conv1d(tensor=relu2,
                                         output_channels=out_dims,
                                         rate=dilation,
                                         is_causal=True,
                                         name="dec_dilated_conv_layer{}".format(layer_no)
                                         )

        relu3 = normalize_activate(dilated_conv,
                                   normalization_type='ln',name='dec_norm_3_layer{}'.format(layer_no))

        relu3 = tf.nn.dropout(relu3, keep_prob=self.keep_prob, name="dropout_dec_3{}".format(layer_no))

        conv2 = conv1d(input_=relu3,
                           output_channels=2 * out_dims,
                           name='dec_conv1d_2_layer{}'.format(layer_no))

        result = tf.add(input_, conv2, name="dec_result_{}".format(layer_no))

        return result


    def decoder(self, input_, encoder_embedding=None):
        """
            Args:
              input_: A 3-D tensor.
              encoder_embedding: A 3-D tensor. Output of the encoder layers.

            Returns:
              A tensor of the shape [batch_size, sequence_length, voca_size], which has
              calculated logits through decoder layers.
        """

        curr_input = input_

        if encoder_embedding != None:
            curr_input = tf.concat([curr_input, encoder_embedding], 2)

        for repeat in range(self.num_layers):
            for layer_no, dilation in enumerate([1,2,4,8,16]):
                layer_output = self.decode_layer(input_=curr_input,
                                                 dilation=dilation,
                                                 layer_no=layer_no + repeat * 5,
                                                 first_layer=True if (layer_no==0 and dilation==1) else False)

                # APPLY MASKING TO EACH LAYERS
                if self.is_masked:
                    layer_output = tf.multiply(layer_output, self.decoder_mask)
                curr_input = layer_output

        processed_output = conv1d(input_=curr_input,
                                      output_channels=self.voca_size,
                                      name='decoder_post_processing')

        # LOGIT MASKING
        if self.is_logit_masked:
            processed_output = tf.multiply(processed_output, self.logit_mask)

        return processed_output

    def make_masks(self):
        """
            Make masks for layers and logits.
        """

        enc_sentence_mask = np.ones(
            (self.voca_size, self.embedding_size), dtype='float32')

        enc_sentence_mask[0, :] = np.zeros(
            (self.embedding_size), dtype='float32')

        self.enc_input_mask = tf.constant(enc_sentence_mask)

        dec_sentence_mask = np.ones(
            (self.voca_size, self.embedding_size), dtype='float32')

        dec_sentence_mask[0, :] = np.zeros(
            (self.embedding_size), dtype='float32')

        self.dec_input_mask = tf.constant(dec_sentence_mask)

        logit_mask = np.ones(
            (self.voca_size, self.voca_size), dtype='float32')
        logit_mask[0, :] = np.zeros(
            self.voca_size, dtype='float32')
        self.logit_mask = tf.constant(logit_mask)

        loss_mask = np.ones(
            self.voca_size, dtype='float32')
        loss_mask[0] = 0
        self.loss_mask = tf.constant(loss_mask)


    def loss(self, decoder_output, target_sentence):
        """
            Args:
              decoder_output: A 3-D tensor. Calculated logits.
              target_sentence: A 3-D tensor. Ground truth.

            Returns:
              A tensor of the shape [batch_size, sequence_length, voca_size], which has
              calculated logits through decoder layers.
        """
        target_one_hot = tf.one_hot(target_sentence,
                                    depth=self.voca_size,
                                    dtype=tf.float32)

        flat_logits = tf.reshape(decoder_output, [-1, self.voca_size])

        flat_targets = tf.reshape(target_one_hot, [-1, self.voca_size])

        loss = tf.nn.softmax_cross_entropy_with_logits(labels=flat_targets, logits=flat_logits, name='decoder_cross_entropy_loss')

        # LOSS
        if self.is_logit_masked:
            flat_loss = tf.reshape(self.loss_mask, [-1])
            loss = tf.multiply(loss, flat_loss, name='masked_loss')
            loss = tf.div(tf.reduce_sum(loss), tf.reduce_sum(flat_loss), name="Reduced_mean_loss")
        else:
            loss = tf.reduce_mean(loss, name="Reduced_mean_loss")

        return loss


if __name__ == '__main__':
    pass
