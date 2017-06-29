import tensorflow as tf

from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.framework import ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn

def atrous_conv1d(tensor, output_channels, is_causal=False, rate=1, pad='SAME', stddev=0.02, name="aconv1d"):
    """
        Args:
          tensor: A 3-D tensor.
          output_channels: An integer. Dimension of output channel.
          is_causal: A boolean. If true, apply causal convolution.
          rate: An integer. Dilation rate.
          pad: Either "SAME" or "VALID". If "SAME", make padding, else no padding.
          stddev: A float. Standard deviation for truncated normal initializer.
          name: A string. Name of scope.

        Returns:
          A tensor of the same shape as `tensor`, which has been
          processed through dilated convolution layer.
    """

    # Set filter size
    size = (3 if is_causal else 3)

    # Get input dimension
    in_dim = tensor.get_shape()[-1].value

    with tf.variable_scope(name):
        # Make filter
        filter = tf.get_variable("w", [1,size, in_dim, output_channels],
                                 initializer=tf.truncated_normal_initializer(stddev=stddev))

        # Pre processing for dilated convolution
        if is_causal:
            # Causal convolution pre-padding
            if pad == 'SAME':
                pad_len = (size - 1) * rate
                x = tf.expand_dims(tf.pad(tensor, [[0, 0], [pad_len, 0], [0, 0]]),axis=1, name="X")
            else:
                x = tf.expand_dims(tensor, axis=1)
            # Apply 2d convolution
            out = tf.nn.atrous_conv2d(x, filter, rate=rate, padding='VALID')
        else:
            # Apply 2d convolution
            out = tf.nn.atrous_conv2d(tf.expand_dims(tensor,axis=1),
                                      filter, rate=rate, padding=pad)
        # Reduce dimension
        out = tf.squeeze(out, axis=1)

    return out

def conv1d(input_, output_channels, filter_width = 1, stride = 1, stddev=0.02, name = 'conv1d'):
    """
        Args:
          tensor: A 3-D tensor.
          output_channels: An integer. Dimension of output channel.
          filter_width: An integer. Size of filter.
          stride: An integer. Stride of convolution.
          stddev: A float. Standard deviation for truncated normal initializer.
          name: A string. Name of scope.

        Returns:
          A tensor of the shape as [batch size, timesteps, output channel], which has been
          processed through 1-D convolution layer.
    """

    # Get input dimension
    input_shape = input_.get_shape()
    input_channels = input_shape[-1].value

    with tf.variable_scope(name):
        # Make filter
        filter_ = tf.get_variable('w', [filter_width, input_channels, output_channels],
            initializer=tf.truncated_normal_initializer(stddev=stddev))

        # Convolution layer
        conv = tf.nn.conv1d(input_, filter_, stride = stride, padding = 'SAME')
        biases = tf.get_variable('biases', [output_channels], initializer=tf.constant_initializer(0.0))

        # Add bias
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv



def normalize_activate(tensor, name="name",
                       normalization_type="ln"):
    '''
    Args:
      tensor: A 3-D or 4-D tensor.
      normalization_type: Either `ln` or `bn`.
      is_training: A boolean. Phase declaration for batch normalization.

    Returns:
      A tensor of the same shape as `tensor`, which has been
      normalized and subsequently activated by Relu.
    '''

    if normalization_type == "ln":  # layer normalization

        result = layer_norm(inputs=tensor, center=True, scale=True,
                                         activation_fn=tf.nn.relu, name=name)
        return result
    else:  # batch normalization
        masks = tf.sign(tf.abs(tensor))
        return tf.contrib.layers.batch_norm(inputs=tensor, center=True, scale=True,
                                            activation_fn=tf.nn.relu, updates_collections=None,
                                            is_training=True, batch_weights=masks)


def layer_norm(inputs,
               center=True,
               scale=True,
               activation_fn=None,
               reuse=None,
               name=None,
               outputs_collections=None,
               trainable=True):
  """Adds a Layer Normalization layer from https://arxiv.org/abs/1607.06450.

    "Layer Normalization"

    "From tensorflow modules, modifying axis=[-1]"

  Args:
    inputs: a tensor with 2 or more dimensions. The normalization
            occurs over all but the first dimension.
    center: If True, add offset of `beta` to normalized tensor. If False, `beta`
      is ignored.
    scale: If True, multiply by `gamma`. If False, `gamma` is
      not used. When the next layer is linear (also e.g. `nn.relu`), this can be
      disabled since the scaling can be done by the next layer.
    activation_fn: activation function, default set to None to skip it and
      maintain a linear activation.
    reuse: whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
    outputs_collections: collections to add the outputs.
    trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).

  Returns:
    A `Tensor` representing the output of the operation.

  Raises:
    ValueError: if rank or last dimension of `inputs` is undefined.
  """
  with tf.variable_scope(name, reuse=reuse) as sc:
    inputs = ops.convert_to_tensor(inputs)
    inputs_shape = inputs.get_shape()
    inputs_rank = inputs_shape.ndims
    if inputs_rank is None:
      raise ValueError('Inputs %s has undefined rank.' % inputs.name)
    dtype = inputs.dtype.base_dtype
    axis = [-1]
    params_shape = inputs_shape[-1:]
    if not params_shape.is_fully_defined():
      raise ValueError('Inputs %s has undefined last dimension %s.' % (
          inputs.name, params_shape))
    # Allocate parameters for the beta and gamma of the normalization.
    beta, gamma = None, None
    if center:
      beta = tf.get_variable(
          'beta',
          shape=params_shape,
          dtype=dtype,
          initializer=init_ops.zeros_initializer(),
          # collections=beta_collections,
          trainable=trainable)
    if scale:
      gamma = tf.get_variable(
          'gamma',
          shape=params_shape,
          dtype=dtype,
          initializer=init_ops.ones_initializer(),
          # collections=gamma_collections,
          trainable=trainable)
    # Calculate the moments on the last axis (layer activations).
    mean, variance = nn.moments(inputs, axis, keep_dims=True)
    # Compute layer normalization using the batch_normalization function.
    variance_epsilon = 1E-12
    outputs = nn.batch_normalization(
        inputs, mean, variance, beta, gamma, variance_epsilon)
    outputs.set_shape(inputs_shape)
    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return utils.collect_named_outputs(outputs_collections,
                                       sc.original_name_scope,
                                       outputs)

