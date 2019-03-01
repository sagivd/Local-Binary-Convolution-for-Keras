import numpy as np

from keras.layers.convolutional import _Conv
from keras.utils import conv_utils
from keras.engine.base_layer import InputSpec
from keras import backend as K

def make_binary_kernel(kernel_shape, sparsity):
    """
    Create a random binary kernel
    """

    filter_1d_sz = kernel_shape[0]
    filter_2d_sz = filter_1d_sz * kernel_shape[1]
    filter_3d_sz = filter_2d_sz * kernel_shape[2]
    filter_4d_sz = filter_3d_sz * kernel_shape[3]

    sparsity_int = np.ceil(sparsity * filter_4d_sz).astype(int)
    pos_cutoff = sparsity_int // 2

    binary_kernel = np.zeros(kernel_shape, dtype=np.int8)

    # We need to pick randomly elements that wont be 0s
    one_d_ind = np.random.choice(range(filter_4d_sz),
                                 sparsity_int,
                                 replace=False)

    # Pick elements to be 1s
    ind = (one_d_ind[:pos_cutoff] % filter_3d_sz
                                  % filter_2d_sz
                                  % filter_1d_sz,
           one_d_ind[:pos_cutoff] % filter_3d_sz
                                  % filter_2d_sz
                                  // filter_1d_sz,
           one_d_ind[:pos_cutoff] % filter_3d_sz
                                  // filter_2d_sz,
           one_d_ind[:pos_cutoff] // filter_3d_sz)
    binary_kernel[ind] = 1

    # Pick elements to be -1s
    ind = (one_d_ind[pos_cutoff:] % filter_3d_sz
                                  % filter_2d_sz
                                  % filter_1d_sz,
           one_d_ind[pos_cutoff:] % filter_3d_sz
                                  % filter_2d_sz
                                  // filter_1d_sz,
           one_d_ind[pos_cutoff:] % filter_3d_sz
                                  // filter_2d_sz,
           one_d_ind[pos_cutoff:] // filter_3d_sz)
    binary_kernel[ind] = -1


class LBC(_Conv):
    """
    Local Binary Convolution layer
    """

    def __init__(self,
                 filters,
                 kernel_size=(3,3),
                 sparsity=.5,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=False,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(LBC, self).__init__(rank=2,
                                  filters=filters,
                                  kernel_size=kernel_size,
                                  strides=strides,
                                  padding=padding,
                                  data_format=data_format,
                                  dilation_rate=dilation_rate,
                                  activation=activation,
                                  use_bias=use_bias,
                                  kernel_initializer=kernel_initializer,
                                  bias_initializer=bias_initializer,
                                  kernel_regularizer=kernel_regularizer,
                                  bias_regularizer=bias_regularizer,
                                  activity_regularizer=activity_regularizer,
                                  kernel_constraint=kernel_constraint,
                                  bias_constraint=bias_constraint,
                                  **kwargs)
        self.sparsity = sparsity

        assert self.use_bias is False, 'No bias in LBC implementation'
        assert self.rank == 2,  'only viable for 2d conv'
        assert 0. < sparsity < 1., 'Expected sparsity level within (0,1)'

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        binary_kernel_shape = self.kernel_size + (input_dim, input_dim)
        self.binary_kernel = make_binary_kernel(binary_kernel_shape,
                                                self.sparsity)

        self.kernel = self.add_weight(shape=(1, 1, input_dim, self.filters),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            raise NotImplementedError
        else:
            self.bias = None

        # Set input spec.
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs):
        # Create difference map
        if self.rank == 2:
            difference_map = K.conv2d(inputs,
                                      tf.cast(self.binary_kernel,
                                      dtype=tf.float32),
                                      strides=self.strides,
                                      padding=self.padding,
                                      data_format=self.data_format,
                                      dilation_rate=self.dilation_rate)

        else:
            raise NotImplementedError

        # Create bitmap. we should always have anactivation here!
        if self.activation is not None:
            bitmap = self.activation(difference_map)
        else:
            assert False, "Expected an activation function in LBC"

        # Calculate output maps
        outputs = K.conv2d(bitmap,
                           self.kernel,
                           data_format=self.data_format)
        return outputs

    def get_config(self):
        config = {
            'sparsity': self.sparsity,
        }
        base_config = super(LBC, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def get_binary_weights(self):
        return self.binary_kernel
