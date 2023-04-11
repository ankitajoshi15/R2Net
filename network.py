"""
For some parts of the code below:
Thanks to voxelmorph: Learning-Based Image Registration, https://github.com/voxelmorph/voxelmorph for this code.
If you use this code, please cite the respective papers in their repo.
"""

import keras.layers as KL
from keras.layers import Add, Input, concatenate, Activation, Conv3D, LeakyReLU, Lambda, Layer,ReLU, Conv3DTranspose
from keras.models import Model
from keras.initializers import RandomNormal
import keras.initializers
import tensorflow as tf
import stn
from SpectralNormalizationKeras import ConvSN3D
import utils

def unet_core(vol_size, enc_nf, dec_nf, src=None, tgt=None, src_feats=1, tgt_feats=1):
    """
    unet architecture for voxelmorph models presented in the CVPR 2018 paper.
    You may need to modify this code (e.g., number of layers) to suit your project needs.
    :param vol_size: volume size. e.g. (256, 256, 256)
    :param enc_nf: list of encoder filters. right now it needs to be 1x4.
           e.g. [16,32,32,32]
    :param dec_nf: list of decoder filters. right now it must be 1x6 (like voxelmorph-1) or 1x7 (voxelmorph-2)
    :return: the keras model
    """
    ndims = len(vol_size)
    assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: %d" % ndims
    #upsample_layer = getattr(KL, 'UpSampling%dD' % ndims)
    # inputs
    if src is None:
        src = Input(shape=[*vol_size, src_feats])
    if tgt is None:
        tgt = Input(shape=[*vol_size, tgt_feats])
    x_in = concatenate([src, tgt])

    # down-sample path (encoder)
    x_enc = [x_in]

    """
    xin_conv1 = Conv3D(8, kernel_size=(3, 3, 3), strides=1, padding='same', kernel_initializer='he_normal')(x_in)
    xin_conv1 = LeakyReLU(0.2)(xin_conv1)
    xin_conv1 = Conv3D(8, kernel_size=(3, 3, 3), strides=1, padding='same', kernel_initializer='he_normal')(xin_conv1)
    xin_conv1 = LeakyReLU(0.2)(xin_conv1)

    xin_conv2 = Conv3D(16, kernel_size=(3, 3, 3), strides=2, padding='same', kernel_initializer='he_normal')(xin_conv1)
    xin_conv2 = LeakyReLU(0.2)(xin_conv2)
    xin_conv2 = Conv3D(16, kernel_size=(3, 3, 3), strides=1, padding='same', kernel_initializer='he_normal')(xin_conv2)
    xin_conv2 = LeakyReLU(0.2)(xin_conv2)

    xin_conv3 = Conv3D(32, kernel_size=(3, 3, 3), strides=2, padding='same', kernel_initializer='he_normal')(xin_conv2)
    xin_conv3 = LeakyReLU(0.2)(xin_conv3)
    xin_conv3 = Conv3D(32, kernel_size=(3, 3, 3), strides=1, padding='same', kernel_initializer='he_normal')(xin_conv3)
    xin_conv3 = LeakyReLU(0.2)(xin_conv3)

    xin_conv4 = Conv3D(64, kernel_size=(3, 3, 3), strides=2, padding='same', kernel_initializer='he_normal')(xin_conv3)
    xin_conv4 = LeakyReLU(0.2)(xin_conv4)
    xin_conv4 = Conv3D(64, kernel_size=(3, 3, 3), strides=1, padding='same', kernel_initializer='he_normal')(xin_conv4)
    xin_conv4 = LeakyReLU(0.2)(xin_conv4)

    xin_conv5 = Conv3D(64, kernel_size=(3, 3, 3), strides=2, padding='same', kernel_initializer='he_normal')(xin_conv4)
    xin_conv5 = LeakyReLU(0.2)(xin_conv5)
    xin_conv5 = Conv3D(64, kernel_size=(3, 3, 3), strides=1, padding='same', kernel_initializer='he_normal')(xin_conv5)
    xin_conv5 = LeakyReLU(0.2)(xin_conv5)

    #xin_conv6 = Conv3D(64, kernel_size=(3, 3, 3), strides=2, padding='same', kernel_initializer='he_normal')(xin_conv5)
    #xin_conv6 = LeakyReLU(0.2)(xin_conv6)
    #xin_conv6 = Conv3D(64, kernel_size=(3, 3, 3), strides=1, padding='same', kernel_initializer='he_normal')(xin_conv6)
    #xin_conv6 = LeakyReLU(0.2)(xin_conv6)


    xin_up1 = concatenate([Conv3DTranspose(32, kernel_size=(3, 3, 3), strides=2, padding='same')(xin_conv5), xin_conv4], axis=4)
    xin_up1 = LeakyReLU(0.2)(xin_up1)
    xin_up1 = Conv3D(32, kernel_size=(3, 3, 3), strides=1, padding='same', kernel_initializer='he_normal')(xin_up1)
    xin_up1 = LeakyReLU(0.2)(xin_up1)

    xin_up2 = concatenate([Conv3DTranspose(16, kernel_size=(3, 3, 3), strides=2, padding='same')(xin_up1), xin_conv3], axis=4)
    xin_up2 = LeakyReLU(0.2)(xin_up2)
    xin_up2 = Conv3D(16, kernel_size=(3, 3, 3), strides=1, padding='same', kernel_initializer='he_normal')(xin_up2)
    xin_up2 = LeakyReLU(0.2)(xin_up2)

    xin_up3 = concatenate([Conv3DTranspose(8, kernel_size=(3, 3, 3), strides=2, padding='same')(xin_up2), xin_conv2], axis=4)
    xin_up3 = LeakyReLU(0.2)(xin_up3)
    xin_up3 = Conv3D(8, kernel_size=(3, 3, 3), strides=1, padding='same', kernel_initializer='he_normal')(xin_up3)
    xin_up3 = LeakyReLU(0.2)(xin_up3)

    xin_up4 = concatenate([Conv3DTranspose(8, kernel_size=(3, 3, 3), strides=2, padding='same')(xin_up3), xin_conv1], axis=4)
    xin_up4 = LeakyReLU(0.2)(xin_up4)
    xin_up4 = Conv3D(8, kernel_size=(3, 3, 3), strides=1, padding='same', kernel_initializer='he_normal')(xin_up4)
    #xin_up4 = LeakyReLU(0.2)(xin_up4)
    x = LeakyReLU(0.2)(xin_up4)
    #xin_up5 = concatenate([Conv3DTranspose(8, kernel_size=(3, 3, 3), strides=2, padding='same')(xin_up4), xin_conv1], axis=4)
    #xin_up5 = LeakyReLU(0.2)(xin_up5)
    #xin_up5 = Conv3D(8, kernel_size=(3, 3, 3), strides=1, padding='same', kernel_initializer='he_normal')(xin_up5)
    #x = LeakyReLU(0.2)(xin_up5)

    """ 
    for i in range(len(enc_nf)):
        x_enc.append(conv_block(x_enc[-1], enc_nf[i], 2))

    # up-sample path (decoder)
    x = conv_block(x_enc[-1], dec_nf[0])
    x = Conv3DTranspose(8, kernel_size=(3, 3, 3), strides=2, padding='same')(x)#upsample_layer()(x)
    x = concatenate([x, x_enc[-2]])
    x = conv_block(x, dec_nf[1])
    x = Conv3DTranspose(8, kernel_size=(3, 3, 3), strides=2, padding='same')(x)#upsample_layer()(x)
    x = concatenate([x, x_enc[-3]])
    x = conv_block(x, dec_nf[2])
    x = Conv3DTranspose(8, kernel_size=(3, 3, 3), strides=2, padding='same')(x)#upsample_layer()(x)
    x = concatenate([x, x_enc[-4]])
    x = conv_block(x, dec_nf[3])
    x = conv_block(x, dec_nf[4])
    x = Conv3DTranspose(8, kernel_size=(3, 3, 3), strides=2, padding='same')(x)#upsample_layer()(x)
    x = concatenate([x, x_enc[0]])
    x = conv_block(x, dec_nf[5])
    # optional convolution at output resolution (used in voxelmorph-2)
    if len(dec_nf) == 7:
        x = conv_block(x, dec_nf[6])
    
    return Model(inputs=[src, tgt], outputs=[x])


def unet(vol_size, enc_nf, dec_nf):
    ndims = len(vol_size)
    assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: %d" % ndims

    # get unet
    unet_model = unet_core(vol_size, enc_nf, dec_nf)
    [src, tgt] = unet_model.inputs
    x_out = unet_model.outputs[-1]
    
    """
    # velocity mean and logsigma layers
    Conv = getattr(KL, 'Conv%dD' % ndims)
    flow_mean = Conv(ndims, kernel_size=3, padding='same',
                     kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5), name='flow')(x_out)
    # we're going to initialize the velocity variance very low, to start stable.
    flow_log_sigma = Conv(ndims, kernel_size=3, padding='same',
                          kernel_initializer=RandomNormal(mean=0.0, stddev=1e-10),
                          bias_initializer=keras.initializers.Constant(value=-10),
                          name='log_sigma')(x_out)
    flow_params = concatenate([flow_mean, flow_log_sigma])
    """

    # velocity sample
    flow0 = Conv3D(3, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal', use_bias=False)(x_out)
    flow0 = Lambda(lambda x: x / 2.0)(flow0)

    # forward integration
    out1 = ConvSN3D(3, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal', use_bias=False)(
        flow0)
    out1 = LeakyReLU(0.2)(out1)
    out1 = ConvSN3D(3, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal', use_bias=False)(
        out1)
    out1 = Activation(tf.keras.activations.tanh)(out1)
    out1 = Lambda(lambda x: x / 2.0)(out1)
    # out1 = resmodel(flow0)
    v1 = Add()([flow0, out1])

    out2 = ConvSN3D(3, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal', use_bias=False)(
        v1)
    out2 = LeakyReLU(0.2)(out2)
    out2 = ConvSN3D(3, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal', use_bias=False)(
        out2)
    out2 = Activation(tf.keras.activations.tanh)(out2)
    out2 = Lambda(lambda x: x / 2.0)(out2)
    # out2 = resmodel(v1)
    v2 = Add()([v1, out2])

    out3 = ConvSN3D(3, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal', use_bias=False)(
        v2)
    out3 = LeakyReLU(0.2)(out3)
    out3 = ConvSN3D(3, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal', use_bias=False)(
        out3)
    out3 = Activation(tf.keras.activations.tanh)(out3)
    out3 = Lambda(lambda x: x / 2.0)(out3)
    # out3 = resmodel(v2)
    v3 = Add()([v2, out3])

    out4 = ConvSN3D(3, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal', use_bias=False)(
        v3)
    out4 = LeakyReLU(0.2)(out4)
    out4 = ConvSN3D(3, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal', use_bias=False)(
        out4)
    out4 = Activation(tf.keras.activations.tanh)(out4)
    out4 = Lambda(lambda x: x / 2.0)(out4)
    # out4 = resmodel(v3)
    v4 = Add()([v3, out4])

    out5 = ConvSN3D(3, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal', use_bias=False)(
        v4)
    out5 = LeakyReLU(0.2)(out5)
    out5 = ConvSN3D(3, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal', use_bias=False)(
        out5)
    out5 = Activation(tf.keras.activations.tanh)(out5)
    out5 = Lambda(lambda x: x / 2.0)(out5)
    # out5 = resmodel(v4)
    v5 = Add()([v4, out5])

    out6 = ConvSN3D(3, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal', use_bias=False)(v5)
    out6 = LeakyReLU(0.2)(out6)
    out6 = ConvSN3D(3, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal', use_bias=False, )(out6)
    out6 = Activation(tf.keras.activations.tanh)(out6)
    out6 = Lambda(lambda x: x / 2.0)(out6)
    # out6 = resmodel(v5)
    v6 = Add()([v5, out6])

    out7 = ConvSN3D(3, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal', use_bias=False)(v6)
    out7 = LeakyReLU(0.2)(out7)
    out7 = ConvSN3D(3, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal', use_bias=False)(out7)
    out7 = Activation(tf.keras.activations.tanh)(out7)
    out7 = Lambda(lambda x: x / 2.0)(out7)
    v7 = Add()([v6, out7])

    phi = Addgrid()(v7)

    # warp the source with the flow
    deformed_src = stn.SpatialTransformer(interp_method='linear', indexing='ij')([src, phi])

    return Model(inputs=[src, tgt], outputs=[deformed_src, phi])


def sample(args):
    """
    sample from a normal distribution
    """
    mu = args[0]
    log_sigma = args[1]
    noise = tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)
    z = mu + tf.exp(log_sigma/2.0) * noise
    return z

class Sample(Layer):
    """
    Keras Layer: Gaussian sample from [mu, sigma]
    """

    def __init__(self, **kwargs):
        super(Sample, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Sample, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return sample(x)

    def compute_output_shape(self, input_shape):
        return input_shape[0]


def conv_block(x_in, nf, strides=1):
    """
    specific convolution module including convolution followed by leakyrelu
    """
    ndims = len(x_in.get_shape()) - 2
    assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: %d" % ndims

    Conv = getattr(KL, 'Conv%dD' % ndims)
    x_out = Conv(nf, kernel_size=3, padding='same',
                 kernel_initializer='he_normal', strides=strides)(x_in)
    x_out = LeakyReLU(0.2)(x_out)
    return x_out

class Addgrid(Layer):
    """
   #     Keras Layer: Add grid to velocity field for locations.
   """
    def __init__(self, **kwargs):
        super(Addgrid, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Addgrid, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        volshape = [160,160,128]
        mesh = utils.volshape_to_meshgrid(volshape)  # volume mesh
        loc = [tf.cast(mesh[d], 'float32') + x[..., d] for d in range(3)]
        loc = tf.stack(loc, -1)
        return loc

    def compute_output_shape(self, input_shape):
        return input_shape



