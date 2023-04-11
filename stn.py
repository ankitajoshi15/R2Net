"""
Thanks to voxelmorph: Learning-Based Image Registration, https://github.com/voxelmorph/voxelmorph for this code.
If you use this code, please cite the respective papers in their repo.
"""


# third party
import numpy as np
from keras import backend as K
from keras.layers import Layer
import tensorflow as tf
import itertools
import utils

class SpatialTransformer(Layer):
    """
    N-D Spatial Transformer Tensorflow / Keras Layer

    The Layer can handle both affine and dense transforms.
    Both transforms are meant to give a 'shift' from the current position.
    Therefore, a dense transform gives displacements (not absolute locations) at each voxel,
    and an affine transform gives the *difference* of the affine matrix from
    the identity matrix.

    If you find this function useful, please cite:
      Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration
      Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu
      MICCAI 2018.

    Originally, this code was based on voxelmorph code, which
    was in turn transformed to be dense with the help of (affine) STN code
    via https://github.com/kevinzakka/spatial-transformer-network

    Since then, we've re-written the code to be generalized to any
    dimensions, and along the way wrote grid and interpolation functions
    """

    def __init__(self,
                 interp_method='linear',
                 indexing='ij',
                 single_transform=False,
                 **kwargs):
        """
        Parameters:
            interp_method: 'linear' or 'nearest'
            single_transform: whether a single transform supplied for the whole batch
            indexing (default: 'ij'): 'ij' (matrix) or 'xy' (cartesian)
                'xy' indexing will have the first two entries of the flow
                (along last axis) flipped compared to 'ij' indexing
        """
        self.interp_method = interp_method
        self.ndims = None
        self.inshape = None
        self.single_transform = single_transform

        assert indexing in ['ij', 'xy'], "indexing has to be 'ij' (matrix) or 'xy' (cartesian)"
        self.indexing = indexing
        super(self.__class__, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        input_shape should be a list for two inputs:
        input1: image.
        input2: transform Tensor
            if affine:
                should be a N x N+1 matrix
                *or* a N*N+1 tensor (which will be reshape to N x (N+1) and an identity row added)
            if not affine:
                should be a *vol_shape x N
        """

        if len(input_shape) > 2:
            raise Exception('Spatial Transformer must be called on a list of length 2.'
                            'First argument is the image, second is the transform.')

        # set up number of dimensions
        self.ndims = len(input_shape[0]) - 2
        self.inshape = input_shape
        vol_shape = input_shape[0][1:-1]
        trf_shape = input_shape[1][1:]

        # confirm built
        self.built = True

    def call(self, inputs):
        """
        Parameters
            inputs: list with two entries
        """

        # check shapes
        assert len(inputs) == 2, "inputs has to be len 2, found: %d" % len(inputs)
        vol = inputs[0]
        trf = inputs[1]

        # necessary for multi_gpu models...
        vol = K.reshape(vol, [-1, *self.inshape[0][1:]])
        trf = K.reshape(trf, [-1, *self.inshape[1][1:]])

        return tf.map_fn(self._single_transform, [vol, trf], dtype=tf.float32)

    def _single_transform(self, inputs):
        return transform(inputs[0], inputs[1], interp_method=self.interp_method)


def transform(vol, loc_shift, interp_method='linear'):
    """
    transform (interpolation N-D volumes (features) given shifts at each location in tensorflow

    Essentially interpolates volume vol at locations determined by loc_shift.
    This is a spatial transform in the sense that at location [x] we now have the data from,
    [x + shift] so we've moved data.

    Parameters:
        vol: volume with size vol_shape or [*vol_shape, nb_features]
        loc_shift: shift volume [*new_vol_shape, N]
        interp_method (default:'linear'): 'linear', 'nearest'
        indexing (default: 'ij'): 'ij' (matrix) or 'xy' (cartesian).
            In general, prefer to leave this 'ij'

    Return:
        new interpolated volumes in the same size as loc_shift[0]

    Keyworks:
        interpolation, sampler, resampler, linear, bilinear
    """

    # parse shapes
    if isinstance(loc_shift.shape, (tf.compat.v1.Dimension, tf.TensorShape)):
        volshape = loc_shift.shape[:-1].as_list()
    else:
        volshape = loc_shift.shape[:-1]
    nb_dims = len(volshape)

    # location should be mesh and delta

    # test single
    return interpn(vol, loc_shift)

def interpn(vol, loc):
    """
    N-D gridded interpolation in tensorflow

    vol can have more dimensions than loc[i], in which case loc[i] acts as a slice
    for the first dimensions

    Parameters:
        vol: volume with size vol_shape or [*vol_shape, nb_features]
        loc: a N-long list of N-D Tensors (the interpolation locations) for the new grid
            each tensor has to have the same size (but not nec. same size as vol)
            or a tensor of size [*new_vol_shape, D]
        interp_method: interpolation type 'linear' (default) or 'nearest'

    Returns:
        new interpolated volume of the same size as the entries in loc

    TODO:
        enable optional orig_grid - the original grid points.
        check out tf.contrib.resampler, only seems to work for 2D data
    """

    # since loc can be a list, nb_dims has to be based on vol.
    nb_dims = loc.shape[-1]

    if nb_dims != len(vol.shape[:-1]):
        raise Exception("Number of loc Tensors %d does not match volume dimension %d"
                        % (nb_dims, len(vol.shape[:-1])))

    if nb_dims > len(vol.shape):
        raise Exception("Loc dimension %d does not match volume dimension %d"
                        % (nb_dims, len(vol.shape)))

    if len(vol.shape) == nb_dims:
        vol = K.expand_dims(vol, -1)

    # flatten and float location Tensors
    loc = tf.cast(loc, 'float32')

    if isinstance(vol.shape, (tf.compat.v1.Dimension, tf.TensorShape)):
        volshape = vol.shape.as_list()
    else:
        volshape = vol.shape


    loc0 = tf.floor(loc)

    # clip values
    max_loc = [d - 1 for d in vol.get_shape().as_list()]
    clipped_loc = [tf.clip_by_value(loc[..., d], 0, max_loc[d]) for d in range(nb_dims)]
    loc0lst = [tf.clip_by_value(loc0[..., d], 0, max_loc[d]) for d in range(nb_dims)]

    # get other end of point cube
    loc1 = [tf.clip_by_value(loc0lst[d] + 1, 0, max_loc[d]) for d in range(nb_dims)]
    locs = [[tf.cast(f, 'int32') for f in loc0lst], [tf.cast(f, 'int32') for f in loc1]]

    # compute the difference between the upper value and the original value
    # differences are basically 1 - (pt - floor(pt))
    #   because: floor(pt) + 1 - pt = 1 + (floor(pt) - pt) = 1 - (pt - floor(pt))
    diff_loc1 = [loc1[d] - clipped_loc[d] for d in range(nb_dims)]
    diff_loc0 = [1 - d for d in diff_loc1]
    weights_loc = [diff_loc1, diff_loc0]  # note reverse ordering since weights are inverse of diff.

    # go through all the cube corners, indexed by a ND binary vector
    # e.g. [0, 0] means this "first" corner in a 2-D "cube"
    cube_pts = list(itertools.product([0, 1], repeat=nb_dims))
    interp_vol = 0

    for c in cube_pts:
        # get nd values
        # note re: indices above volumes via https://github.com/tensorflow/tensorflow/issues/15091
        #   It works on GPU because we do not perform index validation checking on GPU -- it's too
        #   expensive. Instead we fill the output with zero for the corresponding value. The CPU
        #   version caught the bad index and returned the appropriate error.
        subs = [locs[c[d]][d] for d in range(nb_dims)]

        # tf stacking is slow for large volumes, so we will use sub2ind and use single indexing.
        # indices = tf.stack(subs, axis=-1)
        # vol_val = tf.gather_nd(vol, indices)
        # faster way to gather than gather_nd, because the latter needs tf.stack which is slow :(
        idx = sub2ind(vol.shape[:-1], subs)
        vol_reshape = tf.reshape(vol, [-1, volshape[-1]])
        vol_val = tf.gather(vol_reshape, idx)

        # get the weight of this cube_pt based on the distance
        # if c[d] is 0 --> want weight = 1 - (pt - floor[pt]) = diff_loc1
        # if c[d] is 1 --> want weight = pt - floor[pt] = diff_loc0
        wts_lst = [weights_loc[c[d]][d] for d in range(nb_dims)]
        # tf stacking is slow, we we will use prod_n()
        # wlm = tf.stack(wts_lst, axis=0)
        # wt = tf.reduce_prod(wlm, axis=0)
        wt = prod_n(wts_lst)
        wt = K.expand_dims(wt, -1)

        # compute final weighted value for each cube corner
        interp_vol += wt * vol_val

    return interp_vol


def prod_n(lst):
    prod = lst[0]
    for p in lst[1:]:
        prod *= p
    return prod


def sub2ind(siz, subs):
    """
    assumes column-order major
    """
    # subs is a list
    assert len(siz) == len(subs), 'found inconsistent siz and subs: %d %d' % (len(siz), len(subs))

    k = np.cumprod(siz[::-1])

    ndx = subs[-1]
    for i, v in enumerate(subs[:-1][::-1]):
        ndx = ndx + v * k[i]
    return ndx
