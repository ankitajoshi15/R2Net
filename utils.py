"""
Thanks to voxelmorph: Learning-Based Image Registration, https://github.com/voxelmorph/voxelmorph for this code.
If you use this code, please cite the respective papers in their repo.
"""


import tensorflow as tf



def volshape_to_meshgrid(volshape):
    """
    compute Tensor meshgrid from a volume size

    Parameters:
        volshape: the volume size
        **args: "name" (optional)

    Returns:
        A list of Tensors

    See Also:
        tf.meshgrid, meshgrid, ndgrid, volshape_to_ndgrid
    """

    isint = [float(d).is_integer() for d in volshape]
    if not all(isint):
        raise ValueError("volshape needs to be a list of integers")

    linvec = [tf.range(0, d) for d in volshape]
    return meshgrid(*linvec)


def meshgrid(*args):

    ndim = len(args)
    s0 = (1,) * ndim

    # Prepare reshape by inserting dimensions with size 1 where needed
    output = []
    for i, x in enumerate(args):
        output.append(tf.reshape(tf.stack(x), (s0[:i] + (-1,) + s0[i + 1::])))
    # Create parameters for broadcasting each tensor to the full size
    sz = [x.get_shape().as_list()[0] for x in args]

    for i in range(len(output)):
        output[i] = tf.tile(output[i], tf.stack([*sz[:i], 1, *sz[(i + 1):]]))

    return output
