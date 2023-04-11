"""
For some part of code below:
Thanks to voxelmorph: Learning-Based Image Registration, https://github.com/voxelmorph/voxelmorph for this code.
If you use this code, please cite the respective papers in their repo.
"""

# Third party inports
import tensorflow as tf
import keras.backend as K
import numpy as np
import sys

class IR():
    """
    image matching term
    """

    def __init__(self, image_sigma):
        self.image_sigma = image_sigma
        #self.prior_lambda = prior_lambda
        #self.D = None
        #self.flow_vol_shape = flow_vol_shape
    def recon_loss(self, y_true, y_pred):
        """ reconstruction loss """
        return 1. / (self.image_sigma ** 2) * K.mean(K.square(y_true - y_pred))


class Smoothness():

    def __init__(self, penalty=None):
        self.penalty = penalty

    def gradient_loss(self, y_true, y_pred):
        dx = tf.abs(y_pred[:, 1:, :, :, :] - y_pred[:, :-1, :, :, :])
        dy = tf.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dz = tf.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])

        if (self.penalty == 'l2'):
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz
        d = tf.reduce_mean(dx) + tf.reduce_mean(dy) + tf.reduce_mean(dz)
        return d / 3.0

    def folding_loss(self, y_true, y_pred):
        '''
        Penalizing locations where Jacobian has negative determinants
        '''

        jac = self.Get_Ja(y_pred)
        Neg_Jac = 0.5 * (tf.abs(jac) - jac)
        return tf.reduce_sum(Neg_Jac)


    def Get_Ja(self, displacement):
        '''
        Calculate the Jacobian value at each point of the displacement map having
        size of b*h*w*d*3 and in the cubic volumn of [-1, 1]^3
        '''

        D_x = (displacement[:, 1:, :-1, :-1, :] - displacement[:, :-1, :-1, :-1, :])

        D_y = (displacement[:, :-1, 1:, :-1, :] - displacement[:, :-1, :-1, :-1, :])

        D_z = (displacement[:, :-1, :-1, 1:, :] - displacement[:, :-1, :-1, :-1, :])

        D1 = (D_x[..., 0] + 1) * ((D_y[..., 1] + 1) * (D_z[..., 2] + 1) - D_z[..., 1] * D_y[..., 2])

        D2 = (D_x[..., 1]) * (D_y[..., 0] * (D_z[..., 2] + 1) - D_y[..., 2] * D_x[..., 0])

        D3 = (D_x[..., 2]) * (D_y[..., 0] * D_z[..., 1] - (D_y[..., 1] + 1) * D_z[..., 0])

        return D1 - D2 + D3










