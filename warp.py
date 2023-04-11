import numpy as np
import tensorflow as tf
import sys
import os
import stn_nearest

phi = np.load('../outputs/lddmm_outputs/phi.npy') #Load the deformation fields
source_segments = np.load('../../registrationProject/data/testsourceMask.npy') #load the segments to deform

# tensorflow device handling
#gpu_handling
gpuid = "0"
nb_devices = len(gpuid.split(','))
device = '/gpu:'+gpuid
os.environ['CUDA_VISIBLE_DEVICES'] = gpuid

tf.config.set_soft_device_placement(True)
for pd in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(pd,True)


with tf.device(device):
	o1 = stn_nearest.SpatialTransformer(interp_method='linear', indexing='ij')([source_segments, phi])
	r1 = o1.numpy()
	
#save outputs
np.save('',r1)



