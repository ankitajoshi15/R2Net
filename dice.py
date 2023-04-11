"""
Thanks to voxelmorph: Learning-Based Image Registration, https://github.com/voxelmorph/voxelmorph for this code.
If you use this code, please cite the respective papers in their repo.
"""

import numpy as np
import matplotlib.pyplot as plt

x = np.load('../outputs/lddmm_outputs/generated_segments.npy') #generated output
y = np.load('../../registrationProject/data/testtargetMask.npy') #groundtruth


def dice(array1, array2):
    """
    Computes the dice overlap between two arrays for a given set of integer labels.
    Parameters:
        array1: Input array 1.
        array2: Input array 2.
    """
    labels = np.arange(1,87) #this should be the number of regions in the mask. For example, if it is a whole organ mask, like liver,this will be binary, so it will have only 1.
    dicem = np.zeros(len(labels))
    for idx, label in enumerate(labels):
        top = 2 * np.sum(np.logical_and(array1 == label, array2 == label))
        bottom = np.sum(array1 == label) + np.sum(array2 == label)
        bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon
        dicem[idx] = top / bottom
    return dicem

dice_all = []


for i in range(0,##): #iterate over all masks, so place the number of images in ##.
    d = dice(x[i,:,:,:,:],y[i,:,:,:,:])
    dice_all.append(d)

dice_all = np.array(dice_all)
np.save('', dice_all) #give path to save output 
print(np.mean(dice_all))
