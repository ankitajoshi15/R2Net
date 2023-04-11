import numpy as np

def Get_Ja(loc):
    '''
    Calculate the Jacobian value at each point of the displacement map having
    size of b*h*w*d*3 and in the cubic volumn of [-1, 1]^3
    '''
    # check inputs
    #volshape = disp.shape[:-1]
    #nb_dims = len(volshape)
    #assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'

    # compute grid
    #grid_lst = volsize2ndgrid(volshape)
    #grid = np.stack(grid_lst, len(volshape))

    # compute gradients
    D_x = (loc[1:, :-1, :-1, :] - loc[:-1, :-1, :-1, :])

    D_y = (loc[:-1, 1:, :-1, :] - loc[:-1, :-1, :-1, :])

    D_z = (loc[:-1, :-1, 1:, :] - loc[:-1, :-1, :-1, :])




    D1 = (D_x[..., 0]+1) * ((D_y[..., 1]+1) * (D_z[..., 2]+1) - D_z[..., 1] * D_y[..., 2])
    D2 = (D_x[..., 1]) * (D_y[..., 0] * (D_z[..., 2]+1) - D_y[..., 2] * D_x[..., 0])
    D3 = (D_x[..., 2]) * (D_y[..., 0] * D_z[..., 1] - (D_y[..., 1]+1) * D_z[..., 0])
    return D1 - D2 + D3


phi = np.load('') #load the phi numpy array
absdet = []
foldings = []

for i in range(0,##): #go through each deformation field, put the number in place of ##
    disp = phi[i,:,:,:,:]
    det_jac = Get_Ja(disp)
    foldings.append(np.count_nonzero(det_jac < 0.))
    absdet.append(- np.sum(det_jac[det_jac < 0.]))

print("Foldings mean", np.mean(np.array(foldings)))
print("Foldings std",np.std(np.array(foldings)))

print(np.mean(np.array(absdet)))
print(np.std(np.array(absdet)))

#np.save('../outputs/lddmm_outputs/foldings',np.array(foldings))
#np.save('../outputs/lddmm_outputs/abs_det',np.array(absdet))




