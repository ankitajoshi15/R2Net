
import os
import sys
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
tf.compat.v1.experimental.output_all_intermediates(True)
import network
import losses


#gpu_handling
gpuid = "0"
nb_devices = len(gpuid.split(','))
device = '/gpu:'+gpuid
os.environ['CUDA_VISIBLE_DEVICES'] = gpuid

tf.config.set_soft_device_placement(True)
for pd in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(pd,True)

#training parameters

model_dir = '../models/' #give the name of directory where you want to save your model params
data_dir = '../../registrationProject/data/' #name of directory which contains the training, test, validation as numpy arrays
output_dir='../outputs/' #directory path to store output

#unet architecture
enc_nf = [16,32,32,32] #unet encoder features
dec_nf = [32,32,32,32,16,16,3] #unet decoder features
vol_size = [160,160,128] #3D input image shape
#other NN parameters
image_sigma = 1.0 
initial_epoch = 0
steps_per_epoch = 1
nb_epochs = 1500
batch_size = 1
lr = 1e-4
save_filename = os.path.join(model_dir,'{epoch:04d}.h5')

#model
model = network.unet(vol_size, enc_nf, dec_nf)
#print(model.summary()) #uncomment this line to see the model

#Loss functions
image_loss_func = losses.IR(image_sigma).recon_loss
smooth_loss_func = losses.Smoothness().folding_loss

losses = [image_loss_func, smooth_loss_func]
weights = [1, 0.001] #weights this parameter may have to be changed as training progresses

#Load data
trainSource = np.load(data_dir+'trainSource.npy')
trainTarget = np.load(data_dir+'trainTarget.npy')
zeros = np.zeros((239,160,160,128,1)) #zeros because foldings loss does not need groundtruth. Zeros should be same shape as input

#Load test data
testSource = np.load(data_dir+'testSource.npy')
testTarget = np.load(data_dir+'testTarget.npy')


with tf.device(device):
	model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=losses, loss_weights=weights)
	model.fit([trainSource,trainTarget],[trainTarget, zeros],batch_size=batch_size,epochs=nb_epochs,verbose=1)
	model.save(model_dir)
	predictions = model.predict([testSource,testTarget], batch_size=1)
	np.save(output_dir+'targets',np.asarray(predictions[0]))
	np.save(output_dir+'phi',np.asarray(predictions[1]))



