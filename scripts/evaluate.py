import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import time
from PIL import Image
from genData import MNIST, MNISTModel

adv = np.load('adv.npy')
ip = np.load('ip_img.npy')
labels = np.load('labels.npy')
# print(ip.shape)
# print(labels[0])
# print(ip[0][:,:,0].shape)
# c = 628
# plt.imshow(ip[c][:,:,0])
# if __name__ == "__main__":
# 	with tf.Session() as sess:
# 		data, model =  MNIST(), MNISTModel("models/mnist", sess)
# 		print("before attack:", model.model.predict(ip[c:c+1]))
# plt.show()

digits = np.arange(0,10)
flags = np.zeros(10)

if __name__ == "__main__":
	if not os.path.exists("results"):
		os.mkdir("results")

	with tf.Session() as sess:

		data, model =  MNIST(), MNISTModel("models/mnist", sess)
		for i in range(69):
			c = 9*i
			
			l = model.model.predict(ip[c:c+1])
			index = np.argmax(l)
			print(i)
			print(l)
			print(index)
			print("next")						
			
			if flags[index] == 0:
				if not os.path.exists("results/"+ str(index)):
					os.mkdir("results/" + str(index))
				for j in range(10):
					I = adv[c+j][:,:,0]
					I8 = (((I - I.min()) / (I.max() - I.min())) * 255.9).astype(np.uint8)

					img = Image.fromarray(I8)
					img.save("results/" + str(index) + "/" + str(j) + ".png")
					#plt.imsave("results/" + str(index) + "/" + str(j), adv[c+j][:,:,0])
			flags[index] = 1		
			
			if np.sum(flags) == 10:
				break
				
				
			