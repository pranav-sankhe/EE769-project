import tensorflow as tf
import numpy as np
import time

from genData import MNIST, MNISTModel


from genAdv import l2_attack



def show(img):
    """
    Show MNSIT digits in the console.
    """
    remap = "  .*#"+"#"*100
    img = (img.flatten()+.5)*3
    if len(img) != 784: return
    print("START")
    for i in range(28):
        print("".join([remap[int(round(x))] for x in img[i*28:i*28+28]]))


def generate_data(data, samples, targeted=True, start=0, inception=False):
    inputs = []
    targets = []
    for i in range(samples):
        if targeted:
            if inception:
                seq = random.sample(range(1,1001), 10)
            else:
                seq = range(data.test_labels.shape[1])

            for j in seq:
                if (j == np.argmax(data.test_labels[start+i])) and (inception == False):
                    continue
                inputs.append(data.test_data[start+i])
                targets.append(np.eye(data.test_labels.shape[1])[j])
        else:
            inputs.append(data.test_data[start+i])
            targets.append(data.test_labels[start+i])

    inputs = np.array(inputs)
    targets = np.array(targets)
    np.save("ip_img", inputs)
    np.save("labels", targets)
    return inputs, targets


if __name__ == "__main__":
    with tf.Session() as sess:
        data, model =  MNIST(), MNISTModel("models/mnist", sess)
        
        attack = l2_attack(sess, model, batch_size=9, max_iterations=1000, confidence=0)
        

        inputs, targets = generate_data(data, samples=70, targeted=True,
                                        start=0, inception=False)
        timestart = time.time()
        adv = attack.attack(inputs, targets)
        timeend = time.time()
        np.save("adv",adv)
        print(adv.shape)
        
        # print("Took",timeend-timestart,"seconds to run",len(inputs),"samples.")
        # print("adv:  ", adv.shape, adv)        
        print(len(adv))
        for i in range(len(adv)):
            #print("Valid:")
            #show(inputs[i])
            #print("Adversarial:")
            #show(adv[i])
            #np.save("adv",adv)
            #np.save("input",inputs)
            print("Classification:", model.model.predict(adv[i:i+1]))

            print("Total distortion:", np.sum((adv[i]-inputs[i])**2)**.5)


