# Structure of the scripts: 

This folder contains all the scripts which address the paper "Towards Evaluating the Robustness of Neural Networks" by Nicholas Carlini and David Wagner, University of California, Berkeley. 

- main.py:     Performing the attack on the neural net 
- genData.py:  Generate input vector and label vector  
- genAdv.py:   Generate Adverserial images and store them
- train.py:    train the MNIST model
- evaluate.py: classify and store the adverserial images 

## Instructions to run the code: 

- To train and save MNIST trained CNN model `python train.py `
- To run the poisoning algorithm run `python main.py`
- To store and view adverserial images, run `python evaluate.py` 


 

