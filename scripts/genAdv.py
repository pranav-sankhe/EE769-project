import sys
import tensorflow as tf
import numpy as np

BINARY_SEARCH_STEPS = 9  # number of times to adjust the constant with binary search
MAX_ITERATIONS = 10000   # number of iterations to perform gradient descent
ABORT_EARLY = True       # if we stop improving, abort gradient descent early
LEARNING_RATE = 1e-2     # larger values converge faster to less accurate results
TARGETED = True          # should we target one specific class? or just be wrong?
CONFIDENCE = 0           # how strong the adversarial example should be
INITIAL_CONST = 1e-3     # the initial constant c to pick as a first guess

class l2_attack:
    def __init__(self, sess, model, batch_size=1, confidence = CONFIDENCE,
                 targeted = TARGETED, learning_rate = LEARNING_RATE,
                 binary_search_steps = BINARY_SEARCH_STEPS, max_iterations = MAX_ITERATIONS,
                 abort_early = ABORT_EARLY, 
                 initial_const = INITIAL_CONST,
                 boxmin = -0.5, boxmax = 0.5):

        image_size, num_channels, num_labels = model.image_size, model.num_channels, model.num_labels
        self.sess = sess
        self.TARGETED = targeted
        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.BINARY_SEARCH_STEPS = binary_search_steps
        self.ABORT_EARLY = abort_early
        self.CONFIDENCE = confidence
        self.initial_const = initial_const
        self.batch_size = batch_size

        self.repeat = binary_search_steps >= 10

        self.I_KNOW_WHAT_I_AM_DOING_AND_WANT_TO_OVERRIDE_THE_PRESOFTMAX_CHECK = False

        shape = (batch_size,image_size,image_size,num_channels)
        
        # the variable we're going to optimize over
        modifier = tf.Variable(np.zeros(shape,dtype=np.float32))

        #------------------------------------------------------------------------------------------------------------
        
        self.timg = tf.Variable(np.zeros(shape), dtype=tf.float32)						#create a graph element for holding images
        self.tlab = tf.Variable(np.zeros((batch_size,num_labels)), dtype=tf.float32)    #create a graph element for holding labels
        self.const = tf.Variable(np.zeros(batch_size), dtype=tf.float32)				

        
        #------------------------------------------------------------------------------------------------------------

        self.assign_timg = tf.placeholder(tf.float32, shape)						    #placeholder for images
        self.assign_tlab = tf.placeholder(tf.float32, (batch_size,num_labels))          #placeholder for labels
        self.assign_const = tf.placeholder(tf.float32, [batch_size])					#the constant "c"
        
        #------------------------------------------------------------------------------------------------------------

        '''
        reference: https://github.com/carlini/nn_robust_attacks/blob/master/l2_attack.py
        '''
        self.boxmul = (boxmax - boxmin) / 2.
        self.boxplus = (boxmin + boxmax) / 2.
        self.newimg = tf.tanh(modifier + self.timg) * self.boxmul + self.boxplus        #convert to tanh space and restrict the image to boxmin and boxmax.
        																				#We convert into tanh space to solve the optimization problem		
		#------------------------------------------------------------------------------------------------------------        

        
        self.output = model.predict(self.newimg)		#output of the model 
        
        
        # Calculate the l2 distance (eucledian distance) between the image being poisoned and the original image
        self.l2dist = tf.reduce_sum(tf.square(self.newimg-(tf.tanh(self.timg) * self.boxmul + self.boxplus)),[1,2,3])  
        
        
        # compute the probability of the label class versus the maximum other
        real = tf.reduce_sum((self.tlab)*self.output,1)
        other = tf.reduce_max((1-self.tlab)*self.output - (self.tlab*10000),1)


        #------------------------------------------------------------------------------------------------------------        
        '''
        reference: https://github.com/carlini/nn_robust_attacks/blob/master/l2_attack.py
        '''

        if self.TARGETED:
            # if targetted, optimize for making the other class most likely
            loss1 = tf.maximum(0.0, other-real+self.CONFIDENCE)
        else:
            # if untargeted, optimize for making this class least likely.
            loss1 = tf.maximum(0.0, real-other+self.CONFIDENCE)
        #------------------------------------------------------------------------------------------------------------        
        
        self.loss2 = tf.reduce_sum(self.l2dist)
        self.loss1 = tf.reduce_sum(self.const*loss1)
        self.loss = self.loss1+self.loss2						#objective function
        
        #------------------------------------------------------------------------------------------------------------        
        # add the optimizer to the graph. As mentioned in the paper we are using Adam's optimizer.  
        start_vars = set(x.name for x in tf.global_variables())
        optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE)  			 		  
        self.train = optimizer.minimize(self.loss, var_list=[modifier])
        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]

        #------------------------------------------------------------------------------------------------------------        
        #create a variable to store the progress during the execution of the graph 
        self.setup = []
        self.setup.append(self.timg.assign(self.assign_timg))
        self.setup.append(self.tlab.assign(self.assign_tlab))
        self.setup.append(self.const.assign(self.assign_const))
        
        self.init = tf.variables_initializer(var_list=[modifier]+new_vars)




    #------------------------------------------------------------------------------------------------------------        
    #implement the attack 
    def attack(self, imgs, targets):

        """
        If self.targeted is true, then the targets represents the target labels.
        If self.targeted is false, then targets are the original class labels.
        reference: https://github.com/carlini/nn_robust_attacks/blob/master/l2_attack.py
        """
        r = []
        print('go up to',len(imgs))
        for i in range(0,len(imgs),self.batch_size):
            print('tick',i)
            r.extend(self.attack_batch(imgs[i:i+self.batch_size], targets[i:i+self.batch_size]))
        return np.array(r)
        
     #------------------------------------------------------------------------------------------------------------        




	#------------------------------------------------------------------------------------------------------------        
	#Execute the attack 
    def attack_batch(self, imgs, labs):


    	#------------------------------------------------------------------------------------------------------------        
    	# reference: https://github.com/carlini/nn_robust_attacks/blob/master/l2_attack.py
        def compare(x,y):
            if not isinstance(x, (float, int, np.int64)):
                x = np.copy(x)
                if self.TARGETED:
                    x[y] -= self.CONFIDENCE
                else:
                    x[y] += self.CONFIDENCE
                x = np.argmax(x)
            if self.TARGETED:
                return x == y
            else:
                return x != y
        #------------------------------------------------------------------------------------------------------------        
        

        batch_size = self.batch_size

        # reference: https://github.com/carlini/nn_robust_attacks/blob/master/l2_attack.py

        imgs = np.arctanh((imgs - self.boxplus) / self.boxmul * 0.999999)	# convert to tanh-space 

        # set the lower and upper bounds accordingly
        lower_bound = np.zeros(batch_size)
        CONST = np.ones(batch_size)*self.initial_const
        upper_bound = np.ones(batch_size)*1e10

        # initialize the best l2, score, and image attack
        o_bestl2 = [1e10]*batch_size
        o_bestscore = [-1]*batch_size
        o_bestattack = [np.zeros(imgs[0].shape)]*batch_size
        

        #------------------------------------------------------------------------------------------------------------        
        #binary search algorithm to find the constant 'c'
        # reference: https://github.com/carlini/nn_robust_attacks/blob/master/l2_attack.py
        for outer_step in range(self.BINARY_SEARCH_STEPS):
            print("o_bestl2" ,o_bestl2)
            
            self.sess.run(self.init)			# completely reset adam's internal state.
            batch = imgs[:batch_size]				
            batchlab = labs[:batch_size]		#labels	
    
            bestl2 = [1e10]*batch_size			
            bestscore = [-1]*batch_size

            # The last iteration (if we run many steps) repeat the search once.
            if self.repeat == True and outer_step == self.BINARY_SEARCH_STEPS-1:
                CONST = upper_bound

            # set the variables so that we don't have to send them over again
            self.sess.run(self.setup, {self.assign_timg: batch,
                                       self.assign_tlab: batchlab,
                                       self.assign_const: CONST})
            
            prev = 1e6
            for iteration in range(self.MAX_ITERATIONS):	# perform the attack 
                
                _, l, l2s, scores, nimg = self.sess.run([self.train, self.loss,   	#run the session which executes gardient descent			
                                                         self.l2dist, self.output, 
                                                         self.newimg])

                if np.all(scores>=-.0001) and np.all(scores <= 1.0001):
                    if np.allclose(np.sum(scores,axis=1), 1.0, atol=1e-3):
                        if not self.I_KNOW_WHAT_I_AM_DOING_AND_WANT_TO_OVERRIDE_THE_PRESOFTMAX_CHECK:
                            raise Exception("The output of model.predict should return the pre-softmax layer. It looks like you are returning the probability vector (post-softmax). If you are sure you want to do that, set attack.I_KNOW_WHAT_I_AM_DOING_AND_WANT_TO_OVERRIDE_THE_PRESOFTMAX_CHECK = True")
                
                # print out the losses every 10%
                if iteration%(self.MAX_ITERATIONS//10) == 0:
                    print("Losses:  ",iteration,self.sess.run((self.loss,self.loss1,self.loss2)))

                # check if we should abort search if we're getting nowhere.
                if self.ABORT_EARLY and iteration%(self.MAX_ITERATIONS//10) == 0:
                    if l > prev*.9999:
                        break
                    prev = l

                # adjust the best result found so far
                for e,(l2,sc,ii) in enumerate(zip(l2s,scores,nimg)):
                    if l2 < bestl2[e] and compare(sc, np.argmax(batchlab[e])):
                        bestl2[e] = l2
                        bestscore[e] = np.argmax(sc)
                    if l2 < o_bestl2[e] and compare(sc, np.argmax(batchlab[e])):
                        o_bestl2[e] = l2
                        o_bestscore[e] = np.argmax(sc)
                        o_bestattack[e] = ii

            # adjust the constant 'c' as needed
            for e in range(batch_size):
                if compare(bestscore[e], np.argmax(batchlab[e])) and bestscore[e] != -1:
                    # success, divide const by two
                    upper_bound[e] = min(upper_bound[e],CONST[e])
                    if upper_bound[e] < 1e9:
                        CONST[e] = (lower_bound[e] + upper_bound[e])/2
                else:
                    # failure, either multiply by 10 if no solution found yet
                    #          or do binary search with the known upper bound
                    lower_bound[e] = max(lower_bound[e],CONST[e])
                    if upper_bound[e] < 1e9:
                        CONST[e] = (lower_bound[e] + upper_bound[e])/2
                    else:
                        CONST[e] *= 10

        # return the best solution found
        o_bestl2 = np.array(o_bestl2)
        return o_bestattack
