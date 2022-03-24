'''
# This is the ESN package
# My code is partly inspired from https://github.com/mrdragonbear/EchoStateNetworks and https://github.com/cknd/pyESN, 
# and they are under MIT license.
'''

import numpy as np

def identity(x):
    return x


class ESN():
    def __init__(self, adjacency_matrix, noise, random_state, input_dim, output_dim, feedback, print_performance, spectral_radius=0.95, out_activation = identity, inv_out_activation = identity):
        '''
        Args:
        input_dim: dimensions of the input
        output_dim: dimensions of the output
        spectral_radius: the spectral radius of the recurrent weight matrix 
        out_activation: the output activation function
        adjacency_matrix: the adjacency matrix of a network
        noise: noise added to neurons, regularization
        random_state: a positive integer seed for random state
        inv_out_activation: inverse of the out activation function
        feedback: a bullin variable determines whether the feed the output back to the reservoir or not
        print_performance: a bullin variable determines to print out the training error or not
        '''
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.spectral_radius = spectral_radius
        self.adjacency_matrix = adjacency_matrix
        self.out_activation = out_activation
        self.inv_out_activation = inv_out_activation
        self.noise = noise
        self.random_state = random_state
        self.random_state_ = np.random.RandomState(random_state)
        self.feedback = feedback
        self.print_performance = print_performance
        
        self.initweights()
        
    def initweights(self):
        # generate weighted matrices
        # random matrix around 0:
        W_temp = self.random_state_.rand(self.adjacency_matrix.shape[0], self.adjacency_matrix.shape[0]) - 0.5 
        # randomize the original adjacency matrix
        W = W_temp * self.adjacency_matrix
        # spectral radius of W
        radius = np.max(np.abs(np.linalg.eigvals(W)))
        # rescale the radius to reach the given radius
        self.W = W * (self.spectral_radius/radius)
        
        # randomize input weighted matrix
        self.W_in = self.random_state_.rand(self.adjacency_matrix.shape[0], self.input_dim) * 2 - 1
        
        # randomize feedback weighted matrix
        self.W_feedb = self.random_state_.rand(self.adjacency_matrix.shape[0], self.output_dim) * 2 - 1
        
    def _update(self, state, input_array, output_array):
        '''
        update the model by one step
        
        '''
        if self.feedback: 
            preactivation = (np.dot(self.W, state) + np.dot(self.W_in, input_array) + np.dot(self.W_feedb, output_array)) 
            
        else: 
            preactivation = (np.dot(self.W, state) + np.dot(self.W_in, input_array))
        
        return (np.tanh(preactivation) + self.noise * (self.random_state_.rand(self.adjacency_matrix.shape[0]) - 0.5))
            
            
            
    def fit(self, inputs, outputs):
        '''
        train the output weighted matrix
        
        Args:
         inputs: array of dimensions (N_training_samples x input_dim)
         outputs: array of dimensions (N_training_samples x output_dim)
         
        Returns:
            the network's output on the training data, using the trained weights
        '''
        inputs = np.reshape(inputs, (len(inputs), -1))
        outputs = np.reshape(outputs, (len(outputs), -1))

        
        states = np.zeros((inputs.shape[0], self.adjacency_matrix.shape[0]))
        for n in range(1, inputs.shape[0]):
            states[n, :] = self._update(states[n - 1], inputs[n, :], outputs[n - 1, :])
            
        
        # learn the weights, i.e. find the linear combination of collected
        # network states that is closest to the target output
        # disregard the first few states:
        transient = min(int(inputs.shape[1] / 10), 100)
        # include the raw inputs:
        extended_states = np.hstack((states, inputs))
        # Solve for W_out:
        self.W_out = np.dot(np.linalg.pinv(extended_states[transient:, :]), self.inv_out_activation(outputs[transient:, :])).T

        # remember the last state for later:
        self.laststate = states[-1, :]
        self.lastinput = inputs[-1, :]
        self.lastoutput = outputs[-1, :]
        
        
        # apply learned weights to the collected states:
        pred_train = self.out_activation(np.dot(extended_states, self.W_out.T))
        if self.print_performance == True:
            print("The training error is", np.sqrt(np.mean((pred_train - outputs)**2)))
        return pred_train
    
    def predict(self, inputs, continuation=True):
        '''
        Apply the learned weights to the network's reactions to new input.
        Args:
            inputs: array of dimensions (N_test_samples x n_inputs)
            continuation: if True, start the network from the last training state
        Returns:
            Array of output activations
        '''
        inputs = np.reshape(inputs, (len(inputs), -1))
        n_samples = inputs.shape[0]


        if continuation:
            laststate = self.laststate
            lastinput = self.lastinput
            lastoutput = self.lastoutput
        else:
            laststate = np.zeros(self.adjacency_matrix.shape[0])
            lastinput = np.zeros(self.input_dim)
            lastoutput = np.zeros(self.output_dim)

        inputs = np.vstack([lastinput, inputs])
        states = np.vstack([laststate, np.zeros((n_samples, self.adjacency_matrix.shape[0] ))])
        outputs = np.vstack([lastoutput, np.zeros((n_samples, self.output_dim))])

        for n in range(n_samples):
            states[n + 1, :] = self._update(states[n, :], inputs[n + 1, :], outputs[n, :])
            outputs[n + 1, :] = self.out_activation(np.dot(self.W_out,np.concatenate([states[n + 1, :], inputs[n + 1, :]])))

        return self.out_activation(outputs[1:])