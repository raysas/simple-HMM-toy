'''
HMM implementation

```
from HMM import HMM
model = HMM(S,O,A,B,pi)
```
'''


#----------------------------------------------
# "you won't understand it until you try it"
#----------------------------------------------

import numpy as np

class HMM:
    '''
    # Hidden Markov Model class:
    --------------------------
        
    ## Attributes:
    ------------
    S: set of states  
    O: set of observations  
    A: transition probabilities matrix  
    B: emission probabilities matrix  
    pi: initial probabilities vector  

    λ=(A, B, π)

    Each attribute is provided with a setter and a getter method
    
    ## Methods:
    ------------
    - forward_algorithm: computes the forward algorithm  
    - backward_algorithm: computes the backward algorithm  
    - viterbi_algorithm: computes the viterbi algorithm

    '''
    def __init__(self, S, O, A, B, pi):
        self.S= S
        # self.O= O
        self.A = A
        self.B = B
        self.pi = pi
        self.lambda_ = (self.A, self.B, self.pi)

    def __init__(self):
        '''
        Add manually using setters and getters the following attributes:
        S: set of states
        O: set of observations
        A: transition probabilities matrix
        B: emission probabilities matrix
        pi: initial probabilities vector

        e.g.,  
        ```
        model= HMM()
        model.set_states(S)
        model.set_observations(O)
        model.set_transition_probabilities(A)
        model.set_emission_probabilities(B)
        model.set_initial_probabilities(pi)
        ```
        '''
        self.S= None
        # self.O = None
        self.A = None
        self.B = None
        self.pi = None
        self.lambda_ = None

    # -- setters and getters
    def set_states(self, S):
        self.S =S
    def get_states(self):
        return self.S
    
    # def set_observations(self, O):
    #     self.O = O
    # def get_observations(self):
    #     return self.O
    
    def set_transition_probabilities(self, A):
        self.A = A
        self.update_lambda(self.A, self.B, self.pi)

    def get_transition_probabilities(self):
        return self.A
    
    def set_emission_probabilities(self, B):
        self.B = B
        self.update_lambda(self.A, self.B, self.pi)

    def get_emission_probabilities(self):
        return self.B
    
    def set_initial_probabilities(self, pi):
        self.pi = pi
        self.update_lambda(self.A, self.B, self.pi)
    def get_initial_probabilities(self):
        return self.pi
    
    def set_lambda(self, A, B, pi):
        self.lambda_ = (A, B, pi)
        self.A = A
        self.B = B
        self.pi = pi

    def get_lambda(self):
        return self.lambda_
    

    # -- algorithms

    def forward(self, O):
        A, B, pi = self.get_lambda()

        N= A.shape[0]
        T= len(O)

        alpha = np.zeros((N,T))

        for i in range(N):
            for j in range(T):
                if j==0: #base case
                    alpha[i,0]= pi[i]*B[i,O[0]]
                else:
                    for k in range(N):
                        alpha[i,j]+= alpha[k,j-1]*A[k,i]
                    alpha[i,j]*= B[i,O[j]] #can add it to the loop, independent of k

        return alpha
    
    def backward(self, O, termination=None):
        A, B, pi = self.get_lambda()
        N= A.shape[0]
        T= len(O)
        beta = np.zeros((N,T))
        for i in range(N):
            for j in range(T-1, -1, -1):
                if j==T-1:
                    beta[i, j]= termination[i] if termination is not None else 1
                else:
                    for k in range(N):
                        beta[i,j]+= A[i,k]*B[k,O[j+1]]*beta[k,j+1]

        return beta



    