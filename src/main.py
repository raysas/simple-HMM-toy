from HMM import HMM
import numpy as np
import pandas as pd

def main():
    states={0, 1, 2}
    observations={0, 1, 2 ,3}
    transitions = np.array([[0,1,0,0,0],
                  [0,0.5,0.5,0,0],
                  [0,0,0,1,0],
                  [0,0,0,0,1],
                  [0,0,0,0,0]])
    emissions= np.array([[0.25,0.25,0.25,0.25],
                 [0.05,0,0.95,0],
                 [0.4,0.1,0.1,0.4]])
    
    initial_proba = np.array([1,0,0])

    model= HMM(S=states,
               O=observations,
               A=transitions,
               B=emissions,
               pi=initial_proba)