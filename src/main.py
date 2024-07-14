from HMM import HMM
import numpy as np
import pandas as pd

def main():
    states={'start','exon',"5cap",'intron','end'}
    observations={'A','C','G','T'}
    A = np.array([[0,1,0,0,0],
                  [0,0.5,0.5,0,0],
                  [0,0,0,1,0],
                  [0,0,0,0,1],
                  [0,0,0,0,0]])
    E= np.array([[0.25,0.25,0.25,0.25],
                 [0.05,0,0.95,0],
                 [0.4,0.1,0.1,0.4]])
    
    pi = np.array([1,0,0,0,0])

    model= HMM(S=states,
               O=observations,
               A=A,
               B=E,
               pi=pi)