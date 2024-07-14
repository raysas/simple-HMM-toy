from HMM import HMM
import numpy as np


def map_nucc_to_obs(nucc):
    if nucc == 'A':
        return 0
    elif nucc == 'C':
        return 1
    elif nucc == 'G':
        return 2
    elif nucc == 'T':
        return 3
    else:
        return -1
    
def map_seq_to_obs(seq):
    '''
    takes a seq of nucc and converts it into an array of obsevrables
    '''
    return [map_nucc_to_obs(nucc) for nucc in seq]

def map_state_to_exon_intron(state):
    if state == 0:
        return 'E'
    elif state == 1:
        return '5'
    elif state == 2:
        return 'I'
    else:
        return 'U'
    
def map_state_seq_to_exon_intron(state_seq):
    return [map_state_to_exon_intron(state) for state in state_seq]

def main():
    states={0, 1, 2}
    observations={0, 1, 2 ,3}
    transitions = np.array([[0.9,0.1,0],
                            [0,0,1],
                            [0,0,1]])
    emissions= np.array([[0.25,0.25,0.25,0.25],
                 [0.05,0,0.95,0],
                 [0.4,0.1,0.1,0.4]])
    
    initial_proba = np.array([1,0,0])

    model= HMM(num_S=len(states),
               num_O=len(observations),
               A=transitions,
               B=emissions,
               pi=initial_proba)
    
    seq='CTTCATGTGAAGCAGACGTAAGTCA'
    O= map_seq_to_obs(seq)
    states = model.virterbi(O)

    print(''.join(map_state_seq_to_exon_intron(states)))
    print(seq)

main()