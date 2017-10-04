# coding: utf-8

"""A Python function for studying correlations in symbolic sequences, by computing the Mutual Information
	 of symbols at distance n. The result is closely related to the autocorrelation function,
    as described by W. Ebeling and T. Poschel in Entropy and Long-Range Correlations in Literary English,
    EPL(Europhysics Letters) 26.4 (1994): 241."""

#    Copyright (C) 2017 by
#    Iacopo Iacopini <i.iacopini@qmul.ac.uk>
#    All rights reserved.
#    BSD license.

import numpy as np
from collections import Counter


def get_MI(seq, n):
    """Returns the the Mutual Information MI(n) of symbols at distance n along a symbolic sequence (see [1]).
        
        Args
        ----
        seq: list
        List of symbols. It is convenient to map it into a sequence of integers befor passing it to the function.
        
        n: int
        Distance between the symbols for the computation of the Mutual Information MI(n).
        
        Returns
        -------
        MI: float
        Mutual Information MI(n)
                
        References
        ----------
        .. [1] W. Ebeling and T. Poschel (1994).
        "Entropy and Long-Range Correlations in Literary English".
        EPL(Europhysics Letters) 26.4 (1994): 241.
        
        """
        
    L = len(seq)
    labels = np.unique(seq)
    
    freq = Counter(seq)
    p={k: v/float(L) for k,v in freq.items()} #Prob. of having the label i in the sequence (n=1)
    del freq
    
    MI=0. #initial value for the sum
    
    for i in labels:
        idxs_i =  [idx for idx, s in enumerate(seq) if s==i] #where is the label A_i
        for j in labels:
            idxs_j =  [idx for idx, s in enumerate(seq) if s==j] #where is the label A_j
            
            #I take the position where is the A_i, I move to +n and I check how many times I have A_j:
            freq_ij=0
            for idx_i in idxs_i:
                if idx_i+n>=L: continue #this position doesn't exist, end of the sequence
                
                if seq[idx_i+n]==j: freq_ij+=1
            
            #if freq_ij==0: continue #this term will not contribute to I_ij, but the log will give me trouble...skip it
            
            #I take the position where is the A_j, I move to +n and I check how many times I have A_i:
            for idx_j in idxs_j:
                if idx_j+n>=L: continue #this position doesn't exist, end of the sequence
                
                if seq[idx_j+n]==i: freq_ij+=1
            
            if freq_ij==0: continue #this term will not contribute to I_ij, but the log will give me trouble...skip it
            
            #normalizing
            p_ij = 1.*freq_ij/(L-n)
            I_ij = p_ij*np.log2(p_ij/(p[i]*p[j]))
            MI += I_ij
        
    return MI
