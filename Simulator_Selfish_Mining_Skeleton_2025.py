"""
Simulation of selfish mining 
Analyze and simulate relative pool revenue (Rpool) with confidence intervals

David Starobinski
April 11, 2025
"""



# import required packages - 
# do not add other packages

import math
import numpy as np
import scipy as sp
import scipy.stats as stats
import matplotlib.pyplot as plt



'''
Define simulation Global Parameters
'''

ALPHA = [0.1, 0.2, 0.25, 0.3,0.33333, 0.4, 0.45, 0.475] # Selfish Mining Power (Pool Size)
NUMALPHA = len(ALPHA)
GAMMA = 0.5 # Change this parameter as needed
ITERATIONS = 30 # number of independent simulations
CONF_LEVEL = 0.05 # confidence interval is 100*(1-CONF_LEVEL) percent
STEPS = 10**4 #Number of steps in each iteration




# Average relative pool revenue in each iteration
Rpool = np.zeros((ITERATIONS,NUMALPHA)) 

'''
Main Simulator Loop
'''
for a in range(NUMALPHA):
    for itr in range(ITERATIONS):
        curr_state = 0  # initialize DTMC state
        r_pool =       # fill code - initialize running sum of mining pool revenue
        r_other =      # fill code - initialize running sum of honest nodes revenue
        flag =         # fill code - initialize flag variable (1 only if in state 0')
        for s in range(STEPS): # At each step, update state and collected revenue, as appropriate

            '''
            Fill code here
            '''
            
        Rpool[itr,a] = r_pool /(r_pool+r_other)
            
    
    
'''
Compute Statistics     
'''
SampleM_Rpool =  # Fill code - Sample Mean of Rpool averaged over all iterations
Error =  # Fill code - Compute confidence interval 

'''
Print simulation results against analytical results
'''
print('Statistical results for Gamma = %.3f' %GAMMA)
Analytical_Rpool = np.zeros(NUMALPHA) # Analytical values
for a in range(NUMALPHA):

        '''
        Fill code here
        '''


    
'''
Plot simulation results against analytical results
'''

# Plot of Honest mining
'''
Fill code here
'''
# Plot of Analytical Rpool    
'''
Fill code here
'''
#Plot of Simulated Rpool with confidence intervals
'''
Fill code here
'''
#Add title, label, legend, and display
'''
Fill code here
'''


