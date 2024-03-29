


# Implementations of the sparse coding algorithm from [Olshausen & Field, *Nature* 1996]. 
# Parts of the code were from the user [takyamamoto](https://github.com/takyamamoto/SparseCoding-OlshausenField-Model) 

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from tqdm.notebook import tqdm
import math
import utils


### Have the different thresholding functions as a separate class and let the other classes use them via composition
class Thresholding:

    def __init__(self,lr_r,numNonZeroR=20):

        self.lr_r = lr_r
        self.numNonZeroR = numNonZeroR

    #DIFFERENT THRESHOLDING FUNCTIONS

    # thresholding function of S(x)=|x|
    def softΤhresholding(self,r,threshold):
        return np.maximum(r - threshold, 0) - np.maximum(-r - threshold, 0)


    #hard thresholding. everything below a threshold is zero  https://www.pure.ed.ac.uk/ws/files/17821312/BD_JFAA07.pdf
    def hardΤhresholding1(self,r,threshold):
        #print(f'shape of r is {x.shape}')
        thresholdSqrt = threshold**0.5
        z = np.copy(r)
        z[np.where(np.abs(r)<thresholdSqrt)] = 0
        return z


    #helper functions for iterative_halfΤhresholding
    def phi(self,x,threshold):
        k = (threshold/8)*((np.abs(x)/3)**(-1.5))
        return np.arccos(k)

    def fHalf(self,x, threshold):
    
        phita = self.phi(x,threshold)
        val = ((2*np.pi)/3) - (2/3)*phita
        return (2/3)*x*(1 + np.cos(val))

    #https://ieeexplore.ieee.org/document/6205396
    def halfΤhresholding(self,r,threshold):

        H = np.zeros(r.shape)
        thresh = (np.cbrt(54)/4)*((threshold)**(2/3))
        ind = np.where(np.abs(r)> thresh)
        H[ind] = self.fHalf(r[ind],threshold) 
        return H

    #from Luca's Master student Marta Lazzaretti
    def CEL0Thresholding(self,r,threshold):
        a = 1
        num = (np.abs(r) - math.sqrt(2*threshold)*a*self.lr_r)
        num[num<0] = 0
        den = 1-a**2*self.lr_r
        return np.sign(r)*np.minimum(np.abs(r),np.divide(num,den))*(a**2*self.lr_r<1)



##########GENERAL FUNCTION used for both sparse classes
def calculateError(error):
    reconError = np.mean(error**2)
    #sparsity_r = self.lmda*np.mean(np.abs(self.r)) 
    #return recon_error + sparsity_r
    return reconError




### Network implementation

#Implement the network based on different regularizers that sparsify the activities
# Execute the `__call__` function to update the `r` coeficients and the `Phi` weight matrix. 
#Update `r` until it converges, and when it converges, set `training` to `True` and update `Phi`. 

#FINDS BOTH PHI AND R
class SparseModel:
    def __init__(self, numInputs, numUnits, batchSize,lmda,flagMethod, lr_r=1e-2, lr_Phi=1e-2):

        self.lr_r = lr_r # learning rate of r
        self.lr_Phi = lr_Phi # learning rate of Phi
        self.numIterations = 5000
        self.decayRateR  = self.lr_r/50
        self.decayRatePhi = self.lr_Phi/50
        iterationsVec = np.arange(self.numIterations)
        self.lr_rV = (1/(1+self.decayRateR*iterationsVec))*self.lr_r
        self.lrPhiV = (1/(1+self.decayRatePhi*iterationsVec))*self.lr_Phi
        self.iterPhi = 0

        self.lmda = lmda # regularization parameter
        self.threshold = self.lmda*self.lr_r
        #self.threshold = self.lmda

        self.numInputs = numInputs
        self.numUnits = numUnits
        self.batchSize = batchSize
        self.flagMethod = flagMethod

        self.objThreshold = Thresholding(self.lr_r)

        # Weights
        Phi = np.random.randn(self.numInputs, self.numUnits).astype(np.float32)
        self.Phi = Phi * np.sqrt(1/self.numUnits)
        
    
    def initializeStates(self,rInit):
        #it will be of size batchSizeXnumUnits
        self.r = rInit
        #print(np.sum(self.r))
        
    def normalizeRows(self):
        #the numerator is a vector the np.maximum() operator makes sure that the elements are greater than 1e-8
        #so basically the phi vectors are normalized to unit vectors
        self.Phi = self.Phi / np.maximum(np.linalg.norm(self.Phi, ord=2, axis=0, keepdims=True), 1e-8)

    #we run a learning decay of the learning rates for r and Phi so we need the number of iterations: iterR AND iterPhi
    def __call__(self, inputs, iterR,training=True):
        # Updates  

        error = inputs - self.r @ self.Phi.T
        r = self.r + self.lr_rV[iterR] * error @ self.Phi
        
        if self.flagMethod == 'soft':
            self.r = self.objThreshold.softΤhresholding(r, self.threshold)
        elif self.flagMethod == 'hard1':
            self.r = self.objThreshold.hardThresholding1(r, self.threshold)
        elif self.flagMethod == 'half':
            self.r = self.objThreshold.halfThresholding(r,self.threshold)
        elif self.flagMethod == 'CEL0':
            self.r = self.objThreshold.CEL0Thresholding(r,self.lmda)

        if training:  
            #print(f'iteration for learning Phi is {self.iterPhi} and the learning rate of phi is {self.lrPhiV[self.iterPhi]}')
            error = inputs - self.r @ self.Phi.T
            dPhi = error.T @ self.r
            self.Phi += self.lrPhiV[self.iterPhi] * dPhi
            self.iterPhi +=1

            
        return error, self.r



## Function running simulation for SparseModel
#The for loop updates until r converges, and then updates the weight matrix Phi.
#ntMax = maximum number of iterations for convergence
#eps  small value which determines convergence

def runModelSim(model,numIter,batchSize,inputsAll,rAll, ntMax = 5000,eps = 1e-2):
    
    errorList = [] # List to save errors
    rAll_ = [] #gather all r to do the analysis 
    
    # Run simulation
    for iter_ in tqdm(range(numIter)):
        
        inputs = inputsAll[iter_*batchSize:(iter_+1)*batchSize,:] # Input image patches

        rInit = rAll[iter_*batchSize:(iter_+1)*batchSize,:]
        #print(np.sum(rInit))
        #print(f'rInit shape from runModelSim function is {rInit.shape}')
        model.initializeStates(rInit) # Reset r's
        model.normalizeRows() # Normalize weights
    
        # Input an image patch until latent variables are converged 
        rTm1 = model.r # set previous r (t minus 1)

        for t in range(ntMax):
            # Update r without update weights 
            error, r = model(inputs, t,training=False)
            dr = r - rTm1 

            # Compute norm of r
            drNorm = np.linalg.norm(dr, ord=2) / (eps + np.linalg.norm(rTm1, ord=2))
            rTm1 = r # update rTm1
        
            # Check convergence of r, then update weights
            if drNorm < eps:
                #after the r's batch converges, you update the phi
                
                error, r = model(inputs, t,training=True)
                rAll_.append(r)
                break
        
            # If failure to convergence, break and print error
            if t >= ntMax-2: 
                print("Error at patch:", iter_)
                print(drNorm)
                break
   
        errorList.append(calculateError(error)) # Append errors
    
        # Print moving average error
        if iter_ % 100 == 99:  
            print("iter: "+str(iter_+1)+"/"+str(numIter)+", Moving error:",
                  np.mean(errorList[iter_-99:iter_]))

    #return model,rAll_, errorList       
    #############################################################
    rAll = np.array(rAll_)
    print(f'r shape is initially {rAll.shape}')
    rAll = np.reshape(rAll,(rAll.shape[0]*rAll.shape[1],-1))
    print(f'... and now {rAll.shape}')
    
    rActivityFreqUnit = np.sum(rAll!=0,axis=0)/rAll.shape[0]
    print('did the operation for rActivityFreqUnit')
    rNumActiveInstances = np.sum(rAll!=0,axis=1,keepdims = True)
    print('did the operation for rNumActiveInstances')
    errorRates = np.array(errorList)
    print('did the operation for errorRates')

    return model.Phi,rActivityFreqUnit, rNumActiveInstances, errorRates     
    



