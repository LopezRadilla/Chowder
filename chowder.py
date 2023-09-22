#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 13:21:38 2022

@author: jdmolinam
"""

import signal


from numpy import zeros, arange, log, array, diff, exp, savetxt, loadtxt, sqrt, where, ones
from scipy.stats import randint, uniform, rv_discrete
from matplotlib.pyplot import subplots
from numba import jit
import matplotlib.pyplot as plt

def rand1_1():
    """Random 1 or -1."""
    if uniform.rvs() < 0.5:
        return -1
    else:
        return 1

@jit(nopython=True)
def _Changedelta( d_prev, d_new, n_i, trees_gl_i, m_delta, S, T, mk):
    ### First, remove the part in the statistics belonging to the i-th sequence
    for k  in range(d_prev, d_prev+n_i):
        #print( i, j, j-GL['delta'][i] )
        tmp = trees_gl_i[k - d_prev]
        S[k-m_delta] -= tmp
        T[k-m_delta] -= tmp**2
        mk[k-m_delta] -= 1
    
    ### Now add in the sums the new contribution with the new delta
    for k in range(d_new, d_new+n_i):
        #print( i, j, j-GL['delta'][i] )
        tmp = trees_gl_i[k - d_new]
        S[k-m_delta] += tmp
        T[k-m_delta] += tmp**2
        mk[k-m_delta] += 1


class chowder:
    """Class to handle and performe Bayesain analysis for matching annual
       proxy data, typically tree rings.
       name: Name of project
       trees_gl: list with tree ring widths, latests first
       delta: true or inputed (initial value) yr BP of ring 0 for each sequence
       delta_type: 0, delta known. 1 delta unknown.
       delta_prior: prior dist for each delta, as a list of array's or a 1 (uniform),
           default = 1 (uniform).
       data_transf: Convert data, 'raw' (nothing), differences 'diff',
          or proportional incremets 'prop' (default).
    a, alpha, beta: prior hyper parameters.
    """
    
    def __init__(self, name, trees_gl, delta, delta_type, delta_prior=1, data_transf='prop', min_delta=2022-1950, a=0.1, alpha=0.5, beta=1.0):

        self.name = name
        self.trees_gl = trees_gl

        self.a = a
        self.alpha = alpha
        self.beta = beta
        ### It will read all files in the directory

        
        if data_transf == 'raw':
            self.trees_raw = self.trees_gl 
        if data_transf == 'diff':
            self.trees_raw = self.trees_gl.copy()
            self.trees_gl = []
            for tree in self.trees_raw:
                self.trees_gl.append( diff(tree) ) #Differences
        if data_transf == 'prop':
            self.trees_raw = self.trees_gl.copy()
            self.trees_gl = []
            for tree in self.trees_raw:
                self.trees_gl.append( diff(tree)/tree[:-1]) #Proportional increments            
        if data_transf == 'log':
            self.trees_raw = self.trees_gl.copy()
            self.trees_gl = []
            for tree in self.trees_raw:
                self.trees_gl.append( log(tree[1:]/tree[:-1]))
        self.m = len( self.trees_gl ) ##Number of trees 

        self.delta = array(delta)
        if delta_prior is not list:
            delta_prior = ones(self.m)
        self.delta_sel = where(array(delta_type) == 1)[0]
        self.m_unknown = self.delta_sel.size
        self.delta_known_sel = where(array(delta_type) == 0)[0]
        self.min_delta = min_delta
        
        ## Sizes of tree ring records for each tree
        self.n = zeros( self.m , dtype=int)
        for i in range(self.m):
            self.n[i] = len(self.trees_gl[i])
        self.Sn = sum(self.n)
            
        self.min_n = min(self.n) # 45
        self.max_n = max(self.n) # 182
        
        self.m_delta = 1950-2022 #2022 BP
        self.M_delta = 1000 # BP
        self.S = zeros(self.M_delta-self.m_delta)
        self.T = zeros(self.M_delta-self.m_delta)
        self.mk = zeros(self.M_delta-self.m_delta, dtype=int)
        
        self.CalPost1()
        
        """
        ### Testing
        
        print( self.sumlogak, self.sumVk)        
        #self.CalPost2()
        #print( self.sumlogak, self.sumVk)
        
        ### Change the delta an recalculate
        i=4
        self.Changedelta( i, 20)
        print( self.sumlogak, self.sumVk, self.logpost)
        self.CalPost1() # Test
        print( self.sumlogak, self.sumVk, self.logpost)
        """
    
    def ReadGibbsOutput(self, fnam):
        """Read the previouly generated Gibbs output."""
        self.sample = loadtxt( fnam, delimiter=',')
        #Set unknown deltas as the last iteration)
        self.delta[self.delta_sel] = self.sample[-1,:-1].astype(int) 
        self.CalPost1()
        
    
    def CalPost1(self):
        """Calculate the current statistics,
           using the master chronology.
        """
        self.S *= 0.0
        self.T *= 0.0
        self.mk *= 0
        for k in range( min(self.delta), max(self.delta+self.n)):
            for i in range(self.m):
                if 0 <= k - self.delta[i] < self.n[i]:
                    #print( i, j, j-GL['delta'][i] )
                    tmp = self.trees_gl[i][k - self.delta[i]]
                    self.S[k-self.m_delta] += tmp
                    self.T[k-self.m_delta] += tmp**2
                    self.mk[k-self.m_delta] += 1
        #print( sum(self.S), sum(self.T), sum(self.mk))
        self.sumlogak = sum(log(self.a + self.mk))
        self.sumVk = sum(self.T - (self.S**2)/(self.a + self.mk))
        ### Calculate the current log posterior.
        self.logpost = -0.5*self.sumlogak\
            -(self.alpha + (self.Sn)/2)*log(self.beta + 0.5*self.sumVk)            

    def Changedelta(self, i, d_new):
        """Change delta_i to the value of d_new, and update the stats and log post."""
        
        d_prev = self.delta[i]
        if d_new == d_prev:
            ### Nothing to do
            return
        ### The range for k is only relevant for the i-th sequence

        _Changedelta( d_prev, d_new, self.n[i], self.trees_gl[i],\
                   self.m_delta, self.S, self.T, self.mk)
        ### First, remove the part in the statistics belonging to the i-th sequence
        """
        for k  in range(d_prev, d_prev+self.n[i]):
            #print( i, j, j-GL['delta'][i] )
            tmp = self.trees_gl[i][k - d_prev]
            self.S[k-self.m_delta] -= tmp
            self.T[k-self.m_delta] -= tmp**2
            self.mk[k-self.m_delta] -= 1
        
        ### Now add in the sums the new contribution with the new delta
        for k in range(d_new, d_new+self.n[i]):
            #print( i, j, j-GL['delta'][i] )
            tmp = self.trees_gl[i][k - d_new]
            self.S[k-self.m_delta] += tmp
            self.T[k-self.m_delta] += tmp**2
            self.mk[k-self.m_delta] += 1
        """
        self.delta[i] = d_new
        self.sumlogak = sum(log(self.a + self.mk))
        self.sumVk = sum(self.T - (self.S**2)/(self.a + self.mk))
        ### Calculate the current log posterior.
        self.logpost = -0.5*self.sumlogak\
            -(self.alpha + (self.Sn)/2)*log(self.beta + 0.5*self.sumVk)            


    def CalPost2(self):
        """Calculate the current statistics,
           without relaying on the master chronology.
        """
        m_delta = min(self.delta)
        M_delta = max(self.delta) + self.max_n
        self.sS2 = 0.0
        self.sT  = 0.0
        self.sumlogak =0.0
        for k in range( m_delta, M_delta):
            mk = 0
            S = 0.0
            T = 0.0
            for i in range(self.m):
                if 0 <= k - self.delta[i] < self.n[i]:
                    #print( i, j, j-GL['delta'][i] )
                    tmp = self.trees_gl[i][k - self.delta[i]]
                    S += tmp
                    T += tmp**2
                    mk += 1
            self.sS2 += (S**2)/(self.a + mk)
            self.sT += T
            self.sumlogak += log(self.a + mk)
        self.sumVk = self.sT - self.sS2
        ### Calculate the current log posterior.
        self.logpost = -0.5*self.sumlogak\
            -(self.alpha + (self.Sn)/2)*log(self.beta + 0.5*self.sumVk)            
        
    def LogPost(self):
        return self.logpost
     
    def PlotBasic(self, sel=None, color='black', linestyle='solid', ax=None):
        rgb=("b","g","r","c","m","y","coral","olive","tan","lime","violet","plum","pink","peru","maroon","orange","aqua","teal","darkorange","indigo","brown","b","g","r","m")
        if ax is None:
            fig, ax = subplots()
            ax.invert_xaxis()
        if sel is None:
            sel = range(self)
        for i in sel:
            
            t = arange(self.delta[i]-143, self.delta[i]+self.n[i]-143, 1)
            ax.plot(t, self.trees_gl[i], linestyle=linestyle, linewidth=1, color=rgb[i],alpha=0.4)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
        return ax
    
    def PlotSignal( self, mul=1, ax=None):
        """Plot the signal, currently only for the current beta's."""
        if ax is None:
            fig, ax = subplots()
            ax.invert_xaxis()
        self.M = self.S/(self.a + self.mk) # Mean
        self.V = self.T/(self.a + self.mk) - self.M**2 + 1e-8 # Variance
        savetxt( "Mean.csv", (self.M,self.V), delimiter=',')

        t = arange(self.m_delta, self.M_delta, 1)
        ### Plot the signal-to-noise ratio
        #ax.plot( t, M/sqrt(V), '-', color="magenta", linewidth=2)
        ax.plot( t, self.M, '-', color="gold", linewidth=2)
        ax.plot( t, self.M-mul*sqrt(self.V), '--', color="gold", linewidth=1)
        ax.plot( t, self.M+mul*sqrt(self.V), '--', color="gold", linewidth=1)
        ax.plot( t, (self.a + self.mk)/(self.a + self.m), '-', color="brown", linewidth=2)
        ax.set_xlim((max(self.delta+self.n), min(self.delta)))
        
        
        
    def MCMC(self, T=1000):
        lp = self.logpost #Current log post
        self.sample = zeros((T+1, self.m+1))
        self.sample[0,:-1] = self.delta
        self.sample[0,-1] = lp
        for t in range(T):
            if (t % 1000) == 0:
                print("Iteration %6d of %6d." % (t,T))
            ###Proposal: select one delta from 1 to m and move it up or down by 1
            i = randint.rvs( low=1, high=self.m)
            d_current = self.delta[i]
            d_prop = d_current + rand1_1()
            self.Changedelta( i, d_prop)
            if uniform.rvs() < exp(self.logpost-lp): #Accept
                lp = self.logpost
            else:
                self.Changedelta( i, d_current)
            self.sample[ t+1,:-1] = self.delta
            self.sample[ t+1,-1] = lp
    
    def Swipe(self, i):
        """Iterate through delta i calculating the log post."""
        sel = list(range(self.m))
        sel.pop(i)
        sel = array(sel)
        d_erliest = max(self.delta[sel]+self.n[sel])
        d_latest = min(self.delta[sel]) - self.n[i]
        d_latest = max( d_latest, self.min_delta)
        self.out = zeros((d_erliest-d_latest,2))
        self.Changedelta( i, d_latest) ##Set and calculate post at first k
        self.out[0,0] = d_latest
        self.out[0,1] = self.logpost
        for d in range(d_latest+1,d_erliest):
            self.Changedelta( i, d)
            self.out[d-d_latest,0] = d
            self.out[d-d_latest,1] = self.logpost

        self.out[:,1] = exp(300+self.out[:,1]-max(self.out[:,1]))
        self.out[:,1] /= sum(self.out[:,1])
        tmp = rv_discrete(name='tmp', values=(self.out[:,0], self.out[:,1]))
        return tmp.rvs() 

    def sigint_handler( self, signal, frame):
        print('Gibbs interrupted, finishing current swipe ...')
        self.stop_gibbs = True

    def Gibbs( self, T):
        self.M = self.S/(self.a + self.mk) # Mean
        self.V = self.T/(self.a + self.mk) - self.M**2 + 1e-8 # Variance
        z=len(self.M)
        self.sample = zeros((T+1, self.m_unknown+1+2*z))
        
        self.sample[0,:self.m_unknown] = self.delta[self.delta_sel]
        self.sample[0,self.m_unknown:self.m_unknown+z] = self.M
        self.sample[0,self.m_unknown+z:self.m_unknown+2*z] = self.V
        self.sample[0,-1] = self.logpost #Current log post
        self.stop_gibbs = False # Catch the Keyboar interrupt
        dfl_handler = signal.signal( signal.SIGINT, self.sigint_handler)
        print( "%6s, %3s, %s" % ("It.", "Sel", "LogPost"))
        for t in range(T):
            ####select a sequences from the unkonwn delta's
            i = self.delta_sel[randint.rvs(low=0, high=self.m_unknown)]
            d = self.Swipe(i) ###Random scan Gibbs sampler
            self.Changedelta( i, d) ### Move, accepted
            self.M = self.S/(self.a + self.mk) # Mean
            self.V = self.T/(self.a + self.mk) - self.M**2 + 1e-8 # Variance
            self.sample[ t+1,:self.m_unknown] = self.delta[self.delta_sel]
            self.sample[t+1,self.m_unknown:self.m_unknown+z] = self.M
            self.sample[t+1,self.m_unknown+z:self.m_unknown+2*z] = self.V
            self.sample[ t+1,-1] = self.logpost
            
            print( "%6d, %3d, %e" % (t, i, self.logpost) )
            if self.stop_gibbs:
                print("done.")
                signal.signal( signal.SIGINT, dfl_handler)
                self.stop_gibbs = False
                break
                
    
    def SaveGibbsOutput(self, fnam="gibbs.csv"):
        self.M = self.S/(self.a + self.mk) # Mean
        self.V = self.T/(self.a + self.mk) - self.M**2 + 1e-8 # Variance
        savetxt( fnam, self.sample, delimiter=',')
        

"""
            #d_prev = self.delta[i]
            d_prev = d-1
            d_new = d
            ### The range for k is only relevant for the i-th sequence
            
            ### Remove the part in the statistics belonging to the
            ### first k sequence
            #for k  in range(d_prev, d_prev+self.n[i]):
            k = d_prev
            #print( i, j, j-GL['delta'][i] )
            tmp = self.trees_gl[i][k - d_prev]
            self.S[k-self.m_delta] -= tmp
            self.T[k-self.m_delta] -= tmp**2
            self.mk[k-self.m_delta] -= 1
            
            ### Now remove and add in the sums the new contribution with the new delta
            for k in range(d_new, d_new+self.n[i]-1):
                #print( i, j, j-GL['delta'][i] )
                tmp0 = self.trees_gl[i][k - d_prev]
                tmp1 = self.trees_gl[i][k - d_new]
                self.S[k-self.m_delta] += tmp1 - tmp0
                self.T[k-self.m_delta] += tmp1**2 -tmp0**2
                self.mk[k-self.m_delta] += 1

            ### Add the contribution of the last k
            #for k in range(d_new, d_new+self.n[i]):
            k = d_new+self.n[i]-1
            #print( i, j, j-GL['delta'][i] )
            tmp = self.trees_gl[i][k - d_new]
            self.S[k-self.m_delta] += tmp
            self.T[k-self.m_delta] += tmp**2
            self.mk[k-self.m_delta] += 1

            self.sumlogak = sum(log(self.a + self.mk))
            self.sumVk = sum(self.T - (self.S**2)/(self.a + self.mk))
            ### Calculate the current log posterior.
            self.logpost = -0.5*self.sumlogak\
                -(self.alpha + (self.Sn-1)/2)*log(self.beta + 0.5*self.sumVk)            
 
"""
