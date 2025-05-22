import scipy as spy
#import torch 
import autograd.numpy as np
#import jax.numpy as np
#from jax import grad, jit
import matplotlib.pyplot as plt
from pymanopt.manifolds import Euclidean
from pymanopt.optimizers.trust_regions import TrustRegions
import skimage as ski
import rectification_methods as rectif
import pickle, os
from pymanopt import Problem
import pymanopt.manifolds as manifolds
from pymanopt.function import autograd
#from pymanopt.function import jax, pytorch


#from SL3 import SpecialLinearGroup, SpecialLinearGroupModTranslation
from pymanopt.optimizers import SteepestDescent
from pymanopt.function import numpy

def homogeneous(y):
    return np.array([[1,0],[0,1],[0,0]])@y + np.array([0,0,1])

def gradient(H, yi):
    num = H[:2, :2] - np.outer(yi, H[2, :2])
    den = H[2, :] @ (np.linalg.inv(H) @ homogeneous(yi))  # Inversion ici
    return num / den



def embed(H):
    add = np.zeros((3,3))
    add[2,2] = 1
    return  H@np.array([[1,0,0],[0,1,0]]) + add

def get_cost(As, ys, manifold, ord="fro-2-1"):
    n = len(As)
    assert n == len(ys)
    @autograd(manifold)
    def cost(H):
        gradients = np.array([gradient(embed(H), yi) for yi in ys])
        if ord == 'fro':
            return np.sum([np.sum((As[i]-gradients[i])**2) for i in range(n)])/n
        if ord == 'fro-2-1':
            return np.sum([np.sqrt(np.sum((As[i]-gradients[i])**2)) for i in range(n)])/n
    return cost


def optimize(As, ys, filter_function=lambda A: (np.linalg.norm(A-np.eye(2), ord='fro') > 1e-5) and (np.linalg.cond(A) > 1.01) and (np.linalg.cond(A) < 1.5), ord='fro'):
    # throw out As, ys where A is identity or A is badly conditioned
    keep = [filter_function(A) for A in As]
    As = [A for i, A in enumerate(As) if keep[i]]
    ys = [y for i, y in enumerate(ys) if keep[i]]
    
    # get cost function
    manifold = Euclidean(3,2)
    cost = get_cost(As, ys, manifold, ord=ord)
    
    # run optimization problem in R^6
    problem = Problem(manifold, cost=cost)
    solver = TrustRegions()
    X_opt = solver.run(problem)


    return X_opt.point, X_opt.cost 


