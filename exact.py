#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 12:46:33 2019

@author: juan
"""
import sys
import numpy as np
from numpy.linalg import eig
import scipy.sparse as sparse

print('g:', float(sys.argv[1]), '\ndelta: ', float(sys.argv[2]), '\nN_exc: ', int(sys.argv[3]), '\nN_sites: ', int(sys.argv[4]))

g = float(sys.argv[1])
delta = float(sys.argv[2])
N_exc = int(sys.argv[3])
N = int(sys.argv[4])

w0 = 1.
J = 0.4
ks = np.arange(N)
w = w0 - 2 * J * np.cos(2 * np.pi / N * ks)


def f_deltar(deltar, delta, g):
    return delta * np.exp(-2 * np.sum((-g / (np.sqrt(N) * (deltar + w))) ** 2))

def f_f(deltar, g):
    return -g / (np.sqrt(N) * (deltar + w))

def f_analisis(deltas, gs):
    deltar = 0

    analisis = {}

    for delta in deltas:
        delta = round(delta, 2)
        analisis[delta] = {}

        for g in gs: 
            g = round(g, 4)

            while True:
                change = abs(deltar - f_deltar(deltar, delta, g))
                deltar = f_deltar(deltar, delta, g)

                if change < 1e-9: #cuando la mejora es ya muy pequeña cierro el bucle
                    break 

            analisis[delta][g] = deltar #guardo los valores de deltar para poder usarlos más tarde

    return analisis

def sorted_eigsystem(H):
    vals, vects = eig(H)
    idx = np.argsort(vals)
    vals = vals[idx]
    vects = vects[:,idx]
    
    return Eigensystem(vals, vects)

class Eigensystem:
    def __init__(self, vals, vects):
        self.vals = vals
        self.vects = vects
        self.size = len(vals)

def gen_a(dim):
    aux = np.zeros((dim - 1, dim - 1))
    
    for n in range(dim - 1):
        aux[n, n] = np.sqrt(n + 1)
    
    aux = np.append(aux, [np.zeros(dim - 1)], axis=0)
    aux = np.append(np.array([np.zeros(dim)]).T, aux, axis=1)
    
    return sparse.coo_matrix(aux)
  
    

class exact_Diag:
    def __init__(self, g, Delta=1, N_exc = 2):
        self.g = g
        self.Delta = Delta
        self.DeltaR = f_analisis([self.Delta], [self.g])[self.Delta][self.g]
        self.fk = f_f(self.DeltaR, self.g)
        self.L = self.fk.size
        self.N_exc = N_exc
        
        # Qubit operators
        sx = sparse.coo_matrix([[0,1],[1,0]])
        sz = sparse.coo_matrix([[-1,0],[0,1]])
        Hq = self.DeltaR * sparse.kron(sz, sparse.eye(self.N_exc ** self.L)) / 2.0
        # creation-anhilation
        a  = gen_a(self.N_exc)
        ad = a.T
        
        Hph = np.zeros((2 * self.N_exc ** (self.L), 2 * self.N_exc ** (self.L)))
        for i in range(self.L):
            Hph += w[i] * kron_4(sparse.eye(2), sparse.eye(self.N_exc ** i), ad @ a, sparse.eye(self.N_exc ** (self.L - i - 1)))
        
        Hc = np.zeros((2 * self.N_exc ** (self.L), 2 * self.N_exc ** (self.L)))
        for i in range(self.L):
            Hc += g / np.sqrt(self.L) * kron_4(sx, sparse.eye(self.N_exc ** i), a + ad, sparse.eye(self.N_exc ** (self.L - i - 1)))
        
        self.H = Hq + Hph + Hc
    
    def diag(self):
        self.eigsys = sorted_eigsystem(self.H)
        
    def n_photons(self):
        try:
            a = gen_a(self.N_exc)
            ad = a.T

            self.GSphotons = np.zeros(self.L, dtype = 'complex')
            self.E1photons = np.zeros(self.L, dtype = 'complex')
            for n in range(self.L):
                for k in range(self.L):
                    for p in range(self.L):
                        if k < p:
                            Hkp = kron_6(sparse.eye(2), sparse.eye(self.N_exc ** k), ad, sparse.eye(self.N_exc ** (p - k - 1)), a, sparse.eye(self.N_exc ** (self.L - p - 1)))

                        elif k == p:
                            Hkp = kron_4(sparse.eye(2), sparse.eye(self.N_exc ** k), ad @ a, sparse.eye(self.N_exc ** (self.L - k - 1)))

                        elif k > p:
                            Hkp = kron_6(sparse.eye(2), sparse.eye(self.N_exc ** p), a, sparse.eye(self.N_exc ** (k - p - 1)), ad, sparse.eye(self.N_exc ** (self.L - k - 1)))
                            
                        self.GSphotons[n] += (1 / N) * np.exp(1j * 2 * np.pi * (k - p) * (n - self.L / 2) / self.L) * (np.conjugate(self.eigsys.vects[:, 0]).T @ Hkp @ self.eigsys.vects[:, 0])[0, 0]
                        self.E1photons[n] += (1 / N) * np.exp(1j * 2 * np.pi * (k - p) * (n - self.L / 2) / self.L) * (np.conjugate(self.eigsys.vects[:, 1]).T @ Hkp @ self.eigsys.vects[:, 1])[0, 0]
        
        except AttributeError:
            self.diag()
            self.n_photons()
            
    def save(self):
        filename = 'data_files/exact_' + str(self.L) + '_' + str(self.N_exc) + '_' + str(self.Delta).replace('.', ',') + '_' + str(self.g).replace('.', ',') + '.txt'
        f = open(filename, 'w')
        
        for n in range(self.L):
            if n < self.L - 1:
                f.write(str(self.GSphotons[n].real) + ' ' + str(self.E1photons[n].real) + '\n')
            else:
                f.write(str(self.GSphotons[n].real) + ' ' + str(self.E1photons[n].real))
            
        f.close()

def kron_3(A, B, C):
    return sparse.kron(sparse.kron(A, B), C)

def kron_4(A, B, C, D):
    return sparse.kron(sparse.kron(A, B), sparse.kron(C, D))

def kron_6(A, B, C, D, E, F):
    return sparse.kron(sparse.kron(sparse.kron(A, B), sparse.kron(C, D)), sparse.kron(E, F))

def f_energy(deltar, g):
    f = f_f(deltar, g)
    return -0.5 * deltar + np.sum(f * (w * f + 2 * g / np.sqrt(N)))

H_exact = exact_Diag(g, delta, N_exc)
H_exact.n_photons()
H_exact.save()