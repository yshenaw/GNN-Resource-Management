import numpy as np
import math
import time
import scipy.io as sio
import matplotlib.pyplot as plt

def get_mu(Pmax,Hkk,btmp,wf):
    Lmu = 0
    N = Hkk.shape[1]

    I = np.eye(N)
    Hkk_H = (Hkk.conj()).T
    
    if(np.linalg.matrix_rank(btmp) == N 
        and np.linalg.norm(np.matmul(np.linalg.inv(btmp),Hkk_H ) * wf) < np.sqrt(Pmax)):
        return np.squeeze(np.matmul(np.linalg.inv(btmp),Hkk_H ) * wf)

    Lambda, D = np.linalg.eig(btmp)
    Lambda = np.diag(Lambda)
    HUW = Hkk_H*wf
    Phitmp = np.matmul(HUW,(HUW.conj()).T)
    DH = (D.conj()).T

    Phi = np.matmul(np.matmul(DH,Phitmp),D)
    Phimm = np.real(np.diag(Phi))
    Lambdamm = np.real(np.diag(Lambda))

    
    Rmu = 1
    Pcomp = np.sum(Phimm/(Lambdamm + Rmu)**2)
    while(Pcomp > Pmax):
        Rmu = Rmu*2
        Lmu = Rmu
        Pcomp = np.sum(Phimm/(Lambdamm + Rmu)**2)
    while(Rmu-Lmu > 1e-4):
        midmu = (Rmu + Lmu)/2
        Pcomp = np.sum(Phimm/(Lambdamm + midmu)**2)
        if(Pcomp < Pmax ):
            Rmu = midmu
        else: 
            Lmu = midmu
    ans = np.squeeze(np.matmul(np.linalg.inv(btmp + Rmu*I),Hkk_H ) * wf)

    return ans


def np_WMMSE_vector(b_int, H, Pmax, var_noise):
    # fix transpose and conjudgate
    K = b_int.shape[0]
    N = b_int.shape[1]
    vnew = 0
    b = b_int
    f = np.zeros(K,dtype=complex)
    w = np.zeros(K,dtype=complex)

    mask = np.eye(K)

    btmp = np.reshape(b, (K,1,N))
    rx_power = np.multiply(H, b)
    rx_power = np.sum(rx_power,axis=-1)
    valid_rx_power = np.sum(np.multiply(rx_power, mask), axis=0)
    interference_rx = np.square(np.abs(rx_power))
    interference = np.sum(interference_rx, axis=1) + var_noise
    f = np.divide(valid_rx_power, interference)
    w = 1/(1 - valid_rx_power*f.conj())
    vnew = np.sum(np.log2(np.abs(w)))
    
    for iter in range(100):
        vold = vnew
        H_H = np.expand_dims(H.conj(),axis=-1)
        H_tmp = np.expand_dims(H,axis=-2)
        HH = np.matmul(H_H,H_tmp)

        UWU = np.reshape(w * (f.conj()).T * f,(K,1,1,1))
        btmp = np.sum(HH * UWU, axis=0)
        for ii in range(K):
            Hkk = np.expand_dims(H[ii,ii,:],axis=0)
            b[ii,:] = get_mu(Pmax,Hkk,btmp[ii,:,:],w[ii] * f[ii])

        btmp = np.reshape(b, (K,1,N))
        rx_power = np.multiply(H, b)
        rx_power = np.sum(rx_power,axis=-1)
        valid_rx_power = np.sum(np.multiply(rx_power, mask), axis=0)
        interference_rx = np.square(np.abs(rx_power))
        interference = np.sum(interference_rx, axis=1) + var_noise
        f = np.divide(valid_rx_power, interference)
        w = 1/(1 - valid_rx_power*f.conj())
        vnew = np.sum(np.log2(np.abs(w)))
        if abs(vnew - vold) <= 1e-3:
           break
    return b


def IC_sum_rate(H,p, var_noise):
    # H1 K*K*N
    # p1 K*N
    K = H.shape[1]
    N = H.shape[-1]
    p = p.reshape((-1,K,1,N))
    rx_power = np.multiply(H, p)
    rx_power = np.sum(rx_power,axis=-1)
    rx_power = np.square(np.abs(rx_power))
    mask = np.eye(K)
    valid_rx_power = np.sum(np.multiply(rx_power, mask), axis=1)
    interference = np.sum(np.multiply(rx_power, 1 - mask), axis=1) + var_noise
    rate = np.log2(1 + np.divide(valid_rx_power, interference))
    sum_rate = np.mean(np.sum(rate, axis=1))
    return sum_rate
