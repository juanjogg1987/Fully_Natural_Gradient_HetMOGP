#Code written by Juan Jose Giraldo Gutierrez from University of Sheffield 2020
#This code is a new version of the Heterogeneous MOGP model that includes convolution processes

#this code is based on the original HetMOGP code written by: 
#Pablo Moreno-Munoz Universidad Carlos III de Madrid and University of Sheffield (2018)

import sys
import numpy as np
import GPy
from GPy.inference.latent_function_inference import LatentFunctionInference
from GPy.inference.latent_function_inference.posterior import Posterior
from GPy.util import choleskies
from GPy.util import linalg
from convhetmogp2 import util
from collections import namedtuple
from scipy.linalg.blas import dtrmm
import matplotlib.pyplot as plt
import random

qfd = namedtuple("q_fd", "m_fd v_fd Kfdu Afdu S_fd")
qu = namedtuple("q_U", "mu_u chols_u")
pu = namedtuple("p_U", "Kuu Luu Kuui")

class SVMOGPInf(LatentFunctionInference):

    def inference(self, q_u_means, q_u_chols, X, Y, Z, kern_list, kern_list_Gdj, kern_aux,likelihood, B_list, Y_metadata, KL_scale=1.0,
                  batch_scale=None, predictive=False,Gauss_Newton=False):
        M = Z.shape[0]
        T = len(Y)
        if batch_scale is None:
            batch_scale  = [1.0]*T
        Ntask = []
        [Ntask.append(Y[t].shape[0]) for t in range(T)]
        Q = len(kern_list)
        D = likelihood.num_output_functions(Y_metadata)
        Kuu, Luu, Kuui = util.latent_function_covKuu(Z, B_list, kern_list, kern_list_Gdj, kern_aux)
        p_U = pu(Kuu=Kuu, Luu=Luu, Kuui=Kuui)
        q_U = qu(mu_u=q_u_means, chols_u=q_u_chols)

        # for every latent function f_d calculate q(f_d) and keep it as q(F):
        q_F = []
        posteriors_F = []
        f_index = Y_metadata['function_index'].flatten()
        d_index = Y_metadata['d_index'].flatten()

        for d in range(D):
            Xtask = X[f_index[d]]
            q_fd = self.calculate_q_f(X=Xtask, Z=Z, q_U=q_U, p_U=p_U, kern_list=kern_list, kern_list_Gdj=kern_list_Gdj, kern_aux=kern_aux, B=B_list,
                                      M=M, N=Xtask.shape[0], j=d)
            # Posterior objects for output functions (used in prediction)
            #I have to get rid of function below Posterior for it is not necessary
            posterior_fd = Posterior(mean=q_fd.m_fd.copy(), cov=q_fd.S_fd.copy(),
                                     K=util.conv_function_covariance(X=Xtask, B=B_list, kernel_list=kern_list, kernel_list_Gdj=kern_list_Gdj, kff_aux=kern_aux, d=d),
                                     prior_mean=np.zeros(q_fd.m_fd.shape))
            posteriors_F.append(posterior_fd)
            q_F.append(q_fd)

        mu_F = []
        v_F = []
        for t in range(T):
            mu_F_task = np.empty((X[t].shape[0],1))
            v_F_task = np.empty((X[t].shape[0], 1))
            for d, q_fd in enumerate(q_F):
                if f_index[d] == t:
                    mu_F_task = np.hstack((mu_F_task, q_fd.m_fd))
                    v_F_task = np.hstack((v_F_task, q_fd.v_fd))

            mu_F.append(mu_F_task[:,1:])
            v_F.append(v_F_task[:,1:])

        # posterior_Fnew for predictive
        if predictive:
            return posteriors_F
        # inference for rest of cases
        else:
            # Variational Expectations
            VE = likelihood.var_exp(Y, mu_F, v_F, Y_metadata)
            VE_dm, VE_dv = likelihood.var_exp_derivatives(Y, mu_F, v_F, Y_metadata,Gauss_Newton)
            for t in range(T):
                VE[t] = VE[t]*batch_scale[t]
                VE_dm[t] = VE_dm[t]*batch_scale[t]
                VE_dv[t] = VE_dv[t]*batch_scale[t]

            # KL Divergence
            KL = self.calculate_KL(q_U=q_U, p_U=p_U,M=M, J=D)

            # Log Marginal log(p(Y))
            F = 0
            for t in range(T):
                F += VE[t].sum()

            log_marginal = F - KL

            # Gradients and Posteriors
            dL_dS_u = []
            dL_dmu_u = []
            dL_dL_u = []
            dL_dKmm = []
            dL_dKmn = []
            dL_dKdiag = []
            posteriors = []
            for j in range(D):
                (dL_dmu_j, dL_dL_j, dL_dS_j, posterior_j, dL_dKjj, dL_dKdj, dL_dKdiag_j) = self.calculate_gradients(q_U=q_U, p_U=p_U,
                                                            q_F=q_F,VE_dm=VE_dm, VE_dv=VE_dv, Ntask=Ntask, M=M, Q=Q, D=D, f_index=f_index, d_index=d_index, j=j)
                dL_dmu_u.append(dL_dmu_j)
                dL_dL_u.append(dL_dL_j)
                dL_dS_u.append(dL_dS_j)
                dL_dKmm.append(dL_dKjj)
                dL_dKmn.append(dL_dKdj)
                dL_dKdiag.append(dL_dKdiag_j)
                posteriors.append(posterior_j)

            gradients = {'dL_dmu_u':dL_dmu_u, 'dL_dL_u':dL_dL_u,'dL_dS_u':dL_dS_u, 'dL_dKmm':dL_dKmm, 'dL_dKmn':dL_dKmn, 'dL_dKdiag': dL_dKdiag}

            return log_marginal, gradients, posteriors, posteriors_F

    def calculate_gradients(self, q_U, p_U, q_F, VE_dm, VE_dv, Ntask, M, Q, D, f_index, d_index,j):
        """
        Calculates gradients of the Log-marginal distribution p(Y) wrt variational
        parameters mu_q, S_q
        """
        # Algebra for q(u) and p(u):
        m_u = q_U.mu_u.copy()
        L_u = choleskies.flat_to_triang(q_U.chols_u.copy())
        #S_u = np.empty((Q, M, M))
        S_u = np.dot(L_u[j, :, :], L_u[j, :, :].T)    #This could be done outside and recieve it to reduce computation
        #[np.dot(L_u[q, :, :], L_u[q, :, :].T, S_u[q, :, :]) for q in range(Q)]
        Kuu = p_U.Kuu.copy()
        Luu = p_U.Luu.copy()
        Kuui = p_U.Kuui.copy()
        S_qi, _ = linalg.dpotri(np.asfortranarray(L_u[j, :, :]))

        if np.any(np.isinf(S_qi)):
            raise ValueError("Sqi: Cholesky representation unstable")

        # KL Terms
        dKL_dmu_j = np.dot(Kuui[j,:,:],m_u[:, j, None])
        dKL_dS_j = 0.5 * (Kuui[j,:,:] - S_qi)
        dKL_dKjj = 0.5 * Kuui[j,:,:] - 0.5 * Kuui[j,:,:].dot(S_u).dot(Kuui[j,:,:]) \
                   - 0.5 * np.dot(Kuui[j,:,:],np.dot(m_u[:, j, None],m_u[:, j, None].T)).dot(Kuui[j,:,:].T)

        # VE Terms
        dVE_dmu_j = np.zeros((M, 1))
        dVE_dS_j = np.zeros((M, M))
        dVE_dKjj = np.zeros((M, M))
        dVE_dKjd = []
        dVE_dKdiag = []

        Nt = Ntask[f_index[j]]
        dVE_dmu_j += np.dot(q_F[j].Afdu.T, VE_dm[f_index[j]][:,d_index[j]])[:, None]
        Adv = q_F[j].Afdu.T * VE_dv[f_index[j]][:,d_index[j],None].T
        Adv = np.ascontiguousarray(Adv)
        AdvA = np.dot(Adv.reshape(-1, Nt), q_F[j].Afdu).reshape(M, M)
        dVE_dS_j += AdvA

        # Derivatives dKuquq
        tmp_dv = np.dot(AdvA, S_u).dot(Kuui[j,:,:])
        dVE_dKjj += AdvA - tmp_dv - tmp_dv.T
        Adm = np.dot(q_F[j].Afdu.T, VE_dm[f_index[j]][:,d_index[j],None])
        dVE_dKjj += - np.dot(Adm, np.dot(Kuui[j,:,:], m_u[:, j, None]).T)

        # Derivatives dKuqfd
        tmp = np.dot(S_u, Kuui[j,:,:])
        tmp = 2. * (tmp - np.eye(M))
        dve_kjd = np.dot(np.dot(Kuui[j,:,:], m_u[:, j, None]), VE_dm[f_index[j]][:,d_index[j],None].T)
        dve_kjd += np.dot(tmp.T, Adv)
        dVE_dKjd.append(dve_kjd)

        # Derivatives dKdiag
        dVE_dKdiag.append(VE_dv[f_index[j]][:,d_index[j]])

        dVE_dKjj = 0.5 * (dVE_dKjj + dVE_dKjj.T)
        # Sum of VE and KL terms
        dL_dmu_j = dVE_dmu_j - dKL_dmu_j
        dL_dS_j = dVE_dS_j - dKL_dS_j
        dL_dKjj = dVE_dKjj - dKL_dKjj
        dL_dKdj = dVE_dKjd[0].copy() #Here we just pass the unique position
        dL_dKdiag = dVE_dKdiag[0].copy() #Here we just pass the unique position

        # Pass S_q gradients to its low-triangular representation L_q
        chol_u = q_U.chols_u.copy()
        L_j = choleskies.flat_to_triang(chol_u[:,j:j+1])
        dL_dL_j = 2. * np.array([np.dot(a, b) for a, b in zip(dL_dS_j[None,:,:], L_j)])
        dL_dL_j = choleskies.triang_to_flat(dL_dL_j)

        # Posterior
        posterior_j = Posterior(mean=m_u[:, j, None], cov=S_u, K=Kuu[j,:,:], prior_mean=np.zeros(m_u[:, j, None].shape))

        return dL_dmu_j, dL_dL_j, dL_dS_j, posterior_j, dL_dKjj, dL_dKdj, dL_dKdiag


    def calculate_q_f(self, X, Z, q_U, p_U, kern_list, kern_list_Gdj, kern_aux, B, M, N, j):
        """
        Calculates the mean and variance of q(f_d) as
        Equation: E_q(U)\{p(f_d|U)\}
        """
        # Algebra for q(u):
        m_u = q_U.mu_u.copy()
        L_u = choleskies.flat_to_triang(q_U.chols_u.copy())
        #S_u = np.empty((M, M))
        S_u = np.dot(L_u[j, :, :], L_u[j, :, :].T) + 1e-6*np.eye(M)
#        [np.dot(L_u[j, :, :], L_u[j, :, :].T, S_u[j, :, :]) for j in range(J)]
        #for j in range(J): S_u[j,:,:] = S_u[j,:,:] + 1e-6*np.eye(M)

        # Algebra for p(f_d|u):
        #Kfdu = util.conv_cross_covariance_full(X, Z, B, kern_list, kern_list_Gdj, kern_aux,j)
        #Kff = util.conv_function_covariance(X, B, kern_list, kern_list_Gdj, kern_aux,j)
        Kff,Kfdu = util.both_convoled_Kff_and_Kfu_full(X, Z, B, kern_list, kern_list_Gdj, kern_aux, j)

        Kuu = p_U.Kuu.copy()
        Luu = p_U.Luu.copy()
        Kuui = p_U.Kuui.copy()
        Kff_diag = np.diag(Kff)

        # Algebra for q(f_d) = E_{q(u)}[p(f_d|u)]
        #Afdu = np.empty((N, M)) #Afdu = K_{fduq}Ki_{uquq}
        m_fd = np.zeros((N, 1))
        v_fd = np.zeros((N, 1))
        S_fd = np.zeros((N, N))
        v_fd += Kff_diag[:,None] #+ 1e-1
        S_fd += Kff #+ 1e-1*np.eye(N)

        # Expectation part
        #R, _ = linalg.dpotrs(np.asfortranarray(Luu[q, :, :]), Kfdu[:, q * M:(q * M) + M].T)
        #R = np.dot(Kuui[q, :, :], Kfdu[:, q * M:(q * M) + M].T)
        R = np.linalg.solve(Kuu[j, :, :], Kfdu.T)
        Afdu = R.T    #Afdu = K_{fduq}Ki_{uquq}
        m_fd += np.dot(Afdu, m_u[:, j, None]) #exp
        #tmp = dtrmm(alpha=1.0, a=L_u[q, :, :].T, b=R, lower=0, trans_a=0)
        #v_fd += np.sum(np.square(tmp), 0)[:,None] - np.sum(R * Kfdu[:, q * M:(q * M) + M].T,0)[:,None] #exp
        S_fd += np.dot(np.dot(R.T,S_u),R) - np.dot(Kfdu,R)
        #S_fd += np.dot(np.dot(R.T, S_u[q, :, :]), R) - np.dot(np.dot(R.T, Kuu[q, :, :]), R) # - np.dot(Kfdu[:, q * M:(q * M) + M], R)

        v_fd = np.diag(S_fd)[:,None]
        if (v_fd<0).any():
            #v_fd = np.abs(v_fd)
            #v_fd[v_fd < 0] = 1.0e-6
            print('v negative!')
            #print(np.linalg.eig(S_u[q, :, :]))

        q_fd = qfd(m_fd=m_fd, v_fd=v_fd, Kfdu=Kfdu, Afdu=Afdu, S_fd=S_fd)
        return q_fd

    def calculate_KL(self, q_U, p_U, M, J):
        """
        Calculates the KL divergence (see KL-div for multivariate normals)
        Equation: \sum_Q KL{q(uq)|p(uq)}
        """
        # Algebra for q(u):
        m_u = q_U.mu_u.copy()
        L_u = choleskies.flat_to_triang(q_U.chols_u.copy())

        S_u = np.empty((J, M, M))
        [np.dot(L_u[j, :, :], L_u[j, :, :].T, S_u[j, :, :]) for j in range(J)]

        # Algebra for p(u):
        Kuu = p_U.Kuu.copy()
        Luu = p_U.Luu.copy()
        Kuui = p_U.Kuui.copy()

        KL = 0
        for j in range(J):
            KL += 0.5 * np.sum(Kuui[j, :, :] * S_u[j, :, :]) \
                  + 0.5 * np.dot(m_u[:, j, None].T,np.dot(Kuui[j,:,:],m_u[:, j, None])) \
                  - 0.5 * M \
                  + 0.5 * 2. * np.sum(np.log(np.abs(np.diag(Luu[j, :, :])))) \
                  - 0.5 * 2. * np.sum(np.log(np.abs(np.diag(L_u[j, :, :]))))
        return KL
