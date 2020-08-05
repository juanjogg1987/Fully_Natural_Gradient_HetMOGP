import numpy as np
import GPy
from GPy.util import choleskies,linalg
import scipy
from numpy.linalg.linalg import LinAlgError
from vo_util import nearestPD
import climin
import time

"Note that with this code were run all experiments for HetMOGP"

def compute_stoch_grads_for_qu_HetMOGP(model):

    N_posteriors = model.q_u_means.shape[1]
    dL_dm = [0.0] * N_posteriors
    dL_dV = [0.0] * N_posteriors

    for i in range(N_posteriors):

        if not model.q_u_means.is_fixed and not model.q_u_chols.is_fixed:
            dL_dm[i] = dL_dm[i] + (-model.q_u_means.gradient[:, i].copy())
            dL_dV[i] = dL_dV[i] + (-model.gradients['dL_dS_u'][i].copy())

    # These lines below are for numerical stability
    dL_dm, dL_dV = [np.clip(dL, -5.0e0, 5.0e0) for dL in dL_dm], [dL for dL in dL_dV]

    return dL_dm, dL_dV

def optimise_HetMOGP(model,Xval=None,Yval=None, max_iters=1000, step_rate=0.01, decay_mom1=1-0.9,decay_mom2=1-0.999,fng=False,q_s_ini = 0.0,prior_lamb_or_offset=None):
    if prior_lamb_or_offset is None:
        prior_lamb_or_offset = 1e-8
    global mk_ant,mk_aux,mk,V_i,Vk,Lk, Vk,Vki_ant
    def natural_grad_qu(model, n_iter=1, step_size=step_rate, momentum=0.0):
        global mk_ant, mk_aux, mk, V_i, Vk, Lk, Vk, Vki_ant
        """"Initialize the step-sizes"""""
        beta2_k = step_size  #use step_size*0.1 for Convolutional MOGP
        gamma2_k = momentum
        alpha2_k = step_size
        N_posteriors = model.q_u_means.shape[1]

        if n_iter==1:
            V_i = choleskies.multiple_dpotri(choleskies.flat_to_triang(model.q_u_chols.values)).copy()
            Vk = np.zeros_like(V_i)
            for i in range(N_posteriors):
                Vk[i, :, :] = 0.5*(model.posteriors[i].covariance.copy() + model.posteriors[i].covariance.T.copy())

            Lk = np.zeros_like(Vk)
            mk = model.q_u_means.values.copy()

            Vki_ant = V_i.copy()
            mk_aux = mk.copy()


        dL_dm, dL_dV = compute_stoch_grads_for_qu_HetMOGP(model=model)

        mk_ant = mk_aux.copy()
        mk_aux = mk.copy()

        if not model.q_u_means.is_fixed and not model.q_u_chols.is_fixed:
            mk_ant = mk_aux.copy()
            mk_aux = mk.copy()

            for i in range(N_posteriors):
                try:
                    V_i[i, :, :] = V_i[i, :, :] + 2 * beta2_k * dL_dV[i] #+ 1.0e-6*np.eye(*Vk[i,:,:].shape)
                    Vk[i, :, :] = np.linalg.inv(V_i[i, :, :])
                    Vk[i, :, :] = 0.5 * (np.array(Vk[i, :, :]) + np.array(Vk[i, :, :].T))
                    Lk[i, :, :] = np.linalg.cholesky(Vk[i, :, :])
                    mk[:, i] = mk[:, i] - alpha2_k * np.dot(Vk[i, :, :], dL_dm[i]) + gamma2_k * np.dot(
                        np.dot(Vk[i, :, :], Vki_ant[i, :, :]), (mk[:, i] - mk_ant[:, i]))
                except LinAlgError:
                    print("Overflow")
                    Vk[i, :, :] = np.linalg.inv(V_i[i, :, :])
                    Vk[i, :, :] = 1.0e-1*np.eye(*Vk[i,:,:].shape) #nearestPD(Vk[i,:,:]) # + 1.0e-3*np.eye(*Vk[i,:,:].shape)
                    Lk[i, :, :] = linalg.jitchol(Vk[i, :, :])
                    V_i[i, :, :] = np.linalg.inv(Vk[i, :, :])
                    mk[:, i] = mk[:, i]*0.0

            Vki_ant = V_i.copy()

            model.L_u.setfield(choleskies.triang_to_flat(Lk.copy()), np.float64)
            model.m_u.setfield(mk.copy(), np.float64)

    global ELBO, myTimes, sched, NLPD
    ELBO = []
    NLPD = []
    myTimes = []
    sched = step_rate
    def callhybrid(i):
        global start
        global ELBO, myTimes, sched, NLPD

        if i['n_iter'] > max_iters:
            model.q_u_means.unfix()
            model.q_u_chols.unfix()
            return True
        model.update_model(False)
        model.q_u_means.unfix()
        model.q_u_chols.unfix()
        if fng: mom = 0.9;
        else: mom = 0.0;
        natural_grad_qu(model, n_iter=i['n_iter'], step_size=step_rate, momentum=mom)

        model.update_model(True)
        model.q_u_means.fix()
        model.q_u_chols.fix()
        #model.update_model(True)

        ELBO.append(model.log_likelihood())
        myTimes.append(time.time())

        if (i['n_iter']) % 50 == 0:
            print(i['n_iter'])
            print(model.log_likelihood())
            if not(Xval==None or Yval==None):
                NLPD.append(model.negative_log_predictive(Xval, Yval, num_samples=1000))

        return False


    model.q_u_means.fix()
    model.q_u_chols.fix()
    if fng is True:
        print('Running Fully NG, check s_ini:',q_s_ini,' and prior_lamb:',prior_lamb_or_offset)
        opt = climin.VarOpt(model.optimizer_array, model.stochastic_grad, step_rate=step_rate, s_ini=q_s_ini,decay_mom1=decay_mom1, decay_mom2=decay_mom2, prior_lambda=prior_lamb_or_offset)
    else:
        print('Running Hybrid (NG+Adam), check offset:',prior_lamb_or_offset)
        opt = climin.Adam(model.optimizer_array, model.stochastic_grad, step_rate=step_rate,decay_mom1=decay_mom1, decay_mom2=decay_mom2,offset=prior_lamb_or_offset)

    ELBO.append(model.log_likelihood())
    if not (Xval == None or Yval == None):
        NLPD.append(model.negative_log_predictive(Xval, Yval, num_samples=1000))
    start = time.time()
    myTimes.append(start)
    info = opt.minimize_until(callhybrid)
    return np.array(ELBO).flatten(),np.array(NLPD), np.array(myTimes)-start