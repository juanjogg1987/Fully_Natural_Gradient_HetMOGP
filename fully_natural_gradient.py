import numpy as np
from GPy.util import choleskies,linalg
from numpy.linalg.linalg import LinAlgError
import climin
import time

def compute_stoch_grads_for_qu_HetMOGP(model):

    dL_dm = [0.0] * model.num_latent_funcs
    dL_dV = [0.0] * model.num_latent_funcs

    for i in range(model.num_latent_funcs):

        if not model.q_u_means.is_fixed and not model.q_u_chols.is_fixed:
            dL_dm[i] = dL_dm[i] + (-model.q_u_means.gradient[:, i])
            dL_dV[i] = dL_dV[i] + (-model.gradients['dL_dS_u'][i])


    # These lines below are for numerical stability
    if model.Ymulti_all.__len__()>1:
    #if model.input_dim <= 5:
        #dL_dm, dL_dV = [dL for dL in dL_dm], [dL for dL in dL_dV]
        dL_dm, dL_dV = [np.clip(dL, -1.0e0, 1.0e0) for dL in dL_dm], [dL for dL in dL_dV]
        #dL_dm, dL_dV = [np.clip(dL, eval('-5.0e' + str(model.input_dim)), eval('5.0e' + str(model.input_dim))) for dL in dL_dm], [dL for dL in dL_dV]
    else:
        if model.input_dim <= 5:
            dL_dm, dL_dV = [np.clip(dL, -5.0e1, 5.0e1) for dL in dL_dm], [dL for dL in dL_dV]
        else:
            dL_dm, dL_dV = [np.clip(dL, -5.0e100, 5.0e100) for dL in dL_dm], [dL for dL in dL_dV]

    #dL_dm, dL_dV = [dL / float(MC) for dL in dL_dm], [dL / float(MC) for dL in dL_dV]

    return dL_dm, dL_dV

def compute_stoch_grads_hyparam_HetMOGP(model, index_ini, hpmu,hpSig,ysample, mu, Sig, Wmu,WSig, MC = 1):

    dL_dtheta = [0.0]*model.num_latent_funcs
    d2L_dtheta = [0.0]*model.num_latent_funcs
    dL_dZ = [0.0]
    d2L_dZ = [0.0]
    dL_dm = [0.0] * model.num_latent_funcs
    dL_dV = [0.0] * model.num_latent_funcs
    dL_dW = [0.0] * model.num_latent_funcs
    d2L_dW = [0.0] * model.num_latent_funcs
    #f_eval = 1.0
    for Nsample in range(MC):
        #f_eval = model.log_likelihood().copy()
        for i in range(model.num_latent_funcs):
            model.update_model(False)
            index = index_ini[i].copy()
            if not index.shape[0] == 0:
                #print(eval('model.' + model.kern_list[i]._name + '.gradient').reshape(-1, 1)[index])
                #print('mean ',hpmu[i][index, :])
                #print('var ',np.sqrt(np.abs(hpSig[i][index])))
                ysample[i][index] = np.random.normal(hpmu[i][index, :], np.sqrt(np.abs(hpSig[i][index]))).copy()
                #print(hpSig[i][index])
                # ysample[i][index] = np.random.multivariate_normal(hpmu[i][index, 0], np.diagflat(hpSig[i][index])).reshape(-1, 1).copy()
                #kern_hyper = np.clip(np.exp(ysample[i][index]), 1.0e-4, 1e5)
                #kern_hyper = np.log(1.0+np.exp(ysample[i][index]))
                #kern_hyper = np.exp(ysample[i][index])  #former implementation
                kern_hyper = (ysample[i][index])**2
                #print(kern_hyper)
                model.kern_list[i][index] = kern_hyper[:, 0].copy()
                #print(model.kern_list[i][index])

            if not eval('model.B_q'+str(i)+'.W.is_fixed'):
                index_W = np.arange(i * model.num_output_funcs, i * model.num_output_funcs + model.num_output_funcs)
                Wsample = np.random.normal(Wmu[index_W, 0], np.sqrt(np.abs(WSig[index_W,0]))).reshape(-1,1).copy()

                "This part below is to allow not to modify specific W positions when fixed and optimise Correlated Chained GP correctly"

                which_fixed = np.setdiff1d(eval('model.B_q' + str(i) + '.W.values'),eval('model.B_q' + str(i) + '.unfixed_param_array'))
                if which_fixed.__len__()==0:
                    model['B_q' + str(i) + '.W'].setfield(Wsample, np.float64)
                else:
                    #print(which_fixed)
                    index_W_fix = np.where(eval('model.B_q' + str(i) + '.W.values')!=which_fixed)[0]
                    #print(index_W_fix)
                    model['B_q' + str(i) + '.W'][index_W_fix] = Wsample[index_W_fix].copy()

        if not model.Z.is_fixed:
            #Zsample = np.random.normal(mu[:, 0], np.abs(np.sqrt(np.diag(Sig))))
            Zsample = np.random.normal(mu[:, 0], np.sqrt(np.abs(Sig[:,0])))
            #model.inducing_inputs.setfield(Zsample.reshape(model.num_inducing, model.input_dim * model.num_latent_funcs),np.float64)
            model.inducing_inputs.setfield(Zsample.reshape(model.num_inducing, -1), np.float64)

        model.update_model(True)

        for i in range(model.num_latent_funcs):
            #model.update_model(False)
            index = index_ini[i].copy()
            if not index.shape[0] == 0:
                #grad_hyp_GN = -np.exp(ysample[i][index]) * eval('model.' + model.kern_list[i]._name + '.gradient').reshape(-1, 1)[index]  #former implementation
                grad_hyp_GN = -2*(ysample[i][index]) * eval('model.' + model.kern_list[i]._name + '.gradient').reshape(-1, 1)[index]
                #print(eval('model.' + model.kern_list[i]._name + '.gradient').reshape(-1, 1)[index])
                #grad_hyp_GN = -(np.exp(ysample[i][index])/(1.0+np.exp(ysample[i][index]))) * eval('model.' + model.kern_list[i]._name + '.gradient').reshape(-1, 1)[index]
                dL_dtheta[i] = dL_dtheta[i] + grad_hyp_GN
                d2L_dtheta[i] = d2L_dtheta[i] + grad_hyp_GN*grad_hyp_GN

            if not eval('model.B_q'+str(i)+'.W.is_fixed'):
                grad_W = -model['B_q' + str(i) + '.W'].gradient.copy()
                dL_dW[i] = dL_dW[i] + grad_W
                d2L_dW[i] = d2L_dW[i] + grad_W * grad_W

            if not model.q_u_means.is_fixed and not model.q_u_chols.is_fixed:
                dL_dm[i] = dL_dm[i] + (-model.q_u_means.gradient[:, i])
                dL_dV[i] = dL_dV[i] + (-model.gradients['dL_dS_u'][i])

        if not model.Z.is_fixed:
            grad_Z = -model.Z.gradient.copy().reshape(-1, 1)
            dL_dZ[0] = dL_dZ[0] + grad_Z
            d2L_dZ[0] = d2L_dZ[0] + grad_Z * grad_Z

    model.update_model(False)
    if (model.num_latent_funcs==1 and model.num_output_funcs==1):
        dL_dtheta, d2L_dtheta = [dL / float(MC) for dL in dL_dtheta], [dL / float(MC) for dL in d2L_dtheta]
        dL_dZ, d2L_dZ = [np.clip(dL / float(MC), -5.0e100, 5.0e100) for dL in dL_dZ], [dL / float(MC) for dL in d2L_dZ]
        dL_dm, dL_dV = [np.clip(dL / float(MC), -5.0e1, 5.0e1) for dL in dL_dm], [dL / float(MC) for dL in dL_dV]
        dL_dW, d2L_dW = [dL / float(MC) for dL in dL_dW], [dL / float(MC) for dL in d2L_dW]

    elif(model.input_dim>0):
        dL_dtheta, d2L_dtheta = [dL / float(MC) for dL in dL_dtheta], [dL / float(MC) for dL in d2L_dtheta]
        dL_dZ, d2L_dZ = [dL / float(MC) for dL in dL_dZ], [dL / float(MC) for dL in d2L_dZ]
        dL_dW, d2L_dW = [dL / float(MC) for dL in dL_dW], [dL / float(MC) for dL in d2L_dW]

    return dL_dtheta, d2L_dtheta, dL_dZ, d2L_dZ, dL_dm, dL_dV, dL_dW, d2L_dW

def fullyng_opt_HetMOGP(model,Xval=None,Yval=None, max_iters=1000, step_size=0.001, momentum=0.0, prior_lambda=1.0e-10, tao_VI=2.0, MC=None):

    model['.*.kappa'].fix()

    """"Initialize the step-sizes"""""
    tao = tao_VI
    tao_Z = tao_VI

    beta1_k = step_size
    gamma1_k = momentum*0.1
    alpha1_k = step_size

    beta2_k = step_size
    gamma2_k = momentum*0.1
    alpha2_k = step_size

    beta3_k = step_size
    gamma3_k = momentum
    alpha3_k = step_size

    beta4_k = step_size*1
    gamma4_k = momentum
    alpha4_k = step_size*1


    """"""""""Extract some important information"""""""""
    if not hasattr(model,'Z') and not hasattr(model,'inducing_inputs'):
        print('There are not inducing_inputs in the model!!\n')
        print('Check the model corresponds to the SVGPMulti')


    """Initialization for inducing points opt."""
    if (model.num_latent_funcs==1 and model.num_output_funcs==1):
        amp = 1.0e-5
        mylamb = 1e-1
    else:
        if model.input_dim == 1: amp = 1.0e-3;
        else: amp = 1.0e0;
        mylamb = prior_lambda
    Ident = np.ones(model.input_dim * model.num_inducing * model.num_latent_funcs)[:,None]
    mu = model.Z.reshape(-1, 1).copy()
    mu_aux = np.zeros_like(mu)
    Sig_i = 1.0 / amp * Ident
    Sig = amp * Ident
    St = Sig_i - mylamb * Ident


    """"Initialization for kern hyper-params"""""
    num_hyp = []
    hpmu = []
    hpmu_aux = []
    hpSig = []
    hpSig_i = []
    ysample = []
    ind_fixed = []  # this list will contain a booleans to indicate when it is fixed a parameter.
    index_ini = []
    if model.input_dim == 1: my_std = 1.0e-3;
    else:  my_std = 1.0e0;

    for i in range(model.num_latent_funcs):
        num_hyp.append(eval('model.' + model.kern_list[i]._name + '.gradient.__len__()'))  # This is when using automatic relevance determination
        ind_fixed.append(np.ones(num_hyp[i], dtype=int))
        hpmu.append(np.sqrt(eval('model.' + model.kern_list[i]._name + '.param_array'))[:, None])

        hpmu_aux.append(np.zeros_like(hpmu[i]))
        hpSig.append(my_std * np.ones_like(hpmu[i]))
        hpSig_i.append(1.0 / hpSig[i])
        for j in range(num_hyp[i]):
            name = model.kern_list[i].parameter_names()[j]
            ind_fixed[i][j] = ind_fixed[i][j] - eval(
                'model.' + model.kern_list[i]._name + '.' + name + '.is_fixed')  # Puts 0 to not accesss

        index_ini.append(np.arange(num_hyp[i])[np.where(ind_fixed[i] == 1)])
        ysample.append(np.zeros_like(hpmu[i]))


    hpSig_i_ant = hpSig_i.copy()
    mylamb3 = prior_lambda

    """""Initialization for Corregionalisation"""
    num_W = int(model['B_q*.W*'].values().shape[0] / 2)  # Here we divide by 2 'cause the kappas are not optimized
    # num_hyp = 2
    """"HERE WE HAVE TO LOAD THE VALUES FROM THE MODEL FOR Wmu !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"""

    if model.kern_list.__len__()==1:
        Wmu = model['.*.W'].values
    else:
        Wmu = model['.*.W'].values()[:,None]

    if model.input_dim == 1: W_std_ini = 1.0e-3;   #it works well with 1.0e-1
    else: W_std_ini = 1.0e0;

    WSig = np.ones((num_W, 1)) * (W_std_ini)
    WSig_i = 1.0 / (WSig)
    WSig_i_ant = WSig_i.copy()
    mylamb4 = prior_lambda    #it works well with 1.0e-1
    Wmu_aux = np.zeros((num_W, 1))

    """Initialization for Variational posteriors"""

    V_i = choleskies.multiple_dpotri(choleskies.flat_to_triang(model.q_u_chols.values)).copy()
    Vk = np.zeros_like(V_i)
    for i in range(model.num_latent_funcs):
        Vk[i,:,:] = model.posteriors[i].covariance.T.copy()

    Lk = np.zeros_like(Vk)
    mk = model.q_u_means.values.copy()

    Vki_ant = V_i.copy()
    mk_aux = mk.copy()

    myTimes=[]
    ELBO = []
    NLPD = []
    incre_tao = 0.999
    incre_tao_Z = 0.999

    for Niter in range(max_iters):


        if (Niter)%50==0:
            print("Iteration:", Niter)
            print(model.log_likelihood())

        ELBO.append(model.log_likelihood())
        if (not ((Xval is None) or (Yval is None))) and ((Niter)%50==0):
            NLPD.append(model.negative_log_predictive(Xval, Yval))
        if Niter==0:
            start=time.time()
            myTimes.append(start)
        else:
            myTimes.append(time.time())

        MC_aux=MC

        dL_dtheta, d2L_dtheta, dL_dZ, d2L_dZ,_, _, dL_dW, d2L_dW = compute_stoch_grads_hyparam_HetMOGP(
            model=model, index_ini=index_ini,
            hpmu=hpmu, hpSig=hpSig,
            ysample=ysample, mu=mu,
            Sig=Sig, Wmu=Wmu, WSig=WSig, MC=MC_aux)

        if (Niter+1)%1==0:


            if not model.Z.is_fixed:

                mu_ant = mu_aux.copy()
                mu_aux = mu.copy()

                num = (np.sqrt(Sig_i - tao_Z * mylamb) + tao_Z * mylamb)
                Sig_i = (1.0 - tao_Z * beta1_k) * Sig_i + beta1_k * (d2L_dZ[0]+tao_Z*mylamb)

                Sig = 1.0 / Sig_i
                den = (np.sqrt(Sig_i - tao_Z * mylamb)  + tao_Z * mylamb) ** -1
                mu = mu - alpha1_k * den * (dL_dZ[0] + tao_Z * mylamb * mu) + gamma1_k * num * den * (mu - mu_ant)

            "The Vprop-mom algorithm for kernel hyper-param"
            hpmu_ant = hpmu_aux.copy()
            hpmu_aux = hpmu.copy()

            for i in range(model.num_latent_funcs):

                index = index_ini[i].copy()

                if not index.shape[0] == 0:  # This to check if it is not empty (hyper-par fixed) and not necessary to compute update
                    num = (np.sqrt(hpSig_i[i][index] - tao * mylamb3) + tao * mylamb3)
                    hpSig_i[i][index] = (1.0 - tao * beta3_k) * hpSig_i[i][index] + beta3_k * (d2L_dtheta[i] + tao * mylamb3)
                    hpSig[i][index] = 1.0 / hpSig_i[i][index]

                    den = (np.sqrt(hpSig_i[i][index] - tao * mylamb3) + tao * mylamb3) ** -1
                    hpmu[i][index] = hpmu[i][index] - alpha3_k * den * (dL_dtheta[i] + tao*mylamb3 * hpmu[i][index]) + gamma3_k * num * den * (hpmu[i][index] - hpmu_ant[i][index])

                    hpSig_i_ant[i][index] = hpSig_i[i][index].copy()

            "The Vprop-mom algorithm for W params for corregionalisation"
            Wmu_ant = Wmu_aux.copy()
            Wmu_aux = Wmu.copy()

            for i in range(model.num_latent_funcs):
                if not eval('model.B_q'+str(i)+'.W.is_fixed'):
                    index = np.arange(i * model.num_output_funcs, i * model.num_output_funcs + model.num_output_funcs)
                    num = (np.sqrt(WSig_i[index] - tao * mylamb4)  + tao * mylamb4)
                    WSig_i[index] = (1.0 - tao * beta4_k) * WSig_i[index] + beta4_k * (d2L_dW[i] + tao * mylamb4)
                    WSig[index] = 1.0 / WSig_i[index]
                    den = (np.sqrt(WSig_i[index] - tao * mylamb4) + tao * mylamb4) ** -1
                    Wmu[index] = Wmu[index] - alpha4_k * den * (dL_dW[i] + tao*mylamb4 * Wmu[index]) + gamma4_k * den * \
                                 num * (Wmu[index] - Wmu_ant[index])
                    WSig_i_ant[index] = WSig_i[index].copy()


        if (Niter)%1==0:

            model.update_model(False)
            model.q_u_means.unfix()
            model.q_u_chols.unfix()

            dL_dm, dL_dV = compute_stoch_grads_for_qu_HetMOGP(model=model)

            mk_ant = mk_aux.copy()
            mk_aux = mk.copy()

            for i in range(model.num_latent_funcs):

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
                    #print(model['B*'])
                    Vk[i, :, :] = 1.0e-1*np.eye(*Vk[i,:,:].shape)
                    Lk[i, :, :] = linalg.jitchol(Vk[i, :, :])
                    V_i[i, :, :] = np.linalg.inv(Vk[i, :, :])
                    mk[:, i] = mk[:, i]*0.0

            Vki_ant = V_i.copy()


            model.L_u.setfield(choleskies.triang_to_flat(Lk), np.float64)
            model.m_u.setfield(mk, np.float64)

            if (Niter<max_iters-1):
                model.q_u_means.fix()
                model.q_u_chols.fix()
                if model.batch_size is not None:
                    model.set_data(*model.new_batch())


        tao = tao * incre_tao
        tao_Z = tao_Z * incre_tao_Z

    print("Iteration:", Niter+1)
    print(model.log_likelihood())
    ELBO.append(model.log_likelihood())
    if (not ((Xval is None) or (Yval is None))):
        NLPD.append(model.negative_log_predictive(Xval, Yval))
    myTimes.append(time.time())
    model.update_model(True)
    return np.array(ELBO).flatten(),NLPD, np.array(myTimes)-start

def hybrid_opt_HetMOGP(model,Xval=None,Yval=None, max_iters=1000, step_rate=0.01, decay_mom1=1-0.9,decay_mom2=1-0.999):
    global mk_ant,mk_aux,mk,V_i,Vk,Lk, Vk,Vki_ant
    def natural_grad_qu(model, n_iter=1, step_size=step_rate, momentum=0.0):
        global mk_ant, mk_aux, mk, V_i, Vk, Lk, Vk, Vki_ant
        """"Initialize the step-sizes"""""
        beta2_k = step_size
        gamma2_k = momentum
        alpha2_k = step_size

        if n_iter==1:
            V_i = choleskies.multiple_dpotri(choleskies.flat_to_triang(model.q_u_chols.values)).copy()
            Vk = np.zeros_like(V_i)
            for i in range(model.num_latent_funcs):
                Vk[i, :, :] = model.posteriors[i].covariance.T.copy()

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

            for i in range(model.num_latent_funcs):

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
                    Vk[i, :, :] = 1.0e-1*np.eye(*Vk[i,:,:].shape)
                    Lk[i, :, :] = linalg.jitchol(Vk[i, :, :])
                    V_i[i, :, :] = np.linalg.inv(Vk[i, :, :])
                    mk[:, i] = mk[:, i]*0.0

            Vki_ant = V_i.copy()

            model.L_u.setfield(choleskies.triang_to_flat(Lk), np.float64)
            model.m_u.setfield(mk, np.float64)

    global ELBO, myTimes, sched, NLPD
    ELBO = []
    NLPD = []
    myTimes = []
    sched = step_rate
    def callhybrid(i):
        global start
        global ELBO, myTimes, sched, NLPD

        model.update_model(False)
        model.q_u_means.unfix()
        model.q_u_chols.unfix()
        natural_grad_qu(model, n_iter=i['n_iter'], step_size=step_rate, momentum=0.0)

        model.q_u_means.fix()
        model.q_u_chols.fix()
        model.update_model(True)

        ELBO.append(model.log_likelihood())
        myTimes.append(time.time())

        if (i['n_iter']) % 50 == 0:
            print("Iteration:",i['n_iter'])
            print(model.log_likelihood())
            if not(Xval==None or Yval==None):
                NLPD.append(model.negative_log_predictive(Xval, Yval, num_samples=1000))

        if i['n_iter'] >= max_iters:
            model.q_u_means.unfix()
            model.q_u_chols.unfix()
            return True
        return False


    model.q_u_means.fix()
    model.q_u_chols.fix()
    opt = climin.Adam(model.optimizer_array, model.stochastic_grad, step_rate=step_rate, decay_mom1=decay_mom1,decay_mom2=decay_mom2)
    ELBO.append(model.log_likelihood())
    if not (Xval == None or Yval == None):
        NLPD.append(model.negative_log_predictive(Xval, Yval, num_samples=1000))
    start = time.time()
    myTimes.append(start)
    info = opt.minimize_until(callhybrid)
    return np.array(ELBO).flatten(),np.array(NLPD), np.array(myTimes)-start