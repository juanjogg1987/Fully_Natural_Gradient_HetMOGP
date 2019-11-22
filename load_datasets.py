import numpy as np
import sys
import scipy.io as sio
import GPy

np.random.seed(101)
import random
random.seed(101)

from likelihoods.bernoulli import Bernoulli
from likelihoods.gaussian import Gaussian
from likelihoods.hetgaussian import HetGaussian
from likelihoods.beta import Beta
from likelihoods.gamma import Gamma
from likelihoods.exponential import Exponential
from likelihoods.poisson import Poisson


from hetmogp.het_likelihood import HetLikelihood

path_ini = '/home/juanj/My_codes_py/'
sys.path.append('/home/juanj/My_codes_py/')

def load_toy1(N=1000,input_dim=1):
    if input_dim==2:
        Nsqrt = int(N**(1.0/input_dim))
    print('input_dim:',input_dim)
    #Q = 5  # number of latent functions

    # Heterogeneous Likelihood Definition
    likelihoods_list = [HetGaussian(), Beta(),Bernoulli()]  # Real + Binary
    likelihood = HetLikelihood(likelihoods_list)
    Y_metadata = likelihood.generate_metadata()
    D = likelihoods_list.__len__()
    Q = 3
    """"""""""""""""""""""""""""""

    Dim = input_dim
    if input_dim ==2:
        xy = np.linspace(0.0, 1.0, Nsqrt)
        xx = np.linspace(0.0, 1.0, Nsqrt)
        XX, XY = np.meshgrid(xx, xy)
        XX = XX.reshape(Nsqrt ** 2, 1)
        XY = XY.reshape(Nsqrt ** 2, 1)
        Xtoy = np.hstack((XX, XY))
    else:
        minis = 0 * np.ones(Dim)
        maxis = 1 * np.ones(Dim)
        Xtoy = np.linspace(minis[0], maxis[0], N).reshape(1, -1)
        for i in range(Dim - 1):
            Xaux = np.linspace(minis[i + 1], maxis[i + 1], N)
            Xtoy = np.concatenate((Xtoy, Xaux[np.random.permutation(N)].reshape(1, -1)), axis=0)
            # Z = np.concatenate((Z, Zaux.reshape(1, -1)), axis=0)
        Xtoy = 1.0 * Xtoy.T


    def latent_functions_prior(Q, lenghtscale=None, variance=None, input_dim=None):
        if lenghtscale is None:
            lenghtscale = np.array([0.5, 0.05, 0.1]) #This is the one used for previous experiments
            #lenghtscale = np.array([1.0, 0.1, 0.2])
        else:
            lenghtscale = lenghtscale

        if variance is None:
            #variance = 1.0*np.random.rand(Q)
            variance = 1 * np.ones(Q)
        else:
            variance = variance

        kern_list = []
        for q in range(Q):
            #print("length:",lenghtscale[q])
            #print("var:", variance[q])
            kern_q = GPy.kern.RBF(input_dim=input_dim, lengthscale=lenghtscale[q], variance=variance[q], name='rbf')
            kern_q.name = 'kern_q' + str(q)
            kern_list.append(kern_q)
        return kern_list

    kern_list = latent_functions_prior(Q, lenghtscale=None, variance=None, input_dim=Dim)

    # True U and F functions
    def experiment_true_u_functions(kern_list, X):
        Q = kern_list.__len__()
        u_latent = np.zeros((X.shape[0], Q))
        np.random.seed(104)
        for q in range(Q):
            u_latent[:, q] = np.random.multivariate_normal(np.zeros(X.shape[0]), kern_list[q].K(X))

        return u_latent

    def experiment_true_f_functions(true_u, X_list, J):

        Q = true_u.shape[1]
        W = W_lincombination(Q, J)
        #
        f_j = np.zeros((X_list.shape[0], J))
        for q in range(Q):
            f_j += (W[q]*true_u[:,q]).T

        return f_j

    # True Combinations
    def W_lincombination(Q, J):
        W_list = []
        for q in range(Q):
            if q==0:
                W_list.append(np.array([-0.1,-0.1, 1.1, 2.1, -1.1])[:,None])
            elif q == 1:
                W_list.append(np.array([1.4, -0.5, 0.3, 0.7, 1.5])[:,None])
            else:
                W_list.append(np.array([0.1, -0.8, 1.3, 1.5, 0.5])[:,None])

        return W_list

    """"""""""""""""""""""""""""""""

    # True functions values for inputs X
    f_index = Y_metadata['function_index'].flatten()
    J = f_index.__len__()
    trueU = experiment_true_u_functions(kern_list, Xtoy)
    trueF = experiment_true_f_functions(trueU, Xtoy,J)

    d_index = Y_metadata['d_index'].flatten()
    F_true = []


    for t in range(D):
        _, num_f_task, _ = likelihoods_list[t].get_metadata()
        f = np.empty((Xtoy.shape[0], num_f_task))
        for j in range(J):
            if f_index[j] == t:
                f[:, d_index[j], None] = trueF[:,j][:,None]

        F_true.append(f)

    # Generating training data Y (sampling from heterogeneous likelihood)
    Ytrain = likelihood.samples(F=F_true, Y_metadata=Y_metadata)
    Yreg = (Ytrain[0] - Ytrain[0].mean(0)) / (Ytrain[0].std(0))
    Ytrain = [Yreg,np.clip(Ytrain[1],1.0e-9,0.99999),Ytrain[2]]
    Xtrain = []
    for d in range(likelihoods_list.__len__()):
        Xtrain.append(Xtoy)

    return Xtrain, Ytrain

def load_toy2(N=1000,input_dim=1):
    if input_dim==2:
        Nsqrt = int(N**(1.0/input_dim))
    print('input_dim:',input_dim)

    # Heterogeneous Likelihood Definition
    likelihoods_list = [HetGaussian(), Beta(), Bernoulli(), Gamma(), Exponential()]  # Real + Binary
    likelihood = HetLikelihood(likelihoods_list)
    Y_metadata = likelihood.generate_metadata()
    D = likelihoods_list.__len__()
    Q = 3
    """"""""""""""""""""""""""""""

    Dim = input_dim
    if input_dim ==2:
        xy = np.linspace(0.0, 1.0, Nsqrt)
        xx = np.linspace(0.0, 1.0, Nsqrt)
        XX, XY = np.meshgrid(xx, xy)
        XX = XX.reshape(Nsqrt ** 2, 1)
        XY = XY.reshape(Nsqrt ** 2, 1)
        Xtoy = np.hstack((XX, XY))
    else:
        minis = 0 * np.ones(Dim)
        maxis = 1 * np.ones(Dim)
        Xtoy = np.linspace(minis[0], maxis[0], N).reshape(1, -1)
        for i in range(Dim - 1):
            Xaux = np.linspace(minis[i + 1], maxis[i + 1], N)
            Xtoy = np.concatenate((Xtoy, Xaux[np.random.permutation(N)].reshape(1, -1)), axis=0)
            # Z = np.concatenate((Z, Zaux.reshape(1, -1)), axis=0)
        Xtoy = 1.0 * Xtoy.T


    def latent_functions_prior(Q, lenghtscale=None, variance=None, input_dim=None):
        if lenghtscale is None:
            lenghtscale = np.array([0.5, 0.05, 0.1]) #This is the one used for previous experiments
        else:
            lenghtscale = lenghtscale

        if variance is None:
            variance = 1 * np.ones(Q)
        else:
            variance = variance

        kern_list = []
        for q in range(Q):
            kern_q = GPy.kern.RBF(input_dim=input_dim, lengthscale=lenghtscale[q], variance=variance[q], name='rbf')
            kern_q.name = 'kern_q' + str(q)
            kern_list.append(kern_q)
        return kern_list

    kern_list = latent_functions_prior(Q, lenghtscale=None, variance=None, input_dim=Dim)

    # True U and F functions
    def experiment_true_u_functions(kern_list, X):
        Q = kern_list.__len__()
        u_latent = np.zeros((X.shape[0], Q))
        np.random.seed(104)
        for q in range(Q):
            u_latent[:, q] = np.random.multivariate_normal(np.zeros(X.shape[0]), kern_list[q].K(X))

        return u_latent

    def experiment_true_f_functions(true_u, X_list, J):
        #true_f = []

        Q = true_u.shape[1]
        W = W_lincombination(Q, J)
        f_j = np.zeros((X_list.shape[0], J))
        for q in range(Q):
            f_j += (W[q]*true_u[:,q]).T

        return f_j

    # True Combinations
    def W_lincombination(Q, J):
        W_list = []
        # q=1
        for q in range(Q):
            if q==0:
                W_list.append(np.array([-0.1,-0.1, 1.1, 2.1, -1.1, -0.5, -0.6, 0.1])[:,None])
            elif q == 1:
                W_list.append(np.array([1.4, -0.5, 0.3, 0.7, 1.5, -0.3, 0.4, -0.2])[:,None])
            else:
                W_list.append(np.array([0.1, -0.8, 1.3, 1.5, 0.5,-0.02, 0.01, 0.5])[:,None])

        return W_list

    """"""""""""""""""""""""""""""""

    # True functions values for inputs X
    f_index = Y_metadata['function_index'].flatten()
    J = f_index.__len__()
    trueU = experiment_true_u_functions(kern_list, Xtoy)
    trueF = experiment_true_f_functions(trueU, Xtoy,J)

    d_index = Y_metadata['d_index'].flatten()
    F_true = []

    for t in range(D):
        _, num_f_task, _ = likelihoods_list[t].get_metadata()
        f = np.empty((Xtoy.shape[0], num_f_task))
        for j in range(J):
            if f_index[j] == t:
                f[:, d_index[j], None] = trueF[:,j][:,None]

        F_true.append(f)

    # Generating training data Y (sampling from heterogeneous likelihood)
    Ytrain = likelihood.samples(F=F_true, Y_metadata=Y_metadata)
    Yreg = (Ytrain[0] - Ytrain[0].mean(0)) / (Ytrain[0].std(0))
    Ytrain = [Yreg,np.clip(Ytrain[1],1.0e-9,0.99999),Ytrain[2],Ytrain[3],Ytrain[4]]
    Xtrain = []
    for d in range(likelihoods_list.__len__()):
        Xtrain.append(Xtoy)

    return Xtrain, Ytrain

def load_toy3(N=1000,input_dim=1):
    if input_dim==2:
        Nsqrt = int(N**(1.0/input_dim))
    print('input_dim:',input_dim)

    # Heterogeneous Likelihood Definition
    likelihoods_list = [HetGaussian(), Beta(), Bernoulli(), Gamma(), Exponential(), Gaussian(sigma=0.1), Beta(), Bernoulli(), Gamma(), Exponential()]  # Real + Binary
    likelihood = HetLikelihood(likelihoods_list)
    Y_metadata = likelihood.generate_metadata()
    D = likelihoods_list.__len__()
    Q = 3
    """"""""""""""""""""""""""""""

    Dim = input_dim
    if input_dim ==2:
        xy = np.linspace(0.0, 1.0, Nsqrt)
        xx = np.linspace(0.0, 1.0, Nsqrt)
        XX, XY = np.meshgrid(xx, xy)
        XX = XX.reshape(Nsqrt ** 2, 1)
        XY = XY.reshape(Nsqrt ** 2, 1)
        Xtoy = np.hstack((XX, XY))
    else:
        minis = 0 * np.ones(Dim)
        maxis = 1 * np.ones(Dim)
        Xtoy = np.linspace(minis[0], maxis[0], N).reshape(1, -1)
        for i in range(Dim - 1):
            Xaux = np.linspace(minis[i + 1], maxis[i + 1], N)
            Xtoy = np.concatenate((Xtoy, Xaux[np.random.permutation(N)].reshape(1, -1)), axis=0)
            # Z = np.concatenate((Z, Zaux.reshape(1, -1)), axis=0)
        Xtoy = 1.0 * Xtoy.T


    def latent_functions_prior(Q, lenghtscale=None, variance=None, input_dim=None):
        if lenghtscale is None:
            lenghtscale = np.array([0.5, 0.05, 0.1]) #This is the one used for previous experiments
        else:
            lenghtscale = lenghtscale

        if variance is None:
            variance = 1 * np.ones(Q)
        else:
            variance = variance

        kern_list = []
        for q in range(Q):
            kern_q = GPy.kern.RBF(input_dim=input_dim, lengthscale=lenghtscale[q], variance=variance[q], name='rbf')
            kern_q.name = 'kern_q' + str(q)
            kern_list.append(kern_q)
        return kern_list

    kern_list = latent_functions_prior(Q, lenghtscale=None, variance=None, input_dim=Dim)

    # True U and F functions
    def experiment_true_u_functions(kern_list, X):
        Q = kern_list.__len__()
        u_latent = np.zeros((X.shape[0], Q))
        np.random.seed(104)
        for q in range(Q):
            u_latent[:, q] = np.random.multivariate_normal(np.zeros(X.shape[0]), kern_list[q].K(X))

        return u_latent

    def experiment_true_f_functions(true_u, X_list, J):

        Q = true_u.shape[1]
        W = W_lincombination(Q, J)
        #print(W)
        #for j in range(J):
        f_j = np.zeros((X_list.shape[0], J))
        for q in range(Q):
            f_j += (W[q]*true_u[:,q]).T
        #true_f.append(f_d)

        return f_j

    # True Combinations
    def W_lincombination(Q, J):
        W_list = []
        # q=1
        for q in range(Q):
            if q==0:
                W_list.append(np.array([-0.1,-0.1, 1.1, 2.1, -1.1, -0.5, -0.6, 0.1,  -1.1, 0.8, 1.5, -0.2, 0.05, 0.06, 0.3])[:,None])
            elif q == 1:
                W_list.append(np.array([1.4, -0.5, 0.3, 0.7, 1.5, -0.3, 0.4, -0.2,  0.4, 0.3, -0.7, -2.1, -0.03, 0.04, -0.5])[:,None])
            else:
                W_list.append(np.array([0.1, -0.8, 1.3, 1.5, 0.5,-0.02, 0.01, 0.5,  0.5, 1.0, 0.8, 3.0,0.1, -0.5, 0.4])[:,None])

        return W_list

    """"""""""""""""""""""""""""""""

    # True functions values for inputs X
    f_index = Y_metadata['function_index'].flatten()
    J = f_index.__len__()
    trueU = experiment_true_u_functions(kern_list, Xtoy)
    trueF = experiment_true_f_functions(trueU, Xtoy,J)

    d_index = Y_metadata['d_index'].flatten()
    F_true = []

    for t in range(D):
        _, num_f_task, _ = likelihoods_list[t].get_metadata()
        f = np.empty((Xtoy.shape[0], num_f_task))
        for j in range(J):
            if f_index[j] == t:
                f[:, d_index[j], None] = trueF[:,j][:,None]

        F_true.append(f)

    # Generating training data Y (sampling from heterogeneous likelihood)
    Ytrain = likelihood.samples(F=F_true, Y_metadata=Y_metadata)
    Yreg1 = (Ytrain[0] - Ytrain[0].mean(0)) / (Ytrain[0].std(0))
    Yreg2 = (Ytrain[5] - Ytrain[5].mean(0)) / (Ytrain[5].std(0))
    Ytrain = [Yreg1,np.clip(Ytrain[1],1.0e-9,0.99999),Ytrain[2],Ytrain[3],Ytrain[4],Yreg2,np.clip(Ytrain[6],1.0e-9,0.99999),Ytrain[7],Ytrain[8],Ytrain[9]]
    Xtrain = []
    for d in range(likelihoods_list.__len__()):
        Xtrain.append(Xtoy)

    return Xtrain, Ytrain

def load_toy4(N=1000,input_dim=1):
    if input_dim==2:
        Nsqrt = int(N**(1.0/input_dim))
    print('input_dim:',input_dim)

    likelihoods_list = [HetGaussian(), Beta()]  # Real + Binary
    likelihood = HetLikelihood(likelihoods_list)
    Y_metadata = likelihood.generate_metadata()
    D = likelihoods_list.__len__()
    Q = 3
    """"""""""""""""""""""""""""""

    Dim = input_dim
    if input_dim ==2:
        xy = np.linspace(0.0, 1.0, Nsqrt)
        xx = np.linspace(0.0, 1.0, Nsqrt)
        XX, XY = np.meshgrid(xx, xy)
        XX = XX.reshape(Nsqrt ** 2, 1)
        XY = XY.reshape(Nsqrt ** 2, 1)
        Xtoy = np.hstack((XX, XY))
    else:
        minis = 0 * np.ones(Dim)
        maxis = 1 * np.ones(Dim)
        Xtoy = np.linspace(minis[0], maxis[0], N).reshape(1, -1)
        for i in range(Dim - 1):
            Xaux = np.linspace(minis[i + 1], maxis[i + 1], N)
            Xtoy = np.concatenate((Xtoy, Xaux[np.random.permutation(N)].reshape(1, -1)), axis=0)
        Xtoy = 1.0 * Xtoy.T


    def latent_functions_prior(Q, lenghtscale=None, variance=None, input_dim=None):
        if lenghtscale is None:
            lenghtscale = np.array([0.5, 0.05, 0.1]) #This is the one used for previous experiments
        else:
            lenghtscale = lenghtscale

        if variance is None:
            variance = 1 * np.ones(Q)
        else:
            variance = variance

        kern_list = []
        for q in range(Q):
            kern_q = GPy.kern.RBF(input_dim=input_dim, lengthscale=lenghtscale[q], variance=variance[q], name='rbf')
            kern_q.name = 'kern_q' + str(q)
            kern_list.append(kern_q)
        return kern_list

    kern_list = latent_functions_prior(Q, lenghtscale=None, variance=None, input_dim=Dim)

    # True U and F functions
    def experiment_true_u_functions(kern_list, X):
        Q = kern_list.__len__()
        u_latent = np.zeros((X.shape[0], Q))
        np.random.seed(104)
        for q in range(Q):
            u_latent[:, q] = np.random.multivariate_normal(np.zeros(X.shape[0]), kern_list[q].K(X))

        return u_latent

    def experiment_true_f_functions(true_u, X_list, J):
        #true_f = []

        Q = true_u.shape[1]
        W = W_lincombination(Q, J)
        #print(W)
        #for j in range(J):
        f_j = np.zeros((X_list.shape[0], J))
        for q in range(Q):
            f_j += (W[q]*true_u[:,q]).T
        #true_f.append(f_d)

        return f_j

    # True Combinations
    def W_lincombination(Q, J):
        W_list = []
        # q=1
        for q in range(Q):
            if q==0:
                W_list.append(np.array([-0.1,-0.1,1.1,2.1])[:,None])
            elif q == 1:
                W_list.append(np.array([1.4, -0.5, 0.3, 0.7])[:,None])
            else:
                W_list.append(np.array([0.1, -0.8, 1.3, 1.5])[:,None])

        return W_list

    """"""""""""""""""""""""""""""""

    # True functions values for inputs X
    f_index = Y_metadata['function_index'].flatten()
    J = f_index.__len__()
    trueU = experiment_true_u_functions(kern_list, Xtoy)
    trueF = experiment_true_f_functions(trueU, Xtoy,J)

    d_index = Y_metadata['d_index'].flatten()
    F_true = []

    for t in range(D):
        _, num_f_task, _ = likelihoods_list[t].get_metadata()
        f = np.empty((Xtoy.shape[0], num_f_task))
        for j in range(J):
            if f_index[j] == t:
                f[:, d_index[j], None] = trueF[:,j][:,None]

        F_true.append(f)

    # Generating training data Y (sampling from heterogeneous likelihood)
    Ytrain = likelihood.samples(F=F_true, Y_metadata=Y_metadata)
    Yreg = (Ytrain[0] - Ytrain[0].mean(0)) / (Ytrain[0].std(0))
    Ytrain = [Yreg,np.clip(Ytrain[1],1.0e-9,0.99999)]
    Xtrain = []
    for d in range(likelihoods_list.__len__()):
        Xtrain.append(Xtoy)

    return Xtrain, Ytrain

def load_toy5(N=1000,input_dim=1):
    if input_dim==2:
        Nsqrt = int(N**(1.0/input_dim))
    print('input_dim:',input_dim)
    #Q = 5  # number of latent functions

    # Heterogeneous Likelihood Definition
    # likelihoods_list = [Gaussian(sigma=1.0), Bernoulli()] # Real + Binary
    likelihoods_list = [HetGaussian(), Beta(), Gamma()]  # Real + Binary
    # likelihoods_list = [Gaussian(sigma=1.0)]
    likelihood = HetLikelihood(likelihoods_list)
    Y_metadata = likelihood.generate_metadata()
    D = likelihoods_list.__len__()
    Q = 3
    """"""""""""""""""""""""""""""

    Dim = input_dim
    if input_dim ==2:
        xy = np.linspace(0.0, 1.0, Nsqrt)
        xx = np.linspace(0.0, 1.0, Nsqrt)
        XX, XY = np.meshgrid(xx, xy)
        XX = XX.reshape(Nsqrt ** 2, 1)
        XY = XY.reshape(Nsqrt ** 2, 1)
        Xtoy = np.hstack((XX, XY))
    else:
        minis = 0 * np.ones(Dim)
        maxis = 1 * np.ones(Dim)
        Xtoy = np.linspace(minis[0], maxis[0], N).reshape(1, -1)
        for i in range(Dim - 1):
            Xaux = np.linspace(minis[i + 1], maxis[i + 1], N)
            Xtoy = np.concatenate((Xtoy, Xaux[np.random.permutation(N)].reshape(1, -1)), axis=0)
            # Z = np.concatenate((Z, Zaux.reshape(1, -1)), axis=0)
        Xtoy = 1.0 * Xtoy.T


    def latent_functions_prior(Q, lenghtscale=None, variance=None, input_dim=None):
        if lenghtscale is None:
            lenghtscale = np.array([0.5, 0.05, 0.1]) #This is the one used for previous experiments
            #lenghtscale = np.array([1.0, 0.1, 0.2])
        else:
            lenghtscale = lenghtscale

        if variance is None:
            #variance = 1.0*np.random.rand(Q)
            variance = 1 * np.ones(Q)
        else:
            variance = variance

        kern_list = []
        for q in range(Q):
            #print("length:",lenghtscale[q])
            #print("var:", variance[q])
            kern_q = GPy.kern.RBF(input_dim=input_dim, lengthscale=lenghtscale[q], variance=variance[q], name='rbf')
            kern_q.name = 'kern_q' + str(q)
            kern_list.append(kern_q)
        return kern_list

    kern_list = latent_functions_prior(Q, lenghtscale=None, variance=None, input_dim=Dim)

    # True U and F functions
    def experiment_true_u_functions(kern_list, X):
        Q = kern_list.__len__()
        # for d,X in enumerate(X_list):
        u_latent = np.zeros((X.shape[0], Q))
        np.random.seed(104)
        for q in range(Q):
            u_latent[:, q] = np.random.multivariate_normal(np.zeros(X.shape[0]), kern_list[q].K(X))

        return u_latent

    def experiment_true_f_functions(true_u, X_list, J):
        #true_f = []

        Q = true_u.shape[1]
        W = W_lincombination(Q, J)
        #print(W)
        #for j in range(J):
        f_j = np.zeros((X_list.shape[0], J))
        for q in range(Q):
            f_j += (W[q]*true_u[:,q]).T
        #true_f.append(f_d)

        return f_j

    # True Combinations
    def W_lincombination(Q, J):
        W_list = []
        # q=1
        for q in range(Q):
            # W_list.append(np.array(([[-0.5], [0.1]])))
            if q==0:
                #W_list.append(0.3*np.random.randn(J, 1))
                W_list.append(np.array([-0.1,-0.1, 1.1, 2.1, -0.5, -0.6])[:,None])
            elif q == 1:
                #W_list.append(2.0 * np.random.randn(J, 1))
                W_list.append(np.array([1.4, -0.5, 0.3, 0.7, -0.3, 0.4])[:,None])
            else:
                #W_list.append(10.0 * np.random.randn(J, 1)+0.1)
                W_list.append(np.array([0.1, -0.8, 1.3, 1.5, -0.02, 0.01])[:,None])

        return W_list

    """"""""""""""""""""""""""""""""

    # True functions values for inputs X
    f_index = Y_metadata['function_index'].flatten()
    J = f_index.__len__()
    trueU = experiment_true_u_functions(kern_list, Xtoy)
    trueF = experiment_true_f_functions(trueU, Xtoy,J)
    # if input_dim==2:
    #     #from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
    #
    #     from matplotlib import cm
    #     from matplotlib.ticker import LinearLocator, FormatStrFormatter
    #     fig = plt.figure(15)
    #     ax = fig.gca(projection='3d')
    #
    #     # Make data.
    #     # X = np.arange(-5, 5, 0.25)
    #     # Y = np.arange(-5, 5, 0.25)
    #     # X, Y = np.meshgrid(X, Y)
    #     # R = np.sqrt(X ** 2 + Y ** 2)
    #     # Z = np.sin(R)
    #
    #     # Plot the surface.
    #     surf = ax.plot_surface(Xtoy[:,0].reshape(Nsqrt,Nsqrt), Xtoy[:,1].reshape(Nsqrt,Nsqrt), trueF[:,4].reshape(Nsqrt,Nsqrt), cmap=cm.coolwarm,linewidth=0, antialiased=False)
    #
    #     # Customize the z axis.
    #     #ax.set_zlim(-1.01, 1.01)
    #     ax.zaxis.set_major_locator(LinearLocator(10))
    #     ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    #
    #     # Add a color bar which maps values to colors.
    #     fig.colorbar(surf, shrink=0.5, aspect=5)
    #
    #     plt.show()
    #
    # else:
    #     plt.figure(15)
    #     plt.plot(trueF[:,4])
    #     plt.figure(16)
    #     plt.plot(trueU)

    d_index = Y_metadata['d_index'].flatten()
    F_true = []
    # for i,f_latent in enumerate(trueF):
    #    if

    for t in range(D):
        _, num_f_task, _ = likelihoods_list[t].get_metadata()
        f = np.empty((Xtoy.shape[0], num_f_task))
        for j in range(J):
            if f_index[j] == t:
                f[:, d_index[j], None] = trueF[:,j][:,None]

        F_true.append(f)

    # Generating training data Y (sampling from heterogeneous likelihood)
    Ytrain = likelihood.samples(F=F_true, Y_metadata=Y_metadata)
    #Yreg = (Ytrain[0]-Ytrain[0].min())/(Ytrain[0].max()-Ytrain[0].min())
    Yreg = (Ytrain[0] - Ytrain[0].mean(0)) / (Ytrain[0].std(0))
    Ytrain = [Yreg,np.clip(Ytrain[1],1.0e-9,0.99999),Ytrain[2]]
    Xtrain = []
    for d in range(likelihoods_list.__len__()):
        Xtrain.append(Xtoy)

    return Xtrain, Ytrain

def load_Hetdata(dataset='london', Ntoy= None, Only_input_dim_toy = None):
    if (dataset=='toy1' or dataset=='toy2' or dataset=='toy3'or dataset=='toy4'or dataset=='toy5') and ((Only_input_dim_toy is None) or (Ntoy is None)):
        print('For the toys you have to provide second and third arguments as Ndata and input dim respectively!')
    path = path_ini+'Database_Heterogeneous'

    if dataset == 'human':
        # Load Data
        dataEB2 = sio.loadmat(path+'/eb2.mat')
        X1 = dataEB2['X']  # Input for both Ybin and Yexp
        Ybin = dataEB2['Ybin']  # Ybin is the at-home-not at home indicator
        Yexp = dataEB2['Yexp']  # Yexp is the distance wandered (make it real valued with the log!)
        Xcount = dataEB2['Xapp']  # Input for the use-of-Whatsapp indicator
        #Y2 = dataEB2['Yapp']  # Use-of-Whatsapp indicator
        Ycount = dataEB2['Ycount']  # Number of active apps

        Yreal = np.log(Yexp)
        Yreal = Yreal - np.mean(Yreal)
        Yreal = Yreal / np.std(Yreal)

        rate1 = 1  #it was 7 for Pablo's case
        Ybin = Ybin[::rate1, :]
        Yreal = Yreal[::rate1, :]
        X1 = X1[::rate1, :]
        X1 = (X1-X1.mean(0))/X1.std(0)

        rate2 = 1    #use 4 to balance equal to the other outputs
        Ycount = Ycount[::rate2, :]
        aux = Ycount - Ycount.min() + 0.1
        Ycount = np.clip(2*(aux / aux.max() - 0.0001),1e-3,0.99)
        #Ycount = aux / aux.max() - 0.0001
        Xcount = Xcount[::rate2, :]
        Xcount = (Xcount-Xcount.mean(0))/Xcount.std(0)

        Ytrain = [Ybin, Yreal, Ycount]
        Xtrain = [X1, X1, Xcount]

        #Ytrain = [Ycount]
        #Xtrain = [Xcount]

        #flag_stand = True
    elif dataset == 'london':
        # Load Data
        dataLondon = sio.loadmat(path+'/london.mat')
        Xinput = dataLondon['X']
        Ycontract = dataLondon['Ycontract']
        Yprice = dataLondon['Yprice']
        Ytype = dataLondon['Ytype']
        #Ynew = dataLondon['Ynew']

        # greater london limits:
        # latitude (North-South) ~55 is Y axis
        # longitude (East-West) ~0 is X axis
        xmin_london = -0.5105
        xmax_london = 0.3336
        ymin_london = 51.2871
        ymax_london = 51.6925

        # dim 0 of X -> xaxis
        # dim 1 of X -> yaxis
        Xloc = np.zeros(np.shape(Xinput))
        Xloc[:, 0] = (Xinput[:, 1] - xmin_london) / (xmax_london - xmin_london)
        Xloc[:, 1] = (Xinput[:, 0] - ymin_london) / (ymax_london - ymin_london)

        Yreal = np.log(Yprice)
        std_Yreal = np.std(Yreal)
        mean_Yreal = np.mean(Yreal)
        Yreal = Yreal - mean_Yreal
        Yreal = Yreal / std_Yreal

        Ytrain = [Yreal, Ycontract]
        Xtrain = [Xloc, Xloc]

    elif dataset=='naval_beta_gamma':
        Xdata = np.loadtxt(path + '/naval.txt')  # the dataset is not delimited by commas ",", so we let the function infer
        Xdata = np.concatenate((Xdata[:, 0:8], Xdata[:, 9:]), 1)
        aux = Xdata[:, -1:] - Xdata[:, -1:].min()
        Ygamma = np.clip(aux/aux.max(),1.0e-5,1.0) #the last column (output) is always positive so we use it for gamma modeling
        aux = Xdata[:, -2:-1] - Xdata[:, -2:-1].min()
        Ybeta = np.clip(aux / aux.max(),1.0e-5,0.999)
        Xdata = (Xdata - Xdata.mean(0)) / Xdata.std(0)

        Xtrain = [Xdata[:, 0:-2],Xdata[:, 0:-2]]
        Ytrain = [Ybeta,Ygamma]

        # Xtrain = [Xdata[:, 0:-2]]
        # Ytrain = [Ygamma]
    elif dataset=='toy1':
        Xtrain , Ytrain = load_toy1(N=Ntoy,input_dim=Only_input_dim_toy)
    elif dataset=='toy2':
        Xtrain, Ytrain = load_toy2(N=Ntoy, input_dim=Only_input_dim_toy)
    elif dataset=='toy3':
        Xtrain, Ytrain = load_toy3(N=Ntoy, input_dim=Only_input_dim_toy)
    elif dataset=='toy4':
        Xtrain, Ytrain = load_toy4(N=Ntoy, input_dim=Only_input_dim_toy)
    elif dataset=='toy5':
        Xtrain, Ytrain = load_toy5(N=Ntoy, input_dim=Only_input_dim_toy)

    else:
        print("The dataset doesn't exist!")
        return 0

    return Xtrain, Ytrain

