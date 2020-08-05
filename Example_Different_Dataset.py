import sys
import climin
import importlib
from functools import partial
import warnings
import os

sys.path.append('..')

import fully_natural_gradient as vo
import load_datasets as pre
from importlib import reload

reload(vo)
reload(pre)

import numpy as np
from scipy.stats import multinomial
from scipy.linalg.blas import dtrmm

import GPy
from GPy.util import choleskies
from GPy.core.parameterization.param import Param
from GPy.kern import Coregionalize
from GPy.likelihoods import Likelihood
from GPy.util import linalg

from likelihoods.poisson import Poisson
from likelihoods.bernoulli import Bernoulli
from likelihoods.gaussian import Gaussian
from likelihoods.categorical import Categorical
from likelihoods.hetgaussian import HetGaussian
from likelihoods.beta import Beta
from likelihoods.gamma import Gamma
from likelihoods.exponential import Exponential

from hetmogp.util import draw_mini_slices
from hetmogp.het_likelihood import HetLikelihood
from hetmogp.svmogp import SVMOGP
from convhetmogp2.CP_hetmogp import ConvHetMOGP
from convhetmogp2 import util

reload(util)

#from hetmogp.util import vem_algorithm as VEM
from GPy.util.univariate_Gaussian import std_norm_cdf, std_norm_pdf
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.pyplot import gca
from matplotlib import rc, font_manager
from matplotlib import rcParams
# from matplotlib2tikz import save as tikz_save
import time
import getopt

warnings.filterwarnings("ignore")
os.environ['PATH'] = os.environ['PATH'] + ':/usr/texbin'
""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""


class commandLine:
    def __init__(self):
        opts, args = getopt.getopt(sys.argv[1:], 'b:m:d:i:s:c:o:w:r:')
        # opts = dict(opts)
        # print(opts)
        self.minibatch = 100
        self.inducing = 20
        self.dimension = 1
        self.N_iter = 1
        self.dataset = 'london'
        self.MC = 1
        self.mom = 0.9
        self.which_model = 'LMC'
        self.which_seed = 101

        for op, arg in opts:
            # print(op,arg)
            if op == '-b':
                self.minibatch = arg
            if op == '-m':
                self.inducing = arg
            if op == '-d':
                self.dimension = arg
            if op == '-i':
                self.N_iter = arg
            if op == '-s':  # this is for (data)set
                self.dataset = arg
            if op == '-c':  # this is for Markov-(c)hain
                self.MC = arg
            if op == '-o':  # this is for M(o)mentum
                self.mom = arg
            if op == '-w':  # (w)hich model CGP, CCGP or HetMOGP or double_HetMOGP
                self.which_model = arg
            if op == '-r':  # (r)and seed
                self.which_seed = arg


""""""""""""""""""""""""""""""
config = commandLine()
#config.N_iter = 2000
num_inducing = int(config.inducing)  # number of inducing points
batch = int(config.minibatch)
input_dim = int(config.dimension)
MC = int(config.MC)
mom = float(config.mom)
""""""""""""""""""""""""""""""
if config.which_model=='CPM':
    convolved = True   #this is to run the Convolutional version
    print('\nModel to use: Convolution Processes Model\n')
else:
    convolved = False
    print('\nModel to use: Linear Model of Coregionalisation\n')
""""""""""""""""""""""""""""""
which_color = {
    "adam": '#d7191c',
    "adad": '#fdbb84',
    "sgd": '#fdbb84',
    "fng": '#2b8cbe',
    "hyb": '#636363',
    "hybf": '#2b8cbe'
}

"""""""""""""""""""""""""""""""""""""""""""""""""""

"""""""""""""""""""""""""""""""""""""""""""""""""""
methods = ['fng']
seeds = [int(config.which_seed)]
path_to_save = '/home/juanjo/Work_at_Home/My_codes_py/Codes_Github_Resubmit/Mock_local_experiments/'+config.which_model+'/'  #Work_at_Home
NLPD_adam = []
NLPD_fng = []
NLPD_adad = []
NLPD_hyb = []
ELBO_adam = []
ELBO_fng = []
ELBO_adad = []
ELBO_hyb = []

for myseed in seeds:
    plt.close('all')

    for posmeth, method in enumerate(methods):
        print('Running with Optimiser '+method)
        np.random.seed(101)
        import random

        random.seed(101)

        """"""""""""""""""""""""""""""""""""""


        def callback(i):
            global n_iter, start, Xtest, Ytest
            global ELBO, myTimes, NLPD

            ELBO.append(model.log_likelihood())
            myTimes.append(time.time())

            if (i['n_iter']) % 50 == 0:
                print(i['n_iter'])
                print(model.log_likelihood())
                #NLPD.append(model.negative_log_predictive(Xtest, Ytest, num_samples=1000))
            if i['n_iter'] > n_iter:
                myTimes = np.array(myTimes) - start
                return True
            return False


        """"""""""""""""""""""""""""""""""""""
        dataset = config.dataset

        Xtrain, Ytrain = pre.load_Hetdata(dataset=dataset, Ntoy=2000, Only_input_dim_toy=input_dim)

        """"""""""""""""""""""""""""""""""""""

        incomplete_out = np.inf
        # Likelihood Definition
        if dataset == 'london':
            likelihoods_list = [HetGaussian(), Bernoulli()]
            Q = 3  # number of latent functions
            q_s_ini = 1000
            prior_lamb = 50  #50 for fng
            #prior_lamb = 1e-1  #for Conv
        elif dataset == 'human':
            Q = 5  # number of latent functions
            likelihoods_list = [Bernoulli(), HetGaussian(), Beta()]
            q_s_ini = 10000
            prior_lamb = 1e-1  #for Conv
        elif dataset == 'naval_beta_gamma':
            Q = 4  # number of latent functions
            q_s_ini = 10
            prior_lamb = 1e-1  #for Conv
            likelihoods_list = [Beta(), Gamma()]
        elif dataset == 'mocap7':
            Q = 3
            q_s_ini = 10000
            #prior_lamb = 1e-8  #for fng
            prior_lamb = 1e-8   #for Conv
            likelihoods_list = [HetGaussian()]* Ytrain.__len__()
        elif dataset == 'mocap8':
            Q = 2
            q_s_ini = 5000
            prior_lamb = 1e-5 #50
            likelihoods_list = [Gamma()]* Ytrain.__len__()
        elif dataset == 'mocap9':
            Q = 3
            q_s_ini = 10000
            prior_lamb = 1e-8    #use for Conv
            likelihoods_list = [Gamma()]* Ytrain.__len__()
        elif dataset == 'sarcos':
            Q = 3
            q_s_ini = 10
            prior_lamb = 50
            likelihoods_list = [HetGaussian()]* Ytrain.__len__()
        elif dataset == 'sarcos_beta':
            Q = 3
            q_s_ini = 10
            prior_lamb = 50
            likelihoods_list = [Beta()]* Ytrain.__len__()
        elif dataset=='traffic':
            Q = 3  # We define the number of latent functions u_q(x)
            q_s_ini = 1000
            #prior_lamb = 50 for LMC
            prior_lamb = 50
            likelihoods_list = [Poisson()]*Ytrain.__len__()
        elif dataset=='mnist':
            Q = 2  # We define the number of latent functions u_q(x)
            q_s_ini = 10
            prior_lamb = 10
            likelihoods_list = [Bernoulli()]*Ytrain.__len__()
        elif dataset == 'toy1':
            Q = 3
            prior_lamb = 1
            my_proportion = [0.75, 0.75, 0.75]
            likelihoods_list = [HetGaussian(), Beta(), Bernoulli()]
        elif dataset == 'toy1c':
            Q = 1
            prior_lamb = 1  #use 50 with Voptimisation
            my_proportion = [0.75, 0.75]
            #likelihoods_list = [Gaussian(sigma=0.1), Gaussian(sigma=0.1)]
            likelihoods_list = [HetGaussian(), HetGaussian()]
        elif dataset == 'toy2':
            Q = 3
            prior_lamb = 1
            my_proportion = [0.75, 0.75, 0.75, 0.75, 0.75]
            likelihoods_list = [HetGaussian(), Beta(), Bernoulli(), Gamma(), Exponential()]
        elif dataset == 'toy3':
            Q = 3
            prior_lamb = 1
            likelihoods_list = [HetGaussian(), Beta(), Bernoulli(), Gamma(), Exponential(), Gaussian(sigma=0.1), Beta(), Bernoulli(),Gamma(),Exponential()]
        elif dataset == 'toy4':
            Q = 3
            prior_lamb = 1
            likelihoods_list = [HetGaussian(), Beta()]
        elif dataset == 'toy5':
            Q = 3
            prior_lamb = 1
            likelihoods_list = [HetGaussian(), Beta(), Gamma()]

        if 'toy' in dataset:
            mydict = {1:10000,2:1000,3:500,4:100,5:50}
            q_s_ini = mydict.get(Xtrain[0].shape[1],10)
            print('input D and q_s_ini',Xtrain[0].shape[1],q_s_ini)

        how_many_outs = 1

        np.random.seed(101)   #this same seed here is to guarrantee always the same data split for train and test
        cut_region = False
        _, Dim = Xtrain[0].shape
        Ntotal_with_test = []
        Ntotal_without_test = []
        index_train = []
        Ntotal_for_test = []
        for_train = 0.75
        for conti in range(likelihoods_list.__len__()):
            Ntotal_with_test.append(Xtrain[conti].shape[0])
            Ntotal_without_test.append(int(Ntotal_with_test[conti] * for_train))
            Ntotal_for_test.append( Ntotal_with_test[conti] - Ntotal_without_test[conti])
            index_train.append( np.random.permutation(np.arange(0, Ntotal_with_test[conti])) )

        rescale = 1
        Xtrain_new = [rescale * Xtrain[index_train[conti][0:Ntotal_without_test[conti]], :].copy() for conti,Xtrain in enumerate(Xtrain)] * how_many_outs
        Ytrain_new = [Ytrain[index_train[conti][0:Ntotal_without_test[conti]]].copy() for conti,Ytrain in enumerate(Ytrain)] * how_many_outs
        Xtest = [rescale * Xtrain[index_train[conti][Ntotal_without_test[conti]:], :].copy() for conti,Xtrain in enumerate(Xtrain)] * how_many_outs
        Ytest = [Ytrain[index_train[conti][Ntotal_without_test[conti]:]].copy() for conti,Ytrain in enumerate(Ytrain)] * how_many_outs

        Xtrain = Xtrain_new
        Ytrain = Ytrain_new

        likelihood = HetLikelihood(likelihoods_list)
        Y_metadata = likelihood.generate_metadata()

        # np.random.seed(101)
        myindex = []
        ind_split_aux = []

        # kmeans for selecting Z
        from scipy.cluster.vq import kmeans

        np.random.seed(myseed)
        # Z = 1.0 * kmeans(Xtrain, num_inducing)[0]

        minis = Xtrain[0].min(0)
        maxis = Xtrain[0].max(0)
        Dim = Xtrain[0].shape[1]
        Z = np.linspace(minis[0], maxis[0], num_inducing).reshape(1, -1)
        for i in range(Dim - 1):
            Zaux = np.linspace(minis[i + 1], maxis[i + 1], num_inducing)
            Z = np.concatenate((Z, Zaux[np.random.permutation(num_inducing)].reshape(1, -1)), axis=0)
            # Z = np.concatenate((Z, Zaux.reshape(1, -1)), axis=0)
        Z = 1.0 * Z.T

        n_iter = int(config.N_iter)
        all_NLPD = []
        Times_all = []
        ELBO_all = []

        random.seed(101)
        np.random.seed(101)
        ELBO = []
        NLPD = []
        myTimes = []

        Y = Ytrain.copy()
        X = Xtrain.copy()

        J = likelihood.num_output_functions(Y_metadata)  # This function indicates how many J latent functions we need
        np.random.seed(myseed)
        if not convolved:
            if ('motor' in dataset):
                ls_q = 0.1 * np.sqrt(Dim) * (np.random.rand(Q) + 0.001)  # 0.05 * np.ones(Q) era 0.1 ambos
            elif ('yacht' in dataset or 'boston' in dataset):
                ls_q = 1 * np.sqrt(Dim) * (np.random.rand(Q) + 0.001)  # 0.05 * np.ones(Q)
            elif ('mocap' in dataset):
                ls_q = 1 * np.sqrt(Dim) * (np.random.rand(Q) + 0.001)  # 0.05 * np.ones(Q)
            elif ('toy' in dataset):
                ls_q = 1 * np.sqrt(Dim) * (np.random.rand(Q) + 0.001)  # 10 * np.ones(Q)   era 0.01
            elif (dataset == 'naval_beta_gamma'):
                ls_q = np.sqrt(Dim) * (np.random.rand(Q) + 0.001)
            elif (dataset == 'london' or dataset == 'human'):
                ls_q = 0.1 * np.sqrt(Dim) * (np.random.rand(Q) + 0.001)
            elif (dataset == 'mnist'):
                ls_q = 0.01 * np.sqrt(Dim) * (np.random.rand(Q) + 0.001)
            else:
                ls_q = 0.1 * np.sqrt(Dim) * (np.random.rand(Q) + 0.001)  # 10 * np.ones(Q)   era 0.01
            if ('mocap8' in dataset):
                ls_q = 1 * np.sqrt(Dim) * (np.random.rand(Q) + 0.001)  # 0.05 * np.ones(Q)

            print("Initial lengthscales uq:", ls_q)
        else:
            if ('mocap7' in dataset):
                ls_q = 0.1 * np.sqrt(Dim) * (np.random.rand(Q) + 0.001)  # 0.05 * np.ones(Q)
                lenghtscale = 0.1 * np.sqrt(Dim)*(np.random.rand(J) * np.random.rand(J))  # 0.1 * np.ones(J) #better with 0.01
            elif ('mocap' in dataset):
                ls_q = 0.1 * np.sqrt(Dim) * (np.random.rand(Q) + 0.001)  # 0.05 * np.ones(Q)
                lenghtscale = 0.01 * np.sqrt(Dim)*(np.random.rand(J) * np.random.rand(J))
            elif ('toy1c' in dataset):
                ls_q = 0.1 * np.sqrt(Dim) * (np.random.rand(Q) + 0.001)  # 10 * np.ones(Q)   era 0.01
                lenghtscale = 1 * np.sqrt(Dim) * (np.random.rand(J) * np.random.rand(J))
            elif (dataset == 'naval_beta_gamma'):
                ls_q = 1*np.sqrt(Dim) * (np.random.rand(Q) + 0.001)
                lenghtscale = 1*np.sqrt(Dim) * (np.random.rand(J) * np.random.rand(J))
            elif (dataset == 'london' or dataset == 'human'):
                ls_q = 0.01 * np.sqrt(Dim) * (np.random.rand(Q) + 0.001)
                lenghtscale = 0.1 * np.sqrt(Dim) * (np.random.rand(J) * np.random.rand(J))
            elif ('sarcos' in dataset):
                ls_q = 0.01 * np.sqrt(Dim) * (np.random.rand(Q) + 0.001)  #
                lenghtscale = 0.01 * np.sqrt(Dim) * (np.random.rand(J) * np.random.rand(J))  #
            elif ('traffic' in dataset):
                ls_q = 0.1 * np.sqrt(Dim) * (np.random.rand(Q) + 0.001)  #
                lenghtscale = 0.1 * np.sqrt(Dim) * (np.random.rand(J) * np.random.rand(J))
            else:
                ls_q = 0.1 * np.sqrt(Dim) * (np.random.rand(Q) + 0.001)  #
                lenghtscale = 0.1 * np.sqrt(Dim) * (np.random.rand(J) * np.random.rand(J))
            if ('mocap8' in dataset):
                ls_q = 1 * np.sqrt(Dim) * (np.random.rand(Q) + 0.001)  # 0.05 * np.ones(Q)
                lenghtscale = 0.1 * np.sqrt(Dim) * (np.random.rand(J) * np.random.rand(J))
            print("Initial lengthscales uq:", ls_q)
            print("Initial lengthscales smoothing G(x):", lenghtscale)


        chose_ARD = True
        var_q = 1.0 * np.ones(Q)  # use 0.1 for toy 0.1 0.5
        kern_list = util.latent_functions_prior(Q, lenghtscale=ls_q, variance=var_q, input_dim=Dim, ARD=chose_ARD,inv_l=False)

        # MODEL
        if convolved:
            # We create the smoothing kernel for each latent function that parametrise the heterogeneous likelihoods
            kern_list_Gx = []
            variance = np.ones(J)  #
            for j in range(J):
                kern_j = GPy.kern.RBF(input_dim=Dim, lengthscale=lenghtscale[j], variance=variance[j], ARD=chose_ARD,
                                      inv_l=False)
                kern_j.name = 'kern_G' + str(j)
                kern_list_Gx.append(kern_j)

            model = ConvHetMOGP(X=X, Y=Y, Z=Z.copy(), kern_list=kern_list,kern_list_Gx=kern_list_Gx, likelihood=likelihood, Y_metadata=Y_metadata,batch_size=batch)
        else:
            model = SVMOGP(X=X, Y=Y, Z=Z.copy(), kern_list=kern_list, likelihood=likelihood, Y_metadata=Y_metadata,batch_size=batch)

        #model.Z.fix()
        model['.*.kappa'].fix()
        """"""""""""""""""""""""""""""""""""""""""""

        """"""""""""""""""""""""""""""""""""""""""""

        for q in range(Q):
            model['B_q' + str(q) + '.W'] = 1 * np.random.randn(model['B_q0.W'].__len__())[:, None]
            model.kern_list[q].variance.fix()
        """"""""""""""""""""""""""""""""""""""""""""""""""""""
        #print(model['B'])
        print('Initial Log Likelihood:\n',model.log_likelihood())

        if method == 'adam':
            opt = climin.Adam(model.optimizer_array, model.stochastic_grad, step_rate=0.005, decay_mom1=1 - 0.9,decay_mom2=1 - 0.999)
            ELBO.append(model.log_likelihood())
            #NLPD.append(model.negative_log_predictive(Xtest, Ytest, num_samples=1000))
            start = time.time()
            myTimes.append(start)
            print('Running Adam...')
            info = opt.minimize_until(callback)

        elif method == 'sgd':
            opt = climin.GradientDescent(model.optimizer_array, model.stochastic_grad, step_rate=1e-15,
                                         momentum=0.0)
            ELBO.append(model.log_likelihood())
            #NLPD.append(model.negative_log_predictive(Xtest, Ytest, num_samples=1000))
            start = time.time()
            myTimes.append(start)
            print('Running SGD...')
            info = opt.minimize_until(callback)
        elif method == 'adad':
            opt = climin.Adadelta(model.optimizer_array, model.stochastic_grad, step_rate=0.005, momentum=0.9)
            ELBO.append(model.log_likelihood())
            #NLPD.append(model.negative_log_predictive(Xtest, Ytest, num_samples=1000))
            start = time.time()
            myTimes.append(start)
            print('Running Adadelta...')
            info = opt.minimize_until(callback)
        elif method == 'fng':
            model.Gauss_Newton = False
            ELBO, NLPD, myTimes = vo.optimise_HetMOGP(model, Xval=None, Yval=None,
                                                        max_iters=n_iter, step_rate=0.005, decay_mom1=1 - 0.9,
                                                        decay_mom2=1 - 0.999, fng=True, q_s_ini=q_s_ini,
                                                        prior_lamb_or_offset=prior_lamb)

        elif 'hyb' in method:
            model.Gauss_Newton = False
            ELBO, NLPD, myTimes = vo.optimise_HetMOGP(model, Xval=None, Yval=None,
                                                      max_iters=n_iter, step_rate=0.005, decay_mom1=1 - 0.9,
                                                      decay_mom2=1 - 0.999, fng=False)

        if NLPD.__len__() == 0:
            NLPD = []
            NLPD.append(model.negative_log_predictive(Xtest, Ytest))

        Times_all.append(np.array(myTimes).flatten())
        ELBO_all.append(np.array(ELBO).flatten())
        all_NLPD.append(np.clip(NLPD, -1.0e100, 1.0e100))
        print('\nNLPD over test set: ',all_NLPD[0][-1],'\n')

        color = which_color[method]

        final_path = path_to_save + dataset + '/mom' + str(mom) + "/MC" + str(config.MC) + "/Z" + str(
            config.inducing) + "/batch" + str(config.minibatch) + "/D" + str(Dim) + "/" + str(myseed)
        if not os.path.exists(final_path):
            os.makedirs(final_path)

        # font = {'family': 'serif',
        #         'weight': 'bold',
        #         'size': 10}
        #
        # plt.rc('font', **font)
        linewidth = 1.5

        d=0
        if(model.Xmulti_all[0].shape[1]==1):
            plt.rc('text', usetex=True)
            # plt.plot(Xtest[0],Ytest[0],'.',color='red')
            Xtest2 = []
            for d in range(likelihoods_list.__len__()):
                xneg = Xtrain[d].min()
                xpos = Xtrain[d].max()
                Xtest2.append(np.linspace(xneg, xpos, 1000)[:, None])

            mpred_c, vpred_c = model.predict(Xtest2)
            fonti = 13
            # ylim = [-10.0,10.0]

            for d in range(likelihoods_list.__len__()):
                plt.figure(d)
                plt.plot(Xtest[d], Ytest[d], '.', color='red')

                plt.plot(X[d], Y[d], 'x', color='black')
                plt.plot(Xtest2[d], mpred_c[d], '-', color='blue', linewidth=2)
                plt.plot(Xtest2[d], mpred_c[d] - 2.0 * np.sqrt(vpred_c[d]), '--', color='blue')
                plt.plot(Xtest2[d], mpred_c[d] + 2.0 * np.sqrt(vpred_c[d]), '--', color='blue')
                plt.title(r'\bf{Model Performance}', fontsize=fonti, fontweight="bold")
                # plt.ylim([Y[d].min(),Y[d].max()])
                plt.ylim([-20, 20])

                plt.xlim([Xtrain[d].min(), Xtrain[d].max()])
                plt.xlabel(r'\bf{Input}', fontsize=fonti, fontweight="bold")
                plt.ylabel(r'\bf{Output}', fontsize=fonti, fontweight="bold")
                plt.legend((r'\bf{Test}', r'\bf{Train}', r'\bf{Prediction}'), loc=(0.65, 0.1), handlelength=1.5,
                           fontsize=12)

        plt.figure(d+1)
        plt.semilogy(-ELBO_all[0], color, linewidth=linewidth, label=method)
        plt.title("Q=" + str(Q) + " P=" + str(Dim) + " M=" + str(config.inducing) + " MB=" + str(config.minibatch),
            fontweight="bold")
        plt.xlabel("Iteration", fontweight="bold")
        plt.ylabel("Neg ELBO", fontweight="bold")
        # if (ELBO_all[0].max() > 0):
        #     plt.yscale('symlog')
        # else:
        #     plt.yscale('log')
        plt.grid(True, which='both')
        plt.gca().legend(methods, loc='upper center', ncol=4, bbox_to_anchor=(0.5, -0.13))
        plt.tight_layout()

        each = 50   #this is a variable to remember how frequently the NLPD is computed in the optimisation process

"""""""""""""""""""""""""""""""""""""""""""""""""""

"""""""""""""""""""""""""""""""""""""""""""""""""""
