
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

w1 = 15
w2 = 0.3
amp = 2
np.random.seed(15)

def g(theta):
    #gtheta = -np.sinc(theta)
    gtheta = amp*np.exp(-(w2*theta)**2)*np.sin(w2*theta*w1)
    return gtheta

def dg_dtheta(theta_in):
    theta = np.array([theta_in]).flatten()
    #dg_dth = -(np.cos(theta)-np.sinc(theta))/(theta)
    #dg_dth = -(np.cos(theta)/theta - np.sin(theta)/(theta**2))
    dg_dth = amp*(np.exp(-(w2*theta)**2)*(-2*(w2**2)*theta)*np.sin(w2*theta*w1)+np.exp(-(w2*theta)**2)*np.cos(w2*theta*w1)*w2*w1)
    #dg_dth[np.where(theta==0)]=0.0
    return dg_dth

def d2g_dtheta2(theta_in):
    theta = np.array([theta_in]).flatten()
    #d2g_dth2 = 1.0/3.0*np.ones_like(theta).flatten()
    #d2g_dth2[np.where(theta>1e-20)] = -((2-theta[np.where(theta>1e-20)]**2)*np.sin(theta[np.where(theta>1e-20)])-2*theta[np.where(theta>1e-20)]*np.cos(theta[np.where(theta>1e-20)]))/(theta[np.where(theta>1e-20)]**3)
    #d2g_dth2 = -((2 - theta ** 2) * np.sin(theta) - 2 * theta * np.cos(theta)) / (theta ** 3)
    #d2g_dth2[np.where(theta==0.0)]=1.0/3.0
    d2g_dth2 = amp*(np.exp(-(w2*theta)**2)*((-2*(w2**2)*theta)**2)*np.sin(w2*theta*w1)+np.exp(-(w2*theta)**2)*(-2*w2**2*np.sin(w2*theta*w1)+w2*w1*(-2*w2**2*theta)*np.cos(w2*theta*w1))+np.exp(-(w2*theta)**2)*(-2*(w2**2)*theta)*np.cos(w2*theta*w1)*w2*w1+np.exp(-(w2*theta)**2)*(-np.sin(w2*theta*w1))*(w2*w1)**2)
    return d2g_dth2

def Eq_g(mean,sig,MC = 1):
    eps = np.random.randn(MC)
    theta_mc = mean + sig*eps
    Eq_gtheta = (1/MC)*np.sum(g(theta_mc))
    return Eq_gtheta

def div_KL(mean1,sig1,mean2,sig2):
    KL = -0.5*(np.log((sig1**2)/(sig2**2))-(sig1**2)/(sig2**2)-((mean1-mean2)**2)/(sig2**2)+1)
    return KL

def L_bound(q_mean,q_sig,prior_lamb,MC=1,KL_compute=True):
    KL_div = 0.0
    if KL_compute:
        KL_div = div_KL(q_mean,q_sig,0,np.sqrt(1.0/prior_lamb))
        #print(KL_div)
    L=Eq_g(q_mean,q_sig,MC=MC)+KL_div
    return L

def dEq_dsig2_MC(mean,sig,MC=1,Gauss_Newton=True):
    eps = np.random.randn(MC)
    theta_mc = mean + sig * eps
    if Gauss_Newton:
        dEq_dsig2 = (0.5 / MC) * np.sum(dg_dtheta(theta_mc)**2) #1/2E[dg_dtheta**2]
    else:
        dEq_dsig2 = (0.5 / MC) * np.sum(d2g_dtheta2(theta_mc))
    return dEq_dsig2

def dEq_dmean_MC(mean,sig,MC=1):
    eps = np.random.randn(MC)
    theta_mc = mean + sig * eps
    dEq_dsig2 = (1 / MC) * np.sum(dg_dtheta(theta_mc))
    return dEq_dsig2

"Bellow you can chose the possible methods: VO, VOKL and Newton"
"VO stands for Variational Optimisation"
"VOKL stands for Variational Optimisation with KL penalisation"
"Newton stands for the Newton method"

method = 'VOKL'
max_iter = 80# 80 #40
MC=50
q_mean = -3.0#-3.0
q_sig = 3.0#3.5
q_sig2_inv = 1/(q_sig**2)

if method =='Newton':
    step_alpha = 0.05  # 0.02 #for mean update
    q_sig = 0.005 #Newton method doesn't have variance for exploration, we put a small one to be consistent in the figures
    Npoints_for_distribution = 1000  #Here we use many points because the distribution tends to a delta
    use_KL_in_plot = False
    prior_lamb = 0
elif method=='VO':
    step_alpha = 1 #for mean update
    step_beta = 0.1  #for sig2 update
    Npoints_for_distribution = 200
    use_KL_in_plot = False
    prior_lamb = 0
elif method=='VOKL':
    step_alpha = 0.1 #for mean update
    step_beta = 0.01  #for sig2 update
    Npoints_for_distribution = 200
    use_KL_in_plot = False
    prior_lamb = 1.5  # 1  #this is a pecision prior N(0,lamb^-1)

"Plot of the L(q_mean,q_sig) function"
Ngrid = 80
xneg = -4.2
xpos = 1.2
q_mean_grid = np.linspace(xneg,xpos,Ngrid)
q_sig_grid = np.linspace(0.000001,2.0,Ngrid)


XX, XY = np.meshgrid(q_mean_grid, q_sig_grid)
X_flat = XX.flatten()[:,None]
Y_flat = XY.flatten()[:,None]
L_grid = []
print('rendering images... wait few seconds!')
for i in range(XX.flatten().shape[0]):
    L_grid.append(L_bound(X_flat[i], Y_flat[i], prior_lamb, MC=20000, KL_compute=use_KL_in_plot))

L_grid = np.array(L_grid)
""""""""""""""""""""""""""""""""
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

#im = np.zeros((20, 20))
#im[5:-5, 5:-5] = 1
#im = ndimage.distance_transform_bf(L_grid.reshape(Ngrid,Ngrid))
#im_noise = im + 0.2*np.random.randn(*im.shape)

#L_grid = ndimage.median_filter(L_grid.reshape(Ngrid,Ngrid), 5).reshape(-1)
""""""""""""""""""""""""""""""""

fig, (ax1,ax2,ax3) = plt.subplots(nrows=3,ncols=1)

# Plot the contours.
n_levels = 20
#ax2.contour(X_flat.reshape(Ngrid,Ngrid), Y_flat.reshape(Ngrid,Ngrid), L_grid.reshape(Ngrid,Ngrid), levels=n_levels, linewidths=0.1, colors='k')
cntr1 = ax2.contourf(X_flat.reshape(Ngrid,Ngrid), Y_flat.reshape(Ngrid,Ngrid), L_grid.reshape(Ngrid,Ngrid), levels=n_levels)#, cmap="RdBu_r")
ax2.set_ylabel(r'\bf{$\sigma$}')
if use_KL_in_plot or method!='VOKL':
    ax2.text(0.2, 1.5, r'${\tilde{\mathcal{L}}(\mu,\sigma)}$', fontsize=14)
else:
    ax2.text(0.2, 1.5, r'$\mathbf{E}_{q\mathbf{(\theta)}}[g\mathbf{(\theta)}]$', fontsize=14)

#levels = np.linspace(L_grid.min(),L_grid.max(),25)
#plt.contourf(X_flat.reshape(Ngrid,Ngrid), Y_flat.reshape(Ngrid,Ngrid), L_grid.reshape(Ngrid,Ngrid), levels=levels)
#fig.colorbar(cntr1, ax=ax2)
""""""""""""""""""""""""""""""""""""""
font = {'family': 'serif',
                'weight': 'bold',
                'size': 10}

plt.rc('font', **font)
plt.rc('text', usetex=True)  #This is to use latex text
#from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
#rc('text', usetex=True)

theta_plot = np.linspace(xneg,xpos,500)
#plt.subplot(311)
ax1.plot(theta_plot,g(theta_plot),linewidth=1.5)
ax1.set_xlim([xneg,xpos])
ax1.set_ylabel(r'$\bf{Function}$ $g(\bf{\theta})$')

L_t = []
#prior_lamb = 1

#L_t.append(L_bound(q_mean,q_sig,prior_lamb))

import scipy.stats as stats

color_intensities = 10
color_gray = np.linspace(0.8,0.0,color_intensities)
for niter in range(max_iter-1):
    color_circle = 'black'
    plt.pause(0.00001)
    #plt.subplot(311)
    ax1.plot(q_mean, g(q_mean), 'o', color=color_circle)
    ax1.plot(q_mean, g(q_mean), 'o', color='r',markersize=0.8)
    #ax1.axvline(x=q_mean, color='r', linestyle='--')
    ax1.plot(q_mean*np.ones(5), np.linspace(-2.0,2,5), '--', color='r',linewidth=0.4)
    ax1.set_ylim([-2.2,2.1])
    plt.pause(0.00001)
    if (q_sig<=2.0):
        ax2.plot(q_mean, q_sig, 'o', color=color_circle)
        ax2.plot(q_mean, q_sig, 'o', color='r', markersize=0.8)
        #if niter%2 == 0:
    ax2.axvline(x=q_mean, color='r', linestyle='--',linewidth=0.4)

    x = np.linspace(xneg, xpos, Npoints_for_distribution)
    plt.pause(0.00001)
    #plt.subplot(313)
    if niter<color_intensities-1: nindex = niter;
    else: nindex=color_intensities-1;

    if niter%2 == 0:
        ax3.plot(x, stats.norm.pdf(x, q_mean, q_sig), color=[color_gray[nindex]]*3)
        ax3.set_xlim([xneg, xpos])
        ax3.axvline(x=q_mean, color='r', linestyle='--',linewidth=0.4)
        ax3.set_ylabel(r'\bf{Distribution q($\theta$)}')
        ax3.set_xlabel(r'\bf{$\theta$}')
        #ax3.plot(q_mean * np.ones(5), np.linspace(0, stats.norm.pdf(q_mean, q_mean, q_sig), 5), '--', color='r')

    if method=='VO':
        dEq_dsig2 = dEq_dsig2_MC(q_mean, q_sig, MC=MC,Gauss_Newton=True)
        dEq_dmean = dEq_dmean_MC(q_mean, q_sig, MC=MC)
        q_sig2_inv = q_sig2_inv + step_beta * 2 * (dEq_dsig2)
        q_mean = q_mean - step_alpha * (1.0 / q_sig2_inv) * (dEq_dmean)
        q_sig = np.sqrt(1.0 / q_sig2_inv)
        #print(q_sig)
    elif method=='VOKL':
        dEq_dsig2 = dEq_dsig2_MC(q_mean, q_sig, MC=MC,Gauss_Newton=True)
        dEq_dmean = dEq_dmean_MC(q_mean, q_sig, MC=MC)
        q_sig2_inv = q_sig2_inv + step_beta * 2 * (dEq_dsig2 + 0.5 * prior_lamb - 0.5 * q_sig2_inv)
        q_mean = q_mean - step_alpha*(1.0/q_sig2_inv)*(dEq_dmean+q_mean*prior_lamb)
        q_sig = np.sqrt(1.0 / q_sig2_inv)
    elif method=='Newton':
        P_inv = 1.0/(d2g_dtheta2(q_mean))
        q_mean = q_mean - step_alpha * (P_inv) * (dg_dtheta(q_mean))
        #q_mean = q_mean - step_alpha * (dg_dtheta(q_mean)**-1)*g(q_mean)
        #print(q_mean)



