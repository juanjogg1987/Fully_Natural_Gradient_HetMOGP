# Fully Natural Gradient Scheme for Improving Inference of the HetMOGP model

This repository contains an implementation of a Fully Natural Gradient (FullyNG) Scheme for improving optimisation of the Heterogeneous Multi-output Gaussian Process (HetMOGP) model based on a Linear Model of Coregionalisation (LMC) (visit https://github.com/pmorenoz/HetMOGP for more details about the model). The FullyNG optimises the HetMOGP model using Natural Gradient optimisation for the posterior parameters and also Natural Gradients for the Hyper-Parameters (inducing points, kernel hyper-parameters and output's corregionalisation parameters).

This repository also contains in the folder 'convhetmogp2' an extension of the HetMOGP model that relies on a Convolution Processes Model (CPM), for details on this type of model check the work in https://arxiv.org/pdf/1911.10225.pdf.

The repository consists of: 

- **fully_natural_gradient.py**: This module contains the functions **fullyng_opt_HetMOGP** and **hybrid_opt_HetMOGP**

- **load_datasets.py**: This module contains different toy examples with different number of Heterogeneous Outputs and loads the real dataset like London, Naval, Sarcos and MOCAP. 

- **Example_Different_Dataset.py**: This python file contains a code that allows to run the different models using the different toy and real datasets, also allows to configure different settings for the optimisation process.

- **FunctionExample_Variational_Optimisation.py**: This file is an example of how variational optimisation together with natural gradients allows to induce exploration for improving inference of an objective function.

![Variational_Optimisation](tmp/toy4.png) 

The following modules are a Forked version of [Pablo Moreno](https://github.com/pmorenoz/HetMOGP)'s implementation. We recommend the user to work using this package version:
- **hetmogp**: This block contains all model definitions, inference, and important utilities. 
- **likelihoods**: General library of probability distributions for the heterogeneous likelihood construction.

## Usage

* The HetMOGP with LMC is created as:
```
model = SVMOGP(X=X, Y=Y, Z=Z, kern_list=kern_list, likelihood=likelihood, Y_metadata=Y_metadata, batch_size=batch)
```

* The HetMOGP with CPM is created as:
```
model = ConvHetMOGP(X=X, Y=Y, Z=Z, kern_list=kern_list,kern_list_Gx=kern_list_Gx, likelihood=likelihood, Y_metadata=Y_metadata,batch_size=batch)
```
where the kern_list_Gx is a list of smoothing kernels associated to each latent parameter function of the HetMOGP model (see https://arxiv.org/pdf/1911.10225.pdf for details).

Then we can bypass the model to the FullyNG optimiser. Apart from the model we should set the maximun number of iterations, the step-size parameter, momentum in the range \[0.0-1.0\], a lambda prior (usually in the range \[0.0,1.0\] for inducing exploration in the inference process). We suggest using by default the values of momentum = 0.9 and prior_lambda = 1.0e-10

* Using FullyNG optimiser: 
```
optimise_HetMOGP(model, max_iters=n_iter, step_rate=0.005, decay_mom1=1 - 0.9, decay_mom2=1 - 0.999, fng=True, q_s_ini=1000, prior_lamb_or_offset=1.0e-10)
```
An alternative inference scheme is the Hybrid (NG + Adam) which optimises the HetMOGP model using Natural Gradient optimisation for the posterior parameters, but Adam for the Hyper-Parameters. We simply have to set fng = False is the line above, we bypass the model to the hybrid optimiser, set the maximum number of iterations, the step-size parameter, decay_mom1 and decay_mom2 are parameters directly bypass to the Adam optimiser together with the step-size parameter, it is recommended to use the default values of decay_mom1=1-0.9 and decay_mom2=1-0.999.

* Using Hybrid (NG+Adam) optimiser:
```
optimise_HetMOGP(model, max_iters=n_iter, step_rate=0.005, decay_mom1=1 - 0.9, decay_mom2=1 - 0.999, fng=False)
```

A complete example of our model usage can be found in this repository at **notebooks > FullyNG_on_ToyData** where different toy examples can be tested.

## Examples
* **Heterogeneous Model using Five Outputs:** The figure shows the Performance for Objective Convergence Using Stochastic Gradient Descent (SGD), Adam (Adaptive Momentum Method), HYB (NG+Adam) and
Our FNG (Fully Natural Gradient) method.
![toy2](tmp/toy4.png)

## Contributors

[Juan-José Giraldo](https://github.com/juanjogg1987) and [Mauricio A. Álvarez](https://sites.google.com/site/maalvarezl/)

For further information or contact:
```
jjgiraldogutierrez1@sheffield.ac.uk or juanjogg1987@gmail.com
