# Physics-Informed Neural Network (PINN) for 1D Viscous Burgers' Equation

## Overview
This project implements a PINN to solve the 1D Viscous Burgers' Equation, a fundamental partial differential equation
occurring in various areas of applied mathematics, such as fluid mechanics, nonlinear acoustics, and gas dynamics.
This project contransts the deep learning approach against a Finite Difference method (implementing an upwind scheme)
and includes extensive diagnostic and statistical analysis to study network convergence, error distribution, and time variance.

### The Physics
The model solves the following partial differential equation:
* $$u_t + u u_x = \nu u_{xx}$$

With the following domain and boundary conditions:
* $x \in (-1, 1)$
* $t \in (0, 1)$
* $u(x, 0) = -\sin(\pi x)$
* $u(-1, t) = u(1, t) = 0$

And with viscosity $\nu = 0.01$

## Key Features
* **PyTorch PINN Implementation:** Utilizes a unified $(x, t)$ state tensor and `torch.autograd` to compute exact first and second-order derivatives for the PDE residual.
* **Classical Baseline:** Includes a finite difference solver for direct accuracy and runtime comparisons.
* **Ensemble Training Pipeline:** Automates multiple training runs to evaluate the variance of random weight initializations.
* **Advanced Diagnostics:**
* Spatiotemporal heatmaps (using `seismic` and `magma` colormaps) for function $u$ and absolute error.
* Statistical curve fitting (`scipy.stats`) modeling ensemble Execution Times and Mean Squared Errors (Gamma/Log-Normal).
* Pearson correlation analysis between runtime and model accuracy.

* ## Dependencies
This project is designed to run natively in a Jupyter/Google Colab environment.
* `numpy`
* `matplotlib`
* `time`
* `torch`
* `copy`
* `scipy`

## Usage
To run this project:
1. Open the `.ipynb` file in Google Colab or your local Jupyter environment.
2. Run all cells sequentially. The ensemble training block may take a few hours depending on the specified number of runs and epochs.
   (For the specified number of runs and epochs given, it took me ~4 hours to run the ensemble training)
