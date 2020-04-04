## Adam
Adaptive Moment Estimation (Adam) is a method that computes adaptive learning rates for each parameter. It stores an exponentially decaying average of past squared gradients 
v and Adam also keeps an exponentially decaying average of past gradients m, similar to momentum.  We compute the decaying averages of past and past squared gradients m and v respectively as follows: 

> *m = β1 m + (1 - β1) g*

> *v = β2 v + (1 - β2) |g|^2*

m and v are estimates of the first moment (the mean) and the second moment (the uncentered variance) of the gradients respectively.
As m and v are initialized as vectors of 0's, the authors of Adam observe that they are biased towards zero, especially during the initial time steps.
They counteract these biases by computing bias-corrected first and second moment estimates:

> *m_hat = m / (1 - β1^t)*

> *v_hat = v / (1 - β2^t)*

Once estimators are calculated and corrected, the parameters are updated using the following formula:

> *theta = theta - alpha * m_hat / (√v_hat + ϵ)*

β1, β2 and ϵ are hyperparameters like alpha (the learning rate), ϵ  is a small scalar used to prevent division by 0 and β1 and β2 control exponential decay.
The authors propose default values of 0.9 for β1, 0.999 for β2, and 10^(−8) for ϵ
Since Adam is derived from SGD, these operations are perfomed for each training sample.

## Amsgrad

In some cases, e.g. for object recognition or machine translation ADAM fails to converge to an optimal solution and is outperformed by SGD with momentum.

Reddi et al. (2018) formalize this issue and pinpoint the exponential moving average of past squared gradients (v) as a reason for the poor generalization behaviour of adaptive learning rate methods. Recall that the introduction of the exponential average was well-motivated: it should prevent the learning rates to become infinitesimally small as training progresses
. However, this short-term memory of the gradients becomes an obstacle in other scenarios. 

ADAGRAD:
- It uses all the past gradients in the update(“long-term memory” of past gradients)
- Rapid decay of the learning rate 
- Non-increasing step size

ADAM:
- The decreasing of the learning rate is slower (v prevent the learning rates to become infinitesimally small as training progresses)
- Limiting the reliance of the update to only the past few gradients (“short-term memory” of past gradients).
	- Limiting the reliance of the update on essentially only the past few gradients can cause significant convergence issues
	- The step size can potentially be indefinite (can increse and decrese)
		- this violation of positive definiteness can lead to undesirable convergence behavior for ADAM

In settings where Adam converges to a suboptimal solution, it has been observed that some minibatches provide large and informative gradients, but as these minibatches only occur rarely, exponential averaging diminishes their influence, which leads to poor convergence.

In order to have guaranteed convergence the optimization algorithm must have “long-term memory” of past gradients.
To resolve this issue, Reddi et al. propose new variants of ADAM which rely on long-term memory of past gradients, but can be implemented in the same time and space requirements as the original ADAM algorithm

To fix this behaviour, the authors propose a new algorithm, AMSGrad that uses the maximum of past squared gradients v rather than the exponential average to update the parameters. 
v is defined the same as in Adam above:

> *v = β2 v + (1 - β2) |g|^2*

Instead of using v (or its bias-corrected version v_hat) directly, we now employ the previous v(t-1) if it is larger than the current one:

> *v_hat = max(v_hat, v)*

This way, AMSGrad results in a non-increasing step size, which avoids the problems suffered by Adam. For simplicity, the authors also remove the debiasing step that we have seen in Adam.

> *m = β1 m + (1 - β1) g*

> *v = β2 v + (1 - β2) |g|^2*

> *v_hat = max(v_hat, v)*

> *theta = theta - alpha * m / (√v_hat + ϵ)*

The key difference of AMSGRAD with ADAM is that it maintains the maximum of all v until the present time step and uses this maximum value for normalizing the running average of the gradient instead of v in ADAM. By doing this, AMSGRAD results in a non-increasing step size and avoids the pitfalls of ADAM. 
In general, any algorithm that relies on an essentially fixed sized window of past gradients to scale the gradient updates will suffer from this problem.

The authors observe improved performance compared to Adam on small datasets and on CIFAR-10






