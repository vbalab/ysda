<!-- markdownlint-disable MD024 MD025 -->

# > [2025-DGM-MIPT-YSDA-course](https://github.com/r-isachenko/2025-DGM-MIPT-YSDA-course)

# **Lecture 1 - Autoregressive Models**

## Generative Models

![alt text](notes_images/map.png)

## CoV (Change of Variable) for Probabilities

If $ z = f(x) $, then

$$
p_X(x)
= p_Z(f(x)) \, \left| \det \frac{\partial f(x)}{\partial x} \right|
= p_Z(z) \, \big| \det J_f(x) \big|
$$

### Proof

For any Borel set $ A \subset \mathbb{R}^d $,

$$
\mathbb{P}(X \in A) = \mathbb{P}(Z \in f(A)).
$$

Using densities,

$$
\int_A p_X(x) \, dx
= \int_{f(A)} p_Z(z) \, dz.
$$

Apply the **multivariate change-of-variables** formula for integrals with
$z = f(x)$:

$$
\int_{f(A)} p_Z(z) \, dz
= \int_A p_Z(f(x)) \, \big| \det J_f(x) \big| \, dx.
$$

Since this holds for every Borel $A$, the integrands are equal almost everywhere:

$$
p_X(x)
= p_Z(f(x)) \, \big| \det J_f(x) \big|.
$$

## LOTUS (Law of Unconscious Statistician)

If $ \mathbf{Y} = f(x) $, then

$$
\mathbb{E}_{Y \sim p_Y}[g(Y)]
= \int_{\mathbb{R}^d} g(\mathbf{y}) \, p_Y(\mathbf{y}) \, d\mathbf{y}
= \int_{\mathbb{R}^d} g(f(x)) \, p_X(x) \, dx
= \mathbb{E}_{X \sim p_X}[g(f(X))]
$$

### Proof

2 CoVs: 1 multivariate & 1 for probabilites.

$$
\int_{\mathbb{R}^d} g(\mathbf{y}) \, p_Y(\mathbf{y}) \, d\mathbf{y} =
$$
$$
\int_{\mathbb{R}^d} g(\mathbf{y}) \, p_X(x) \big|\det J_f^{-1}(x) \big| \big|\det J_f(x) \big| \, dx =
$$
$$
\int_{\mathbb{R}^d} g(f(x)) \, p_X(x) \, dx
$$

## Dirac Delta Function

$$
\delta(x - a) =
\begin{cases}
0, & x \neq a \\
\infty, & x = a
\end{cases},
\quad
\int \delta(x - a) f(x) \, dx = f(a)
$$

Used to represent deterministic mappings:
$ p(x \mid z) = \delta(x - f_\theta(z)) $.

## Forward KL as MLE

$$
\mathrm{KL}(p_{\text{data}} \,\|\, p_\theta) =
$$

$$
\int p_{\text{data}}(x) \log \frac{p_{\text{data}}(x)}{p_\theta(x)} \, dx =
$$

$$
\int p_{\text{data}}(x) \log p_{\text{data}}(x) \, dx
- \int p_{\text{data}}(x) \log p_\theta(x) \, dx =
$$

$$
-\mathbb{E}_{p_{\text{data}}}[\log p_\theta(x)] + \text{const} \approx
$$

$$
-\frac{1}{n} \sum_{i=1}^n \log p_\theta(x_i) + \text{const} =
$$

$$
-\frac{1}{n} \log{L} + const \;\to\; \min_\theta
$$

MLE $\equiv$ minimizing forward $\mathrm{KL}$.

### Reverse KL

$$
\mathrm{KL}(p_\theta \,\|\, p_{\text{data}})
=
$$

$$
\int p_\theta(x) \log \frac{p_\theta(x)}{p_{\text{data}}(x)} \, dx =
$$

$$
\mathbb{E}_{p_\theta}[\log p_\theta(x) - \log p_{\text{data}}(x)] \;\to\; \min_\theta
$$

$\to$ can't simplify.

## Autoregressive Models

$$
p_\theta(x) = \prod_{j=1}^m p_\theta(x_j \mid x_{1:j-1}),
\quad
\log p_\theta(x) = \sum_{j=1}^m \log p_\theta(x_j \mid x_{1:j-1})
$$

### MLE for Autoregressive Models

$$
\theta^* = \arg\max_\theta
\sum_{i=1}^n \sum_{j=1}^m
\log p_\theta(x_{ij} \mid x_{i,1:j-1})
$$

Equivalent to next-step prediction.

### Sampling

$$
\hat{x}_1 \sim p_\theta(x_1), \quad
\hat{x}_2 \sim p_\theta(x_2 \mid \hat{x}_1), \quad
\dots, \quad
\hat{x}_m \sim p_\theta(x_m \mid \hat{x}_{1:m-1})
$$

The generated sample is $ \hat{x} = (\hat{x}_1, \hat{x}_2, \ldots, \hat{x}_m) $.

### PixelCNN

Autoregressive model for images.  
Each pixel conditioned on previous ones (in raster scan order).

### Limitations

- Loss of spatial structure  
- $ O(n^2) $ steps (for images) with $ O(n^2) $ attention per step $\to$ $ \sum_{i=1}^{n^2} i^2 = \frac{n^2 (n^2 + 1)(2n^2 + 1)}{6} = O(n^6) $
- No bidirectional structure

# **Lecture 2 - Normalizing Flows**

## Inverse Function Theorem (Jacobian Determinant)

If $f$ is **diffeomorphism** - invertible and its Jacobian is continuous and non-singular, then for $ z=f(x) $:

$$
J_{f^{-1}}(z) = \big(J_f(x)\big)^{-1},
\qquad
\left|\det J_{f^{-1}}(z)\right| = \frac{1}{\left|\det J_f(x)\right|}.
$$

In order $J$ to be invertable it needs to be squared $\to$ $\text{dim}(x) = \text{dim}(z)$.

![alt text](notes_images/jacobian.png)

## Normalizing Flows

### Fitting Normalizing Flows

**Normalizing Flow** — differentiable, invertible mapping that transforms data $ x $ to latent noise $ z $.

By **CoV**:

$$
\log p_X(x) =
$$

$$
\log \left(
    p_Z(z) \left| \det\!\left(\frac{\partial z}{\partial x}\right) \right| \right) =
$$

$$
\log \left(
    p_Z\!\left(f_{\theta}(x)\right) \left| \det\!\left(\frac{\partial f_{\theta}(x)}{\partial x}\right) \right|
 \right) =
$$

$$
\log p_Z\!\left(f_{\theta}(x)\right)
+ \log \left|\det(J_{f_\theta})\right|
\;\to\; \max_{\theta}
$$

![alt text](notes_images/normalizing_flows.png)

### Composition of Normalizing Flows

![alt text](notes_images/normalizing_flows_composition1.png)

**Theorem:**  
If every $ \{ f_k \}_{k=1}^K $ satisfies the conditions of the change-of-variables theorem,  
then the composition $ f(x) = f_K \circ \ldots \circ f_1(x) $ also satisfies them.

$$
p_{X}(x)
= p_{Z_k}(f(x)) \left| \det\!\left( \frac{\partial f(x)}{\partial x} \right) \right|
= p_{Z_k}(f(x)) \left| \det\!\left( \frac{\partial f_K}{\partial f_{K-1}} \cdots \frac{\partial f_1}{\partial x} \right) \right|
$$

![alt text](notes_images/normalizing_flows_composition2.png)

## NF Examples: 1. Linear NF

Simplest flow: linear transformation

$$
z = f(x) = Ax + \mathbf{b}, \quad A \in \mathbb{R}^{n\times n}
$$

Then

$$
\log p_{\theta}(x) = \log p(Ax + \mathbf{b}) + \log|\det A|
$$

Efficient if $ \det(A) $ is easy to compute.

### LU Decomposition

Decompose $ A = LU $,  
where $ L $ — lower-triangular with 1s on diagonal, $ U $ — upper-triangular.  
Then:

$$
\log|\det A| = \sum_i \log|U_{ii}|
$$

Efficient: $ O(n) $ determinant and easy inverse.

### QR Decomposition

Decompose $ A = QR $,  
$ Q $ orthogonal ($ \det Q = \pm1 $), $ R $ upper-triangular.  
Then:

$$
\log|\det A| = \log|\det R| = \sum_i \log|R_{ii}|
$$

Often used to stabilize flow training.

## NF Examples: 2. Gaussian Autoregressive NF

For each dimension $ i $:

$$
z_i = \frac{x_i - \mu_i(x_{1:i-1})}{\sigma_i(x_{1:i-1})}
\quad \to \quad
x_i = \mu_i(z_{1:i-1}) + \sigma_i(z_{1:i-1}) \, z_i
$$

This defines an invertible autoregressive transformation.

Jacobian is triangular $\to$
$\log|\det J_{f_\theta}| = -\sum_i \log \sigma_i(x_{1:i-1})$

Implemented in **MAF** (Masked Autoregressive Flow) and **IAF** (Inverse Autoregressive Flow).

## NF Examples: 3. Coupling Layer (RealNVP)

$$
x = [x_1, x_2] = [x_{1:d}, x_{d+1:m}],
$$

$$
z = [z_1, z_2] = [z_{1:d}, z_{d+1:m}]
$$

$$
\begin{cases}
x_1 = z_1 \\
x_2 = z_2 \odot \sigma_\theta(z_1) + \mu_\theta(z_1)
\end{cases}
\qquad
\begin{cases}
z_1 = x_1 \\
z_2 = (x_2 - \mu_\theta(x_1)) \odot \frac{1}{\sigma_\theta(x_1)}
\end{cases}
$$

Jacobian is block-triangular:

$$
\det\!\left( \frac{\partial z}{\partial x} \right)
= \det\!\begin{pmatrix}
I_d & 0_{d \times (m-d)} \\
\frac{\partial z_2}{\partial x_1} & \frac{\partial z_2}{\partial x_2}
\end{pmatrix}
= \prod_{j=1}^{m-d} \frac{1}{\sigma_{j,\theta}(x_1)}.
$$

Efficient, invertible, scalable — basis for **RealNVP** and **Glow** architectures.

# **Lecture 3 - Latent Variable Models**

> In NF latent variable $z$ was translated into $x$, but in LVM $z$ is translated into $p(x | z)$ (whole distribution).

## MLE Problem

$$
\theta^* = \arg\max_{\theta} p_{\theta}(x) = \arg\max_{\theta} \prod_{i=1}^{n} p_{\theta}(x_i) = \arg\max_{\theta} \sum_{i=1}^{n} \log p_{\theta}(x_i).
$$

The distribution $ p_{\theta}(x) $ can be highly **complex and often intractable** (just like the true data distribution $ p_{\text{data}}(x) $).

## Extended Probabilistic Model

Introduce a latent variable $ z $ for each observed sample $ x $:

$$
p_{\theta}(x) = \int p_{\theta}(x, z) dz = \int p_{\theta}(x|z) p(z) dz.
$$

Both $ p_{\theta}(x|z) $ and $ p(z) $ are usually much simpler than $ p_{\theta}(x) $.

![alt text](notes_images/mixture_of_gaussians.png)

### Naive Monte Carlo Estimation

$$
\log p_{\theta}(x) = \log \mathbb{E}_{p(z)}[p_{\theta}(x|z)]
\ge \mathbb{E}_{p(z)}[\log p_{\theta}(x|z)]
\approx \frac{1}{K} \sum_{k=1}^{K} \log p_{\theta}(x|z_k),
$$
where $ z_k \sim p(z) $.

#### **Problem**

As the dimensionality of $ z $ increases, the number of samples $K$ needed to adequately cover the latent space grows exponentially.

## ELBo (Evidence Lower Bound)

### Inequality Derivation

$$
\log p_{\theta}(x) =
\log \int p_{\theta}(x, z) dz =
\log \int \frac{q(z)}{q(z)} p_{\theta}(x, z) dz =
$$
$$
= \log \mathbb{E}_{q} \!\left[ \frac{p_{\theta}(x, z)}{q(z)} \right] \ge
\mathbb{E}_{q} \log \frac{p_{\theta}(x, z)}{q(z)} =
\mathcal{L}_{q,\theta}(x) \leftarrow  \text{ELBo}
$$

- Here, $ q(z) $ is any distribution such that $ \int q(z) dz = 1 $.
- We assume that $ \mathrm{supp}(q(z)) = \mathrm{supp}(p_{\theta}(z|x)) = \mathbb{R}^d $.

### Equality Derivation

$$
\mathcal{L}_{q,\theta}(x)
= \int q(z) \log \frac{p_{\theta}(x, z)}{q(z)} dz
= \int q(z) \log \frac{p_{\theta}(z|x) p_{\theta}(x)}{q(z)} dz =
$$

$$
= \log p_{\theta}(x) \cdot \int q(z) dz
+ \int q(z) \log \frac{p_{\theta}(z|x)}{q(z)} dz  =
$$
$$
= \log p_{\theta}(x) - \mathrm{KL}(q(z) \| p_{\theta}(z|x))
$$

### Log-Likelihood Decomposition

$$
\log p_{\theta}(x)
$$
$$
= \mathcal{L}_{q,\theta}(x)
+ \mathrm{KL}(q(z) \| p_{\theta}(z|x))
$$
$$
= \mathbb{E}_{q} \log p_{\theta}(x|z)
- \mathrm{KL}(q(z) \| p(z))
+ \mathrm{KL}(q(z) \| p_{\theta}(z|x)).
$$

- Instead of maximizing the likelihood, maximize the **ELBO**:

    $$
    \max_{\theta} p_{\theta}(x)
    \;\;\to\;\;
    \max_{q, \theta} \mathcal{L}_{q,\theta}(x)
    $$

- Maximizing the ELBO with respect to the **variational distribution** $ q $
is equivalent to minimizing the KL divergence:

    $$
    \arg\max_{q} \mathcal{L}_{q,\theta}(x)
    \equiv
    \arg\min_{q} \mathrm{KL}(q(z) \| p_{\theta}(z|x)).
    $$

    **Intuition**: each $ q(z) $ ("each", because we have $q$ for each $x$) estimates the posterior $ p_{\theta^*}(z|x) $ (note, having $\theta^*$).

## Amortized Inference

### Variational Posterior

$$
q(z) = \arg \max_q \mathcal{L}_{q, \theta^*}(x)
= \arg \min_q \mathrm{KL}(q \| p) = p_{\theta^*}(z | x)
$$

- $ p_{\theta^*}(z|x) $ may be **intractable**;
- $ q(z) $ is **individual for each data point** $ x $.

### Amortized Variational Inference

We restrict the family of possible distributions $ q(z) $ to a parametric class $ q_\phi(z|x) $,  
**conditioned on data** $ x $ and **parameterized by** $ \phi $.

$$
\log p_\theta(x) = \mathcal{L}_{\phi, \theta}(x) +
\mathrm{KL}(q_\phi(z|x) \| p_\theta(z|x))
\ge \mathcal{L}_{\phi, \theta}(x)
$$

$$
\mathcal{L}_{q, \theta}(x) =
\mathbb{E}_q [\log p_\theta(x | z)] -
\mathrm{KL}(q_\phi(z|x) \| p(z))
$$

## ELBo Gradients

### Gradient Update

$$
\begin{bmatrix}
\phi_k \\
\theta_k
\end{bmatrix}
=
\begin{bmatrix}
\phi_{k-1} + \eta \cdot \nabla_\phi \mathcal{L}_{\phi, \theta}(x) \\
\theta_{k-1} + \eta \cdot \nabla_\theta \mathcal{L}_{\phi, \theta}(x)
\end{bmatrix}
\Bigg|_{(\phi_{k-1}, \theta_{k-1})}
$$

- $ \phi $ denotes the parameters of the variational posterior $ q_\phi(z|x) $
- $ \theta $ represents the parameters of the generative model $ p_\theta(x|z) $

### Gradient $ \nabla_\theta \mathcal{L}_{\phi, \theta}(x) $

Unlike the $ \theta $-gradient, the density $ q_\phi(z|x) $
now depends on $ \phi $, so standard Monte Carlo estimation can’t be applied:

$$
\nabla_\phi \mathcal{L}_{\phi, \theta}(x)
= \nabla_\phi \int q_\phi(z|x) \log p_\theta(x|z) dz
- \nabla_\phi \mathrm{KL}(q_\phi(z|x) \| p(z))
$$

$$
\ne \int q_\phi(z|x) \nabla_\phi \log p_\theta(x|z) dz
- \nabla_\phi \mathrm{KL}(q_\phi(z|x) \| p(z))
$$

### Reparametrization Trick (LOTUS Trick)

$$
\nabla_\phi \int q_\phi(z|x) f(z) dz
= \nabla_\phi \int p(\boldsymbol{\epsilon}) f(\mathbf{g}_\phi(x, \boldsymbol{\epsilon})) d\boldsymbol{\epsilon}
$$

$$
= \int p(\boldsymbol{\epsilon}) \nabla_\phi f(\mathbf{g}_\phi(x, \boldsymbol{\epsilon})) d\boldsymbol{\epsilon}
\approx \nabla_\phi f(\mathbf{g}_\phi(x, \boldsymbol{\epsilon}^*)), \quad
\boldsymbol{\epsilon}^* \sim p(\boldsymbol{\epsilon})
$$

### Variational Assumption

$$
p(\boldsymbol{\epsilon}) = \mathcal{N}(0, \mathbf{I}); \quad
z = \mathbf{g}_\phi(x, \boldsymbol{\epsilon}) =
\sigma_\phi(x) \odot \boldsymbol{\epsilon} + \mu_\phi(x);
$$

$$
q_\phi(z|x) = \mathcal{N}(\mu_\phi(x), \sigma_\phi^2(x))
$$

Here, $ \mu_\phi(\cdot) $ and $ \sigma_\phi(\cdot) $ are parameterized functions
(outputs of a neural network).  
Thus, we can write $ q_\phi(z|x) = \text{NN}_{e, \phi}(x) $,
the **encoder**.

$$
\nabla_\phi \mathcal{L}_{\phi, \theta}(x)
= \nabla_\phi \int q_\phi(z|x) \log p_\theta(x|z) dz
- \nabla_\phi \mathrm{KL}(q_\phi(z|x) \| p(z))
$$

#### **1. Reconstruction Term**

$$
\nabla_\phi \int q_\phi(z|x) \log p_\theta(x|z) dz
= \int p(\boldsymbol{\epsilon}) \nabla_\phi
\log p_\theta(x|\mathbf{g}_\phi(x, \boldsymbol{\epsilon})) d\boldsymbol{\epsilon}
$$

$$
\approx \nabla_\phi \log p_\theta(x|
\sigma_\phi(x) \odot \boldsymbol{\epsilon}^* + \mu_\phi(x)),
\quad \boldsymbol{\epsilon}^* \sim \mathcal{N}(0, \mathbf{I})
$$

The generative distribution $ p_\theta(x|z) $
can be implemented as a neural network.  
We may write $ p_\theta(x|z) = \text{NN}_{d, \theta}(z) $,
called the **decoder**.

#### **2. KL Term**

$ p(z) $ is the prior over latents $ z $, typically $ p(z) = \mathcal{N}(0, \mathbf{I}) $.

$$
\nabla_\phi \mathrm{KL}(q_\phi(z|x) \| p(z))
= \nabla_\phi \mathrm{KL}(\mathcal{N}(\mu_\phi(x), \sigma_\phi^2(x)) \| \mathcal{N}(0, \mathbf{I}))
$$

This expression admits a **closed-form analytic solution**.

## VAE (Variational AutoEncoder)

![alt text](notes_images/vae.png)

### Training

- Pick a batch of samples $ \{ x_i \}_{i=1}^B $ (here we use Monte Carlo technique).
- Compute the objective for each sample (apply the reparametrization trick):

$$
\boldsymbol{\epsilon}^* \sim p(\boldsymbol{\epsilon}); \quad
z^* = \mathbf{g}_\phi(x, \boldsymbol{\epsilon}^*);
$$

$$
\mathcal{L}_{\phi, \theta}(x)
\approx \log p_\theta(x|z^*) -
\mathrm{KL}(q_\phi(z|x) \| p(z)).
$$

- Update parameters via stochastic gradient steps with respect to $ \phi $ and $ \theta $.

### Inference

- Sample $ z^* $ from the prior $ p(z) $ ($ \mathcal{N}(0, \mathbf{I}) $);
- Generate data from the decoder $ p_\theta(x|z^*) $.

**Note:**  
The encoder $ q_\phi(z|x) $ isn’t needed during generation.

# **Lecture 4 - ...**

haven't started DGM4
