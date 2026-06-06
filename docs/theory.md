# Theory & Algorithms

This page describes the mathematical foundations of Tomocam, including the geometry of laminography, the forward projection model, and the iterative reconstruction algorithms.

## Introduction

**Laminography** is a tomographic imaging technique particularly well-suited for thin samples with preferred orientation. Unlike conventional tomography, laminography uses projection data from limited angles and can achieve reconstruction from undersampled data through mathematical modeling.

Tomocam implements efficient laminography reconstruction using:

- **NUFFT (Non-Uniform Fast Fourier Transform)** for forward and backward projection on polar grids
- **Conjugate Gradient (CG)** method for unconstrained least-squares problems
- **Split Bregman optimization** with total variation (TV) regularization for constrained problems

## Geometry
![Laminography Geometry](figs/laminography.png)
### Coordinate Systems

We define two coordinate systems:

**Lab frame (detector coordinates):**
- $X$: horizontal detector axis
- $Y$: vertical detector axis  
- $Z$: X-ray beam direction

**Sample frame (material coordinates):**
- $x$: rotation axis (sample rotation)
- $y$: in-plane, perpendicular to rotation axis
- $z$: normal to sample surface

### Rotation Matrices

The sample is oriented with two angles:

- $\alpha$: tomographic tilt angle (rotation about the $y$-axis in sample frame)
- $\gamma$: rotation angle about the $Z$-axis in lab frame

The combined rotation matrix is:

$$R = R_\gamma \, R_\alpha$$

where:

$$R_\alpha = \begin{pmatrix}
\cos \alpha & 0 & \sin \alpha \\
0 & 1 & 0 \\
-\sin \alpha & 0 & \cos \alpha
\end{pmatrix}$$

$$R_\gamma = \begin{pmatrix}
\cos \gamma & -\sin \gamma & 0 \\
\sin \gamma & \cos \gamma & 0 \\
0 & 0 & 1
\end{pmatrix}$$

### Fourier Space Mapping

Projection onto the detector in real space corresponds to a sampling pattern in Fourier space. The detector Fourier coordinates $(q_X, q_Y)$ map to sample Fourier coordinates $(q_x, q_y, q_z)$ via:

$$(q_x, q_y, q_z)^T = R^T \begin{pmatrix} q_X \\ q_Y \\ 0 \end{pmatrix}$$

This defines a **non-uniform sampling pattern** in 3D Fourier space, which is efficiently handled by NUFFT.

## Forward Projection Model

### Projection Operator

Let $\rho(x, y, z)$ denote the 3D scalar density (reconstructed volume) and $y(\xi, \eta)$ denote the 2D projections (detector image for a given tilt angle $\theta$).

The forward projection operator $P$ maps the volume to projections:

$$y = \mathcal{P} \, \rho$$

In the Fourier domain, the projection operator factorizes as:

$$\mathcal{P} = \mathbb{F}^{-1} \, \mathcal{F}_{\text{cal}}$$

where:
- $\mathcal{F}_{\text{cal}}$ computes Fourier coefficients at the non-uniform sampling points (NUFFT forward)
- $\mathbb{F}^{-1}$ transforms back to real space (IFFT)

The adjoint operator is:

$$\mathcal{P}^T = \mathcal{F}_{\text{cal}}^* \, \mathbb{F}$$

### Least-Squares Problem

The unconstrained tomographic reconstruction problem is:

$$\rho = \arg\min_{\rho} \left\| \mathcal{P} \rho - y \right\|_2^2$$

The gradient of the loss function is:

$$\nabla J = A \rho - b$$

where:
- $A = \mathcal{P}^T \mathcal{P}$ (Hessian, or normal equations matrix)
- $b = \mathcal{P}^T y$ (right-hand side)

## Conjugate Gradient Algorithm

For unconstrained reconstruction (no regularizer in config), Tomocam solves $A \rho = b$ using the **Preconditioned Conjugate Gradient (PCG)** method.

### Algorithm

```
Input: A, b, preconditioner M, tolerance tol
Initialize: ρ₀ = 0, r₀ = b, z₀ = M⁻¹r₀, p₀ = z₀

for k = 0, 1, 2, ... do
  αₖ = (zₖ · rₖ) / (pₖ · A pₖ)
  ρₖ₊₁ = ρₖ + αₖ pₖ
  rₖ₊₁ = rₖ - αₖ A pₖ
  
  if ‖rₖ₊₁‖₂ < tol then
    return ρₖ₊₁
  end if
  
  zₖ₊₁ = M⁻¹ rₖ₊₁
  βₖ = (zₖ₊₁ · rₖ₊₁) / (zₖ · rₖ)
  pₖ₊₁ = zₖ₊₁ + βₖ pₖ
end for
```

**Notes:**
- The preconditioner $M$ accelerates convergence; a common choice is diagonal preconditioning or the inverse of the Laplacian
- Convergence is controlled by `tol` and `max_iters` in the configuration
- $\|\cdot\|_2$ denotes the Euclidean norm

## Split Bregman for Total Variation Regularization

For regularized reconstruction with total variation, Tomocam solves:

$$\hat{\rho} = \arg\min_{\rho} \left\| P \rho - y \right\|_2^2 + \lambda \left\| \nabla \rho \right\|_1$$

where:
- $\lambda$ is the regularization weight (controls smoothness)
- $\nabla \rho$ is the gradient of the volume (3-component vector field)
- $\|\cdot\|_1$ is the $\ell^1$ norm (sum of absolute values)

This is solved using **Split Bregman iteration**, which decouples the smooth data fidelity term from the non-smooth regularization.

### Algorithm

Introduce an auxiliary variable $d = \nabla \rho$ and Bregman variable $b$:

```
Input: P, y, λ, μ, max_iters, tol

Initialize: ρ⁰ = 0, d⁰ = 0, b⁰ = 0, k = 0

while k < max_iters do
  
  # u-subproblem: solve linear system
  (P^T P + μ ∇^T ∇) ρ^(k+1) = P^T y + μ ∇^T (d^k - b^k)
  [solved via Preconditioned Conjugate Gradient]
  
  # d-subproblem: soft thresholding (isotropic TV shrinkage)
  d^(k+1) = shrink(∇ ρ^(k+1) + b^k, λ/μ)
  
  # Bregman update
  b^(k+1) = b^k + ∇ ρ^(k+1) - d^(k+1)
  
  # Check convergence
  if ‖ρ^(k+1) - ρ^k‖₂ < tol then
    return ρ^(k+1)
  end if
  
  k = k + 1
end while
```

### Shrinkage Operator

The isotropic soft thresholding operator (shrink) applied component-wise:

$$\text{shrink}(\mathbf{v}, \tau) = \max\left(1 - \frac{\tau}{\|\mathbf{v}\|}, 0\right) \mathbf{v}$$

where $\|\mathbf{v}\|$ is the Euclidean norm of the 3-vector at each voxel, and the factor $\tau = \lambda / \mu$ is the shrinkage threshold.

### Parameters

- `lambda`: Regularization weight; higher values enforce more smoothness
- `mu`: Augmented Lagrangian penalty; higher values enforce constraints more strictly
- `inner_iters`: Number of CG iterations to solve the u-subproblem in each outer iteration
- `max_iters`: Maximum number of outer (Bregman) iterations

**Typical ranges:**
- `lambda`: $0.01$ to $1.0$ (depends on noise level and image content)
- `mu`: $5$ to $100$ (higher for noisier data)
- `inner_iters`: $1$ to $10$ (more inner iterations improve convergence but increase cost)

## References

- [Marcus et al., 2022] M. A. Marcus. "Information content of and the ability to reconstruct dichroic X-ray tomography and laminography." *Optics Express*, 30(22), 2022.

- [Myagotin et al., 2013] A. Myagotin, A. Voropaev, L. Helfen, D. Hanschke, and T. Baumbach. "Efficient Volume Reconstruction for Parallel-Beam Computed Laminography by Filtered Back Projection on Multi-Core Clusters." *IEEE Transactions on Image Processing*, 22(10), 2013.

- [Wikipedia] "Magnetic Circular Dichroism." https://en.wikipedia.org/wiki/Magnetic_circular_dichroism (accessed 2025)
