# Configuration Reference

All reconstruction parameters are specified in a TOML file passed to the `recon` command.

## Input Section

The `[[input]]` array-of-tables block specifies one or more projection datasets. Use double brackets to enable multiple datasets (for multi-tilt XMCD).

```toml
[[input]]
filename = "/path/to/projections.tiff"     # TIFF file with projection data (required)
angles   = "/path/to/angles.txt"           # Text file with one angle per line (required)
gamma    = 0                                # Tilt angle in degrees (required)
```

**Notes:**
- You can have multiple `[[input]]` blocks for multi-tilt reconstructions with different gamma values
- Angles file: one value per line; auto-detected as degrees or radians based on magnitude
- If `max(|angle|) > 2π`, angles are treated as degrees and converted to radians

## Output Section

```toml
[output]
filename = "recon.tiff"                 # Output path (relative or absolute)
formats  = ["tiff"]                     # Output formats (optional, default: ["tiff"])
                                        # Valid options: "tiff", "vti"
```

## Reconstruction Parameters

```toml
[recon_params]
max_iters  = 100                        # Maximum outer iterations (default: 100)
tol        = 1e-5                       # Loss convergence tolerance (default: 1e-5)
xtol       = 1e-5                       # Solution change tolerance (default: 1e-5)
recon_dims = [51, 511, 511]             # [thickness, height, width] in voxels
```

**Key constraints:**
- All dimensions in `recon_dims` must be odd (even values are silently decremented)
- `thickness` is the laminography depth, expected to be << `width` (a warning is printed if ratio > 0.15)

## Regularization

### Unconstrained Reconstruction (Conjugate Gradient)

Omit the `[recon_params.regularizer]` block entirely:

```toml
[recon_params]
max_iters  = 50
tol        = 1e-5
xtol       = 1e-5
recon_dims = [51, 511, 511]
```

This runs the least-squares conjugate gradient solver without regularization.

### Split Bregman with Total Variation Regularization

Add a `[recon_params.regularizer]` block:

```toml
[recon_params]
max_iters  = 100
tol        = 1e-5
xtol       = 1e-5
recon_dims = [51, 511, 511]

[recon_params.regularizer]
method = "split_bregman"                # Currently the only supported method

[recon_params.regularizer.split_bregman]
lambda      = 0.5                       # TV regularization weight (default: 0.1)
mu          = 10.0                      # Augmented Lagrangian penalty (default: 10.0)
inner_iters = 1                         # CG iterations per outer iteration (default: 1)
```

**Parameters:**
- `lambda`: Regularization weight; controls balance between data fidelity and smoothness
- `mu`: Augmented Lagrangian penalty parameter; higher values enforce constraints more strictly
- `inner_iters`: Number of CG iterations to solve the u-subproblem in each Bregman iteration

## Complete Example

Here's a complete configuration file for split Bregman reconstruction:

```toml
[[input]]
filename = "data/projections.tiff"
angles   = "data/angles.txt"
gamma    = 0

[output]
filename = "reconstruction.tiff"
formats  = ["tiff", "vti"]

[recon_params]
max_iters  = 100
tol        = 1e-6
xtol       = 1e-6
recon_dims = [51, 511, 511]

[recon_params.regularizer]
method = "split_bregman"

[recon_params.regularizer.split_bregman]
lambda      = 0.5
mu          = 10.0
inner_iters = 2
```

## Multi-tilt XMCD Example

For multi-tilt magnetic circular dichroism with three datasets at different tilts:

```toml
[[input]]
filename = "data/gamma_0.tiff"
angles   = "data/angles.txt"
gamma    = 0

[[input]]
filename = "data/gamma_45.tiff"
angles   = "data/angles.txt"
gamma    = 45

[[input]]
filename = "data/gamma_-45.tiff"
angles   = "data/angles.txt"
gamma    = -45

[output]
filename = "recon_multi_tilt.tiff"

[recon_params]
max_iters  = 150
tol        = 1e-6
xtol       = 1e-6
recon_dims = [51, 511, 511]

[recon_params.regularizer]
method = "split_bregman"

[recon_params.regularizer.split_bregman]
lambda      = 0.5
mu          = 10.0
inner_iters = 2
```
