# Quick Start

## Basic Reconstruction

The simplest reconstruction runs the command-line tool with a TOML configuration file:

```bash
./release/recon config.toml
```

If you call the binary with no arguments, it prints usage and generates a template config file:

```bash
./release/recon
# Generates: config_template.toml
```

## Minimal Configuration

Here is a minimal TOML configuration for an unconstrained conjugate gradient reconstruction:

```toml
[[input]]
filename = "data/projections.tiff"
angles   = "data/angles.txt"
gamma    = 0

[output]
filename = "recon.tiff"

[recon_params]
max_iters  = 50
tol        = 1e-5
xtol       = 1e-5
recon_dims = [51, 511, 511]
```

## Configuration Explanation

- **`[[input]]`** — Array-of-tables block containing one dataset. The double brackets mean you can have multiple inputs (for multi-tilt XMCD).
  - `filename`: Path to a TIFF file with projection data
  - `angles`: Path to a text file with one projection angle per line (degrees or radians, auto-detected)
  - `gamma`: Tilt angle in degrees (required, e.g., 0, ±45 for multi-tilt)

- **`[output]`** — Output settings
  - `filename`: Where to save the reconstructed volume

- **`[recon_params]`** — Reconstruction parameters
  - `max_iters`: Maximum iterations (default: 100)
  - `tol`: Loss convergence tolerance
  - `xtol`: Solution-change convergence tolerance
  - `recon_dims`: `[thickness, height, width]` in voxels
    - All dimensions must be odd
    - `thickness` is laminography depth (expected << width)

## Input Data

### Projection Data (TIFF)

The projection file should be a TIFF containing 2D projections, either as a single multi-frame image or as individual frames.

### Angles File

Create a text file with one angle per line:

```
0.0
1.5
3.0
4.5
...
```

Angles can be in degrees or radians; the library auto-detects based on magnitude (`max(|angle|) > 2π` → degrees).

Alternatively, generate angles with the helper script:

```bash
scripts/angles.sh 0 90 > angles.txt
```

This creates angles from 0° to 90°.

## Example: Multi-tilt XMCD

For multi-tilt magnetic circular dichroism, provide multiple inputs with different gamma angles:

```toml
[[input]]
filename = "data/projections_0.tiff"
angles   = "data/angles.txt"
gamma    = 0

[[input]]
filename = "data/projections_45.tiff"
angles   = "data/angles.txt"
gamma    = 45

[[input]]
filename = "data/projections_-45.tiff"
angles   = "data/angles.txt"
gamma    = -45

[output]
filename = "recon.tiff"

[recon_params]
max_iters  = 100
tol        = 1e-5
xtol       = 1e-5
recon_dims = [51, 511, 511]
```

## Next Steps

- See [Configuration Reference](configuration.md) for the full TOML schema, including regularization options
- See [Usage Guide](usage.md) for batch reconstruction and utility scripts
- See [Theory & Algorithms](theory.md) for mathematical details
