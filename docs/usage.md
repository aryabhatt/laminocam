# Usage Guide

## Command-Line Reconstruction

### Basic Invocation

```bash
./release/recon <config.toml>
```

The tool reads the TOML configuration, performs reconstruction, and writes output to the path specified in `[output] filename`.

### Batch Reconstruction

Use the provided shell script to run multiple configurations:

```bash
./run.sh config1.toml config2.toml config3.toml
```

This runs each configuration sequentially.

Alternatively, manually loop over configurations:

```bash
for config in configs/*.toml; do
  ./release/recon "$config"
done
```

## Input Data

### Projection TIFF Files

Input projection data should be stored in TIFF format. The file can contain:
- Multiple 2D frames (multi-frame TIFF)
- A single 3D stack (as a grayscale TIFF volume)

### Angles File

Create a text file with one projection angle per line:

```
0.0
1.5
3.0
...
```

**Auto-detection:**
- If any angle has magnitude > 2π, all angles are treated as degrees and converted to radians
- Otherwise, angles are treated as radians

**Generate angles with a helper script:**

```bash
scripts/angles.sh 0 90 > angles.txt
```

This creates integer degree angles from 0° to 90°.

## Output Files

### TIFF Output

By default, reconstructions are saved as TIFF. The output is a 3D array with dimensions specified by `recon_dims` in the configuration.

### VTI Output (ParaView Format)

To also save the reconstruction in VTI format (compatible with ParaView/VisIt):

```toml
[output]
filename = "recon.tiff"
formats  = ["tiff", "vti"]
```

This creates both `recon.tiff` and `recon.vti`.

## Python Utility Scripts

### Estimate GPU Memory

Analytically estimate the peak GPU memory required for split Bregman + conjugate gradient:

```bash
python scripts/estimate_bregman_memory.py --shape 256 256 256
```

Options:
- `--shape N N N`: Volume dimensions (required)
- `--dtype float32` or `--dtype float64`: Data type (default: float32)
- `--verbose`: Print detailed breakdown

### Test MBIR Pipeline Memory

Estimate GPU memory from a configuration file:

```bash
python scripts/test_bregman_memory.py dino/tv/config.toml
```

Options:
- `--verbose`: Print detailed breakdown
- `--sino-shape Na Nr Nc`: Override sinogram shape (optional)

### Visualize Reconstruction

Display slices of a reconstructed TIFF volume:

```bash
python scripts/plot_chess.py recon.tiff
```

Opens a matplotlib window showing three cross-sectional slices. Useful for quick quality assessment.

Alias:
```bash
python scripts/plot_projs.py recon.tiff  # Identical functionality
```

### Plot NUFFT Error

Plot numerical error vs. kernel radius for NUFFT parameters:

```bash
python scripts/run_nufft.py          # Generate error data
python scripts/plot_error.py         # Visualize results
```

This sweeps over different NUFFT oversampling and kernel parameters, runs the NUFFT forward/backward test, and plots the reconstruction error.

## Performance Optimization

### Parallel Execution

The library uses multiple parallelization strategies:

- **OpenMP**: Multi-threaded CPU execution (set `OMP_NUM_THREADS` to control cores)
- **Intel TBB**: Fine-grained task parallelization
- **FFTW**: Optimized fast Fourier transforms (multi-threaded)
- **FINUFFT**: Non-uniform FFT for polar grid operations
- **CUDA (optional)**: GPU-accelerated forward/backward projections and optimization

### Using GPU

If compiled with CUDA support, the tool automatically uses GPU acceleration if available. Disable with:

```bash
export CUDA_VISIBLE_DEVICES=  # Empty string disables CUDA
./release/recon config.toml
```

### Controlling Thread Count

For OpenMP:

```bash
export OMP_NUM_THREADS=4
./release/recon config.toml
```

## Troubleshooting

### Configuration Errors

If the TOML file has errors, the tool prints a message and generates `config_template.toml` as a reference.

### Memory Issues

Use the memory estimation scripts to check if your system has sufficient RAM or VRAM:

```bash
python scripts/estimate_bregman_memory.py --shape 325 303 303 --verbose
```

### Convergence Issues

If reconstruction does not converge or has artifacts:

1. Reduce `lambda` for less aggressive regularization
2. Increase `max_iters` for more iterations
3. Increase `inner_iters` to solve the u-subproblem more accurately
4. Adjust `mu` (higher enforces constraints more strictly)

## Example Workflows

### Single-Tilt Reconstruction

```bash
# Create angles
scripts/angles.sh 0 180 > angles.txt

# Create config.toml with [[input]] gamma = 0

# Run reconstruction
./release/recon config.toml

# Visualize
python scripts/plot_chess.py recon.tiff
```

### Multi-Tilt XMCD

```bash
# Prepare three datasets with gamma = 0, ±45

# Create config.toml with three [[input]] blocks

# Run once (combines all three tilts)
./release/recon config.toml

# Result is a single 3D reconstruction
python scripts/plot_chess.py recon.tiff
```

### Batch Processing

```bash
# Run all configs in a directory
for config in experiments/*/config.toml; do
  echo "Processing $config..."
  ./release/recon "$config"
done
```
