# EASI-FISH Spots Tools

A Python toolkit for detecting and analyzing fluorescent spots in EASI-FISH imaging data. This package provides tools for spot detection using FishSpots, as well as utilities for extracting statistics from spots detected by various tools including airlocalize, RS-FISH, and FishSpots.

## Installation

### Using Conda

Create a conda environment with the required dependencies:

```bash
conda env create -f conda-env.yml
conda activate easifish-spots-tools
```

### Install the package

```bash
pip install -e .
```

## Requirements

- Python >= 3.12
- numpy > 2, < 3
- pandas = 2.3.3
- scikit-image = 0.25.2
- zarr
- dask/distributed
- See `pyproject.toml` for complete dependency list

## Tools

### 1. Spot Detection

Run spot detection on image data using the FishSpots algorithm with distributed processing support.

**Get help:**
```bash
python -m easifish_spots_tools.main_spot_extraction -h
```

**Basic usage:**
```bash
python -m easifish_spots_tools.main_spot_extraction \
  --input /path/to/input.zarr \
  --output /path/to/output.csv \
  --channels 0 1 2 \
  --fishspots-config config.yaml \
  --local-dask-workers 4
```

**Key parameters:**
- `--input`: Path to input image container (Zarr/OME-Zarr format)
- `--output`: Path to output CSV file for detected spots
- `--channels`: Channel indices to process
- `--fishspots-config`: YAML configuration file for FishSpots parameters
- `--local-dask-workers`: Number of Dask workers for parallel processing
- `--blocksize`: Block size for processing (X,Y,Z)
- `--intensity-threshold`: Intensity threshold for spot detection
- `--apply-voxel-spacing`: Apply voxel spacing to spot coordinates

### 2. Extract Spot Counts

Count spots within labeled regions and aggregate results across multiple spot files.

**Get help:**
```bash
python -m easifish_spots_tools.labeled_spots_counts -h
```

**Basic usage:**
```bash
python -m easifish_spots_tools.labeled_spots_counts \
  --labels-container /path/to/labels.zarr \
  --labels-subpath /labels \
  --spots-pattern "/path/to/spots/*.csv" \
  --output counts.csv
```

**Key parameters:**
- `--labels-container`: Path to the labeled segmentation container
- `--spots-pattern`: Glob pattern for spot CSV files
- `--timeindex`: Time index for the labels
- `--channel`: Channel index for the labels
- `--voxel-spacing`: Voxel spacing (X,Y,Z) if not in metadata

### 3. Extract Spot Region Properties

Extract detailed region properties and statistics for spots within labeled regions.

**Get help:**
```bash
python -m easifish_spots_tools.labeled_spots_props -h
```

**Basic usage:**
```bash
python -m easifish_spots_tools.labeled_spots_props \
  --labels-container /path/to/labels.zarr \
  --image-container /path/to/image.zarr \
  --spots-pattern "/path/to/spots/*.csv" \
  --output properties.csv
```

### 4. Post-Processing for RS-FISH

Normalize spot data format across different detection tools (airlocalize, RS-FISH, FishSpots) to ensure consistent output format.

**Get help:**
```bash
python -m easifish_spots_tools.post_rs_fish -h
```

**Purpose:**
This tool standardizes spot information regardless of the detection tool used, ensuring all spots follow the same CSV format with columns: x, y, z, t, c, intensity, sx, sy, sz.

## Output Format

All tools produce CSV files with spot information in the following format:

- **x, y, z**: Spatial coordinates (in voxels or micrometers, depending on `--apply-voxel-spacing`)
- **t**: Time index
- **c**: Channel index
- **intensity**: Spot intensity
- **sx, sy, sz**: Spot sigma values in each dimension

## Common Workflows

### Full Pipeline: Detection to Analysis

1. **Detect spots:**
```bash
python -m easifish_spots_tools.main_spot_extraction \
  --input data.zarr \
  --output spots.csv \
  --channels 0 1 \
  --local-dask-workers 8 \
  --apply-voxel-spacing
```

2. **Count spots per labeled region:**
```bash
python -m easifish_spots_tools.labeled_spots_counts \
  --labels-container labels.zarr \
  --spots-pattern "spots.csv" \
  --output spot_counts.csv
```

3. **Extract region properties:**
```bash
python -m easifish_spots_tools.labeled_spots_props \
  --labels-container labels.zarr \
  --image-container data.zarr \
  --spots-pattern "spots.csv" \
  --output spot_properties.csv
```

## Configuration

### FishSpots Configuration

Create a YAML configuration file for spot detection parameters:

```yaml
white_tophat_args:
  # White top-hat filter parameters

psf_estimation_args:
  # PSF estimation parameters

deconvolution_args:
  # Deconvolution parameters

spot_detection_args:
  # Spot detection parameters
```

### Dask Configuration

For distributed processing, you can provide a Dask configuration file:

```bash
python -m easifish_spots_tools.main_spot_extraction \
  --dask-config dask_config.yaml \
  --dask-scheduler tcp://scheduler:8786
```

## License

BSD-3-Clause

## Version

0.3.0
