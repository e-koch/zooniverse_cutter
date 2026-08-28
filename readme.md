# zooniverse_cutter

## Summary

Generates image cutouts of galaxies from multiple input images (e.g. JWST/HST/MUSE) for the
Zooniverse citizen-science "Bubble Zoo" project. The active pipeline is `bubblezoo_spectralcube.py`, driven
by a TOML config file that describes which bands/targets to use and how to color and scale each band.

## Setup

```
conda env create -f hydra_jobs/zooniverse_environment.yml
conda activate bubblezoo
```

`reproject` needs its dev version installed from source rather than from conda:

```
git clone https://github.com/astropy/reproject.git
cd reproject
pip install -e .
```

(08/2026: try the most recent release of reproject; it's likely a previous release contained the needed fix.)

## TLDR

1. Write a TOML config (see `colors_config.toml` or `testgalaxies_colors_config_*.toml` for examples).
2. Run `bubblezoo_spectralcube.py <config.toml> <job_id_or_None>` from within this directory.
3. To process a batch of galaxies on a cluster, submit an SGE job array that calls the same script once per
   target (see `hydra_jobs/jobarray_zooniverse_cutout_tests.job`).

* * *

## Detailed usage

### 1. The TOML config

The config file lists where the data lives, which bands and galaxies to process, and the colorization
parameters for each band:

```toml
data_path = "/path/to/images_for_zooniverse"
output_path = "/path/to/output_for_zooniverse"

bands = [
    "f770w",      # jwst
    "hst_bband",  # hst
]

targets = [
    "ngc7496",
    "ic5332",
    "ngc0628",
]

# One [parameters.<band>] table is required for every entry in `bands`.
[parameters.f770w]
file_search_str = "_f770w_i2d_align.fits"  # glob matched under data_path/<target>/
hdu_ext = 1
color = "#FF0000"

[parameters.f770w.greyRGBize_kwargs]
rescalefn = "asinh"     # stretch: linear/sqrt/squared/log/power/sinh/asinh
min_max = [0.1, 10]
scaletype = "abs"        # 'abs' (fixed min/max), 'perc' (percentiles), or 'maxent' (max-entropy stretch)

[parameters.f770w.colorize_image_kwargs]
colorintype = "hex"
gammacorr_color = 2.2
```

Each `[parameters.<band>]` table also accepts optional flags: `apply_bkgsub`/`bgksub_percent` (background
subtraction), `apply_cd` (constrained-diffusion filtering), `apply_radial_scaling`/`radscale_thresh`
(down-weight bright galaxy centers), and `edge_clip`/`edge_clip_size`/`edge_clip_match` (binary-erode chip
edges, mainly for HST). `greyRGBize_kwargs` also accepts `vmin`/`vmax` (hard overrides) and
`min_vmax`/`max_vmin` (floors/ceilings for percentile scaling in noisy images).

Every FITS file is located by globbing `data_path/<target>/*<file_search_str>*`, so exactly one match must
exist per band per galaxy.

### 2. Running the cutter script

From within this directory:

```
python bubblezoo_spectralcube.py <config.toml> <job_id>
```

- `<job_id>` selects a single galaxy by index into the config's `targets` list (used for HPC job arrays).
  Pass `None` to process every target listed in the config in one run.

For each galaxy, the script reprojects all bands onto a common WCS, tiles a grid of overlapping cutouts,
colorizes and combines the bands, and writes out (per cutout) 4 rotated PNG/JPG images, optional per-band
greyscale cutouts, a WCS header text file, and a CSV of the vmin/vmax used per cutout. When finished, it tars
up the output directory into `cutouts_<color_tag>_target_<galaxy_or_"set">_jobid_<id_or_"None">.tar` in the
parent of `output_path` (`<color_tag>` is taken from the last `_`-separated token of the config filename,
e.g. `..._maxent.toml` → `maxent`).

Example — process a single galaxy directly:

```
python bubblezoo_spectralcube.py testgalaxies_colors_config_geminiccancer_maxent.toml None
```

### 3. Batch processing on a cluster (job array example)

**The key info in the batch script is running over a set of galaxy names. The details on jobarray and submission is specific to the cluster/job scheduler.**

`hydra_jobs/jobarray_zooniverse_cutout_tests.job` shows how to run the cutter over a batch of galaxies as an
SGE job array, with one task per galaxy in the config's `targets` list:

```csh
#$ -t 1-3            # one task per target galaxy in the config
#$ -pe mthread 1
#$ -l mres=4G,h_data=4G,h_vmem=4G

conda activate bubblezoo

set CONFIG_FILE = $HOME/zooniverse_cutter/testgalaxies_colors_config_geminiccancer_maxent.toml

cd $HOME/zooniverse_cutter
$my_python -u $HOME/zooniverse_cutter/bubblezoo_spectralcube.py $CONFIG_FILE $SGE_TASK_ID
```

Submit it with:

```
qsub hydra_jobs/jobarray_zooniverse_cutout_tests.job
```

Each task gets `$SGE_TASK_ID` as `job_id`, which the script uses to pick one galaxy out of `targets` — so the
`-t 1-3` range must match the number of targets in the config. Each task's output is tarred up separately
(the job ID is embedded in the tar filename), so results from a batch run don't overwrite one another.
