# BOSS

**Bayesian Orthogonal-channel Spline Separation for LISA**

This repository contains the code and notebooks associated with the paper:

**Bayesian P-spline recovery of stochastic gravitational-wave backgrounds in LISA**

BOSS is a Bayesian framework for jointly estimating the LISA instrumental noise and a stochastic gravitational-wave background (SGWB). The instrumental test-mass and optical metrology system noise spectra are modeled using log-P-splines, while the SGWB can be modeled using either a parametric power law or a flexible log-P-spline model.

## Repository contents

### `sgwbsep.py`

Contains the main functions used for the SGWB and instrumental-noise analysis.

### `paper_plots_sgwb_det.ipynb`

Notebook used to reproduce the figures presented in the paper.

The data required by this notebook are available in the accompanying Zenodo dataset as:

- `sgwb_paper_data.h5`
- `bayes_factor_results.zip`

### `sgwb_sim.ipynb`

Contains examples for simulating stochastic gravitational-wave backgrounds with different parameters.

The simulation inputs used by this notebook are available in the accompanying Zenodo dataset as:

- `sim_inputs.h5`

## Data

The data associated with this work are available on Zenodo:

**Zenodo:** 10.5281/zenodo.21995299

The Zenodo record contains:

- `bayes_factor_results.zip` — stepping-stone sampling results used to calculate the Bayes factors and reproduce the detection boundaries shown in Fig. 2 of the paper;
- `sgwb_paper_data.h5` — data used by `paper_plots_sgwb_det.ipynb` to reproduce the paper figures;
- `sim_inputs.h5` — data used by `sgwb_sim.ipynb` for the SGWB simulation examples.
