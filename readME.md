# Surrogate-Assisted Optimization of Lattice Unit Cells for Desired Force-Displacement Response

**Akshay Kumar**, **Saketh Sridhara**, **Krishnan Suresh**

Department of Mechanical Engineering, University of Wisconsin–Madison

# Frame Optimization Workflow

This repository provides a complete workflow for optimizing frame geometries using finite element simulations and surrogate modeling. It includes tools to:

1. Generate input data and geometry
2. Run Abaqus simulations on the input data to obtain force-displacement curves
3. Train a surrogate model on the simulation results
4. Perform optimization using the surrogate with five random initial guesses
5. Validate the optimized solutions by re-running Abaqus
6. Plot and compare results from both the surrogate model and Abaqus

## Workflow

### 1. Data Generation and Geometry Creation

Use `frameOptimize.py` to:
- Define the geometry and design parameters
- Generate input data for simulation and optimization

### 2. Run Abaqus Simulation

Update the parameters in `run_abaqus.py` as needed, then run the following command to generate force-displacement data using Abaqus:

```bash
abaqus cae noGUI=run_abaqus.py
```

### 3. Train the Surrogate Model

Use the Jupyter notebook `SurrogateBuild.ipynb` to:

* Train a surrogate model using the Abaqus-generated data
* Save the trained model for use in optimization

### 4. Run Optimization

In `frameOptimize.py`, use the surrogate model to:

* Perform optimization of the design using five random initial guesses
* Re-run Abaqus on the optimized solutions for validation

### 5. Plot and Compare Results

`frameOptimize.py` also:

* Loads results from both Abaqus and the optimization process
* Plots them for comparison to evaluate performance and consistency

## Requirements

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

## License

This project currently has no license. All rights reserved.

## Acknowledgments

The authors would like to thank the support of the U.S. Office of Naval Research under PANTHER award number N00014-21-1-2916 through Dr. Timothy Bentley.

