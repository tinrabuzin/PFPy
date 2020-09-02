# PFPy - PowerFactory with Python

[![Documentation Status](https://readthedocs.org/projects/pfpy/badge/?version=latest)](https://pfpy.readthedocs.io/en/latest/?badge=latest)

PFPy provides a wrapper around PowerFactory's Python API with additional features in  `pfpy.pf`. 
Furthermore,  `pfpy.models` provides a high-level access to PowerFactory models, e.g. via:
- `model.simulate()`
- `model.linearize()`
- `model.set_params()`

It has been developed at the Division of Electric Power and Energy Systems, KTH Royal Institute of Technology.

The code has been tested with PowerFactory 2019 and PowerFactory 2020. 

## Installation

The module can be installed using pip as follows:
```
pip install pfpy
```

All the requirements except `powerfactory` API should be installed. Note that $PYTHONPATH needs to point to the PowerFactory Python API to be able to use it. It is located in `%POWERFACTORYPATH/Python/%version`. The code has been tested with Python 3.7 and 3.8.

## Examples
The PowerFactory model used in all of the examples is located in [./examples/data/models/PFPY_DEMO.pfd](https://github.com/tinrabuzin/PFPy/blob/master/examples/data/models/PFPY_DEMO.pfd)

### Simulation Example

The simulation of the aforementioned model is demonstrated in the [simulation example](https://github.com/tinrabuzin/PFPy/blob/master/examples/simulation_with_inputs.py).
The model contains a controllable voltage source at the PCC. CSV files are generated that are replayed at the PCC, simulation is executed and the results are plotted.

### Linearization Examples

The same model is linearized as seen from the PCC in the following [Jupyter notebook](https://github.com/tinrabuzin/PFPy/blob/master/examples/linearization.ipynb).
When compared to the linearization in PowerFacory, this code, in addition to the A matrix, provides also B,C, and D matrices.

## Documentation

Automatically generated documentation from docstrings can be found [here](https://pfpy.readthedocs.io/en/latest/).

### Acknowledgments
Parts of this code were inspired by the following book:
*Francisco Gonzalez-Longatt, José Luis Rueda Torres, Advanced Smart Grid Functionalities Based on PowerFactory, Springer, 2017*
