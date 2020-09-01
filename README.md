# PFPy

PFPy provides a wrapper around PowerFactory's Python API with additional features in  `pfpy.pf`. 
Furthermore,  `pfpy.models` provides a high-level access to PowerFactory models, e.g. via:
- `model.simulate()`
- `model.linearize()`
- `model.set_params()`
It has been developed at the Division of Electric Power and Energy Systems, KTH Royal Institute of Technology.

The code has been tested with PowerFactory 2019 and PowerFactory 2020. It should be noted that $PYTHONPATH needs to point to the PowerFactory Python API.

## Examples
The PowerFactory model using in all of the examples is located in [./examples/data/models/PFPY_DEMO.pfd](https://github.com/tinrabuzin/PFPy/blob/master/examples/data/models/PFPY_DEMO.pfd)

### Simulation Example

The simulation of the aforementioned model is demonstrated in the [simulation example](https://github.com/tinrabuzin/PFPy/blob/master/examples/simulation_with_inputs.py).
The model contains a controllable voltage source at the PCC. CSV files are generated that are replayed at the PCC, simulation is executed and the results are plotted.

### Linearization Examples

The same model is linearized as seen from the PCC in the following [Jupyter notebook](https://github.com/tinrabuzin/PFPy/blob/master/examples/linearization.ipynb).

## Documentation

Automatically generated documentation from docstrings can be found [here](https://tinrabuzin.github.io/PFPYTest/).
