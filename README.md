# py-fmas 

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

`py-fmas` is a Python package for the accurate numerical simulation of the
:math:`z`-propagation dynamics of ultrashort optical pulses in single mode
nonlinear waveguides. The simulation approach is based on nonlinear propagation
models for the analytic signal of the optical field.  The implemented models
include, e.g., third-harmonic generation and the Raman effect.

To perfom z-propagation, the initial real-valued optical field is defined on a
periodic one-dimensional temporal grid and converted to the complex-valued
analytic signal. z-stepping is performed by pseudospectral methods. The
software implements a selection of algorithms with fixed or adaptive stepsize,
commonly used in nonlinear optics for solving nonlinear Schr\"odinger type
equations.

The provided software implements various models for the Raman response and
allows to calculate spectrograms, detailing the time-frequency composition of
the analytic signal. Additionally, a convenience class for analyzing
propagation constants is provided.

## Installation

Simply clone the repository

``$ git clone https://github.com/omelchert/py-fmas``

and install the Python3 Wheel via

``$ python3 -m pip install ./py-fmas/dist/py_fmas-1.3.0.whl``

## Usage Examples

`py-fmas` comes with various step-by-step demonstrations in folder `tutorials`
and usage examples in folder `galleries`.

## Further information 

TWB
