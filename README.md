# py-fmas 

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

`py-fmas` is a Python package for the accurate numerical simulation of the
z-propagation dynamics of ultrashort optical pulses in single mode nonlinear
waveguides.  The simulation approach is based on propagation models for the
analytic signal of the optical field.
The software implements various models of the Raman response and allows to
calculate spectrograms, detailing the time-frequency composition of the
analytic signal. Additionally, a convenience class for analyzing propagation
constants is provided.

Further details such as an extended user guide with step-by-step
demonstrations, various usage examples, and a reference manual can be found on
the [py-fmas project page on gitHub pages](https://omelchert.github.io/py-fmas/).

## Installation

The software can be installed by cloning the repository and installing the
provided Python3 wheel in the following way

```
$ git clone https://github.com/omelchert/py-fmas
$ cd py-fmas/dist 
$ python3 -m pip install py_fmas-1.3.0-py3-none-any.whl
```

## Further information 

- Step-by-step demonstrations of the implemented funtionality are shown in the
  [extended user guide](https://omelchert.github.io/py-fmas/auto_tutorials/index.html)

- Usage examples demonstrating exemplary propagation scenarios are shown in the 
  [examples gallery](https://omelchert.github.io/py-fmas/auto_examples/index.html)

- The `py-fmas` package is documented in the online 
  [reference manual](https://omelchert.github.io/py-fmas/reference_manual/index.html)
