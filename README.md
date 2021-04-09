# py-fmas 

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

`py-fmas` is a Python package for the accurate numerical simulation of the
z-propagation dynamics of ultrashort optical pulses in single mode nonlinear
waveguides. The simulation approach is based on nonlinear propagation models
for the analytic signal of the optical field.  The implemented models include,
e.g., third-harmonic generation and the Raman effect.

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
and usage examples in folder `galleries`. After completing the installation,
these examples can be run directly from the command line.

## Further information 

An extended user guide with step-by-step demonstrations, further usage examples
and a reference manual can be found on the  [project
page](https://omelchert.github.io/py-fmas/) on gitHub pages.

### Pulse propagation in terms ot the analytic signal

Below you find a list of articles that discuss ultrashort optical pulse
propagation in terms of the analytic signal and its variants:


- A. Demircan, Sh. Amiranashvili, C. Bree, U. Morgner, G. Steinmeyer,
  Supercontinuum generation by multiple scatterings at a group velocity
  horizon, Opt. Exp. 22 (2014) 3866, https://doi.org/10.1364/OE.22.003866.

- A. Demircan, Sh. Amiranashvili, C. Bree, C. Mahnke, F. Mitschke, G.
  Steinmeyer, Rogue wave formation by accelerated solitons at an optical event
  horizon, Appl. Phys. B 115 (2014) 343,
  http://dx.doi.org/10.1007/s00340-013-5609-9

- Sh. Amiranashvili, A. Demircan, Ultrashort Optical Pulse Propagation in
  terms of Analytic Signal, Adv. Opt. Tech. 2011 (2011) 989515,
  http://dx.doi.org/10.1155/2011/989515.

- Sh. Amiranashvili, A. Demircan, Hamiltonian structure of
  propagation equations for ultrashort optical pulses, Phys. Rev. E 10
  (2010) 013812, http://dx.doi.org/10.1103/PhysRevA.82.013812.


### Research articles employing `py-fmas`

Below you find a list of articles that used some of the :math:`z`-proapgation
models and algorithms implemented by the `py-fmas` package:

- O. Melchert, C. Bree, A. Tajalli, A. Pape, R. Arkhipov, S. Willms, I.
  Babushkin, D. Skryabin, G. Steinmeyer, U. Morgner, A. Demircan, All-optical
  supercontinuum switching, Communications Physics 3 (2020) 146,
  https://doi.org/10.1038/s42005-020-00414-1.
 
- O. Melchert, S. Willms, S. Bose, A. Yulin, B. Roth, F. Mitschke, U.
  Morgner, I. Babushkin, A. Demircan, Soliton Molecules with Two Frequencies,
  Phys. Rev. Lett. 123 (2019) 243905,
  https://doi.org/10.1103/PhysRevLett.123.243905.

- O. Melchert, B. Roth, U. Morgner, A. Demircan, OptFROG — Analytic signal
  spectrograms with optimized time–frequency resolution, SoftwareX 10 (2019)
  100275, https://doi.org/10.1016/j.softx.2019.100275, code repository:
  https://github.com/ElsevierSoftwareX/SOFTX_2019_130.

- O. Melchert, U. Morgner, B. Roth, I. Babushkin, A. Demircan, Accurate
  propagation of ultrashort pulses in nonlinear waveguides using propagation
  models for the analytic signal, Proc. SPIE 10694, Computational Optics II,
  106940M (2018), https://doi.org/10.1117/12.2313255. 

