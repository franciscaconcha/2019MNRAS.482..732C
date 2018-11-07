# 2019MNRAS.482..732C The viscous evolution of circumstellar discs in young star clusters
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1465931.svg)](https://doi.org/10.5281/zenodo.1465931) [![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) ![python](https://img.shields.io/badge/python-2.7-yellow.svg)

(Code to run the simulations and generate figures of the paper: [2019MNRAS.482..732C: The viscous evolution of circumstellar discs in young star clusters](https://academic.oup.com/mnras/article/482/1/732/5124387).)

This is a code to run simulations of star clusters where the stars have protoplanetary disks around them. Protoplanetary disks are subject to viscous growth and dynamical truncations. The disks are parametrized using the descriptions of Lynden-Bell & Pringle 1974 ([1974MNRAS.168..603L](https://doi.org/10.1093/mnras/168.3.603)) and Hartmann et al 1998 ([1998ApJ...495..385H](https://doi.org/10.1086/305277)). The simulations are run using the [AMUSE framework](http://amusecode.org) (Astrophysical Multipurpose Software Environment). See paper for more details.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for testing purposes. You will be able to run the simulations as specified in the paper and to reproduce the figures.

### Prerequisites
* Python 2.7. Should work fine with Python 3 but it has not been tested.
* AMUSE: https://github.com/amusecode/amuse
* scipy

### Running the simulations

You can run an individual simulation by using the AMUSE script directly:

```
amuse.sh cluster_with_viscous_disks.py
```

The script has extensive options which can be passed through the command line. For a list of these options run:

```
amuse.sh cluster_with_viscous_disks.py --help
```

### Creating the plots

Figures 1, 2, and 3 of the paper were created with the script ```plot_disks.py```. You can run it as:

```
amuse.sh plot_disks.py
```
A list of options is available for this script, including the path to the files that you want to use for the plots. To see the list of options add ```--help``` or ```-h``` to the line above.

Figures 4, 5, and 6 of the paper were created with the script ```find_models.py```. Instructions are the same as above.


## Authors


* **Francisca Concha-Ramírez** - [francisca.cr](https://francisca.cr) [![Twitter Follow](https://img.shields.io/twitter/follow/espadrine.svg?style=social&label=Follow)](http://twitter.com/franconchar)
* **Eero Vaher** - *Initial work* 
* **Simon Portegies Zwart**

## License

This project is licensed under the GPL v3 License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* We thank [Arjen van Elteren](https://github.com/arjenve) for invaluable help with AMUSE!
