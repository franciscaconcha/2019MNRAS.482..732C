# viscous-disks-submit
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1465972.svg)](https://doi.org/10.5281/zenodo.1465972) [![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

(Code to run the simulations and generate figures of the paper: )

This is a code to run simulations of star clusters where the stars have protoplanetary disks around them. The disks are parametrized using the descriptions of Lynden-Bell & Pringle 1974 (1974MNRAS.168..603L) and Hartmann et al 1998 (1998ApJ...495..385H). The simulations are run using the AMUSE (Astrophysical Multipurpose Software Environment) framework.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for testing purposes. You will be able to run the simulations as specified in the paper and to reproduce the figures.

### Prerequisites
* Python 2.7 or higher
* AMUSE: https://github.com/amusecode/amuse
* scipy

### Running the simulations

You can run an individual simulation by using the python script directly:

```
amuse.sh cluster_with_viscous_disks.py
```

Or run a series of simulations using the shell script:

```
source run_simulations.sh
```

Both scripts have an extensive options which can be passed through the command line. For a list of these options run:

```
amuse.sh cluster_with_viscous_disks.py --help
```
### Creating the plots



## Authors


* **Francisca Concha-Ramírez** - [francisca.cr](https://francisca.cr) [![Twitter Follow](https://img.shields.io/twitter/follow/espadrine.svg?style=social&label=Follow)](http://twitter.com/franconchar)
* **Eero Vaher** - *Initial work* 
* **Simon Portegies Zwart**

## License

This project is licensed under the GPL v3 License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
