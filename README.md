# Parrallel Solver for Conic Optimisation Problems using the Alternating Direction Method of Multipliers

This respository contains a Python prototype solver that solves sparse convex optimsation problems using a parallel implementation of the alternating direction method of multipliers. 

This project was completed with the guidance of Dr. Giovanni Fantuzzi as my final year Aeronautical Engineering Masters thesis at Imperial College London. A copy of my thesis is included in the project root directory. 

## Guide to Project Directories

The following directories contain experimental code from earlier stages of the project. Feel free the explore their contents, however they do not contribute to the final running of the code and thus will not be described further:

- ``no-splitting``
- ``nonlinear_parsing_old``

Below is a summary of the functional/useful directories for hti

#### ``popData``

This is a large directory containing sample convex optimisation problems of various sizes. These problems were produced from the relaxation of polynomial optimisation problems.

The suffixes for these data files represent the number of independent variables as the order of relaxation of the polynomial optimisation problem. For example: ``pop_data_'k'_'o'.mat`` represents a convex optimisation problem with $n=2\cdot(k-1)$ independent variables and relaxation order `o`.

#### ``refactored_splitting`` and ``no-splitting``

Code in these directories implement the sparse parallel and dense ADMM algorithm for convex optimisation problems respectively. Instructions on running the solvers can be found below. 

#### ``results``

The complete set of data produced during the project are found in this directory. In there, two directories of identical size contain the solver results for the sparse and dense implementations for the same respective problems. The ``_iterations_`` files contain data about the algorithm's status and progress at every 50 iterations, such as the primal and dual residuals. The corresponding ``_meta_`` file stores data about the overall run characteristics, such as the total CPU time for the run and the time taken for each individual step.

The naming convention for these data files follow the same as that of the problem set files.

#### ``Plotting``

Code for plotting various graphs for my thesis are contained in this directory. You are free to explore these, and even make use of the plotting format generate new graphs. The images I used for my thesis have also been left included in this directory.

## Installation and Getting Started

The project includes several dependencies that need to be installed before the solvers can be run, and they are listed in the ``requirements.txt`` file in the root directory. 

#### __Virtual Environment Setup__

As per usual, it is recommended that a virtual environment is set up as a wrapper for these dependencies, and activated each time before running the solvers.

Visit this [link](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/) for instructions on how to install `venv`. Once installed, navigate to your project root directory and create a virtual environment with: 
```
bash python3 -m venv venv
``` 
for MacOS, or
```
py -m venv venv
```
for Windows.

To activate the virtual environment, run:
``` bash
source ./venv/bin/activate
``` 
This needs to be done each time a new terminal is initialised for running code. You will know that the virtual environment is active if a ``(venv)`` appears to the left of the terminal cursor.

#### __Installing Dependencies__

To install the project dependencies, navigate to the root directory with the virtual environment activated and run:

```bash
pip3 install -r requirements.txt
```
After the installation completes, you should see the list of installed dependencies in the ``bin`` folder of the ``venv`` directory.

## Running the Solvers

There is a ``bash`` script in the root directory that will automatically activate the virtual environment and run both solvers consecutively. To use it, simply navigate to the root directory and run:

```bash
bash runsolvers.sh
```

Information about solver's status and will be displaeyd in the terminal as both solvers are running. Once the algorithms have finished, data for both solvers will be saved into the ``results`` directory with the naming convention as described above.

If you wish to run each solver individually, simply navigate to the respective solver directory and run either 
```bash
python3 initialisation.py 
```
for the sparse solver, or
```bash
python3 example.py 
```
for the dense solver.

You can easily change the number of independent variables and the relaxation order for the problem you wish to solve from the first few lines of code. The respective problem size must be contained within the ``popData`` directory.

##### Thank you for visitng my project!