# Training Quantum Circuit Born Machines

In this project a Quantum Circuit Born Machine (QCBM) was implemented and trained. The main goal was to test the effects of sparsity as a regularization method. For a detailed description of the project please read the Report.pdf.

## File description
- `bas_dataset.py`: Generates the Bars and Stripe (BAS) dataset.
- `learning_functions.py`: Contains functions related with the training and benchmarking of the QCBM. 
- `circuit_design.py`: Contains functions related with the construction of the variational circuit.
- `shallow_qcbm.py`: Code that trains a deep QCBM to generate the output probability distribution of another shallow QCBM.
- `bars_and_stripes.py`: Code that trains a deep QCBM to generate the BAS dataset.

## Usage
1. clone the repository `git clone https://gitlab.kwant-project.org/computational_physics/projects/Project-2---Ising_idonfernandezg_smitchaudhary_ysotiropoulos.git`
2. Run the simulation for your choice of algorithm
    * `metropolis_bool = True` in `collect_data.py` for Metropolis Algorithm.
    * `metropolis_bool = False` in `collect_data.py` for Wolff Algorithm.
3. Run `plots.py` to plot average absolute magnetisation, energy, susceptibility, and specific heat.
    * `method = 'metropolis'` in `plots.py` for Metropolis Algorithm.
    * `method = 'wolff'` in `plots.py` for Metropolis Algorith.
    
## Requirements
- Python 3
- numpy
- scipy
- matplotlib

## Author
- Ignacio Fernández Graña
