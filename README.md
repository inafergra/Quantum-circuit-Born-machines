# Training Quantum Circuit Born Machines

In this project a Quantum Circuit Born Machine (QCBM) was implemented and trained. The main goal was to test the effects of sparsity as a regularization method. For a detailed description of the project please read the Report.pdf.

## File description
- `bas_dataset.py`: Generates the Bars and Stripe (BAS) dataset.
- `learning_functions.py`: Contains functions related with the training and benchmarking of the QCBM. 
- `circuit_design.py`: Contains functions related with the construction of the variational circuit.
- `shallow_qcbm.py`: Code that trains a deep QCBM to generate the output probability distribution of another shallow QCBM.
- `bars_and_stripes.py`: Code that trains a deep QCBM to generate the BAS dataset.


## Requirements
- Python 3
- cirq
- numpy
- scipy
- matplotlib

## Author
- Ignacio Fernández Graña
