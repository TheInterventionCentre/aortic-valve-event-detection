# aortic-valve-event-detection

- - -

Source code for recreating "Automatic Detection of Aortic Valve Events Using Deep Neural Networks on Cardiac Signals From Epicardially Placed Accelerometer"

- - -

The Paper: [https://ieeexplore.ieee.org/document/9792198](https://ieeexplore.ieee.org/document/9792198) \
Supplementary Material: https://theinterventioncentre.github.io/aortic-valve-event-detection/ \
The data is avaiable at PhysioNet: https://physionet.org/content/cardiac-accel-canine-porcine/1.0.0/

After downloading the data from PhysioNet, create a folder named 'data_json' at the root level of the repository (at the same level as the 'code' folder). Extract the zip file from PhysioNet, and copy the subfolders (AP, MK, MKCMS, MP, MV) into the 'data_json' folder. The tables VI and VII are reproduced by running the file 'run_evaluate_on_multiple_experiment_folders.py' and will be stored in folder 'experiments_k1_maxpool_batchNorm_lr1e-03_noise0e+00/net_102/summary'.
