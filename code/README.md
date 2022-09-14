To recreate the results found in Table VI and VII in the manuscript:

Run the python files in the follow order:
- run_evaluate_on_multiple_experiment_folders.py
- collect_results_from_single_cross_validations_run.py

The tables will be stored as csv files in the folder: "../experiments_k1_maxpool_batchNorm_lr1e-03_noise0e+00/net_102/summary/"


**Note**: Throughout the code the two time events 
- Aortic valve opening: "AVO" and "ed"
- Aortic valve closure: "AVC" and "es"
