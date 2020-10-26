# neighborhood-uncertainty-scores
Repository for the code of TKDD paper "Mitigating Class-Boundary Label Uncertainty to Reduce Both Model Bias and Variance"

Modeling code runs based on manifest files, which are simply text files in which each line describes an experiment. ```run_manifest.sh``` loops through the lines of the manifest file and runs each of the experiments. A sample manifest is included, and the results logs from which the figure in the paper were computed are included in the results folder. There should be an empty blank line at the bottom of the manifest.

Download the FMA small dataset (repo is here: https://github.com/mdeff/fma) and use the notebook in the compute_mel_spectrograms folder to create the data tensor for training (convert the songs into mel spectrograms, determine the number of samples to separate each song into, etc). This data tensor will be used for model training; the neighborhood scores are computed using code from the distances folder.

Some paths are hardcoded in load_data.py; point the root directory at the top of the file to the FMA top-level folder and the files should be where the code needs them. Store the provided distance matrix at ```{FMA_root}/distances/distance_matrix.npy``` to allow for score computation (when the sw flag=1). We recommend using the provided distance matrix, as computing the distances from scratch is expensive. We do include the code we used to obtain the distances in the distances folder with a series of python scripts and a shell script to parallelize their execution.