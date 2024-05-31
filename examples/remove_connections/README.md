# Connection Removal Example

In this usecase, we will remove connections from existing SONATA circuits. 

- <u>Example 2.1</u>: [Small Test Circuit Model from BlueCelluLab](./test_circuit/)
- <u>Example 2.2</u>: [Primary Somatosensory Cortex Circuit Model](./sscx/)

The first example is standalone, so you don’t need to download any circuit, and it can be used for testing and understanding the connectome-manipulator. The second example is more complex and requires HPC infrastructure. Depending on your use case, you can adapt either example to your circuit of interest.


Within each example, you will find:

	•	A notebook to generate configurations.
	•	manip_config.json to provide parameters for connectome manipulation.
	•	structcomp_config__Orig_vs_Manip.json to be populated with parameters using the notebook.
	•	structcomp_config.json generated from the template JSON file by the notebook, to run Structural Comparator after manipulation.
	•	Bash scripts to initiate manipulation after config generation.
	•	A logs directory to store manipulator logs.
	•	An output directory to store the manipulated connectome

## Steps:

	1.	Follow the notebook example to generate configs for manipulation and connectome comparison.
	2.	Run run_rewiring.sh with the desired output directory, manipulation config path, and number of splits for parallelization.
	3.	Monitor the logs after submitting the job.
	4.	Once the job has finished, check your specified output directory for circuit_config.json.
	5.	(Optional) Run run_struct_comparison.sh to compare connectomes.
    
    
Copyright (c) 2024 Blue Brain Project/EPFL