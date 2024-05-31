# Connection Removal Example

In this usecase, we will remove connections from existing SONATA circuits. 

- <u>Example 2.1</u>: [Small Test Circuit Model from BlueCelluLab](./test_circuit/)
- <u>Example 2.2</u>: [Primary Somatosensory Cortex Circuit Model](./sscx/)

The first example is standalone so you dont need to download any circuit and it can be used for testing, and understanding the connectome-manipulator. The second example, has much more complexity which also requires HPC infastructure. Depending on your usecase, you can adapt either example to your circuit of interest.


Within each example, there is :

- a notebook to generate:
- manip_config.json to provide parameters for Connectome Manipulation
- structcomp_config__Orig_vs_Manip.json to be populated with parameters with the notebook
- structcomp_config.json as a result of notebook from template json file and to run Structural Comparator after the manipulation
- bash scripts to initiate the manipulation after config generation
- logs directory to store manipulator logs
- output directory to store the manipulated connectome

Steps:

1. Follow Notebook Example to generate configs for manipulation and connectome comparison
2. Run run_rewiring.sh with desired output directory, manipulation config path, and number of splits for parallelization
3. follow logs after job has been submitted. 
4. once job has finished, check your given output directory to see if you have a circuit_config.json
5. (Optional) run run_struct_comparison.sh to compare connectomes