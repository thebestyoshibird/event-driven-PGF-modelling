This code implements event-driven simulations of the impact of three probabilities (the probability of becoming homeless, the probability of exiting homelessness, and the probability of a night of homelessness being undetected/unrecorded within an HMIS systems) on systemwide counts of homelessness and chronic homelessness (consecutive versus episodic).

To run the simulations and generate a dataset for visualization, open viz_chron.py. Uncomment the following lines of code, then run: [1, 208]. (Comment out all other lines) Note that the code is seeded for replicability while adjusting visualization code.

To load the dataset in the future without running the simulation, you can comment out lines [193, 212]. Uncomment lines below 212 as desired in order to generate the targeted visualization. Uncomment and adjust line 217 only if you wish to slice the DataFrame along a specified value or values for a probability.

Resulting visualizations will save to the indicated folder within the director housing the code on your local machine - generally either an images or data subfolder. Note that a full set of 10 simulations run on all 5 values for p_e, p_h, and p_u and initialized with 20,000 Persons will generate a file too large to upload to this free repo.

## Development

Install required packages:

`$python -m pip install -r requirements.txt`
