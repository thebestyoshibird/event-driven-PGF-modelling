# Purpose 

This code implements event-driven simulations of the impact of three probabilities (the probability of becoming homeless, the probability of exiting homelessness, and the probability of a night of homelessness being undetected/unrecorded within an HMIS systems) on systemwide counts of homelessness and chronic homelessness (consecutive versus episodic).

# Use 

To run the simulations and generate a dataset for visualization, run `$ python generate_dataset.py`

To create visualizations, use the viz_chron script with the name of the preferred viz as an argument. For more information and available chart types, run `$ python viz_chron.py help`

Resulting visualizations will save to the indicated folder within the directory housing the code on your local machine - generally either an images or data subfolder. Note that a full set of 10 simulations run on all 5 values for p_e, p_h, and p_u and initialized with 20,000 Persons will generate a file too large to upload to this free repo.

## Development

Install required packages:

`$python -m pip install -r requirements.txt`
