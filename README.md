# Bachelor
mostly relevant models and their components for my thesis

transformer 3.1.py is the script for the model I am building to work with 'real' data.
please note that the dependecies are specified in the import statements at the beginning of this file

it will with a selection of 3 sensors (Vhub, Power, Omega) and 5 modes, all truncated to 1000 time steps, for the first wind turbine.
training is done on the U8 scenarion, validation on U12, as it represents similar operating conditions

the python scripts for generating the training and validation datasets are available here.
please note that in the transformer 3.1.py script there are absolute (!) paths indicated for these files
