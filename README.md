# GPR data multipath-summation-with-topo-corrections


Instructions on how to use the m files are in the main script "Script_for_Rustaq_paper.m". It was created in Matlab 2022b. Copy the m-file together with all the files above in the same folder and run it in Matlab. 

For the implementation of multi-path summation we also involve the package "hyperbola recognition", which can be downloaded from https://github.com/lweileeds/hyperbola_recognition.

This package is used to automatically detect hyperbolas within the GPR section. The hyperbolas are used for GPR velocities estimation which in their turn are invlolved in a multipath summation scheme, which finilizes to topographic corrections.
