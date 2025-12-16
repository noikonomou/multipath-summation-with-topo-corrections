# GPR data multipath-summation-with-topo-corrections


Instructions on how to use the m files are in the main script within the directory "code".
For the implementation of multi-path summation we use:
1. The function "topomig2d_varV" downloaded from https://github.com/tinawunderlich/MultichannelGPR, under the GNU 3.0 licence and
2. The package "hyperbola recognition" downloaded from https://github.com/lweileeds/hyperbola_recognition.
The former is used to detect hyperbolas within the GPR section, which are used for GPR velocities estimation and multipath summation and the latter to apply migrations and or topographic corrections.
