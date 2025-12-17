# GPR data multipath-summation-with-topo-corrections


Instructions on how to use the m files are in the main script within the directory "Code".
For the implementation of multi-path summation we use:
1. The package "hyperbola recognition", which can be downloaded from https://github.com/lweileeds/hyperbola_recognition and
2. The function "topomig2d_varV", which can be downloaded from https://github.com/tinawunderlich/MultichannelGPR, under the GNU 3.0 licence.

The former is used to detect hyperbolas within the GPR section. The hyperbolas are used for GPR velocities estimation which in their turn are invlolved in a multipath summation scheme. The latter is used to apply topographic corrections.
