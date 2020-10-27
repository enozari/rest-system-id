
This repository contains the MATLAB codes used for data processing and simulations in the article referenced below. The methods of system identification discussed therein are available under 'System ID Methods', the codes used for the simulation of the linearizing effects of macroscopic neurodynamics and neuroimaging can be found in 'Macroscopic Linearity Simulation', and the piece of code used for generating arbitrarily distinct linear systems with almost identical functional connectivity is placed in 'Generating Systems with Identical FC'. The utility codes are available in 'Utilities'.

## Licensing and Reference

The software available in this repository is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License. If you use codes from this repository for your work, please provide reference to the article

* Nozari A, Bertolero M A, Cornblath E J, Mahadevan A, Caciagli L, He X, Pappas G J, and Bassett D S, "Is the brain macroscopically linear? A system identification of resting state dynamics", 2020.

## External packages and files:

The following MATLAB packages should be downloaded and installed (added to MATLAB path using the addpath() function) before running the codes in this repository:

* MINDy package for neural mass model identification, available at https://github.com/singhmf/MINDy-Beta
* MATLAB distributionPlot package, available at https://www.mathworks.com/matlabcentral/fileexchange/23661-violin-plots-for-plotting-multiple-distributions-distributionplot-m
* MATLAB Hatchfill package, available at https://www.mathworks.com/matlabcentral/fileexchange/30733-hatchfill
* MATLAB Arrow3 package, available at https://www.mathworks.com/matlabcentral/fileexchange/14056-arrow3
* MATLAB export_fig package, available at https://www.mathworks.com/matlabcentral/fileexchange/23629-export_fig
* npy-matlab package, available at https://github.com/kwikteam/npy-matlab, even though it is only used for reading resting scan files which are not included in this distribution

Also, the following files need to be downloaded and available to the MATLAB functions (either placed in the same directory or in a directory on the MATLAB path):

* The parcel specifications for the Schaefer 100x7 parcellation, available at https://github.com/ThomasYeoLab/CBIG/blob/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI/Schaefer2018_100Parcels_7Networks_order.txt
* Color codes for the 7 resting state networks, available at https://github.com/jimmyshen007/NeMo/blob/master/resource/Atlasing/Yeo_JNeurophysiol11_MNI152/Yeo2011_7Networks_ColorLUT.txt

Finally, the file main_fmri.m assumes that preprocessed HCP S1200 rsfMRI time series are available in an 'HCP' subdirectory. The data is not included in this distribution but is publicly available at https://db.humanconnectome.org and our preprocessing pipeline is described in the Methods section of the above reference. Similarly, the file main_ieeg.m assumes that preprocessed RAM rsiEEG time series are available in the rs_5min/rand_segments subdirectory, which are not included in this distribution but publicly available at http://memory.psych.upenn.edu/RAM.
