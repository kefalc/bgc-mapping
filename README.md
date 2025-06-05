# bgc-mapping
Repository for bivariate GPR Biogeochemical Mapping

Instructions for use in this document.

Repository for bivariate GPR Biogeochemical Mapping

Steps for Using Mapping Code with Synthetic Floats

Please note that this code is set up to run on the NERSC Perlmutter machine using Dask. To use on other machines, tweaks may have to be made.

Go to the data folder and open the save_floats_E3SM_mapping_idealized_distributions.ipynb script. In the second code block, set years, region, variables, and sampling frequency to select data from. If you are reducing data by profile, run the 13th code block to randomly select profiles. Set the percentage of data you want to keep. Otherwise run this block but set reduce_floats to FALSE. If you are reducing the amount of data by float, run the 17th cell and select the amount of floats you want to keep. Do this to map oxygen or another BGC variable.

Connect to remote computing resources by running the following line in your terminal: ./launch-dask-module-cpu.sh

Go to the ls-mean folder and open the run_ma_ts.ipynb script. Run this script to calculate the large scale mean and residuals for temperature and salinity. In the 5th code block, ensure that file names are correct. Set the test number. In the 6th code block, set the depth levels that you want to save data for in the range (0, 7).

Go to the ss-covparam folder and open the run_cov_ts.ipynb script. Run this script to calculate the covariance parameters and small scale anomalies. NOTE: The 4th code block can be skipped if you are using the remote clusters. In the 5th code block, ensure that the file names are correct. Set the test number (this should be the same as in step 3). In the 6th code block, set the depth levels that you want to save data for (this should be the same as in step 3).

Go to the gpr-mapping-data folder and open new_netcdf.ipynb. This script is needed to collocate T&S mean/anomalies with oxygen (or other BGC) observations and save only those mean/anomalies in a new file AND the locations of TS observations only in a new file which are needed for calculating oxygen (or other bgc) large scale means and covparams.

Go to the ls-mean folder and open run_mean_bgc.ipynb. Run this script to calculate the large scale mean and residuals for oxygen or other BGC parameter. In the 5th code block, ensure that file names are correct. Set the test number. In the 6th code block, set the depth levels that you want to save data for. This should be the same as in step 3.

For univariate oxygen, go to the ss-covparam folder and open the run_cov_bgc.ipynb script. Run this script to calculate the covariance parameters and small scale anomalies. In the 5th code block, ensure that the file names are correct. Set the test number (this should be the same as in step 3). In the 6th code block, set the depth levels that you want to save data for (this should be the same as in step 3).

For multivariate oxygen, go to the ss-covparam folder and open the run_cov_bgc_jax-SouthernOcean.ipynb file. Run this script to calculate multivariate covparams.

Note: There are also scripts to get Argo float data and use that in place of synthetic float data. This will have to be used in place of the files in step 1 if not using the NERSC machine. These files will put data in the appropriate format to run the rest of the code.
