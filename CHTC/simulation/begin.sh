# Shell script for launching all jobs and cleaning up past runs

# Cleaning up old log files
rm -r log_files/*.log

# Launching all of the jobs 
condor_submit run_sim.sub 
condor_submit run_sim90.sub 
condor_submit run_sim180.sub
condor_submit run_sim270.sub