import os
import subprocess

dir = os.getcwd()

# Run R script to create the main table
path2script = dir + "/R files/Main_Results_Prep.R"
try:
    subprocess.call(['Rscript', path2script, dir])
except subprocess.CalledProcessError as e:
    raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))

# Run R script to create ATE RMSE graphs

path2script = dir + "/R files/ATE_RMSE.R"
try:
    subprocess.call(['Rscript', path2script, dir])
except subprocess.CalledProcessError as e:
    raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))

