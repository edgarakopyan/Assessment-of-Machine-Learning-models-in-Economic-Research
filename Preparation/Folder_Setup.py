import os

dir = os.getcwd()
os.mkdir(dir + "/Results")
os.mkdir(dir + "/Graphs_and_Tables")
os.mkdir(dir + "/ACIC")
for i in range(1, 78):
    os.mkdir(dir + "/ACIC/" + str(i))