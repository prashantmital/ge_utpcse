import matplotlib.pyplot as plt
from math import log
def parse_screenout(outfile):
  f_open = outfile
  f_read = open(f_open,'r')
  data = f_read.readlines()
  times = []

  for str in data:
      if (str.find("took")!=-1):
	  tmp = str.split()[3]
	  times.append(float(tmp))
  
  return times

def compute_speedup(serial_time,exec_times):
  return_time = [serial_time/x for x in exec_times]
  return return_time

def compute_diff(times1,times2):
  return_times = [t1-t2 for t1,t2 in zip(times1,times2)]
  return return_times
#---------------------------------------------------------------
#----------------PLOTTING ROUTINE FOR SPEEDUP-------------------
#--Feed the input file names to the function calls below and  --
#--let the magic unfold. No boundary case checks may mean     --
#--that script produces unexpected results                    --
#-----------------Written By: Prashant Mital--------------------
#---------------------------------------------------------------
data_1 = parse_screenout("screenout.weak_scaling")
i=0
data_1[0] = 1
data_1[1] = (0.202036/data_1[1])/4
data_1[2] = (2.255988/data_1[2])/16
     
data_2 = parse_screenout("strong_scaling_problem_2.out")
#data_2 = [2.165706/x for x in data_2]
p1,=plt.plot([4,16],data_1[1::],'b',linewidth=1.5)
#p1,=plt.plot([1,2,4,8,16,32,64],data_2[:],'r',label="Problem 2",linewidth=1.5)
#p1,=plt.plot([1,2,4,8,16,32,64],[1,2,4,8,16,32,64],'g',label="Idealized Speedup",linewidth=1.5)
plt.legend(loc=0)
plt.xlabel("NPROC")
plt.ylabel("Efficiency [Speedup/P]")
plt.title("Weak Scaling of Gaussian Elimination")
plt.grid(True)
plt.show()


