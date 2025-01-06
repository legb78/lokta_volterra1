import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.integrate import odeint 

df = pd.read_csv("populations_lapins_renards.csv")
df_cp = df.copy()
df_cp["days"] = range(1, 1001)

days = np.arange(0, 1001, 1 )

alpha = 2/3 # taux de reproduction 
beta = 4/3 # taux de mortalitée 
delta = 1 # de reproduction des prédateurs 
gamma = 1 #taux de mortalitée des prédateurs 
step = 0.001 
time = [0]

params = [alpha,beta,delta,gamma,step]


rabbit = [1]
fox = [2]

     
alpha = params[0]
beta  = params[1]
delta = params[2]
gamma = params[3]
    
   
for valo  in range(100_000):
    dt = time[-1] + step 
    dx = (rabbit[valo-1] * (alpha - beta * fox[valo-1])) * step + rabbit[-1]
    dy = (fox[valo-1] * (delta * rabbit[valo-1] - gamma)) * step + fox[-1]
        
    time.append(dt)
    rabbit.append(dx)
    fox.append(dy)

np_time = np.array(time)
np_rabbit = np.array(rabbit)
np_fox = np.array(fox)

rabbit_1000 = [np_rabbit[::1000]]
print((rabbit_1000*1000))
  
  
#for value in range  ()    
'''
plt.figure(figsize=(10,8))
plt.plot(time, rabbit, color= "red")
plt.plot(time, fox, color= "blue")
plt.show()
 '''
        
        
    
        
 
        



