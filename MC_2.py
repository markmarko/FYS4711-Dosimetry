import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math
from statistics import stdev
#---------------------------------- Task 4 -------------------------------------

nv = 3.43e23

Emax = 2000
Emin = 50

x = np.logspace(-2, 3, 100)

alpha = x/0.511

sigma_e = 2*math.pi*(2.818e-13)**2*((1+alpha)/(alpha**2)*(2*(1+alpha)/(1+2*alpha)-np.log(1+2*alpha)/alpha)+np.log(1+2*alpha)/alpha/2-(1+3*alpha)/((1+2*alpha)**2))

sigma_O_mass = 6.022e23*8/16*sigma_e
sigma_H_mass = 6.022e23*1/1*sigma_e

sigma_tot = 2/18*sigma_H_mass+16/18*sigma_O_mass


plt.plot(x,sigma_e,'r--',label='Electronic cross-section')
plt.xlabel('Energy, MeV')
plt.ylabel('$\sigma_e$, cm$^2$')
plt.title('Electronic cross section for Compton scattering')
plt.xscale("log")
plt.yscale("log")
plt.grid(True)
plt.xlim(0.01,500)
plt.ylim(1e-27,1e-24)
plt.legend()
plt.show()

plt.plot(x,sigma_tot,'r--',label='Attenuation coefficient $\mu$')
plt.xlabel('Energy, MeV')
plt.ylabel('$\mu$, cm$^{-1}$')
plt.title('Attenuation coefficient, Compton scattering, H$_2$O')
plt.xscale("log")
plt.yscale("log")
plt.plot([2,2],[0.0001,0.049],"--",color="cadetblue", linewidth=1)
plt.plot([0.0001,2],[0.049,0.049],"--",color="cadetblue", linewidth=1)
plt.text(2.2, 0.002, r'2 MeV photons', fontsize=13,color="cadetblue")
plt.plot([0.2,0.2],[0.0001,0.137],"--",color="indigo", linewidth=1)
plt.plot([0.0001,0.2],[0.137,0.137],"--",color="indigo", linewidth=1)
plt.text(0.22, 0.004, r'200 keV photons', fontsize=13,color="indigo")
plt.grid(True)
plt.xlim(0.01,500)
plt.ylim(0.0001,0.3)
plt.legend()
plt.show()

alpha = 2/0.511

sigma_e_2 = 2*math.pi*(2.818e-13)**2*((1+alpha)/(alpha**2)*(2*(1+alpha)/(1+2*alpha)-np.log(1+2*alpha)/alpha)+np.log(1+2*alpha)/alpha/2-(1+3*alpha)/((1+2*alpha)**2))

sigma_O_mass_2 = 6.022e23*8/16*sigma_e_2
sigma_H_mass_2 = 6.022e23*1/1*sigma_e_2

sigma_tot_2 = 2/18*sigma_H_mass_2+16/18*sigma_O_mass_2

print(sigma_tot_2)

#----------------------------------------------- Task 5 --------------------------------------------

s = np.linspace(0, 40, 100)

PDF_2 = sigma_tot_2*np.exp(-s*sigma_tot_2) 
CDF_2 = 1 - np.exp(-s*sigma_tot_2) 

plt.plot(s,PDF_2,"--",color = 'lightseagreen',label='PDF, f(s)')
plt.plot(s,CDF_2,"--",color="darkorange",label='CDF, F(s)')
plt.xlabel('s, cm')
plt.ylabel('PDF/CDF value, au')
plt.title('PDF and CDF for 2 MeV photons')
plt.grid(True)
plt.xlim(0,40)
plt.ylim(0,1)
plt.legend(loc = 'upper left')
plt.show()


alpha = 0.2/0.511

sigma_e_3 = 2*math.pi*(2.818e-13)**2*((1+alpha)/(alpha**2)*(2*(1+alpha)/(1+2*alpha)-np.log(1+2*alpha)/alpha)+np.log(1+2*alpha)/alpha/2-(1+3*alpha)/((1+2*alpha)**2))

sigma_O_mass_3 = 6.022e23*8/16*sigma_e_3
sigma_H_mass_3 = 6.022e23*1/1*sigma_e_3

sigma_tot_3 = 2/18*sigma_H_mass_3+16/18*sigma_O_mass_3

print(sigma_tot_3)


PDF_3 = sigma_tot_3*np.exp(-s*sigma_tot_3) 
CDF_3 = 1 - np.exp(-s*sigma_tot_3) 

plt.plot(s,PDF_3,"--",color = 'lightseagreen',label='PDF, f(s)')
plt.plot(s,CDF_3,"--",color="darkorange",label='CDF, F(s)')
plt.xlabel('s, cm')
plt.ylabel('PDF/CDF value, au')
plt.title('PDF and CDF for 200 keV photons')
plt.xlim(0,40)
plt.ylim(0,1)
plt.grid(True)
plt.legend(loc = 'upper left')
plt.show()

#----------------------------------------------- Task 6 --------------------------------------------

NumSim = 1000
Pathlengths = np.zeros(NumSim)

for i in range(NumSim):
	ksi = np.random.uniform(0,1,1)
	Pathlengths[i] = - np.log(1-ksi)/sigma_tot_2

print(np.mean(Pathlengths))
print(1/sigma_tot_2)
print(stdev(Pathlengths))

x_1 = np.arange(170)
y_1 = 74.61415*np.exp(-x_1/20.421772096725537)

plt.hist(Pathlengths, bins = 100, facecolor='lightseagreen', alpha=0.75,label='Photon distribution')
plt.plot(x_1,y_1,'r--',label='Predicted value')
plt.xlabel('Pathlength, cm')
plt.ylabel('Number of photons')
plt.title('Distribution of the pathlengths for 2 MeV photons')
plt.grid(True)
plt.xlim(0,140)
plt.legend()
plt.show()

Pathlengths = np.zeros(NumSim)

for i in range(NumSim):
	ksi = np.random.uniform(0,1,1)
	Pathlengths[i] = - np.log(1-ksi)/sigma_tot_3

print(np.mean(Pathlengths))
print(1/sigma_tot_3)
print(stdev(Pathlengths))

x_1 = np.arange(46)
y_1 = 55.17867*np.exp(-x_1/7.662871195243166)

plt.hist(Pathlengths, bins = 100, facecolor='lightseagreen', alpha=0.75,label='Photon distribution')
plt.plot(x_1,y_1,'r--',label='Predicted value')
plt.xlabel('Pathlength, cm')
plt.ylabel('Number of photons')
plt.title('Distribution of the pathlengths for 200 keV photons')
plt.grid(True)
plt.xlim(0,50)
plt.legend()
plt.show()




