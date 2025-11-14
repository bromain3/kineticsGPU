# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 16:19:12 2024

@author: bromain
"""
####### After testing, it seems that cantera implementation extend ranges of validity of Nasa7 polynomials, even if is not valid strictly speaking
# optimisation du temps de calcul (95% du temps de calcul vient de l'ajout de la réaction backwards, et donc probablement de la façon dont sont gérées les tables thermo. Il serait peut être judicieux de faire des parties en C ou en Fortran genre les tables termo et les elementary reactions)
import numpy as np


R = 8.314_462_618_153_24  # SI
Pref = 101325.0

class MM:
    O = 15.999e-3
    N = 14.0067e-3
    Ar = 39.948e-3
    H = 1.00784e-3
    C = 12.011e-3


class ThermoData:
    ######## something has to be done about interpolation ranges
    # This class is far form being optimized. This issue should be adressed because the downgrade of performances caused by it is huge
    # Can be done by precomputing values and using linear interpolation, or by doing the computation in a subroutine in C or Fortran
    MODEL = 'Model'
    DATA = 'data'
    TRANGE = 'TRange'
    LOWTDATA = 'Low T data'
    HIGHTDATA = 'High T data'
    MolarMass = 'MolarMass'
    Composition = 'Composition'
    
    def __init__(self):
        self.__Species = {}
        
    def AddSpeciePorperty(self, SpecieName, Composition, ModelType, **kwargs):
        if ModelType == 'NASA7':
            if 'TRange' not in kwargs or 'data' not in kwargs:
                msg = 'The temperature range of data or data itself is missing'
                raise ValueError(msg)
            self.__Species[SpecieName] = {ThermoData.MODEL: 'NASA7',
                                          ThermoData.MolarMass: sum([number * getattr(MM, atom) for number, atom in Composition]),
                                          ThermoData.TRANGE: kwargs[ThermoData.TRANGE],
                                          ThermoData.LOWTDATA: np.array(kwargs[ThermoData.DATA][0]),
                                          ThermoData.HIGHTDATA: np.array(kwargs[ThermoData.DATA][1]),
                                          ThermoData.Composition: Composition,
                                          }
        else:
            msg = 'The thermodynamics model named {} is not understood'.format(ModelType)
            raise KeyError(msg)

    @property
    def Species(self):
        return(self.__Species)

    @property
    def speciesNames(self):
        return(list(self.__Species.keys()))

    def __CheckIfSpecieExists(self, SpecieName):
        if SpecieName not in self.__Species:
            msg = 'Thermodynamics data of {} has not been defined but is required for the simulated system'.format(SpecieName)
            raise KeyError(msg)

    def cp(self, SpecieName, T, P):
        # Independant of P for ideal gas
        self.__CheckIfSpecieExists(SpecieName)
        if self.__Species[SpecieName][ThermoData.MODEL] == 'NASA7':
            Temp = np.array([1,T,T**2,T**3,T**4])
            Tmin, Tmid, Tmax = self.__Species[SpecieName][ThermoData.TRANGE]
            if T <= Tmid and T >= Tmin:
                coef = self.__Species[SpecieName][ThermoData.LOWTDATA][:5]
                return(np.sum(R * coef * Temp))
            else:#if T <= Tmax:
                coef = self.__Species[SpecieName][ThermoData.HIGHTDATA][:5]
                return (np.sum(R * coef * Temp))
            # else:
            #     raise ValueError(f'Impossible to interpolate cp out of the range {Tmin} K - {Tmax} K')

    def cp_R(self, specieName, T, P):
        return(self.cp(specieName, T, P) / R)


    def cv(self, SpecieName, T, P):
        return(self.cp(SpecieName, T, P) - R)
    
    def h(self, SpecieName, T, P):
        # dH/dP = V (1-alpha*T) = 0 for perfect gaz as alpha = 1/T
        self.__CheckIfSpecieExists(SpecieName)
        if self.__Species[SpecieName][ThermoData.MODEL] == 'NASA7':
            Temp = np.array([1, T/2, T ** 2/3, T ** 3/4, T ** 4/5, 1/T])
            Tmin, Tmid, Tmax = self.__Species[SpecieName][ThermoData.TRANGE]
            if T <= Tmid and T >= Tmin:
                coef = self.__Species[SpecieName][ThermoData.LOWTDATA][:6]
                return (np.sum(R * T * coef * Temp))
            else:#if T <= Tmax:
                coef = self.__Species[SpecieName][ThermoData.HIGHTDATA][:6]
                return (np.sum(R * T * coef * Temp))
            # else:
            #     raise ValueError(f'Impossible to interpolate h out of the range {Tmin} K - {Tmax} K')

    def h_RT(self, specieName, T, P):
        return(self.h(specieName, T, P) / R / T)

    def s(self, SpecieName, T, P):
        self.__CheckIfSpecieExists(SpecieName)
        if self.__Species[SpecieName][ThermoData.MODEL] == 'NASA7':
            Temp = np.array([np.log(T), T , T ** 2 / 2, T ** 3 / 3, T ** 4/4, 0, 1])
            Tmin, Tmid, Tmax = self.__Species[SpecieName][ThermoData.TRANGE]
            if T <= Tmid and T >= Tmin:
                coef = self.__Species[SpecieName][ThermoData.LOWTDATA]
                return (np.sum(R * coef * Temp) - R * np.log(P/Pref))
            else:#if T <= Tmax:
                coef = self.__Species[SpecieName][ThermoData.HIGHTDATA]
                return (np.sum(R * coef * Temp) - R * np.log(P/Pref))
            # else:
            #     raise ValueError(f'Impossible to interpolate s out of the range {self.Tmin} - {self.Tmax} K')

    def s_R(self, SpecieName, T, P):
        return(self.s(SpecieName, T, P) / R)

    def g(self, SpecieName, T, P):
        return(self.h(SpecieName, T, P) - self.s(SpecieName, T, P) * T)
    
    def g0(self, SpecieName, T):
        P = Pref
        return(self.h(SpecieName, T, P) - self.s(SpecieName, T, P) * T)
    
    def MM(self, SpecieName):
        return(self.__Species[SpecieName][ThermoData.MolarMass])
    
    
if __name__ == '__main__':
    import cantera as ct
    import matplotlib.pyplot as plt
    
    th = ThermoData()
    th.AddSpeciePorperty('O', [(1, 'O')], 'NASA7', TRange=[200.0, 1000.0, 3500.0], data= [[3.1682671, -3.27931884e-03, 6.64306396e-06, -6.12806624e-09, 2.11265971e-12, 2.91222592e+04, 2.05193346],[2.56942078, -8.59741137e-05, 4.19484589e-08, -1.00177799e-11, 1.22833691e-15, 2.92175791e+04, 4.78433864]])
    th.AddSpeciePorperty('N', [(1, 'N')], 'NASA7', TRange=[200.0, 1000.0, 6000.0], data= [[2.5, 0.0, 0.0, 0.0, 0.0, 5.6104637e+04, 4.1939087],[2.4159429, 1.7489065e-04, -1.1902369e-07, 3.0226245e-11, -2.0360982e-15, 5.6133773e+04, 4.6496096]])
    th.AddSpeciePorperty('NO', [(1, 'O'), (1, 'N')], 'NASA7', TRange=[200.0, 1000.0, 6000.0], data= [[4.2184763, -4.638976e-03, 1.1041022e-05, -9.3361354e-09, 2.803577e-12, 9844.623, 2.2808464],[3.2606056, 1.1911043e-03, -4.2917048e-07, 6.9457669e-11, -4.0336099e-15, 9920.9746, 6.3693027]])
    th.AddSpeciePorperty('N2', [(2, 'N')], 'NASA7', TRange=[300.0, 1000.0, 5000.0], data= [[3.298677, 1.4082404e-03, -3.963222e-06, 5.641515e-09, -2.444854e-12, -1020.8999, 3.950372],[2.92664, 1.4879768e-03, -5.68476e-07, 1.0097038e-10, -6.753351e-15, -922.7977, 5.980528]])
    th.AddSpeciePorperty('NO2', [(2, 'O'), (1, 'N')], 'NASA7', TRange=[300.0, 1000.0, 6000.0], data= [[3.9440312, -1.585429e-03, 1.6657812e-05, -2.0475426e-08, 7.8350564e-12, 2896.6179, 6.3119917],[4.8847542, 2.1723956e-03, -8.2806906e-07, 1.574751e-10, -1.0510895e-14, 2316.4983, -0.11741695]])
    th.AddSpeciePorperty('AR', [(1,'Ar')], 'NASA7', TRange=[300.0, 1000.0, 5000.0], data=[[2.5, 0.0, 0.0, 0.0, 0.0, -745.375, 4.366],[2.5, 0.0, 0.0, 0.0, 0.0, -745.375, 4.366]])
    th.AddSpeciePorperty('O2', [(2,'O')], 'NASA7', TRange=[300.0, 1000.0, 3500.0], data=[[3.78245636, -2.99673416e-03, 9.84730201e-06, -9.68129509e-09, 3.24372837e-12, -1063.94356, 3.65767573],[3.28253784, 1.48308754e-03, -7.57966669e-07, 2.09470555e-10, -2.16717794e-14, -1088.45772, 5.45323129]])
    th.AddSpeciePorperty('N2O', [(1,'O'), (2, 'N')], 'NASA7', TRange=[200.0, 1000.0, 6000.0], data=[[2.2571502, 0.011304728, -1.3671319e-05, 9.6819806e-09, -2.9307182e-12, 8741.7744, 10.757992],[4.8230729, 2.6270251e-03, -9.5850874e-07, 1.6000712e-10, -9.7752303e-15, 8073.4048, -2.2017207]])
    th.AddSpeciePorperty('HO2', [(1,'H'), (2, 'O')], 'NASA7', TRange=[200.0, 1000.0, 3500.0], data=[[4.30179801, -4.74912051e-03, 2.11582891e-05, -2.42763894e-08, 9.29225124e-12, 294.80804, 3.71666245],[4.0172109, 2.23982013e-03, -6.3365815e-07, 1.1424637e-10, -1.07908535e-14, 111.856713, 3.78510215]])
    th.AddSpeciePorperty('H', [(1,'H')], 'NASA7', TRange=[200.0, 1000.0, 3500.0], data=[[2.5, 7.05332819e-13, -1.99591964e-15, 2.30081632e-18, -9.27732332e-22, 2.54736599e+04, -0.446682853],[2.50000001, -2.30842973e-11, 1.61561948e-14, -4.73515235e-18, 4.98197357e-22, 2.54736599e+04, -0.446682914]])

    P = 100e5
    gas = ct.Solution('air.yaml')
    gas.TP = 300, P
    print(gas.species_names)
    print(gas.T)
    print(gas.P)
    
    
    T = np.linspace(300,6000,1000)
    
    ###############################
    ####### Visual analysis #######
    ###############################
    for i, name in enumerate(gas.species_names):
        fig, ax = plt.subplots(2, 2)
        fig.suptitle('Visual comparison of thermodynamical data\n computed by Canter and by pol-ene-kinetics for specie ' + name)
        ct = []
        hm = []
        for t in T:
            gas.TP = t, P
            ct.append(gas.standard_gibbs_RT[i])
            hm.append(th.g(name,t,P)/R/t)
        ax[0,0].plot(T, ct,label='Cantera')
        ax[0,0].plot(T, hm,label='pol-ene-kinetics')
        ax[0,0].legend()
        ax[0,0].set_title('Gibs energy')
        
        ct = []
        hm = []
        for t in T:
            gas.TP = t, P
            ct.append(gas.standard_entropies_R[i])
            hm.append(th.s(name,t,P)/R)
        ax[0,1].plot(T, ct,label='Cantera')
        ax[0,1].plot(T, hm,label='pol-ene-kinetics')
        ax[0,1].legend()
        ax[0,1].set_title('Entropy')
        
        ct = []
        hm = []
        for t in T:
            gas.TP = t, P
            ct.append(gas.standard_enthalpies_RT[i])
            hm.append(th.h(name,t,P)/R/t)
        ax[1,0].plot(T, ct,label='Cantera')
        ax[1,0].plot(T, hm,label='pol-ene-kinetics')
        ax[1,0].legend()
        ax[1,0].set_title('Enthalpy')
        
        ct = []
        hm = []
        for t in T:
            gas.TP = t, P
            ct.append(gas.standard_cp_R[i])
            hm.append(th.cp(name,t,P)/R)
        ax[1,1].plot(T, ct,label='Cantera')
        ax[1,1].plot(T, hm,label='pol-ene-kinetics')
        ax[1,1].legend()
        ax[1,1].set_title('cp')
        
        
    
    ###################################
    ####### Stochastic analysis #######
    ###################################
    import random as rd
    nb_spe = len(gas.species_names)
    mean_P, std_P = 50e5, 35e5
    mean_T, std_T = 1500, 500
    nb_computation = 1000
    ERRH = []
    ERRS = []
    ERRG = []
    ERRCP = []
    Ps = []
    Ts = []
    for _ in range(nb_computation):
        i, P, T = rd.randint(0, nb_spe-1), max(1e3, rd.gauss(mean_P, std_P)), max(300, min(rd.gauss(mean_T, std_T), 3499))
        Ps.append(P)
        Ts.append(T)
        gas.TP = T, P
        name = gas.species_names[i]
        ERRH.append(abs(th.h(name,T,P)/R/T - gas.standard_enthalpies_RT[i] / gas.standard_enthalpies_RT[i]))
        ERRS.append(abs(th.s(name,T,P)/R - gas.standard_entropies_R[i] / gas.standard_entropies_R[i]))
        ERRG.append(abs(th.g(name,T,P)/R/T - gas.standard_gibbs_RT[i] / gas.standard_gibbs_RT[i]))
        ERRCP.append(abs(th.cp(name,T,P)/R - gas.standard_cp_R[i] / gas.standard_cp_R[i]))
    fig, ax = plt.subplots()
    a = ax.scatter(Ts, Ps, c=ERRH)
    fig.colorbar(a)
    fig, ax = plt.subplots()
    a = ax.scatter(Ts, Ps, c=ERRS)
    fig.colorbar(a)
    fig, ax = plt.subplots()
    a = ax.scatter(Ts, Ps, c=ERRG)
    fig.colorbar(a)
    fig, ax = plt.subplots()
    a = ax.scatter(Ts, Ps, c=ERRCP)
    fig.colorbar(a)
    plt.show()
    
     
    print('Correlation coefficient of Temperature and enthalpy difference: {}'.format(np.corrcoef(Ts, ERRH)[0,1]**2))
    print('Correlation coefficient of Temperature and entropy difference: {}'.format(np.corrcoef(Ts, ERRS)[0,1]**2))
    print('Correlation coefficient of Temperature and Gibbs free energy difference: {}'.format(np.corrcoef(Ts, ERRG)[0,1]**2))
    print('Correlation coefficient of Temperature and cp difference: {}'.format(np.corrcoef(Ts, ERRCP)[0,1]**2))
    print('Correlation coefficient of Pressure and enthalpy difference: {}'.format(np.corrcoef(Ps, ERRH)[0,1]**2))
    print('Correlation coefficient of Pressure and entropy difference: {}'.format(np.corrcoef(Ps, ERRS)[0,1]**2))
    print('Correlation coefficient of Pressure and Gibbs free energy difference: {}'.format(np.corrcoef(Ps, ERRG)[0,1]**2))
    print('Correlation coefficient of Pressure and cp difference: {}'.format(np.corrcoef(Ps, ERRCP)[0,1]**2))