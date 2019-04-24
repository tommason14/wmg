#!/usr/bin/env python3

"""
File: wmg_graphs.py
Date: Summer 2017
Author: Tom Mason
Email: tom.mason14@gmail.com
Description: Two styles of graph- determining recystallisation times at given temperatures, calculating Avrami exponents
"""

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.stats import linregress
import seaborn


class JMAK():
    """Output is a graph showing the classical Johnson-Mehl-Avrami-Kolmogorov
    recrystallisation kinetics"""

    @classmethod
    def get_time(cls):
        print()
        time = int(input('Time for recrystallisation (secs): '))
        return time

    @classmethod
    def getTemp(cls):
        """Asks user for annealing temperature"""
        print()
        temp = int(input('Recrystallisation temperature (\u00b0C): ')) + 273
        return temp

    @classmethod
    def getStrain(cls):
        """Asks user for number of layers and corresponding micro-strain values"""
        print()
        layers = int(input('How many layers? '))
        strainList = []
        for i in range(layers):
            strain = float(input('Strain for layer {}: '.format(i + 1)))
            strainList.append(strain)
        df1 = pd.DataFrame(strainList, columns = ['Strain'])
        return df1

    @classmethod
    def getK(cls, df1, temp):
        """Calculation of recrystallisation rate constant, k
        and recrystallisation incubation time.
        """
        RsList = []
        kList = []
        for i in df1['Strain']:
            exponent = (300000 / (8.3145 * temp))
            Rs = (6.76 * 10 ** -20) * (40000) * (float(i) ** -4) * math.exp(exponent)
            # print('Rs for layer {} = {}'.format(i, Rs))
            RsList.append(Rs)
            k = -math.log(0.95) / (Rs ** 3)
            # print('k for layer {} = {}'.format(i, k))
            kList.append(k)
        df1['Rs'] = RsList
        df1['k'] = kList
        return df1

    @classmethod
    def calcWeightings(cls, df1):
        """Calculates weighting coefficients for weighted recrystallisation
        fractions- used to calculate average recrystallisation fraction of
        heterogeneous samples"""
        total = 0
        weightingList = []
        for i in df1['k']:
            total += i
        for i in df1['k']:
            w = i / total
            weightingList.append(w)
        df1['Weightings'] = weightingList
        return df1

    @classmethod
    def calcFractions(cls, df1, timespan=200):
        """Calculates recrystallisation fractions for each layer of sample.
        The default time is 200 seconds, can be changed in the main function."""
        times = np.arange(0, timespan, 0.1)
        df2 = pd.DataFrame(times)
        df2.columns = ['Time']
        i = 0
        for k in df1['k']:
            Frac = [1 - math.exp(-k * (t ** 3)) for t in times]
            df2['frac ' + str(i + 1)] = Frac
            i += 1
        return df2

    @classmethod
    def calcWeightedFrac(cls, df1, df2):
        """
        Calculates weighted recrystallisation fractions
        as weighting (from calcWeightings function) multiplied
        by recrystallisation fraction
        """
        
        i = 0
        while i < df1['Strain'].count():
            df2['weighted frac ' + str(i + 1)] = df2['frac ' + str(i + 1)] * df1.loc[i, 'Weightings']
            i += 1
        return df2

    @classmethod
    def aveFrac(cls, df2):
        """
        Sums all weighted fractions, returns the average
        recrystallisation fraction
        """

        filtered = df2.filter(regex='weighted')
        df2['average'] = filtered.sum(axis=1)
        return df2

    @classmethod
    def printRecryst(cls, df1, df2):
        """
        Plot of JMAK curve produced using matplotlib
        """
        
        d = {}
        print()
        for i in range(len(df1['Weightings'])):
            d["frac " + str(i + 1)] = [x for x in df2['frac ' + str(i + 1)]]
            valid = False
            while not valid:
                for j in d["frac " + str(i + 1)]:
                    if j > 0.5:
                        valid = True
                    else:
                        pass
            if valid:
                print('50% recrystallisation of layer {0} occurs at {1:.2f} seconds'.format(str(i + 1), np.interp(0.5, df2['frac ' + str(i + 1)], df2['Time'])))
            else:
                print('50% recrystallisation has not occured for layer {0}.'.format(str(i + 1)))

    @classmethod
    def plot(cls, df1, df2):
        """
        Plot of JMAK curve produced using matplotlib
        """

        if len(df1['Weightings']) == 1:
            plt.figure(1)
            plt.plot(df2['Time'], df2['frac 1'])

        if len(df1['Weightings']) > 1:
            for i in range(len(df1['Weightings'])):
                plt.figure(1)
                plt.plot(df2['Time'], df2['frac ' + str(i + 1)], label=str(i + 1))
                #print('50% recrystallisation for frac', str(i+ 1), 'occurs at','{0:.2f}'.format(np.interp(0.5, df2['frac ' + str(i + 1)], df2['Time'])), 'seconds')
            plt.plot(df2['Time'], df2['average'], '--', label='Average')
            plt.legend(loc='best')
        plt.title('JMAK Curve')
        plt.ylabel('Recrystallisation Fraction')
        plt.xlabel('Time (s)')
        return plt.show()

    @classmethod
    def avramiData(cls, df1, df2):
        """
        Asks user if Avrami data is desired, then calls method of the Avrami class containing the data.
        """
        
        if len(df1['Weightings']) > 1:
            print()
            
            correct = False
            while not correct:
                response = input('Would you like the averaged data using an Avrami Plot? (y/n) ')
                if response in ('y', 'n'):
                    correct = True
                else:
                    print("Please choose either 'y' or 'n'")
                  
            if response == 'y':
                calc = Avrami.calcAvramiExp(df2)  # calculates values for avrami plots
                Avrami.plot(calc)
                Avrami.avramiStats(calc)
            else:
                print('Thanks for using this program!')
           
    @classmethod
    def main(cls):
        # ask user for temp and strain values of layers
        r_time = JMAK.get_time()
        t = JMAK.getTemp()
        askuser = JMAK.getStrain()
        # calculate k and Rs
        df1_k = JMAK.getK(askuser, t)
        # calculate weightings for average fraction later on
        df1_weights = JMAK.calcWeightings(df1_k)
        # calculate recrystallisation fractions for a given time (default = 200)
        df2_fracs = JMAK.calcFractions(df1_weights, r_time)
        # calculate weighted recryst. fractions as fraction * weighting
        df2_weighted_fracs = JMAK.calcWeightedFrac(df1_weights, df2_fracs)
        # average recryst. fraction = sum(weighted fractions)
        df2_ave_frac = JMAK.aveFrac(df2_weighted_fracs)
        JMAK.printRecryst(df1_weights, df2_ave_frac)
        # plot using matplotlib
        JMAK.plot(df1_weights, df2_ave_frac)
        JMAK.avramiData(df1_weights, df2_ave_frac)


class Avrami:
    """
    Calculates the Avrami exponent, along with average recrystallisation rate constant and start time.
    Plots this data according to the Avrami equation.
    """

    @classmethod
    def calcAvramiExp(cls, df2):
        """Calculation according to Avrami equation"""
        # convert values of t=0 into nan, then drop na
        df2 = df2.replace(0, np.nan).dropna(axis=0, how='all')
        time = np.array(df2['Time'])
        ln_time = np.array(np.log(time))
        ave = np.array(df2['average'])
        start = np.array(1 / (1 - ave))
        ln = np.array(np.log(start))
        lnln = np.array(np.log(ln))
        df_avrami = pd.DataFrame({'ln_time': ln_time,
                                  '1/1-ave': start,
                                  'ln(1/1-ave)': ln,
                                  'lnln': lnln})
        return df_avrami

    @classmethod
    def plot(cls, df_avrami):
        '''
        Plot an 'Avrami plot', as a method to find the average starting recryst. time,
        average recryst. rate constant and Avrami exponent (accounts for grain growth)
        '''
        
        plt.figure(2)
        plt.plot(df_avrami['ln_time'], df_avrami['lnln'])
        plt.title('Avrami')
        plt.ylabel(r"$ln(ln(\frac{1}{1-X}))$")
        plt.xlabel('ln(time)')
        plt.show()

    @classmethod
    def avramiStats(cls, df_avrami):
        print()
        stats = linregress(df_avrami['ln_time'], df_avrami['lnln'])
        slope, intercept, r_value, p_value, std_err = stats 
        print('{} = {:.2f}'.format('Avrami exponent, n', slope)) 
        ave_k = math.exp(intercept)
        print('{} = {:.3E} per sec'.format('Average k', ave_k))
        ave_Rs = (-math.log(0.95) / ave_k) ** (1 / slope)
        print('{} = {:.2f} seconds'.format('Average Rs', ave_Rs))


class TimeCurves():
    """
    Graph showing initial and final recrystallisation times,
    for a given macro-strain values.
    - Assumes grain size of 200 microns (D0^2 = 40000, in formula for Rs)
    - Assuming 300 kJ/mol activation energy (Ea)
    - Assuming a pre exponential factor of 6.76x10^-20 in Rs
    - Rs = 6.76x10^-20 * D0^2 * (strain)^-4 * exp(Ea / RT)
    - Rf = 85% Recrystallisation
    Note that the initial temp must be lower than the final temperature
    """

    @classmethod
    def makeGraph(cls):
        initTemp = int(input("Initial temp (\u00b0C): "))
        finalTemp = int(input("Final temp (\u00b0C): "))
        degC = [x for x in range(initTemp, finalTemp)]
        tempList = [x for x in range(initTemp + 273, finalTemp + 273)]

        strain = float(input("Strain: "))
        initTime = []
        for i in tempList:
            exponent = (300000 / (8.3145 * i))
            Rs = (6.76 * 10 ** -20) * (40000) * (strain ** -4) * math.exp(exponent)
            initTime.append(Rs)

        finalTime = []
        for i in initTime:
            Rf = i * (math.log(0.15) / math.log(0.95)) ** 0.5
            finalTime.append(Rf)

        plt.plot(initTime, degC, label="$\mathregular{R_s}$")  # Use LaTex in matplotlib!
        plt.plot(finalTime, degC, label="$\mathregular{R_f}$")
        plt.ylabel('Annealing Temperature({})'.format('$^\circ$C'))
        plt.xlabel('Time(s)')
        plt.legend(loc="best", fontsize=15)
        plt.show()


class Menu:

    def __init__(self):
        self.choices = {
            1: TimeCurves.makeGraph,
            2: JMAK.main,
            3: exit
        }

    def displayMenu(self):
        """
        Displays menu and responds to choice
        """
        
        print("""
Welcome to the recrystallisation graphing program!

1. Curves of initial (5%) and final (85%) recrystallisation time.
2. JMAK curve- with option of Avrami plot.
3. Exit program.

NB:

If Avrami plot returns values of NaN (not a number), check the timescale of recrystallisation
from the graph above and re-submit the data with a more appropriate timescale.
""")

    def run(self):
        valid = False
        while not valid:
            self.displayMenu()
            choice = int(input('Please select an option: '))
            action = self.choices.get(choice) 
            if action: 
                action()
            else:
                print('Please choose a valid option')

def exit():
    sys.exit('\nThanks for using this program!\n')

def main():
    menu = Menu()
    menu.run()

if __name__ == "__main__":
    main()
