from scipy import optimize
import numpy as np


def myega(t, eta, t_input):
    '''This funciton fits the MYEGA equation by Mauro et al. 2... to a data set. The it uses input temperatures
    to calculate viscosites from the fitted model. Temperature are in Kelvin and viscosities in Pa s'''
    
    t = np.asarray(t)
    eta = np.asarray(np.squeeze(eta))
    
    ### Define MYEGA equation. The infinite temperature viscosity paramters A (here eta_d) is set to a constant value of -2.9. ###
    ### For further information see Langhammer et al. (2021). ###
    
    def myega_fit(T, tg, m):
        eta_d = -2.9
        return eta_d + (tg/T)*(12-eta_d)*np.exp(((m/(12-eta_d))-1)*((tg/T)-1))
    
    ### Fit MYEGA equation to data. ###
    
    param, cov = optimize.curve_fit(myega_fit, t, eta, p0 = [900, 30], bounds = (0, np.inf))

    
    ### Calculate viscosities form the fitted MYEGA model for the input temperature and the plot ###
    t_plot = np.arange(min(t), max(t), 1)
    fit = myega_fit(t_plot, *param)
    eta_goal = myega_fit(t_input, *param)
    
    

   
    
    return param, fit, t_plot, eta_goal