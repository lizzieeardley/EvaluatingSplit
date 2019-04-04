from validate_stats import *
from validate_calculate import *

import pandas as pd 

from statsmodels.stats.power import NormalIndPower, tt_ind_solve_power
from statsmodels.stats.weightstats import _zconfint_generic, _tconfint_generic

from IPython.display import Markdown, display
def printmd(string):
    display(Markdown(string))
    



def printResults(sdf , bdf, treatments = ['on', 'off']):
    for treatment in treatments:
        print ('\n'+treatment+'\n')
        sessions_df_t = sdf.loc[sdf['Treatment']==treatment]    
        bookings_df_t = bdf.loc[bdf['Treatment']==treatment]
        print ('Users : ', sessions_df_t.userID.nunique())  
        print ('Users That Book : ', getProportionMetric(sessions_df_t))
        print ('Average Booking Value : ', getAverageValue(bookings_df_t))    
        print ('Total Booking Value : ', getTotalValue(bookings_df_t,sessions_df_t))     
        print ('Bookings Per User : ', getEventsPerUser(bookings_df_t, sessions_df_t)[0])     
        print ('Bookings Per Session : ', getEventsPerSession(bookings_df_t, sessions_df_t)[0])      
        print ('Total Bookings in Treatment : ', getAcrossBookings(bookings_df_t))
        print ('Total Sessions in Treatment : ', getAcrossBookings(sessions_df_t, event = 'sessionID'))

def getStatTestResults( sdf, bdf, baseline = 'off', comparison = 'on', alpha_test = 0.05, type_two = 0.2):
        power = 1.0 - type_two
        sessions_df_b = sdf.loc[sdf['Treatment']==baseline]    
        sessions_df_c = sdf.loc[sdf['Treatment']==comparison]   
        nusers_b =    sessions_df_b.userID.nunique()
        nusers_c =    sessions_df_c.userID.nunique()        
        bookings_df_b = bdf.loc[bdf['Treatment']==baseline]
        bookings_df_c = bdf.loc[bdf['Treatment']==comparison]        
        nusers_booked_b =    bookings_df_b.userID.nunique()
        nusers_booked_c =    bookings_df_c.userID.nunique()   
        b_data = getProportionMetric(sessions_df_b)
        c_data = getProportionMetric(sessions_df_c)        
        p_b = b_data[0]
        p_c = c_data[0]        
        p = chi2([[b_data[1], c_data[1]],[b_data[2], c_data[2]]] )[1]
        error=dpropci_wilson_cc(c_data[1], c_data[3], b_data[1], b_data[3], alpha = alpha_test)
        error2 = 100.0* getTConfInt(b_data[3], math.sqrt(p_b*(1.0-p_b)), c_data[3], math.sqrt(p_c*(1.0-p_c))) / p_b

        mlde = getMLDE_Lizzie(b_data[1], b_data[3], sig = alpha_test, power = type_two)
        mlde2 = getMLDE_z(b_data[3], c_data[3], alpha = alpha_test, default_typ2 = type_two)*math.sqrt(p_b*(1.0 - p_b))
        printmd('**Users That Book**')
        print ('''\n\tb = %5.4f%%, c = %5.4f%%\n\timpact = %5.4f%%, error margin = %5.4f%%, t-error = %5.4f%%, MLDE (relative) = %4.3f%%\n\tp = %5.4f (%s)\n''' % ( 100.0*p_b, 100.0*p_c, 100.0*((p_c - p_b) / p_b),error[0],error2,100.0*(mlde2/p_b), p, getSignificanceText(p, p_t = alpha_test)))        
        
        
        m_b, sd_b, n_b = getAverageValue(bookings_df_b)[0:3]
        m_c, sd_c, n_c = getAverageValue(bookings_df_c)[0:3]   
        t_pval = ttest(m_b, sd_b, n_b, m_c, sd_c, n_c, equal_var=True)[1]
        mlde = 100.0*getMLDE_t(n_b, n_c,alpha = alpha_test, default_typ2 = power)*sd_b/m_b
        error = getTConfInt(n_b, sd_b, n_c, sd_c, alpha = alpha_test)
        printmd('**Average Booking Value**')
        print ('\n\tb = %5.4f, c = %5.4f\n\timpact = %5.4f%%, error margin = %5.4f%%, relative = %5.4f%%, MLDE (relative) = %5.4f%%, \n\tp = %5.4f (%s) \n\tstd (b, c) = (%5.4f, %5.4f)\n' % ( m_b, m_c, 100.0*((m_c - m_b) / m_b),error,100.0*error/m_b, mlde, t_pval,getSignificanceText(t_pval, p_t = alpha_test), sd_b, sd_c))
                
        m_b, sd_b, n_b = getTotalValue(bookings_df_b, sessions_df_b)[0:3]
        m_c, sd_c, n_c = getTotalValue(bookings_df_c, sessions_df_c)[0:3]   
        t_pval = ttest(m_b, sd_b, n_b, m_c, sd_c, n_c, equal_var=True)[1]
        error = getTConfInt(n_b, sd_b, n_c, sd_c, alpha = alpha_test)    
        mlde = 100.0*getMLDE_t(n_b, n_c,alpha = alpha_test, default_typ2 = power)*sd_b/m_b
        
        #print ('Total Booking Value : b = %5.4f, c = %5.4f, impact = %5.4f%%, p = %5.4f \n\tstd (b, c) = (%5.4f, %5.4f)\n' % ( m_b, m_c, 100.0*((m_c - m_b) / m_b), t_pval, sd_b, sd_c))
        printmd('**Total Booking Value**')
        print ('\n\tb = %5.4f, c = %5.4f\n\timpact = %5.4f%%, error margin = %5.4f%%, relative = %5.4f%%, MLDE (relative) = %5.4f%%, \n\tp = %5.4f (%s) \n\tstd (b, c) = (%5.4f, %5.4f)\n' % ( m_b, m_c, 100.0*((m_c - m_b) / m_b),error,100.0*error/m_b, mlde, t_pval,getSignificanceText(t_pval, p_t = alpha_test), sd_b, sd_c))
                
        m_b, sd_b, n_b = getEventsPerUser(bookings_df_b, sessions_df_b)[0:3]
        m_c, sd_c, n_c = getEventsPerUser(bookings_df_c, sessions_df_c)[0:3]   
        t_pval = ttest(m_b, sd_b, n_b, m_c, sd_c, n_c, equal_var=True)[1]    
        error = getTConfInt(n_b, sd_b, n_c, sd_c, alpha = alpha_test)       
        mlde = 100.0*getMLDE_t(n_b, n_c,alpha = alpha_test, default_typ2 = power)*sd_b/m_b
        
        printmd('**Bookings Per User**')
        print ('\n\tb = %5.4f, c = %5.4f\n\timpact = %5.4f%%, error margin = %5.4f%%, relative = %5.4f%%, MLDE (relative) = %5.4f%%, \n\tp = %5.4f (%s) \n\tstd (b, c) = (%5.4f, %5.4f)\n' % ( m_b, m_c, 100.0*((m_c - m_b) / m_b),error,100.0*error/m_b, mlde, t_pval,getSignificanceText(t_pval, p_t = alpha_test), sd_b, sd_c))
                
        m_b, sd_b, n_b = getEventsPerSession(bookings_df_b, sessions_df_b)[0:3]
        m_c, sd_c, n_c = getEventsPerSession(bookings_df_c, sessions_df_c)[0:3]   
        t_pval = ttest(m_b, sd_b, n_b, m_c, sd_c, n_c, equal_var=True)[1]   
        error = getTConfInt(n_b, sd_b, n_c, sd_c, alpha = alpha_test)       
        mlde = 100.0*getMLDE_t(n_b, n_c,alpha = alpha_test, default_typ2 = power)*sd_b/m_b
        
        printmd('**Bookings Per Session**')
        print ('\n\tb = %5.4f, c = %5.4f\n\timpact = %5.4f%%, error margin = %5.4f%%, relative = %5.4f%%, MLDE (relative) = %5.4f%%, \n\tp = %5.4f (%s) \n\tstd (b, c) = (%5.4f, %5.4f)\n' % ( m_b, m_c, 100.0*((m_c - m_b) / m_b),error,100.0*error/m_b, mlde, t_pval,getSignificanceText(t_pval, p_t = alpha_test), sd_b, sd_c))
        
        m_b, n_b = getAcrossBookings(bookings_df_b)[0:2]
        m_c, n_c = getAcrossBookings(bookings_df_c)[0:2]        
        #print ('Total Bookings in Treatment : b = %5.4f, c = %5.4f, impact = %5.4f%% \n' % ( m_b, m_c, 100.0*((m_c - m_b) / m_b)))
        printmd('**Total Bookings in Treatment**')
        print ('\n\tb = %5i, c = %5i \n\timpact = %5.4f%% \n' % ( m_b, m_c, 100.0*((m_c - m_b) / m_b)))
                
        m_b, n_b = getAcrossBookings(sessions_df_b, event = 'sessionID')[0:2]
        m_c, n_c = getAcrossBookings(sessions_df_c, event = 'sessionID')[0:2]        
        printmd('**Total Sessions in Treatment**')
        print ('\n\tb = %i, c = %i \n\timpact = %5.4f%% \n' % ( m_b, m_c, 100.0*((m_c - m_b) / m_b)))
                
    
        #scipy.stats.ttest_ind_from_stats(mean1, std1, nobs1, mean2, std2, nobs2, equal_var=True)[source]