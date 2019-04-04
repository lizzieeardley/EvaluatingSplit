import pandas as pd 
import numpy as np
import math

from scipy.stats import binom, norm, t
from scipy.stats import ttest_ind_from_stats as ttest
from scipy.stats import chi2_contingency as chi2

from statsmodels.stats.power import NormalIndPower, tt_ind_solve_power
from statsmodels.stats.weightstats import _zconfint_generic, _tconfint_generic


def getProportionMetric(df_in, event='bookings'):
    user_df = df_in.groupby('userID').agg({'sessionID':'nunique', event:'sum'}).reset_index()
    user_df[event+'_user'] = user_df.apply(lambda x: 1 if x[event] > 0 else 0, axis=1)    
    proportionMetric = user_df[event+'_user'].sum()/user_df.userID.nunique()
    contributors = user_df.userID.nunique()
    return proportionMetric, int (user_df[event+'_user'].sum()), int(user_df.userID.nunique() - user_df[event+'_user'].sum()), contributors

def getAverageValue(df_in, event = 'bookingID',  value='bookingValue'):
    user_df = df_in.groupby('userID').agg({ value:'sum', event:'nunique'}).reset_index()  
    user_df['averageValue'] = user_df.apply(lambda x: x[value] / x[event], axis=1)
    averageValue_var = user_df['averageValue'].var()
    averageValue_std = user_df['averageValue'].std()
    averageValue_min = user_df['averageValue'].min()
    averageValue_max = user_df['averageValue'].max()
    averageValue_median = user_df['averageValue'].quantile(0.5)
    averageValue_95 = user_df['averageValue'].quantile(0.95)

    averageValue_contributors = user_df.userID.nunique()

    averageValue = user_df['averageValue'].mean()#/user_df[event].sum()    
    return averageValue, averageValue_std, averageValue_contributors, averageValue_var, averageValue_min, averageValue_max, averageValue_median, averageValue_95

def getTotalValue(event_df,all_users_df, event = 'bookingID',  value='bookingValue'):
    
    users = all_users_df.groupby('userID').agg({'sessionID':'min'}).reset_index()  
    #users[value]=0
    eventValue_per_user = event_df.groupby('userID').agg({ value:'sum'}).reset_index() 
    
    combined_df = users.merge(eventValue_per_user, how='left', on='userID')
    #print (combined_df.head())
    combined_df[value] = combined_df[value].fillna(0)    
    
    totalValue_variance = combined_df[value].var()
    totalValue_std = combined_df[value].std()
    
    totalValue = combined_df[value].mean()#/user_df[event].sum()    
    
    totalValue_contributors = combined_df.userID.nunique()
    
    return totalValue, totalValue_std, totalValue_contributors

def getEventsPerUser(event_df,all_users_df, event = 'bookingID'):
    users = all_users_df.groupby('userID').agg({'sessionID':'min'}).reset_index()  
    events_per_user = event_df.groupby('userID').agg({ event:'nunique'}).reset_index() 
    combined_df = users.merge(events_per_user, how='left', on='userID')
    combined_df[event] = combined_df[event].fillna(0)
    
    eventsPerUser_variance = combined_df[event].var()
    eventsPerUser = combined_df[event].mean()
    eventsPerUser_std = combined_df[event].std()
        
    contributors = combined_df.userID.nunique()    
    
    return eventsPerUser, eventsPerUser_std, contributors

def getEventsPerSession(event_df,all_users_df, denom_event = 'sessionID', num_event = 'bookingID'):
    sessions_per_user = all_users_df.groupby('userID').agg({ denom_event:'nunique'}).reset_index()  
    events_per_session = event_df.groupby('userID').agg({ num_event:'nunique'}).reset_index()      
    combined_df = sessions_per_user.merge(events_per_session, how='left', on='userID')
    combined_df[num_event] = combined_df[num_event].fillna(0)
    combined_df['events_per_session'] = combined_df[num_event] / combined_df[denom_event]

    eventsPerSession_std = combined_df['events_per_session'].std()
    eventsPerSession = combined_df['events_per_session'].mean()
    
    contributors = combined_df.userID.nunique()    
    
    return eventsPerSession, eventsPerSession_std, contributors

def getAcrossBookings(event_df, event = 'bookingID'):
    return event_df[event].nunique(), event_df['userID'].nunique()