__all__ = [
    'adjust_annual_precipitation', 
    'multisite_disaggregation',
    'waterday_range'
    ]

import pandas as pd
from datetime import timedelta
import numpy as np
from scipy.stats import truncnorm
from math import log
from typing import List, Dict

from ibmweathergen.constants_ import DATE, PRECIPITATION, SAMPLE_DATE


def waterday_range(day: pd.Timedelta, window: int)->List[int]: 
    
    """ Compute the days within a given window.
        
        Parameters
        ----------
 
        day : 
            Day to be the center of a given window size. 

        window : int
            Size of the window centered on a given day.

        Returns
        ----------
            A list with 'window'size of the days of the years.
    """
    
    if (window % 2 == 1): 
        l = (window - 1) / 2 
        u = (window - 1) / 2 
    
    else:
        l = window / 2
        u = window / 2 - 1
        
    rng = [(day - timedelta(days=i)).dayofyear for i in range(-int(l), int(u)+1, 1)]
    
    return rng 


def variables_monthly_stats(df: pd.DataFrame,
                            weather_variables: list,
                            date_column=DATE) -> Dict:
    
    """ Compute the monthly mean and the standard deviation for each weather variable.
        
        Parameters
        ----------

        df : pd.DataFrame
            Selected daily precipitation labeled from the observed data. 

        weather_variables : list
            Weather variables names.

        Returns
        ----------
            A dict with the monthly 'mean' and 'standard deviation' of each variable.
    """

    df['month'] = df[date_column].dt.month
    # Group by month and calculate mean and std for each weather variable
    grouped = df.groupby('month')[weather_variables].agg(['mean', 'std'])
    # Flatten the MultiIndex columns and rename
    grouped.columns = ['_'.join([var, stat]) for var, stat in grouped.columns]
    # Reset index to turn 'month' back into a column
    return grouped.reset_index().to_dict('records')


def multisite_disaggregation(simulation_dates, weather_data_df, frequency,
                             date_column=DATE) -> pd.DataFrame:
    days_multisite = list()

    if frequency != 0:
        column_name = 'date_'
    else:
        column_name = date_column
    
    for i in range(len(simulation_dates)):
        tmp = weather_data_df[weather_data_df[column_name] == simulation_dates[SAMPLE_DATE][i]].rename(
            columns={date_column: SAMPLE_DATE})
        tmp[date_column] = simulation_dates[date_column][i]

        if frequency:
            tmp[date_column] = tmp[date_column].astype('str') + ' ' + tmp[SAMPLE_DATE].dt.time.astype('str')
        
        tmp[date_column] = tmp[date_column].astype('datetime64[ns]')
        days_multisite.append(tmp)

    df = pd.concat(days_multisite).reset_index(drop=True)

    return df


def adjust_annual_precipitation(df, predicted,
                                precipitation_column=PRECIPITATION,
                                date_column=DATE) -> pd.DataFrame:

    df_annual = df.groupby(df[date_column].dt.date)[[precipitation_column]].mean().reset_index()
    
    df_annual[date_column] = pd.to_datetime(df_annual[date_column])

    df_annual = df_annual.groupby(df_annual[date_column].dt.year)[precipitation_column].sum().values[0]

    if (df_annual < predicted['mean_ci_lower'].values[0]) or (df_annual > predicted['mean_ci_upper'].values[0]):

        myclip_a = predicted['mean_ci_lower'].values[0]
        myclip_b = predicted['mean_ci_upper'].values[0]
        my_mean = predicted['mean'].values[0]
        my_std = (myclip_b - myclip_a)/(2 * np.sqrt(2*log(2)))

        a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
        final_prcp = truncnorm.rvs(a, b, loc=my_mean, scale=my_std, size=1)[0]
        
        RATIO = final_prcp/df_annual
        df[precipitation_column] = df[precipitation_column] * RATIO
    
    return df

