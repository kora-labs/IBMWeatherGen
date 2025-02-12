import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from typing import Optional
import pandas as pd
from random import sample
import itertools

from ibmweathergen.bootstrap_sampling import BootstrapSampling
from ibmweathergen.markov_chain import FirstOrderMarkovChain
from ibmweathergen.lag_one import LagOne
from ibmweathergen.annual_forecaster import autoArimaFourierFeatures, Utils
from ibmweathergen.utilities import multisite_disaggregation, adjust_annual_precipitation
from ibmweathergen.constants import PRECIPITATION, DATE, LONGITUDE, LATITUDE, SAMPLE_DATE, T_MIN, T_MAX


class IBMWeatherGen:

    DEFAULT_WET_EXTREME_THRESHOLD = 0.999

    """
    Semi-parametric stochastic weather generator capable of generate syntetic timeseries of weather observations at the time 
    resolution provided by having the precipitation as the "key variable". 

    Args
    ----------
    file_in_path : str
        Full path with the .csv file of the historic data.

    years : list
        List of years to be simulated.

    file_out_path : str
        Path where the .zip file will be stored.

    nsimulations : int
        Number of simulation requested for each year.

    Properties
    ----------
    file_out_path : pd.DataFrame
        Interval and mean of the total annual precipitation forecasted.

    number_of_simulations : pd.DataFrame
        Total annual precipitation for each year of the observed data.

    file_in_path : pd.DataFrame
        Daily precipitation of the observed data.

    simulation_year_list: dict
        The values to be used in the monthly quantile computation and in the labeling proccess.
    
    raw_data : pd.DataFrame
        Data in original spatial and temporal format, without aggregations ("multisite", "subhourly"). Receives 'Date', 'Latitude',
    'Longitude', 'precipitation' [, 't_min', 't_max', 'wind_10m', wind_100m']

    daily_data : pd.DataFrame
        Data in daily format and "single-site".

    annual_data : pd.DataFrame
        Data in annual format.

    weather_variables : list[str]
        Original names of each weather variables to be used in the new timeseries.

    weather_variables_mean : list[str]
        Weather variable names after any needed calculation (e.g mean).

    """

    def __init__(self, 
                 file_in_path,
                 years,
                 wet_extreme_quantile_threshold: Optional[float]=DEFAULT_WET_EXTREME_THRESHOLD,
                 nsimulations=1,
                 precipitation_column=PRECIPITATION,
                 date_column=DATE,
                 t_min_column=T_MIN,
                 t_max_column=T_MAX,
                 raw_data=None):

        self.best_annual_forecaster = None
        self.number_of_simulations = nsimulations
        self.file_in_path = file_in_path
        self.simulation_year_list = years
        self.precipitation_column = precipitation_column
        self.date_column = date_column
        self.t_min_column = t_min_column
        self.t_max_column = t_max_column
        self.raw_data = raw_data
        self.daily_data = None 
        self.annual_data = None
        self.frequency = None
        self.weather_variables = list()

        self.wet_extreme_quantile_threshold = wet_extreme_quantile_threshold
        self.randomly_clip = False
        
        self.weather_variables_mean = list() 

    def closest(self, lst, K):
    
        return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))]

    def read_data(self):
        if self.raw_data is None:
            self.raw_data = pd.read_csv(self.file_in_path, parse_dates=[self.date_column]).dropna()

    def select_bbox(self, df):
        
        lat0 = sample(sorted(list(df[LATITUDE].unique()))[1:len(list(df[LATITUDE].unique()))-10], 1)[0]
        lat1 = self.closest(lst=sorted(list(df[LATITUDE].unique())), K=lat0+1)

        
        lon0 = sample(sorted(list(df[LONGITUDE].unique()))[1:len(list(df[LONGITUDE].unique()))-10], 1)[0]
        lon1 = self.closest(lst=sorted(list(df[LONGITUDE].unique())), K=lon0+1)
        
        return [lon0, lon1, lat0, lat1]

    def generate_daily(self, frequency, df):
        
        if frequency!= 0:
            self.daily_data = df.groupby(by=[df[self.date_column].dt.date, LONGITUDE, LATITUDE]).sum()*(frequency/60)
            self.daily_data.reset_index(inplace=True)
            self.daily_data[self.date_column] = pd.to_datetime(self.daily_data[self.date_column])
            self.raw_data['date_'] = self.raw_data[self.date_column].dt.date

        else:
            self.daily_data = df.copy()
        
        return self.daily_data

    def compute_daily_variables(self)->pd.DataFrame:

        self.read_data()

        if (self.t_min_column and self.t_max_column in self.raw_data.columns):
            self.raw_data = self.raw_data.assign(temperature=
                                                 (self.raw_data[self.t_min_column] +
                                                  self.raw_data[self.t_max_column])/2)

        self.weather_variables_mean = [element for element in list(self.raw_data.columns)
                                       if element not in [self.date_column, LONGITUDE, LATITUDE,
                                                          self.t_min_column, self.t_max_column]]
        
        self.weather_variables = [weather_var for weather_var in self.raw_data.columns
                                  if weather_var not in [self.date_column, LONGITUDE, LATITUDE]]
        
        self.frequency = self.raw_data[self.date_column].diff().min().seconds//60

        #TODO: FOR N_SIMULATIONS == 1 --> AT THE CENTER?
        if ((max(self.raw_data.Latitude) - min(self.raw_data.Latitude)) > 1
                or (max(self.raw_data.Longitude) - min(self.raw_data.Longitude)) > 1):
            
            selected_bbox = self.select_bbox(self.raw_data)
            self.sub_raw_data = self.raw_data[(self.raw_data.Longitude <= selected_bbox[1]) &
                                              (self.raw_data.Longitude >= selected_bbox[0]) &
                                              (self.raw_data.Latitude >= selected_bbox[2]) &
                                              (self.raw_data.Latitude <= selected_bbox[3])]
            
            self.daily_data = self.generate_daily(self.frequency, self.sub_raw_data)
        else:
            self.daily_data = self.generate_daily(self.frequency, self.raw_data)

        return self.daily_data.groupby(self.daily_data[self.date_column])[self.weather_variables].mean().reset_index()

    def compute_annual_prcp(self) -> pd.DataFrame:
        self.daily_data = self.compute_daily_variables()

        self.annual_data = self.daily_data.groupby(self.daily_data[self.date_column].dt.year)[
            self.precipitation_column].sum()
        
        self.annual_data.index = pd.period_range(str(self.annual_data.index[0]), 
                                                 str(self.annual_data.index[-1]), freq='Y')

        return self.annual_data
    
    def generate_forecasted_values(self):
        list_autoArimaFourierFeatures = []
        l_m = list(range(2, len(self.annual_data.index), 3))
        l_m = sample(list(l_m), k=4)
        comb_list = [list(itertools.product([m], list(range(1, (int(m/2)+1))))) for m in list(l_m)]
        comb_list = list(itertools.chain(*comb_list))

        for m, k in comb_list:
            list_autoArimaFourierFeatures.append(autoArimaFourierFeatures(m=m, k=k))
        
        list_models = list_autoArimaFourierFeatures
        return Utils.model_selection(list_models, self.annual_data) 
    
    def adjust_prediction(self, prediction) -> pd.DataFrame:
        year = prediction['mean_ci_lower'].index.values[0]
        stp = self.annual_data.values.std() * 0.8

        if prediction['mean_ci_lower'].values[0] < 0:
            prediction['mean_ci_lower'] = self.annual_data[str(year)]
            prediction['mean'] = prediction['mean_ci_lower'] + stp
            prediction['mean_ci_upper'] = prediction['mean_ci_lower'] + 2*stp

        else:
            vle = ((prediction['mean'] - self.annual_data[str(year)])/self.annual_data[str(year)]).values[0]
            
            if vle < -0.05: #annual > predicted_mean
                diff = prediction['mean'] - prediction['mean_ci_lower'] 
                prediction['mean'] = prediction['mean'] - (0.95*(prediction['mean'] - self.annual_data[str(year)])).values[0]
                prediction['mean_ci_lower'] = (prediction['mean_ci_lower'] + diff)
                prediction['mean_ci_upper'] = (prediction['mean_ci_upper'] + diff)
                #print(f'MENOR: {prediction}')
            if vle > 0.05:
                diff = prediction['mean'] - prediction['mean_ci_lower'] 
                prediction['mean'] = prediction['mean'] - (0.95*abs(prediction['mean'] - self.annual_data[str(year)])).values[0]
                prediction['mean_ci_lower'] = (prediction['mean_ci_lower'] - diff)
                prediction['mean_ci_upper'] = (prediction['mean_ci_upper'] - diff)
        
        return prediction

    def generate_weather_series(self):
        simulations = []
        self.annual_data = self.compute_annual_prcp()
        stp = self.annual_data.values.std() * 0.8

        for simulation_year in self.simulation_year_list:
            predicted = {}
            print('Predicting the range for the year.')
            if str(simulation_year) in self.annual_data.index:
                predicted['mean'] = self.annual_data[str(simulation_year)]
            else:
                predicted['mean'] = self.annual_data.mean()
            predicted['mean_ci_lower'] = predicted['mean'] - stp
            predicted['mean_ci_upper'] = predicted['mean'] + stp
            predicted = pd.DataFrame(index=[str(simulation_year)], data=predicted)

            for num_simulation in range(self.number_of_simulations):
                print(f'\nYear: [[{simulation_year}]] | Simulation: [[{num_simulation+1}/{self.number_of_simulations}]]')

                bootstrap = BootstrapSampling(predicted, self.annual_data.to_frame(), self.daily_data,
                                              self.wet_extreme_quantile_threshold,
                                              precipitation_column=self.precipitation_column,
                                              date_column=self.date_column)
                training_data, thresh = bootstrap.get_labels_states()

                prcp_occurence = FirstOrderMarkovChain(training_data, simulation_year, self.weather_variables,
                                                       precipitation_column=self.precipitation_column,
                                                       date_column=self.date_column)
                df_simulation, thresh_markov_chain = prcp_occurence.simulate_state_sequence()
                
                single_timeseries = LagOne(training_data, df_simulation, self.weather_variables,
                                           self.weather_variables_mean, date_column=self.date_column)
                df_simulation = single_timeseries.get_series()

                df_simulation = multisite_disaggregation(df_simulation, self.raw_data, self.frequency,
                                                         date_column=self.date_column)

                df_simulation = adjust_annual_precipitation(df_simulation, predicted,
                                                            precipitation_column=self.precipitation_column,
                                                            date_column=self.date_column)
                
                df_simulation = df_simulation.assign(n_simu=num_simulation+1)
                df_simulation = df_simulation.drop([SAMPLE_DATE, 'temperature'], axis=1).set_index(self.date_column)
                simulations = simulations + [df_simulation]

        dfnl = pd.concat(simulations)

        return dfnl
