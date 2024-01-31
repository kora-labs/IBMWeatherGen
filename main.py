import numpy as np
from ibmweathergen.ibmweathergen import IBMWeatherGen
import pathlib

DEFAULT_WET_EXTREME_THRESHOLD = 0.999

if __name__ == "__main__":
    wg_weather = IBMWeatherGen(file_in_path=pathlib.Path.cwd().joinpath('data/dset_wg_d.csv'),
                               years=list(np.arange(2014, 2015)),
                               nsimulations=1,
                               date_column='date',
                               precipitation_column='pp',
                               wet_extreme_quantile_threshold=DEFAULT_WET_EXTREME_THRESHOLD)

    import time
    start = time.time()
    df = wg_weather.generate_weather_series()
    print(time.time()-start)
    print(df[['t_min', 't_max']])