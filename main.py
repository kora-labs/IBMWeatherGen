import numpy as np
from ibmweathergen.ibmweathergen import IBMWeatherGen
import pathlib

DEFAULT_WET_EXTREME_THRESHOLD = 0.999

if __name__ == "__main__":
    wg_weather = IBMWeatherGen(file_in_path=pathlib.Path.cwd().joinpath('data/dset_wg_d.csv'),
                               years=list(np.arange(2008, 2009)),
                               nsimulations=1,
                               precipitation_column='precipitation',
                               wet_extreme_quantile_threshold=DEFAULT_WET_EXTREME_THRESHOLD)

    df = wg_weather.generate_weather_series()
    print(df)
    print(df['precipitation'])