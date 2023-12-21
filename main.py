import numpy as np
from src.IBMWeatherGen import IBMWeatherGen
import pathlib


DEFAULT_WET_EXTREME_THRESHOLD = 0.999

if __name__ == "__main__":
    wg_weather = IBMWeatherGen(file_in_path=pathlib.Path.cwd().joinpath('data/dset_wg_d.csv'),
                               years=list(np.arange(2011, 2012)),
                               nsimulations=2,
                               wet_extreme_quantile_threshold=DEFAULT_WET_EXTREME_THRESHOLD)

    df = wg_weather.generate_weather_series()
    print(df)