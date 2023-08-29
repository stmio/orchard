import pandas as pd
import dask_cudf as cd
import pyarrow as pa
import pyarrow.parquet as pq

import os
import glob
from datetime import datetime


# See https://help.ceda.ac.uk/article/4982-midas-open-user-guide
QC_VERSION = 1
DATASET_VERSION = 202207
DATASETS = ["uk-hourly-rain-obs", "uk-hourly-weather-obs"]


def getCounties():
    counties = {}

    paths = glob.glob(
        f"/home/sam/ukmo-midas-open/data/uk-hourly-weather-obs/dataset-version-{DATASET_VERSION}/*",
        recursive=True,
    )

    for path in paths:
        counties[os.path.split(path)[-1]] = path

    return counties


def getCountyStations(county):
    stations = {}
    paths = glob.glob(f"{getCounties()[county]}/*")

    for path in paths:
        stations[os.path.split(path)[-1]] = path

    return stations


def getStationCSVFiles(county, station):
    return sorted(
        glob.glob(f"{getCounties()[county]}/{station}/qc-version-{QC_VERSION}/*.csv")
    )


def getStationParquetFile(county, station):
    return f"{getCounties()[county]}/{station}/{station}.parquet"


def getStationFilePath(county, station):
    return f"{getCounties()[county]}/{station}/"


def readAllData():
    data = []
    for county in getCounties():
        for station in getCountyStations(county):
            station_data = cd.read_parquet(getStationParquetFile(county, station))
            station_data["county"] = county
            station_data["station"] = station
            data.append(station_data)
    return cd.concat(data)


def convertStationData(verbose=False):
    for county in getCounties():
        for station in getCountyStations(county):
            start_time = datetime.now()
            station_data = []
            for file in getStationCSVFiles(county, station):
                df = pd.read_csv(
                    file,
                    index_col="ob_time",
                    parse_dates=["ob_time"],
                    engine="c",
                    skiprows=280,
                    on_bad_lines="warn",
                    usecols=[
                        "ob_time",
                        "wind_direction",
                        "wind_speed",
                        "visibility",
                        "msl_pressure",
                        "vert_vsby",
                        "air_temperature",
                        "dewpoint",
                        "wetb_temp",
                        "rltv_hum",
                        "stn_pres",
                        "alt_pres",
                    ],
                )
                # Remove last row "end data"
                df.drop(df.tail(1).index, inplace=True)
                station_data.append(df)

            station_df = pd.concat(station_data)
            df_pa = pa.Table.from_pandas(station_df)
            pq.write_table(
                df_pa, getStationFilePath(county, station) + station + ".parquet"
            )

            if verbose:
                print(
                    f"Converted data to parquet for station {station} ({county})",
                    f"in {datetime.now() - start_time} seconds",
                )

    print("All data converted to parquet")


def clean_data(df):
    return df
