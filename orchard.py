import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import os
import glob
from datetime import datetime


# See https://help.ceda.ac.uk/article/4982-midas-open-user-guide/
QC_VERSION = 1
DATASET_VERSION = 202207
DATASETS = {
    "uk-hourly-weather-obs": {
        "index": "ob_time",
        "skiprows": 280,
        "columns": [
            "ob_time",
            "wind_direction",
            "wind_speed",
            "air_temperature",
            "rltv_hum",
            "msl_pressure",
            "stn_pres",
            "alt_pres",
        ],
    },
    "uk-hourly-rain-obs": {
        "index": "ob_end_time",
        "skiprows": 61,
        "columns": ["ob_end_time", "ob_hour_count", "prcp_amt", "prcp_dur"],
    },
    "uk-soil-temperature-obs": {
        "index": "ob_time",
        "skiprows": 85,
        "columns": ["ob_time", "q5cm_soil_temp", "q10cm_soil_temp"],
    },
}


def get_counties(dataset="uk-hourly-weather-obs"):
    counties = {}

    paths = glob.glob(
        f"/home/sam/ukmo-midas-open/data/{dataset}/dataset-version-{DATASET_VERSION}/*",
        recursive=True,
    )

    for path in paths:
        counties[os.path.split(path)[-1]] = path

    return counties


def get_stations_metadata():
    cols = [
        "historic_county",
        "station_latitude",
        "station_longitude",
        "station_elevation",
    ]
    metadata = []

    for dataset in DATASETS:
        df = pd.read_csv(
            f"/home/sam/ukmo-midas-open/data/{dataset}/midas-open_{dataset}_dv-{DATASET_VERSION}_station-metadata.csv",
            skiprows=46,
            usecols=["station_file_name"] + cols,
        )

        df.drop(df.tail(1).index, inplace=True)
        df[dataset] = 1

        metadata.append(df)

    return (
        pd.concat(metadata)
        .groupby("station_file_name")
        .aggregate({c: "first" for c in cols} | {d: "sum" for d in DATASETS})
    )


def get_county_stations(dataset, county):
    stations = {}
    paths = glob.glob(f"{get_counties(dataset)[county]}/*")

    for path in paths:
        stations[os.path.split(path)[-1]] = path

    return stations


def get_station_csv_files(dataset, county, station, qc=QC_VERSION):
    return sorted(
        glob.glob(f"{get_counties(dataset)[county]}/{station}/qc-version-{qc}/*.csv")
    )


def get_station_parquet_file(dataset, county, station):
    return f"{get_counties(dataset)[county]}/{station}/{station}.parquet"


def get_station_file_path(dataset, county, station):
    return f"{get_counties(dataset)[county]}/{station}/"


def load_dataset(dataset):
    data = []
    for county in get_counties(dataset):
        for station in get_county_stations(dataset, county):
            station_data = pd.read_parquet(
                get_station_parquet_file(dataset, county, station)
            )
            station_data["county"] = county
            station_data["station"] = station
            data.append(station_data)
    return pd.concat(data)


def convert_data(dataset, verbose=False):
    for county in get_counties(dataset):
        for station in get_county_stations(dataset, county):
            start_time = datetime.now()
            station_data = []

            # First try loading all CSV files from default QC_VERSION directory
            # If no files, try the other QC_VERSION folder
            csv_files = get_station_csv_files(
                dataset, county, station
            ) or get_station_csv_files(dataset, county, station, 1 - QC_VERSION)

            for file in csv_files:
                df = pd.read_csv(
                    file,
                    engine="c",
                    on_bad_lines="warn",
                    index_col=DATASETS[dataset]["index"],
                    parse_dates=[DATASETS[dataset]["index"]],
                    skiprows=DATASETS[dataset]["skiprows"],
                    usecols=DATASETS[dataset]["columns"],
                )
                # Remove last row "end data"
                df.drop(df.tail(1).index, inplace=True)
                station_data.append(df)

            station_df = pd.concat(station_data)
            df_pa = pa.Table.from_pandas(station_df)
            pq.write_table(
                df_pa,
                get_station_file_path(dataset, county, station) + station + ".parquet",
            )

            if verbose:
                print(
                    f"Converted data to parquet for station {station} ({county})",
                    f"in {datetime.now() - start_time} seconds",
                )

    print(f"All data converted to parquet for {dataset}")


def haversine_distance(lat1, lon1, lat2, lon2):
    EARTH_RADIUS_KM = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    return (
        2
        * EARTH_RADIUS_KM
        * np.arcsin(
            np.sqrt(
                (np.sin((lat2 - lat1) / 2) ** 2)
                + (np.cos(lat1) * np.cos(lat2) * (np.sin((lon2 - lon1) / 2) ** 2))
            )
        )
    )
