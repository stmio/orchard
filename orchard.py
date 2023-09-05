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
        ],
    },
    "uk-hourly-rain-obs": {
        "index": "ob_end_time",
        "skiprows": 61,
        "columns": ["ob_end_time", "ob_hour_count", "prcp_amt"],
    },
    "uk-soil-temperature-obs": {
        "index": "ob_time",
        "skiprows": 85,
        "columns": ["ob_time", "q10cm_soil_temp"],
    },
}


def get_available_datasets():
    return list(DATASETS)


def get_counties(dataset="uk-hourly-weather-obs"):
    counties = {}

    paths = glob.glob(
        f"/home/sam/ukmo-midas-open/data/{dataset}/dataset-version-{DATASET_VERSION}/*",
        recursive=True,
    )

    for path in paths:
        counties[os.path.split(path)[-1]] = path

    return counties


# TODO: document moving of metadata file - maybe add function to do this?
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
            usecols=["station_file_name", "src_id"] + cols,
        )

        df.drop(df.tail(1).index, inplace=True)
        df[dataset] = 1

        metadata.append(df)

    metadata = pd.concat(metadata)
    metadata["station"] = metadata["src_id"] + "_" + metadata["station_file_name"]

    return metadata.groupby("station").aggregate(
        {c: "first" for c in cols} | {d: "sum" for d in DATASETS}
    )


def stations_count():
    return len(get_stations_metadata().index)


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
    data = {}
    for county in get_counties(dataset):
        county_data = {}
        for station in get_county_stations(dataset, county):
            station_data = pd.read_parquet(
                get_station_parquet_file(dataset, county, station)
            )
            station_data.index = pd.to_datetime(station_data.index)
            county_data[station] = station_data
        data[county] = county_data
    return data


def load_all():
    metadata = get_stations_metadata()

    datasets = {}
    for dataset in get_available_datasets():
        datasets[dataset] = load_dataset(dataset)

    data = {}
    for county in metadata["historic_county"].unique():
        data[county] = {}

    for index, row in metadata.iterrows():
        station = []
        county = row["historic_county"]

        for dataset in get_available_datasets():
            if row[dataset] == 1:
                station.append(
                    datasets[dataset][county][index]
                    .groupby(level=0, dropna=False)
                    .sum(min_count=1)
                )
            else:
                cols = DATASETS[dataset]["columns"][1:]
                station.append(pd.DataFrame(columns=cols))

        data[county][index] = pd.concat(station, axis=1)
        data[county][index].drop("ob_hour_count", axis=1, errors="ignore", inplace=True)
        data[county][index].index = pd.to_datetime(data[county][index].index)
        data[county][index] = data[county][index].astype(np.float64)

    return data


def spread_rain_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Randomly distribute rain data in its sample,
    to match the other datasets. This makes it so that,
    for example, all rain data from the last 12 hours is
    spread across this time.
    """

    new_rows = {}
    for index, row in df.iterrows():
        amount = row["prcp_amt"]
        duration = row["ob_hour_count"]

        random_dist = np.random.dirichlet(np.ones(int(duration)))

        for i in range(0, int(duration)):
            new_rows[index - pd.Timedelta(hours=i)] = {
                "prcp_amt": amount * random_dist[i]
            }

    rain = pd.DataFrame.from_dict(new_rows, orient="index").sort_index()
    rain.index.name = "ob_time"

    return rain


def fill_missing(data):
    for county in data:
        for station in data[county]:
            data[county][station] = data[county][station].resample("1H").asfreq()

    metadata = get_stations_metadata()
    for county in data:
        for station in data[county]:
            near = find_nearest_stations(
                metadata.loc[station]["station_latitude"],
                metadata.loc[station]["station_longitude"],
            )

            i = 0  # TODO: change to for loop
            while data[county][station].isnull().values.any():
                try:
                    data[county][station].fillna(
                        data[near.iloc[i]["historic_county"]][near.iloc[i].name],
                        axis=0,
                        inplace=True,
                    )
                except IndexError:
                    break
                i += 1

            data[county][station].interpolate(
                method="linear", limit_direction="both", inplace=True
            )

    contains_na = []
    for county in data:
        for station in data[county]:
            if data[county][station].isnull().values.any():
                contains_na.append(station)

    return data, contains_na


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
            station_df.index = pd.to_datetime(station_df.index)

            # TODO: Currently distributes rain data at random - change?
            if station_df.index.name != "ob_time":
                station_df = spread_rain_data(station_df)

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


def find_nearest_stations(lat, lon, n=100, include_self=False):
    stations = get_stations_metadata()[
        ["station_latitude", "station_longitude", "historic_county"]
        + get_available_datasets()
    ]

    stations["distance_to"] = haversine_distance(
        lat, lon, stations["station_latitude"], stations["station_longitude"]
    )

    stations.sort_values("distance_to", ascending=True, inplace=True)
    if not include_self and stations.iloc[0]["distance_to"] == 0:
        stations.drop(stations.head(1).index, inplace=True)

    return stations if n == -1 else stations[:n]
