import numpy as np
import pandas as pd
import tensorflow as tf
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.preprocessing import MinMaxScaler

import os
import glob
import pickle
from tqdm.notebook import tqdm
from datetime import datetime

from typing import Callable, Dict, Tuple, TypeAlias
from typeguard import check_type


# Don't show tf warnings until https://github.com/tensorflow/tensorflow/issues/62963 released
tf.get_logger().setLevel("ERROR")


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

min_max = MinMaxScaler()

DatasetsType: TypeAlias = Dict[str, Dict[str, np.ndarray | pd.DataFrame]]
DatasetType: TypeAlias = np.ndarray | pd.DataFrame


def assert_dataset(d):
    assert check_type(d, DatasetsType)


# Runs function "f" on every station dataset in the dictionary
# TODO: rename map_all?
def run_all(f: Callable, data: DatasetsType, verbose: bool = False) -> DatasetsType:
    assert_dataset(data)

    for county in tqdm(data, delay=2):
        for station in data[county]:
            if verbose:
                print(county + ": " + station)

            data[county][station] = f(data[county][station])

    return data


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


# TODO: currently not called anywhere
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


def pickle_data(data, path=(os.getcwd() + "/orchard-data")):
    file = open(path + ".pkl", "wb")
    pickle.dump(data, file)
    file.close()


def unpickle_data(path=(os.getcwd() + "/orchard-data")):
    file = open(path + ".pkl", "rb")
    data = pickle.load(file)
    file.close()
    return data


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


def min_max_scale(df):
    return min_max.fit_transform(df)


def inverse_min_max_scale(df):
    return min_max.inverse_transform(df)


def get_time_features(
    index: pd.DatetimeIndex,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    # Extract individual time features from index
    mins = pd.Series(index.minute.values, name="mins")
    hours = pd.Series(index.hour.values, name="hours")
    days = pd.Series(index.day.values, name="days")
    months = pd.Series(index.month.values, name="months")
    years = pd.Series(index.year.values, name="years")

    return mins, hours, days, months, years


def window_dataset(data, steps, horizon, batch_size, shuffle_buffer):
    # create a window with n steps back plus the size of the prediction length
    window = steps + horizon

    # create the inital tensor dataset
    with tf.device("CPU"):
        ds = tf.data.Dataset.from_tensor_slices(data)

    # create the window function shifting the data by the prediction length
    ds = ds.window(window, shift=horizon, drop_remainder=True)

    # flatten the dataset and batch into the window size
    ds = ds.flat_map(lambda x: x.batch(window))
    ds = ds.shuffle(shuffle_buffer)

    # create the supervised learning problem x and y and batch
    ds = ds.map(lambda x: (x[:-horizon], x[-horizon:, :1]))

    ds = ds.batch(batch_size).prefetch(1)

    return ds


def get_params() -> Tuple[int, int, int, float]:
    learning_rate = 3e-4
    steps = 24 * 30
    horizon = 24
    features = 12

    return steps, horizon, features, learning_rate


def build_station_dataset(
    data: DatasetType,
    steps=24 * 30,
    horizon=24,
    batch_size=256,
    shuffle_buffer=500,
):
    # TODO: add station number as well?
    times = get_time_features(data.index)
    data = pd.concat([data.reset_index(drop=True), *times], axis=1)

    data = min_max_scale(data)

    # TODO: split data

    data = window_dataset(data, steps, horizon, batch_size, shuffle_buffer)

    return data


def build_dataset(
    data,
    steps=24 * 30,
    horizon=24,
    batch_size=256,
    shuffle_buffer=500,
):
    tf.random.set_seed(64)

    ds = run_all(
        lambda x: build_station_dataset(x, steps, horizon, batch_size, shuffle_buffer),
        data,
    )

    return ds


def lstm_cnn_model(steps, horizon, features, learning_rate):
    tf.keras.backend.clear_session()

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv1D(
                64, kernel_size=6, activation="relu", input_shape=(steps, features)
            ),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Conv1D(64, kernel_size=3, activation="relu"),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.LSTM(72, activation="relu", return_sequences=True),
            tf.keras.layers.LSTM(48, activation="relu", return_sequences=False),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(horizon),
        ],
        name="lstm_cnn",
    )

    loss = tf.keras.losses.Huber()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(loss=loss, optimizer=optimizer, metrics=["mae"])

    return model


# TODO: remove?
def fit_model_to_station(data, model, epochs):
    model_hist = model.fit(data, epochs=epochs)
    return model, model_hist


def run_model(ds, epochs, steps, horizon, features, learning_rate):
    model = lstm_cnn_model(steps, horizon, features, learning_rate=learning_rate)
    model_hist = {"loss": [], "mae": []}

    for county in tqdm(ds):
        for station in ds[county]:
            model, h = fit_model_to_station(ds[county][station], model, epochs)
            model_hist = {
                key: np.hstack([model_hist[key], h.history[key]])
                for key in h.history.keys()
            }

    return model, model_hist
