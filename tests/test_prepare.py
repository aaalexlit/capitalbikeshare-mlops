import os

import pandas as pd
from pandas import Timestamp
from sklearn.feature_extraction import DictVectorizer

from src.data import prepare

os.environ["WANDB_MODE"] = "offline"


def test_preprocess():
    input_dicts = [
        {
            'start_station_id': '31239',
            'end_station_id': '31251',
            'rideable_type': 'docked_bike',
            'member_casual': 'casual',
            'hour': 17,
            'year': 2020,
            'duration': 6.416666666666667,
            'started_at': Timestamp('2020-04-25 17:28:39'),
        },
        {
            'start_station_id': '31205',
            'end_station_id': '31224',
            'rideable_type': 'docked_bike',
            'member_casual': 'member',
            'hour': 7,
            'year': 2020,
            'duration': 2.4166666666666665,
            'started_at': Timestamp('2020-04-06 07:54:59'),
        },
        {
            'start_station_id': '31313',
            'end_station_id': '31313',
            'rideable_type': 'docked_bike',
            'member_casual': 'casual',
            'hour': 17,
            'year': 2020,
            'duration': 62.23333333333333,
            'started_at': Timestamp('2020-04-22 17:06:18'),
        },
    ]
    expected_feature_names = [
        'end_station_id=31224',
        'end_station_id=31251',
        'end_station_id=31313',
        'hour',
        'member_casual=casual',
        'member_casual=member',
        'rideable_type=docked_bike',
        'start_station_id=31205',
        'start_station_id=31239',
        'start_station_id=31313',
        'year',
    ]
    df = pd.DataFrame(input_dicts)
    X, dv = prepare.preprocess(df, DictVectorizer(), fit_dv=True)
    assert dv.feature_names_ == expected_feature_names
    assert X.shape == (3, 11)
