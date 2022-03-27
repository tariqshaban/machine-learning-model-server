import joblib
import pandas as pd

from fastapi import FastAPI
from sklearn import preprocessing
from starlette.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get('/machine_learning_models/winning')
async def winning(club1: str, club2: str):
    club1_is_host = False

    clubs = sorted([club1, club2])

    if [club1, club2] == sorted([club1, club2]):
        club1_is_host = True

    club1 = clubs[0]
    club2 = clubs[1]

    winners = pd.read_csv('assets/dataframes/winners.csv')

    club1_row = winners[(winners['club1'].str.upper() == club1.upper())]
    club1_row = club1_row.drop('winner', axis=1)

    if club1_row.empty:
        club1_row = winners[(winners['club2'].str.upper() == club1.upper())]
        club1_row = club1_row. \
            rename({'club1': 'club2', 'club2': 'club1',
                    'team_1_home': 'team_2_home', 'team_2_home': 'team_1_home',
                    'club1_avg_goals': 'club2_avg_goals', 'club2_avg_goals': 'club1_avg_goals',
                    'club1_avg_assists': 'club2_avg_assists', 'club2_avg_assists': 'club1_avg_assists',
                    'club1_avg_shots': 'club2_avg_shots', 'club2_avg_shots': 'club1_avg_shots',
                    'club1_avg_shots_on_target': 'club2_avg_shots_on_target',
                    'club2_avg_shots_on_target': 'club1_avg_shots_on_target',
                    'club1_avg_saves': 'club2_avg_saves', 'club2_avg_saves': 'club1_avg_saves'
                    }, axis=1)
        club1_row = club1_row[[
            'date', 'club1', 'club2', 'team_1_home', 'team_2_home', 'club1_avg_goals', 'club2_avg_goals',
            'club1_avg_assists', 'club2_avg_assists', 'club1_avg_shots', 'club2_avg_shots',
            'club1_avg_shots_on_target', 'club2_avg_shots_on_target', 'club1_avg_saves', 'club2_avg_saves',
            'winner'
        ]]
        club1_row = club1_row.drop('winner', axis=1)

    club2_row = winners[(winners['club2'].str.upper() == club2.upper())]
    club2_row = club2_row.drop('winner', axis=1)

    if club2_row.empty:
        club2_row = winners[(winners['club1'].str.upper() == club2.upper())]
        club2_row = club2_row. \
            rename({'club1': 'club2', 'club2': 'club1',
                    'team_1_home': 'team_2_home', 'team_2_home': 'team_1_home',
                    'club1_avg_goals': 'club2_avg_goals', 'club2_avg_goals': 'club1_avg_goals',
                    'club1_avg_assists': 'club2_avg_assists', 'club2_avg_assists': 'club1_avg_assists',
                    'club1_avg_shots': 'club2_avg_shots', 'club2_avg_shots': 'club1_avg_shots',
                    'club1_avg_shots_on_target': 'club2_avg_shots_on_target',
                    'club2_avg_shots_on_target': 'club1_avg_shots_on_target',
                    'club1_avg_saves': 'club2_avg_saves', 'club2_avg_saves': 'club1_avg_saves'
                    }, axis=1)
        club2_row = club2_row[[
            'date', 'club1', 'club2', 'team_1_home', 'team_2_home', 'club1_avg_goals', 'club2_avg_goals',
            'club1_avg_assists', 'club2_avg_assists', 'club1_avg_shots', 'club2_avg_shots',
            'club1_avg_shots_on_target', 'club2_avg_shots_on_target', 'club1_avg_saves', 'club2_avg_saves',
            'winner'
        ]]
        club2_row = club2_row.drop('winner', axis=1)

    if club1_row.empty | club2_row.empty:
        return {'winner': 'No club found'}

    club1_row = club1_row.head(1)
    club2_row = club2_row.head(1)

    club1 = club1_row['club1'].item()
    club2 = club2_row['club2'].item()

    club_merged_row = club1_row.copy()

    for column in club2_row:
        if '2' in column:
            club_merged_row[column] = list(club2_row[column].values)

    if club1_is_host:
        club_merged_row['team_1_home'] = 1
        club_merged_row['team_2_home'] = 0
    else:
        club_merged_row['team_1_home'] = 0
        club_merged_row['team_2_home'] = 1

    le = preprocessing.LabelEncoder()
    # noinspection PyProtectedMember
    for i in list(set(club_merged_row.columns) - set(club_merged_row._get_numeric_data().columns)):
        club_merged_row[i] = le.fit_transform(club_merged_row[i])

    clf = joblib.load("assets/machine_learning_models/winners.pkl")

    for column in club_merged_row.columns:
        club_merged_row[column] = float(club_merged_row[column])

    result = clf.predict(club_merged_row)[0]

    if result == 0:
        winner = club1
    else:
        winner = club2

    club_merged_row = club_merged_row.drop(['date', 'club1', 'club2'], axis=1)

    return {
        'winner': winner,
        'details': club_merged_row.iloc[0]
    }


@app.get('/machine_learning_models/goals')
async def goals(shots: int, shots_on_target: int):
    clf = joblib.load("assets/machine_learning_models/goals.pkl")

    df = pd.DataFrame({
        'shots': [shots],
        'shots_on_target': [shots_on_target],
    })

    result = clf.predict(df)[0]

    return {'Estimated Goals': round(float(result), 2)}
