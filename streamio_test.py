import requests
import pandas as pd
import streamlit as st
import streamlit_pandas as sp
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.express as px
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
@st.cache_data
def fastest_laps(df, circuit):
    cols = []
    for i in range(2008, 2024):
        url = f'http://ergast.com/api/f1/{i}/circuits/{circuit}/fastest/1/results.json'
        response = requests.get(url=url)
        temp = response.json()
        if temp["MRData"]["RaceTable"]['Races'] != []:
            latest_race = {
                "season": temp["MRData"]["RaceTable"]['season'],
                "driver": temp["MRData"]["RaceTable"]["Races"][0]['Results'][0]['Driver']['driverId'],
                "contructor": temp["MRData"]["RaceTable"]["Races"][0]['Results'][0]['Constructor']['constructorId'],
                "time": temp["MRData"]["RaceTable"]["Races"][0]['Results'][0]['FastestLap']['Time']['time'],
                "avg_speed": temp["MRData"]["RaceTable"]["Races"][0]['Results'][0]['FastestLap']['AverageSpeed']['speed']
            }
            cols.append(latest_race)
    df_1 = pd.DataFrame(cols)
    df = pd.concat([df, df_1], ignore_index=True)
    df['time'] = df['time'].apply(lambda x: datetime.strptime(x, "%M:%S.%f").strftime("%M:%S.%f")[:-3])
    df['avg_speed'] = pd.to_numeric(df['avg_speed'])
    return df.sort_values(['season'])

def circuit_names():
    url1 = 'http://ergast.com/api/f1/current/last/results.json'
    response = requests.get(url=url1)
    temp = response.json()
    week = temp["MRData"]["RaceTable"]['round']
    circuits = []
    for i in range(1,int(week)+1):
        url1 = f'http://ergast.com/api/f1/current/{i}/results.json'
        response = requests.get(url=url1)
        temp = response.json()
        race_id = temp["MRData"]["RaceTable"]['Races'][0]['Circuit']['circuitId']
        circuits.append(race_id)
    return circuits

def initialize():
    url = "http://ergast.com/api/f1/current/driverStandings.json"
    response = requests.get(url=url)
    temp = response.json()
    season = temp['MRData']['StandingsTable']['StandingsLists'][0]['season']
    round = temp['MRData']['StandingsTable']['StandingsLists'][0]['round']
    drivers = temp['MRData']['StandingsTable']['StandingsLists'][0]['DriverStandings']
    df_data =[]
    for entry in drivers:
        driver_info = entry["Driver"]
        constructor_info = entry["Constructors"][0]

        row = {
         "position": entry["position"],
         "points": entry["points"],
         "wins": entry["wins"],
         "driver": driver_info["code"],
          "constructor": constructor_info["name"],
          "season": season,
          "week": round,
          "driver_Season": str(season) + driver_info['code']

             }

        df_data.append(row)
    df = pd.DataFrame(df_data)
    df['position'] = pd.to_numeric(df['position'])
    df["points"] = pd.to_numeric(df['points'])
    df["wins"] = pd.to_numeric(df['wins'])
    df['driver'] = df['driver'].astype(str)
    df['constructor'] = df['constructor'].astype(str)
    df['season'] = pd.to_numeric(df['season'])
    return df,round

def populate_dataframe(calc:int,stage:int,df,year=2023):
    for i in range((year-calc),(year)):
        url = 'http://ergast.com/api/f1/{}/{}/driverStandings.json'.format(i,stage)
        response = requests.get(url=url)
        temp = response.json()
        season = temp['MRData']['StandingsTable']['StandingsLists'][0]['season']
        round = temp['MRData']['StandingsTable']['StandingsLists'][0]['round']
        drivers = temp['MRData']['StandingsTable']['StandingsLists'][0]['DriverStandings']
        for entry in drivers:
            driver_info = entry["Driver"]
            constructor_info = entry["Constructors"][0]

            new_row = {
                "position": entry["position"],
                "points": entry["points"],
                "wins": entry["wins"],
                "driver": driver_info["code"],
                "constructor": constructor_info["name"],
                "season": season,
                "week":stage,
                "driver_season": str(i) + driver_info['code']
            }
            new_row_df = pd.DataFrame(new_row, index=[0])
            df = pd.concat([df, new_row_df], ignore_index=True)
    df['position'] = pd.to_numeric(df['position'])
    df["points"] = pd.to_numeric(df['points'])
    df["wins"] = pd.to_numeric(df['wins'])
    df['driver'] = df['driver'].astype(str)
    df['constructor'] = df['constructor'].astype(str)
    df['season'] = pd.to_numeric(df['season'])
    return df


def end_of_season(calc:int,year=2023):
    df = pd.DataFrame()
    for i in range((year-calc),(year)):
        url = 'http://ergast.com/api/f1/{}/driverStandings.json'.format(i)
        response = requests.get(url=url)
        temp = response.json()
        drivers = temp['MRData']['StandingsTable']['StandingsLists'][0]['DriverStandings']
        for entry in drivers:
            driver_info = entry["Driver"]
            constructor_info = entry["Constructors"][0]

            new_row = {
                "position_EOS": entry["position"],
                "points_EOS": entry["points"],
                "wins_EOS": entry["wins"],
                "driver": driver_info["code"],
                "season": i,
                "driver_season": str(i) + driver_info['code']
            }
            new_row_df = pd.DataFrame(new_row, index=[0])
            df = pd.concat([df, new_row_df], ignore_index=True)
    df['position_EOS'] = pd.to_numeric(df['position_EOS'])
    df["points_EOS"] = pd.to_numeric(df['points_EOS'])
    df["wins_EOS"] = pd.to_numeric(df['wins_EOS'])
    df['driver'] = df['driver'].astype(str)
    return df



def main():
    st.title('F1 Analytics')
    st.sidebar.title('Settings')
    num_years = st.sidebar.slider('Select number of years', min_value=1, max_value=20, value=10)
    temporada_actual, round = initialize()
    standings = populate_dataframe(num_years, stage=round, df=temporada_actual)
    eos = end_of_season(num_years)
    final_test = standings[standings['season'] != 2023].merge(eos, how='left', on='driver_season')
    X = final_test[['position', 'points', "wins"]]
    y = final_test[['position_EOS', 'points_EOS', 'wins_EOS']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    regr = LinearRegression()
    regr.fit(X_train, y_train)
    y_pred = regr.predict(temporada_actual[['position', 'points', 'wins']])
    temporada_actual = temporada_actual.assign(pred_position=y_pred[:, 0], pred_points=y_pred[:, 1],
                                               pred_wins=y_pred[:, 2])
    hue = 'driver'
    style = 'constructor'
    tab1, tab2 = st.tabs(['Predictions', 'Fastest Times'])

    with tab1:
        st.subheader('Current Points vs Projected Points')
        fig = px.scatter(temporada_actual, x='points', y='pred_points', labels={'x': 'Points', 'y': 'Predicted Points'},
                          color="driver", symbol="constructor")
        st.plotly_chart(fig)

        st.subheader('Current Position vs Projected Position')
        fig = px.scatter(temporada_actual, x='position', y='pred_position', labels={'x': 'Position', 'y': 'Predicted Position'},
                         color="driver", symbol="constructor")
        st.plotly_chart(fig)

        st.subheader('Current Wins vs Projected Wins')
        fig = px.scatter(temporada_actual, x='wins', y='pred_wins', labels={'x': 'Wins', 'y': 'Predicted Wins'},
                         color="driver", symbol="constructor")
        st.plotly_chart(fig)

        st.write(" This prediction model has an R-squared score of: ", regr.score(X_test, y_test))
        driver_wins = standings.groupby('driver')['wins'].sum().sort_values(ascending=False)
        st.subheader(f'Driver Wins in the last {num_years} years')
        st.bar_chart(driver_wins)

    with tab2:
        df = pd.DataFrame()
        circuit = circuit_names()
        col1, col2 = st.columns(2)
        circuits = st.sidebar.selectbox('Select circuit', circuit)
        df = fastest_laps(df, circuits)
        with col1:
            df['time_in_seconds'] = df['time'].apply(lambda x: 60 * int(x.split(':')[0]) + float(x.split(':')[1]))
            create_data = {"constructor": "multiselect",
                           'driver':'multiselect'}

            all_widgets = sp.create_widgets(df, create_data)
            res = sp.filter_df(df, all_widgets)
            st.title('Fastest laps at {}'.format(circuits))
            st.write(res)
            current_year_avg_speed = df['avg_speed'].iloc[-1]
            previous_year_avg_speed = df['avg_speed'].iloc[-2]
            avg_speed_change = current_year_avg_speed - previous_year_avg_speed
            fig = px.line(df, x='season', y='time_in_seconds', labels={'x': 'Year', 'y': 'Fastest Lap Time (sec)'})
            st.plotly_chart(fig)
            difference_in_seconds = df['time_in_seconds'].iloc[-1] - df['time_in_seconds'].iloc[-2]
            fig = px.line(df, x='season', y='avg_speed', labels={'x': 'Year', 'y': 'Average Speed (mph)'})
            st.plotly_chart(fig)

        with col2:
            st.metric('Fastest Lap', df['time'].iloc[-1],
                      delta=f"{difference_in_seconds:.2f} seconds")
            st.metric('Average speed', f"{current_year_avg_speed:.2f}", delta=f"{avg_speed_change:.2f}")


if __name__ == '__main__':
    main()
