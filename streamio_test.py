import requests
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
@st.cache_data
def create_dataframe(df, circuit):
    cols = []
    for i in range(2008, 2024):
        url = f'http://ergast.com/api/f1/{i}/circuits/{circuit}/fastest/1/results.json'
        response = requests.get(url=url)
        temp = response.json()
        if temp["MRData"]["RaceTable"]['Races'] != []:
            latest_race = {
                "Season": temp["MRData"]["RaceTable"]['season'],
                "Driver": temp["MRData"]["RaceTable"]["Races"][0]['Results'][0]['Driver']['driverId'],
                "Contructor": temp["MRData"]["RaceTable"]["Races"][0]['Results'][0]['Constructor']['constructorId'],
                "Time": temp["MRData"]["RaceTable"]["Races"][0]['Results'][0]['FastestLap']['Time']['time'],
                "Avg_Speed": temp["MRData"]["RaceTable"]["Races"][0]['Results'][0]['FastestLap']['AverageSpeed']['speed']
            }
            cols.append(latest_race)
    df_1 = pd.DataFrame(cols)
    df = pd.concat([df, df_1], ignore_index=True)
    df['Time'] = df['Time'].apply(lambda x: datetime.strptime(x, "%M:%S.%f").strftime("%M:%S.%f")[:-3])
    df['Avg_Speed'] = pd.to_numeric(df['Avg_Speed'])
    return df.sort_values(['Season'])

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
         "Position": entry["position"],
         "Points": entry["points"],
         "Wins": entry["wins"],
         "Driver": driver_info["code"],
          "Constructor": constructor_info["name"],
          "Season": season,
          "Week": round,
          "Driver_Season": str(season) + driver_info['code']

             }

        df_data.append(row)
    df = pd.DataFrame(df_data)
    df['Position'] = pd.to_numeric(df['Position'])
    df["Points"] = pd.to_numeric(df['Points'])
    df["Wins"] = pd.to_numeric(df['Wins'])
    df['Driver'] = df['Driver'].astype(str)
    df['Constructor'] = df['Constructor'].astype(str)
    df['Season'] = pd.to_numeric(df['Season'])
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
                "Position": entry["position"],
                "Points": entry["points"],
                "Wins": entry["wins"],
                "Driver": driver_info["code"],
                "Constructor": constructor_info["name"],
                "Season": season,
                "Week":stage,
                "Driver_Season": str(i) + driver_info['code']
            }
            new_row_df = pd.DataFrame(new_row, index=[0])
            df = pd.concat([df, new_row_df], ignore_index=True)
    df['Position'] = pd.to_numeric(df['Position'])
    df["Points"] = pd.to_numeric(df['Points'])
    df["Wins"] = pd.to_numeric(df['Wins'])
    df['Driver'] = df['Driver'].astype(str)
    df['Constructor'] = df['Constructor'].astype(str)
    df['Season'] = pd.to_numeric(df['Season'])
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
                "Position_EOS": entry["position"],
                "Points_EOS": entry["points"],
                "Wins_EOS": entry["wins"],
                "Driver": driver_info["code"],
                "Season": i,
                "Driver_Season": str(i) + driver_info['code']
            }
            new_row_df = pd.DataFrame(new_row, index=[0])
            df = pd.concat([df, new_row_df], ignore_index=True)
    df['Position_EOS'] = pd.to_numeric(df['Position_EOS'])
    df["Points_EOS"] = pd.to_numeric(df['Points_EOS'])
    df["Wins_EOS"] = pd.to_numeric(df['Wins_EOS'])
    df['Driver'] = df['Driver'].astype(str)
    return df



def main():
    st.title('F1 Analytics')
    st.sidebar.title('Settings')
    num_years = st.sidebar.slider('Select number of years', min_value=1, max_value=20, value=10)
    temporada_actual, round = initialize()
    standings = populate_dataframe(num_years, stage=round, df=temporada_actual)
    top_ten_per_season = standings[standings['Position'] <= 10]
    eos = end_of_season(num_years)
    final_test = standings[standings['Season'] != 2023].merge(eos, how='left', on='Driver_Season')
    X = final_test[['Position', 'Points', "Wins"]]
    y = final_test[['Position_EOS', 'Points_EOS', 'Wins_EOS']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    regr = LinearRegression()
    regr.fit(X_train, y_train)
    y_pred = regr.predict(temporada_actual[['Position', 'Points', 'Wins']])
    temporada_actual['Pred_Position'] = y_pred[:, 0]
    temporada_actual['Pred_Points'] = y_pred[:, 1]
    temporada_actual['Pred_Wins'] = y_pred[:, 2]
    hue = 'Driver'
    style = 'Constructor'

    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Current Points vs Projected Points')
        fig, ax = plt.subplots()
        scatter1 = sns.scatterplot(temporada_actual, x='Points', y='Pred_Points', hue=hue, style=style, s=100)
        handles, labels = scatter1.get_legend_handles_labels()
        legend = fig.legend(handles, labels, title='Drivers(Color) and Constructors(Shape)', loc='bottom', ncol=4)
        st.pyplot(fig)
        fig, ax = plt.subplots()
        st.subheader('Current Position vs Projected Position')
        scatter2 = sns.scatterplot(temporada_actual, x='Position', y='Pred_Position', hue=hue, style=style, s=100)
        st.pyplot(fig)
        st.subheader('Current Wins vs Projected Wins')
        fig, ax = plt.subplots()
        scatter3 = sns.scatterplot(temporada_actual, x='Wins', y='Pred_Wins', hue=hue, style=style, s=100)
        st.pyplot(fig)
        st.write(" This prediction model has an R-squared score of: ", regr.score(X_test, y_test))
        driver_wins = standings.groupby('Driver')['Wins'].sum().sort_values(ascending=False)
        st.subheader(f'Driver Wins in the last {num_years} years')
        st.bar_chart(driver_wins)
    df = pd.DataFrame()
    circuit = circuit_names()
    with col2:

        circuits = st.selectbox('Select circuit', circuit)
        df = create_dataframe(df, circuits)
        st.title('Fastest laps at {}'.format(circuits))
        st.table(df)
        current_year_avg_speed = df['Avg_Speed'].iloc[-1]
        previous_year_avg_speed = df['Avg_Speed'].iloc[-2]
        avg_speed_change = current_year_avg_speed - previous_year_avg_speed
        st.metric('Average speed', f"{current_year_avg_speed:.2f}", delta=f"{avg_speed_change:.2f}")
        # Add a new chart that displays the fastest laps
        df['Time_in_seconds'] = df['Time'].apply(lambda x: 60 * int(x.split(':')[0]) + float(x.split(':')[1]))
        fig, ax = plt.subplots()
        sns.lineplot(data=df, x='Season', y='Time_in_seconds')
        ax.set_xlabel('Year')
        ax.set_ylabel('Fastest Lap Time (sec)')
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)
        difference_in_seconds = df['Time_in_seconds'].iloc[-1] - df['Time_in_seconds'].iloc[-2]
        st.metric('Fastest Lap', df['Time'].iloc[-1],
                  delta=f"{difference_in_seconds:.2f} seconds")
        fig, ax = plt.subplots()
        sns.lineplot(data=df, x='Season', y='Avg_Speed')
        ax.set_xlabel('Year')
        ax.set_ylabel('Average Speed (mph)')
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)


if __name__ == '__main__':
    main()
