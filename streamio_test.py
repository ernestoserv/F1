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
    return df



def main():
    temporada_actual, round = initialize()
    temporada_actual['Position'] = pd.to_numeric(temporada_actual['Position'])
    temporada_actual["Points"] = pd.to_numeric(temporada_actual['Points'])
    temporada_actual["Wins"] = pd.to_numeric(temporada_actual['Wins'])
    temporada_actual['Driver'] = temporada_actual['Driver'].astype(str)
    temporada_actual['Constructor'] = temporada_actual['Constructor'].astype(str)
    temporada_actual['Season'] = pd.to_numeric(temporada_actual['Season'])
    num_years = st.slider('Select number of years', min_value=1, max_value=20, value=10)
    standings = populate_dataframe(num_years, stage=round, df=temporada_actual)
    standings['Position'] = pd.to_numeric(standings['Position'])
    standings["Points"] = pd.to_numeric(standings['Points'])
    standings["Wins"] = pd.to_numeric(standings['Wins'])
    standings['Driver'] = standings['Driver'].astype(str)
    standings['Constructor'] = standings['Constructor'].astype(str)
    standings['Season'] = pd.to_numeric(standings['Season'])
    top_ten_per_season = standings[standings['Position'] <= 10]
    eos = end_of_season(num_years)
    eos['Position_EOS'] = pd.to_numeric(eos['Position_EOS'])
    eos["Points_EOS"] = pd.to_numeric(eos['Points_EOS'])
    eos["Wins_EOS"] = pd.to_numeric(eos['Wins_EOS'])
    eos['Driver'] = eos['Driver'].astype(str)
    final_test = standings[standings['Season'] != 2023].merge(eos, how='left', on='Driver_Season')
    X = final_test[['Position', 'Points', "Wins"]]
    y = final_test[['Position_EOS', 'Points_EOS', 'Wins_EOS']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    regr = LinearRegression()
    regr.fit(X_train, y_train)
    print("R-squared score: ", regr.score(X_test, y_test))
    y_pred = regr.predict(temporada_actual[['Position', 'Points', 'Wins']])
    temporada_actual['Pred_Position'] = y_pred[:, 0]
    temporada_actual['Pred_Points'] = y_pred[:, 1]
    temporada_actual['Pred_Wins'] = y_pred[:, 2]
    fig, ax = plt.subplots(1, 4,figsize=(60,15))
    fig.suptitle(f'Current Points, Position, and Wins vs Projected Points, Position, and Wins')
    hue = 'Driver'
    style = 'Constructor'
    scatter1 = sns.scatterplot(temporada_actual, x='Points', y='Pred_Points', hue=hue, style=style, s=100, ax=ax[0])
    scatter2 = sns.scatterplot(temporada_actual, x='Position', y='Pred_Position', hue=hue, style=style, s=100, ax=ax[1])
    scatter3 = sns.scatterplot(temporada_actual, x='Wins', y='Pred_Wins', hue=hue, style=style, s=100, ax=ax[2])
    handles, labels = scatter1.get_legend_handles_labels()
    scatter1.legend_.remove()
    scatter2.legend_.remove()
    scatter3.legend_.remove()
    ax_legend = ax[3].axis('off')
    legend = fig.legend(handles, labels, title='Drivers(Color) and Constructors(Shape)', loc='right', ncol=4)
    ax[0].set_xlabel('Points')
    ax[0].set_ylabel('Predicted Points')
    ax[0].set_title('Points vs. Predicted Points')

    ax[1].set_xlabel('Position')
    ax[1].set_ylabel('Predicted Position')
    ax[1].set_title('Position vs. Predicted Position')

    ax[2].set_xlabel('Wins')
    ax[2].set_ylabel('Predicted Wins')
    ax[2].set_title('Wins vs. Predicted Wins')
    st.title('F1 Analytics')

    col1, col2 = st.columns(2)
    with col1:
        plt.tight_layout()
        st.subheader('Current Points, Position, and Wins vs Projected Points, Position, and Wins')
        st.pyplot(fig)
        driver_wins = standings.groupby('Driver')['Wins'].sum().sort_values(ascending=False)
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
