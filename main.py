import requests
from json import loads, dumps
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

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
temporada_actual,round = initialize()
temporada_actual['Position'] = pd.to_numeric(temporada_actual['Position'])
temporada_actual["Points"] = pd.to_numeric(temporada_actual['Points'])
temporada_actual["Wins"] = pd.to_numeric(temporada_actual['Wins'])
temporada_actual['Driver'] = temporada_actual['Driver'].astype(str)
temporada_actual['Constructor'] = temporada_actual['Constructor'].astype(str)
temporada_actual['Season'] = pd.to_numeric(temporada_actual['Season'])
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

standings = populate_dataframe(10, stage=round, df=temporada_actual)
standings['Position'] = pd.to_numeric(standings['Position'])
standings["Points"] = pd.to_numeric(standings['Points'])
standings["Wins"] = pd.to_numeric(standings['Wins'])
standings['Driver'] = standings['Driver'].astype(str)
standings['Constructor'] = standings['Constructor'].astype(str)
standings['Season'] = pd.to_numeric(standings['Season'])
top_ten_per_season = standings[standings['Position'] <= 10]
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

eos = end_of_season(10)
eos['Position_EOS'] = pd.to_numeric(eos['Position_EOS'])
eos["Points_EOS"] = pd.to_numeric(eos['Points_EOS'])
eos["Wins_EOS"] = pd.to_numeric(eos['Wins_EOS'])
eos['Driver'] = eos['Driver'].astype(str)
#top_ten_eos = eos[eos['Position'] <= 10]
final_test = standings[standings['Season'] != 2023].merge(eos,how='left',on='Driver_Season')
X = final_test[['Position','Points',"Wins"]]
y = final_test[['Position_EOS','Points_EOS','Wins_EOS']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
regr = LinearRegression()
regr.fit(X_train, y_train)
print("R-squared score: ", regr.score(X_test, y_test))
y_pred = regr.predict(temporada_actual[['Position','Points','Wins']])
temporada_actual['Pred_Position'] = y_pred[:,0]
temporada_actual['Pred_Points'] = y_pred[:,1]
temporada_actual['Pred_Wins'] = y_pred[:,2]
fig, ax = plt.subplots(1, 4, figsize=(19, 5))
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

plt.tight_layout()

plt.show()








