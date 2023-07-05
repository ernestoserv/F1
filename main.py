import requests
from json import loads, dumps
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
          "Week": round

             }

        df_data.append(row)
    df = pd.DataFrame(df_data)
    return df,round
temporada_actual,round = initialize()
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
                "Week":stage
            }
            new_row_df = pd.DataFrame(new_row, index=[0])
            df = pd.concat([df, new_row_df], ignore_index=True)
    return df

standings = populate_dataframe(4, stage=round, df=temporada_actual)
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
                "Position": entry["position"],
                "Points": entry["points"],
                "Wins": entry["wins"],
                "Driver": driver_info["code"],
                "Constructor": constructor_info["name"],
                "Season": i,
                "Week": 'Last Week of Season'
            }
            new_row_df = pd.DataFrame(new_row, index=[0])
            df = pd.concat([df, new_row_df], ignore_index=True)
    return df

eos = end_of_season(4)
eos['Position'] = pd.to_numeric(eos['Position'])
eos["Points"] = pd.to_numeric(eos['Points'])
eos["Wins"] = pd.to_numeric(eos['Wins'])
eos['Driver'] = eos['Driver'].astype(str)
eos['Constructor'] = eos['Constructor'].astype(str)
top_ten_eos = eos[eos['Position'] <= 10]
final_test = pd.concat([top_ten_per_season,top_ten_eos],ignore_index=True)
fig, ax = plt.subplots(1,5,figsize=(22,7))
fig.suptitle(f'End of Season Standing vs Standing of Week {round} over the past 4 seasons')
sns.set_style('whitegrid')
ax[0].set_title('2019 Season')
ax[1].set_title('2020 Season')
ax[2].set_title('2021 Season')
ax[3].set_title('2022 Season')
ax[4].set_title('2023 Season')
sns.barplot(data=final_test[final_test['Season'] == 2019],y='Driver',x='Position',hue='Week',ax=ax[0],orient='h',palette='Blues')
sns.barplot(data=final_test[final_test['Season'] == 2020],y='Driver',x='Position',hue='Week',ax=ax[1],orient='h',palette='Reds')
sns.barplot(data=final_test[final_test['Season'] == 2021],y='Driver',x='Position',hue='Week',ax=ax[2],orient='h',palette='RdBu')
sns.barplot(data=final_test[final_test['Season'] == 2022],y='Driver',x='Position',hue='Week',ax=ax[3],orient='h',palette='Greens')
sns.barplot(data=final_test[final_test['Season'] == 2023],y='Driver',x='Position',hue='Week',ax=ax[4],orient='h',palette='Purples')
ax[0].bar_label(ax[0].containers[0])
ax[0].bar_label(ax[0].containers[1])
ax[1].bar_label(ax[1].containers[0])
ax[1].bar_label(ax[1].containers[1])
ax[2].bar_label(ax[2].containers[0])
ax[2].bar_label(ax[2].containers[1])
ax[3].bar_label(ax[3].containers[0])
ax[3].bar_label(ax[3].containers[1])
ax[4].bar_label(ax[4].containers[0])




plt.show()
print(final_test.head())