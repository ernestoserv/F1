import requests
from json import loads, dumps
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
url = 'http://ergast.com/api/f1/current/last/fastest/1/results.json'
response = requests.get(url=url)
temp = response.json()
#print(dumps(temp,indent=2))
race_to_analyze = temp["MRData"]["RaceTable"]["Races"][0]["Circuit"]["circuitId"]
temp1 = []
latest_race = {
    "Season": temp["MRData"]["RaceTable"]['season'],
    "Driver":temp["MRData"]["RaceTable"]["Races"][0]['Results'][0]['Driver']['driverId'],
    "Contructor":temp["MRData"]["RaceTable"]["Races"][0]['Results'][0]['Constructor']['constructorId'],
    "Time":temp["MRData"]["RaceTable"]["Races"][0]['Results'][0]['FastestLap']['Time']['time'],
    "Avg_Speed":temp["MRData"]["RaceTable"]["Races"][0]['Results'][0]['FastestLap']['AverageSpeed']['speed']
}
temp1.append(latest_race)
df_1 = pd.DataFrame(temp1)
def create_dataframe(df,circuit):
    cols = []
    for i in range(2008,2023):
        url = f'http://ergast.com/api/f1/{i}/circuits/{circuit}/fastest/1/results.json'
        response = requests.get(url=url)
        temp = response.json()
        if temp["MRData"]["RaceTable"]['Races'] != []:
            latest_race = {
                "Season": temp["MRData"]["RaceTable"]['season'],
                "Driver": temp["MRData"]["RaceTable"]["Races"][0]['Results'][0]['Driver']['driverId'],
                "Contructor": temp["MRData"]["RaceTable"]["Races"][0]['Results'][0]['Constructor']['constructorId'],
                "Time": temp["MRData"]["RaceTable"]["Races"][0]['Results'][0]['FastestLap']['Time']['time'],
                 "Avg_Speed": temp["MRData"]["RaceTable"]["Races"][0]['Results'][0]['FastestLap']['AverageSpeed'][
                'speed']
        }
            cols.append(latest_race)
    df_1 = pd.DataFrame(cols)
    df = pd.concat([df, df_1], ignore_index=True)
    df['Time'] = df['Time'].apply(lambda x: datetime.strptime(x, "%M:%S.%f").strftime("%M:%S.%f")[:-3])
    df['Avg_Speed'] = pd.to_numeric(df['Avg_Speed'])
    return df.sort_values(['Season'])
print(create_dataframe(df_1, race_to_analyze))
