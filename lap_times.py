import requests
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

url1 = 'http://ergast.com/api/f1/current/last/results.json'
response = requests.get(url=url1)
temp = response.json()
print(temp)