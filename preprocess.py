import pandas as pd

data = pd.read_csv('TA_restaurants_curated.csv')

global_number_rest = data['City'].value_counts(dropna=False)
print(global_number_rest)