import pandas as pd
import duckdb

airports = pd.read_csv('airports.csv')
airports = airports[airports['Country'] == 'United States']
airports = airports[airports['IATA'].str.len() == 3]
airlines = pd.read_csv('airlines.csv')
routes = pd.read_csv('routes.csv')
nr_rows = 10000 # Total dataset size is 5559465
delays = duckdb.query(f'SELECT * FROM "{'./delays.parquet'}" LIMIT {nr_rows};').df()

for _, row in airports.iterrows():
    airport_code = row['IATA']
    involving_airport = delays[(delays['ORIGIN'] == airport_code) | (delays['DEST'] == airport_code)]
    delays_involving_airport = involving_airport[involving_airport['CARRIER_DELAY'] > 0]
    mean = delays_involving_airport['CARRIER_DELAY'].mean()
    print(f'We have {len(involving_airport)} data points involving {airport_code}, {len(delays_involving_airport)} are delayed with mean carrier delay: {mean}')

