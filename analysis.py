import pandas as pd
import matplotlib.pyplot as plt

airports = pd.read_csv('airports.csv')
airports = airports[airports['Country'] == 'United States']
airports = airports[airports['IATA'].str.len() == 3]
airlines = pd.read_csv('airlines.csv')
routes = pd.read_csv('routes.csv')
delays = pd.read_parquet('delays.parquet', engine='fastparquet')

# Filter out airports for which we have no flight information
# Given 100000 rows, this leaves us with 60 airports, less than the 349 they have :(
airports = airports[airports['IATA'].isin(delays['ORIGIN'])]

for i, row in airports.iterrows():
    airport_code = row['IATA']
    involving_airport = delays[(delays['ORIGIN'] == airport_code) | (delays['DEST'] == airport_code)].copy()
    delays_involving_airport = involving_airport[involving_airport['CARRIER_DELAY'] > 0]
    mean = delays_involving_airport['CARRIER_DELAY'].mean()
    print(f'We have {len(involving_airport)} data points involving {airport_code}, {len(delays_involving_airport)} are delayed with mean carrier delay: {mean}')

    # Finding vulnerability of this airport
    # Timezones wont be an issue right?
    involving_airport['DEP_SCH'] = (pd.to_datetime(involving_airport['Scheduled_DEP'], errors="coerce").dt.floor("h"))
    involving_airport['DEP_ACT'] = pd.to_datetime(involving_airport['Actual_DEP_dt_EST'], errors="coerce").dt.floor("h")
    involving_airport['ARR_SCH'] = pd.to_datetime(involving_airport['Scheduled_ARR_Ori'], errors="coerce").dt.floor("h")
    involving_airport['ARR_ACT'] = pd.to_datetime(involving_airport['Actual_ARR_dt_Ori'], errors="coerce").dt.floor("h")
    involving_airport = involving_airport[["DEP_SCH", "DEP_ACT", "ARR_SCH", "ARR_ACT"]]
    involving_airport = involving_airport.dropna()

    events = involving_airport.melt(value_name="timestamp", var_name="event_type")
    events = events[events["timestamp"].dt.hour > 6] # Per paper we don't consider flights between 0 and 6 hours
    counts = events.groupby(["timestamp", "event_type"]).size().reset_index(name="total")
    pivoted = counts.pivot(index="timestamp", columns="event_type", values="total").fillna(0).astype(int)
    pivoted['PLANNED_CAP'] = pivoted['DEP_SCH'] + pivoted['ARR_SCH']
    pivoted['ACTUAL_CAP']  = pivoted['DEP_ACT'] + pivoted['ARR_ACT']
    pivoted['VULN']        = pivoted['ACTUAL_CAP'] > pivoted['PLANNED_CAP'] / 0.9 # Assume alpha = 0.9
    vulnerability = pivoted['VULN'].mean()
    airports.loc[i, 'VULN'] = vulnerability

    print(f'pivoted for {airport_code} is \n{pivoted}\n')
    print(f'Vulnerability for {airport_code} is {vulnerability:.2f}\n')

print(f'\n{airports}\n')

plt.figure(figsize=(8, 5))

plt.hist(airports["VULN"], bins=20, density=True, alpha=0.7, color='blue', edgecolor='black')

plt.xlabel("Vulnerability")
plt.xlim(0, 1)
plt.ylabel("Probability Density")
plt.title("Reproduction of Fig 1")
plt.show()

