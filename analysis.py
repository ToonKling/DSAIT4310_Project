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
# Including ORIGIN and DEST airports brings it up to 341 (this means some airports have only arriving flights?)
airports = airports[airports['IATA'].isin(delays['ORIGIN'])| airports['IATA'].isin(delays['DEST'])]

# Timezones wont be an issue right? Wrong! We use local time for scheduled and actual times.
delays['DEP_DELAY_td'] = pd.to_timedelta(delays['DEP_DELAY'], unit='m')
delays['ARR_DELAY_td'] = pd.to_timedelta(delays['ARR_DELAY'], unit='m')

delays['DEP_SCH'] = pd.to_datetime(delays['Scheduled_DEP'], errors="coerce").dt.floor("h")
delays['ARR_SCH'] = pd.to_datetime(delays['Scheduled_ARR_Ori'], errors="coerce").dt.floor("h")
delays['DEP_ACT'] = (delays['DEP_SCH'] + delays['DEP_DELAY_td']).dt.floor("h")
delays['ARR_ACT'] = (delays['ARR_SCH'] + delays['ARR_DELAY_td']).dt.floor("h")

# Analysis period
start_date = delays['DEP_SCH'].min()# pd.Timestamp('2023-07-01') # delays['Scheduled_DEP'].min()
end_date = delays['ARR_SCH'].max()#pd.Timestamp('2023-07-14')  # inclusive
delays = delays[(delays['DEP_SCH'] >= start_date) & (delays['DEP_SCH'] <= end_date)]
delays = delays[(delays['ARR_SCH'] >= start_date) & (delays['ARR_SCH'] <= end_date)]
num_days = (end_date - start_date).days + 1 
hours_per_day = 18  # Hours 6:00-23:59 (excluding 0:00-05:59)
total_operation_hours = hours_per_day * num_days


for i, row in airports.iterrows():
    airport_code = row['IATA']
    involving_airport = delays[(delays['ORIGIN'] == airport_code) | (delays['DEST'] == airport_code)].copy()
    delays_involving_airport = involving_airport[involving_airport['CARRIER_DELAY'] > 0]
    mean = delays_involving_airport['CARRIER_DELAY'].mean()
    print(f'We have {len(involving_airport)} data points involving {airport_code}, {len(delays_involving_airport)} are delayed with mean carrier delay: {mean}')

    if len(delays_involving_airport) == 0:
        airports.loc[i, 'VULN'] = 0.0 # Skip airports with no delayed flights
        continue

    # Finding vulnerability of this airport
    
    # Filter flights by origin and destination separately 
    # We avoid double counting delays e.g. a delayed departure only counts as a delay for the ORIGIN airport, not DEST, unless arrival is delayed as well
    departures = delays[delays['ORIGIN'] == airport_code].copy()
    arrivals = delays[delays['DEST'] == airport_code].copy()

    dep_events = departures[["DEP_SCH", "DEP_ACT"]].dropna()
    dep_events_melted = dep_events.melt(value_name="timestamp", var_name="event_type")
    arr_events = arrivals[["ARR_SCH", "ARR_ACT"]].dropna()
    arr_events_melted = arr_events.melt(value_name="timestamp", var_name="event_type")
    
    events = pd.concat([dep_events_melted, arr_events_melted], ignore_index=True)

    events = events[events["timestamp"].dt.hour >= 6] # Per paper we don't consider flights between 0 and 6 hours
    
    counts = events.groupby(["timestamp", "event_type"]).size().reset_index(name="total")
    pivoted = counts.pivot(index="timestamp", columns="event_type", values="total").fillna(0).astype(int)

    all_possible_events = ['DEP_SCH', 'ARR_SCH', 'DEP_ACT', 'ARR_ACT']
    pivoted = pivoted.reindex(columns=all_possible_events, fill_value=0) # Deals with empty columns

    pivoted['PLANNED_CAP'] = pivoted['DEP_SCH'] + pivoted['ARR_SCH']
    pivoted['ACTUAL_CAP']  = pivoted['DEP_ACT'] + pivoted['ARR_ACT']
    pivoted['VULN']        = pivoted['ACTUAL_CAP'] > pivoted['PLANNED_CAP'] / 0.9 # Assume alpha = 0.9

    vulnerability = pivoted['VULN'].sum()
    vulnerability = vulnerability / total_operation_hours # Divide by total hours instead of mean() since some hours may not have any flights
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

