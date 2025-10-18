import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def hhmm_to_timedelta(hhmm):
    # NaN check
    if pd.isna(hhmm):
        return pd.NaT
    # Ensure it's treated as an integer
    hhmm = int(hhmm)
    # Extract hours and minutes
    hours = hhmm // 100
    minutes = hhmm % 100
    return pd.Timedelta(hours=hours, minutes=minutes)

airports = pd.read_csv('airports.csv')
airports = airports[airports['Country'] == 'United States']
airports = airports[airports['IATA'].str.len() == 3]
airlines = pd.read_csv('airlines.csv')
routes = pd.read_csv('routes.csv')
delays = pd.read_csv('OnTimePerformance_July2018.csv')#pd.read_parquet('delays.parquet', engine='fastparquet')

# Filter out airports for which we have no flight information
# Given 100000 rows, this leaves us with 60 airports, less than the 349 they have :(
# Including ORIGIN and DEST airports brings it up to 341 (this means some airports have only arriving flights?)
airports = pd.DataFrame(np.unique(np.concatenate([delays['ORIGIN'].unique(), delays['DEST'].unique()])), columns=['IATA'])

date_parts = {
    'year': delays['YEAR'],
    'month': delays['MONTH'],
    'day': delays['DAY_OF_MONTH']
}
delays['DATE'] = pd.to_datetime(date_parts)

start_date = pd.Timestamp('2018-07-01')
end_date = pd.Timestamp('2018-07-14')
delays = delays[(delays['DATE'] >= start_date) & (delays['DATE'] <= end_date)]

delays['DEP_SCH_TD'] = delays['CRS_DEP_TIME'].apply(hhmm_to_timedelta)
delays['ARR_SCH_TD'] = delays['CRS_ARR_TIME'].apply(hhmm_to_timedelta)
# Handle flights that cross midnight (arrival time < departure time)
# delays['ARR_DATE_OFFSET'] = (delays['ARR_SCH_TD'] < delays['DEP_SCH_TD']).astype(int)

# Calculate scheduled times
delays['DEP_SCH'] = (delays['DATE'] + delays['DEP_SCH_TD']).dt.floor("h")
delays['ARR_SCH'] = (delays['DATE'] + delays['ARR_SCH_TD']).dt.floor("h")
# delays['ARR_SCH'] = (delays['DATE'] + pd.to_timedelta(delays['ARR_DATE_OFFSET'], unit='D') + delays['ARR_SCH_TD']).dt.floor("h")
# Calculate actual times (scheduled + delay)
delays['DEP_ACT'] = (delays['DEP_SCH'] + pd.to_timedelta(delays['DEP_DELAY'], unit='m')).dt.floor("h")
delays['ARR_ACT'] = (delays['ARR_SCH'] + pd.to_timedelta(delays['ARR_DELAY'], unit='m')).dt.floor("h")

print(airports)
for i, row in airports.iterrows():
    airport_code = row['IATA']
    involving_airport = delays[(delays['ORIGIN'] == airport_code) | (delays['DEST'] == airport_code)].copy()
    delays_involving_airport = involving_airport[involving_airport['CARRIER_DELAY'] > 0]
    mean = delays_involving_airport['CARRIER_DELAY'].mean()
    print(f'We have {len(involving_airport)} data points involving {airport_code}, {len(delays_involving_airport)} are delayed with mean carrier delay: {mean}')

    # Filter flights by origin and destination separately
    departures = delays[delays['ORIGIN'] == airport_code].copy()
    arrivals = delays[delays['DEST'] == airport_code].copy()

    # Get departure events (only scheduled and actual departures)
    dep_events = departures[["DEP_SCH", "DEP_ACT"]].dropna()
    dep_events_melted = dep_events.melt(value_name="timestamp", var_name="event_type") if len(dep_events) > 0 else pd.DataFrame(columns=["event_type", "timestamp"])
    
    # Get arrival events (only scheduled and actual arrivals)
    arr_events = arrivals[["ARR_SCH", "ARR_ACT"]].dropna()
    arr_events_melted = arr_events.melt(value_name="timestamp", var_name="event_type") if len(arr_events) > 0 else pd.DataFrame(columns=["event_type", "timestamp"])
    
    # Combine departure and arrival events for this airport
    events = pd.concat([dep_events_melted, arr_events_melted], ignore_index=True)
    

    # Finding vulnerability of this airport
    # Timezones wont be an issue right? Wrong!
    
    #events = event_data.melt(value_name="timestamp", var_name="event_type")
    events = events[events["timestamp"].dt.hour >= 6] # Per paper we don't consider flights between 0 and 6 hours
    
    counts = events.groupby(["timestamp", "event_type"]).size().reset_index(name="total")
    pivoted = counts.pivot(index="timestamp", columns="event_type", values="total").fillna(0).astype(int)
    all_possible_events = ['DEP_SCH', 'ARR_SCH', 'DEP_ACT', 'ARR_ACT']
    pivoted = pivoted.reindex(columns=all_possible_events, fill_value=0)

    pivoted['PLANNED_CAP'] = pivoted['DEP_SCH'] + pivoted['ARR_SCH']
    pivoted['ACTUAL_CAP']  = pivoted['DEP_ACT'] + pivoted['ARR_ACT']
    pivoted['VULN']        = pivoted['ACTUAL_CAP'] > pivoted['PLANNED_CAP'] / 0.9 # Assume alpha = 0.9

    total_operation_hours = 18 * 14  # 252 hours
    hours_congested = pivoted['VULN'].sum()
    vulnerability = hours_congested / total_operation_hours
    airports.loc[i, 'VULN'] = vulnerability

    print(f'pivoted for {airport_code} is \n{pivoted}\n')
    print(f'Vulnerability for {airport_code} is {vulnerability:.2f}\n')

print(f'\n{airports}\n')

plt.figure(figsize=(8, 5))
plt.hist(airports["VULN"], bins=20, density=True, alpha=0.7, color='blue', edgecolor='black')

plt.xlabel("Vulnerability")
plt.xlim(0, 1)
plt.ylabel("Probability Density")
plt.title(f'Reproduction of Fig 1 - Original Dataset - ({start_date.date()} - {end_date.date()})')
plt.show()

