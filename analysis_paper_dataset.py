import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

start_date = pd.Timestamp('2018-07-01')
end_date = pd.Timestamp('2018-07-14')

def hhmm_to_timedelta(hhmm):
    # NaN check
    if pd.isna(hhmm):
        return pd.NaT
    # Ensure it's treated as an integer
    hhmm = int(hhmm)
    # Extract hours and minutes
    hours = hhmm // 100
    minutes = hhmm % 100
    assert minutes < 60
    return pd.Timedelta(hours=hours, minutes=minutes)

def get_airports_with_vulnerability():
    # Just some caching since this calculation is expensive
    # Delete this file to rerun calculation
    CACHE_FILE = "./airports_vuln.cache"
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)

    delays = pd.read_csv('data/OnTimePerformance_July2018.csv')

    # Filter out airports for which we have no flight information
    airports = pd.DataFrame(np.unique(np.concatenate([delays['ORIGIN'].unique(), delays['DEST'].unique()])), columns=['IATA'])

    date_parts = {
        'year': delays['YEAR'],
        'month': delays['MONTH'],
        'day': delays['DAY_OF_MONTH']
    }
    delays['DATE'] = pd.to_datetime(date_parts)

    delays = delays[(delays['DATE'] >= start_date) & (delays['DATE'] <= end_date)]

    delays['DEP_SCH_TD'] = delays['CRS_DEP_TIME'].apply(hhmm_to_timedelta)
    delays['ARR_SCH_TD'] = delays['CRS_ARR_TIME'].apply(hhmm_to_timedelta)

    # Calculate scheduled times
    delays['DEP_SCH'] = (delays['DATE'] + delays['DEP_SCH_TD']).dt.floor("h")
    delays['ARR_SCH'] = (delays['DATE'] + delays['ARR_SCH_TD']).dt.floor("h")

    # Calculate actual times (scheduled + delay)
    delays['DEP_ACT'] = delays['DATE'] + delays['DEP_TIME'].apply(hhmm_to_timedelta).dt.floor("h")
    delays['ARR_ACT'] = delays['DATE'] + delays['ARR_TIME'].apply(hhmm_to_timedelta).dt.floor("h")

    for i, row in airports.iterrows():
        airport_code = row['IATA']

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

    with open(CACHE_FILE, "wb") as f:
        pickle.dump(airports, f)
    return airports

if __name__ == "__main__":
    airports = get_airports_with_vulnerability()
    print(f'\n{airports}\n')

    plt.figure(figsize=(8, 5))

    binwidth = 1 / 45
    bins= np.arange(0, 1 , binwidth)
    plt.hist(airports["VULN"], bins=bins, density=True, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel("Vulnerability")
    plt.xlim(0, 1)
    plt.ylabel("Probability Density")
    plt.title(f'Reproduction of Fig 1 - Original Dataset - ({start_date.date()} - {end_date.date()})')
    plt.show()

