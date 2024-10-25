from pyepo_asp import ASPmodel
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
# 


def generate_multi_date_relative_transit_time(data, time_window=3600, flights_per_window=100):
    """
    Generate a new column with the relative transit time for flights across multiple dates,
    distributing them into specified time windows.
    
    Args:
    data (pandas.DataFrame): The input DataFrame containing 'time' and 'expected_transit_time' columns
    time_window (int): The time window in seconds to distribute flights (default: 3600, i.e., 1 hour)
    flights_per_window (int): The target number of flights to distribute within each time window (default: 100)
    
    Returns:
    pandas.DataFrame: DataFrame with the new 'relative_transit_time' column
    """
    # Convert 'time' to datetime if it's not already
    data['time'] = pd.to_datetime(data['time'], utc=True)
    
    # Sort the dataframe by time
    data = data.sort_values('time')
    
    # Function to process each group of flights
    def process_group(group):
        num_flights = len(group)
        num_windows = max(1, num_flights // flights_per_window)
        total_time = num_windows * time_window
        
        # Generate new relative times within the total time range
        new_relative_times = np.linspace(0, total_time, num_flights)
        
        # Add the new relative times to the expected transit time
        group['relative_transit_time'] = group['expected_transit_time'] + new_relative_times
        return group

    # Process the entire dataset
    data = process_group(data)
    
    return data

# Example usage:
# Assuming 'df' is your DataFrame with 'time' and 'expected_transit_time' columns
if __name__ =="__main__":
    
    df = pd.read_csv('filtered_flight_data_with_costs.csv')
    sampled_df = df.sample(n=1000, random_state=42)
    sampled_df = sampled_df.reset_index(drop=True)
    sampled_df = generate_multi_date_relative_transit_time(sampled_df, time_window=3600, flights_per_window=40)
    # Sample 100 flights randomly
    
    
    # Reset the index of the sampled DataFrame
    
    
    print("Sampled 100 flights:")
    print(sampled_df[['relative_transit_time']].head())
    
    # Update df to use the sampled data
    df = sampled_df
    print(df.relative_transit_time)
    # Sort the DataFrame based on relative_transit_time from smallest to largest
    df = df.sort_values('relative_transit_time')
    
    # Reset the index after sorting
    df = df.reset_index(drop=True)
    
    print("DataFrame sorted by relative_transit_time:")
    print(df[['relative_transit_time']].head())  # Display the first few rows to verify sorting
    T = df['relative_transit_time']
    n_aircraft = len(df)
    E = dict(zip(range(n_aircraft), df['relative_transit_time']-60))
    L = dict(zip(range(n_aircraft), df['relative_transit_time']+1800))
    T = dict(zip(range(n_aircraft), df['relative_transit_time']))
    sizes = dict(zip(range(n_aircraft), df['wtc']))
    feats = df[['lat', 'lon', 'velocity', 'heading', 'vertrate', 'baroaltitude', 'wind_score', 
                'dangerous_phenom_score', 'precip_score', 'vis_ceiling_score', 'freezing_score', 
                'aircraft_count_1h', 'expected_transit_time', 'cost']]

    costs = df['cost'] * df['transit_time_difference']
    optmodel = ASPmodel(n_aircraft, E, L, sizes, T)
    optmodel.setObj(costs)  # Set the objective function
    optmodel.solve() 
    # optmodel.setObj(costs)
    # # Set the objective function
    # optmodel.solve()  # Solve the model

    # Perform post-analysis
    analysis = optmodel.post_analysis()

    # Print the results
    print("\nDetailed Comparison:")
    print(analysis['summary'])

    print("\nSummary:")
    print(f"Optimized total cost: {analysis['optimized_cost']:.2f}")
    print(f"FCFS total cost: {analysis['fcfs_cost']:.2f}")
    print(f"Cost improvement: {analysis['cost_improvement']}")
    print(f"Optimized late landings: {analysis['opt_late_landings']}")
    print(f"FCFS late landings: {analysis['fcfs_late_landings']}")
    print(f"Late landings improvement: {analysis['late_landings_improvement']}")

    x_train, x_test, c_train, c_test = train_test_split(feats, costs, test_size=1000, random_state=42)

