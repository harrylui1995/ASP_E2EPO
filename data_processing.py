import pandas as pd
from typing import Union
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def add_relative_transit_time(data):
    """
    Add a column with the relative transit time based on the smallest value of time in data.
    
    Args:
    data (pandas.DataFrame): The input DataFrame for a group of 30 flights
    
    Returns:
    pandas.DataFrame: DataFrame with the new 'relative_transit_time' column
    """
    # Convert 'time' and 'arrival_time' to datetime if they're not already
    data['time'] = pd.to_datetime(data['time'], utc=True)
    data['arrival_time'] = pd.to_datetime(data['arrival_time'], utc=True)
    
    # Find the smallest time in the data
    smallest_time = data['time'].min()
    
    # Calculate T_mean
    data['T_mean'] = data['expected_transit_time'].mean()
    
    # Calculate the relative transit time
    data['relative_transit_time'] = data['expected_transit_time'] + (data['arrival_time'] - smallest_time).dt.total_seconds()
    
    return data

def process_flight_data(input_data: Union[str, pd.DataFrame], flights_per_instance: int = 15, max_time_window: float = 0.75) -> pd.DataFrame:
    """
    Process flight data to create instances with a fixed number of flights per instance,
    ensuring all flights in an instance occur within a specified time window.
    
    Parameters:
    input_data: Either a path to a CSV file (str) or a pandas DataFrame containing flight data
    flights_per_instance: Number of flights to include in each instance (default: 15)
    max_time_window: Maximum time window in hours for each instance (default: 0.75)
    
    Returns:
    pd.DataFrame: Processed DataFrame with fixed-size flight instances
    """
    # Load and prepare data
    if isinstance(input_data, str):
        if not os.path.exists(input_data):
            raise FileNotFoundError(f"The file {input_data} does not exist")
        df = pd.read_csv(input_data)
    elif isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
    else:
        raise TypeError("Input must be either a file path (str) or a pandas DataFrame")
    
    # Verify required columns
    required_columns = ['callsign', 'time', 'wtc', 'cost', 'expected_transit_time', 'transit_time']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Convert time column to datetime and remove timezone info
    df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)
    
    # Filter for high traffic hours (7-22)
    df_filtered = df[df['time'].dt.hour.between(6, 23)].copy()
    df_filtered = df_filtered.sort_values('time')
    
    # Initialize list to store instances
    instances = []
    current_start_idx = 0
    
    while current_start_idx + flights_per_instance <= len(df_filtered):
        # Get exactly 30 flights for this group
        group = df_filtered.iloc[current_start_idx:current_start_idx + flights_per_instance].copy()
        
        # Calculate time window for this group
        time_window = (group['time'].max() - group['time'].min()).total_seconds() / 3600  # in hours
        
        # Check if group meets time window constraint
        if time_window <= max_time_window:
            # Add relative transit time
            group = add_relative_transit_time(group)
            
            # Calculate transit time difference
            group['transit_time_difference'] = group['transit_time'] - group['T_mean']
            group['cost_transit_time_diff'] = group['cost'] * group['transit_time_difference']
            
            instance = {
                'instance_start_time': pd.to_datetime(group['time'].min()),
                'instance_end_time': pd.to_datetime(group['time'].max()),
                'time_window_hours': time_window,
                'costs': group['cost'].tolist(),
                'T': group['expected_transit_time'].tolist(),
                'T_mean': group['T_mean'].tolist(),
                'wtc': group['wtc'].tolist(),
                'callsigns': group['callsign'].tolist(),
                'transit_times': group['transit_time'].tolist(),
                'transit_time_difference': group['transit_time_difference'].tolist(),
                'entry_times': group['time'].tolist(),
                'relative_transit_times': group['relative_transit_time'].tolist(),
                'cost_transit_time_diff': group['cost_transit_time_diff'].tolist()
            }
            instances.append(instance)
            current_start_idx += flights_per_instance
        else:
            # Skip this window and move forward
            current_start_idx += 1
    
    # Convert to DataFrame
    result_df = pd.DataFrame(instances)
    
    if len(result_df) == 0:
        raise ValueError("No valid instances found with the specified constraints")
    
    # Ensure datetime types for time columns
    result_df['instance_start_time'] = pd.to_datetime(result_df['instance_start_time'])
    result_df['instance_end_time'] = pd.to_datetime(result_df['instance_end_time'])
    
    # Sort by instance_start_time
    result_df = result_df.sort_values('instance_start_time')
    
    # Print summary statistics
    print("\nProcessing Summary:")
    print(f"Total number of valid instances: {len(result_df)}")
    print(f"Average time window: {result_df['time_window_hours'].mean():.2f} hours")
    print(f"Date range: {result_df['instance_start_time'].min()} to {result_df['instance_end_time'].max()}")
    
    return result_df

def feature_generation(original_df, processed_df):
    """
    Generate features for the fixed-size flight instances.
    """
    # Ensure time columns are the same datetime type
    original_df['time'] = pd.to_datetime(original_df['time']).dt.tz_localize(None)
    processed_df['instance_start_time'] = pd.to_datetime(processed_df['instance_start_time']).dt.tz_localize(None)
    processed_df['instance_end_time'] = pd.to_datetime(processed_df['instance_end_time']).dt.tz_localize(None)
    
    weather_columns = ['wind_score', 'dangerous_phenom_score', 'precip_score', 
                      'vis_ceiling_score', 'freezing_score']
    
    weather_features = []
    for idx, row in processed_df.iterrows():
        mask = (original_df['time'] >= row['instance_start_time']) & \
               (original_df['time'] <= row['instance_end_time'])
        instance_weather = original_df[mask][weather_columns].mean()
        weather_features.append(instance_weather)
    
    weather_df = pd.DataFrame(weather_features, index=processed_df.index)
    processed_df = pd.concat([processed_df, weather_df], axis=1)
    
    # Store original features for each instance
    original_features = []
    
    # Perform PCA on trajectory features
    pca_columns = ['lat', 'lon', 'velocity', 'heading', 'vertrate', 'baroaltitude']
    
    # Standardize the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(original_df[pca_columns])
    
    # Apply PCA
    pca = PCA(n_components=0.95)
    pca_result = pca.fit_transform(scaled_features)
    
    # Store PCA features and original features for each instance
    pca_features = []
    for idx, row in processed_df.iterrows():
        mask = (original_df['time'] >= row['instance_start_time']) & \
               (original_df['time'] <= row['instance_end_time'])
        instance_indices = original_df[mask].index
        instance_pca = [pca_result[i].tolist() for i in instance_indices]
        pca_features.append(instance_pca)
        
        # Store original features
        instance_original = original_df.loc[instance_indices, pca_columns].values.tolist()
        original_features.append(instance_original)
    
    processed_df['feats'] = pca_features
    processed_df['original_feats'] = original_features
    return processed_df


if __name__ == "__main__":
    n_aircraft = 30
    max_time_window = 2
    df = pd.read_csv('data/filtered_flight_data_aug_sep.csv')
    processed_df = process_flight_data(df, flights_per_instance=n_aircraft, max_time_window=max_time_window)
    enhanced_df = feature_generation(df, processed_df)
    
    # Create a filename that includes n_aircraft and max_time_window
    output_filename = f'instances/traffic_instances_n{n_aircraft}_t{max_time_window:.2f}_aug_sep.csv'
    
    enhanced_df.to_csv(output_filename, index=False)
    print(f"Saved to {output_filename}")
    print(enhanced_df['costs'].apply(len).value_counts())

    