from pyepo_asp import ASPmodel
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
def get_hour_data(df, target_hour):
    """
    Get data for a specific hour from the DataFrame.
    
    Args:
    df (pandas.DataFrame): The input DataFrame
    target_hour (str): The target hour in format 'YYYY-MM-DD HH:MM:SS+00:00'
    
    Returns:
    pandas.DataFrame: Filtered DataFrame for the specified hour
    """
    target_datetime = pd.to_datetime(target_hour)
    return df[df['hour'].dt.floor('H') == target_datetime.floor('H')]

def get_busiest_hour(df, start_date, end_date):
    """
    Find the hour with the most number of rows within a given date range.
    
    Args:
    df (pandas.DataFrame): The input DataFrame
    start_date (str): The start date in format 'YYYY-MM-DD'
    end_date (str): The end date in format 'YYYY-MM-DD'
    
    Returns:
    str: The busiest hour in format 'YYYY-MM-DD HH:00:00+00:00'
    """
    # Convert start and end dates to datetime
    start = pd.to_datetime(start_date).tz_localize('UTC')
    end = pd.to_datetime(end_date).tz_localize('UTC')
    df['hour'] = pd.to_datetime(df['hour'], utc=True)
    # Filter the DataFrame for the given date range
    mask = (df['hour'] >= start) & (df['hour'] <= end)
    date_range_df = df.loc[mask]
    
    # Group by hour and count the number of rows
    hourly_counts = date_range_df.groupby(date_range_df['hour'].dt.floor('H')).size()
    
    # Find the hour with the maximum count
    busiest_hour = hourly_counts.idxmax()
    
    return busiest_hour.strftime('%Y-%m-%d %H:00:00+00:00')

def get_busiest_date(df):
    """
    Find the day with the most number of rows and return the time of 7:00 AM on that day.
    
    Args:
    df (pandas.DataFrame): The input DataFrame
    
    Returns:
    str: The busiest date with time set to 07:00:00+00:00
    """
    df['date'] = pd.to_datetime(df['hour'], utc=True).dt.date
    
    # Group by date and count the number of rows
    daily_counts = df.groupby('date').size()
    
    # Find the date with the maximum count
    busiest_date = daily_counts.idxmax()
    
    # Create a datetime object for 7:00 AM on the busiest date
    busiest_time = pd.Timestamp(busiest_date).replace(hour=7, minute=0, second=0, microsecond=0, tzinfo=pd.UTC)
    
    return busiest_time.strftime('%Y-%m-%d %H:00:00+00:00')



def add_relative_transit_time(data):
    """
    Add a column with the relative transit time based on the smallest value of time in data,
    using R = expected_transit_time + (arrival_time - smallest(time))
    
    Args:
    data (pandas.DataFrame): The input DataFrame for a specific hour
    
    Returns:
    pandas.DataFrame: DataFrame with the new 'relative_transit_time' column
    """
    # Convert 'time' and 'arrival_time' to datetime if they're not already
    data['time'] = pd.to_datetime(data['time'], utc=True)
    data['arrival_time'] = pd.to_datetime(data['arrival_time'], utc=True)
    
    # Find the smallest time in the data
    smallest_time = data['time'].min()
    
    # Calculate the relative transit time
    data['relative_transit_time'] = data['expected_transit_time'] + (data['arrival_time'] - smallest_time).dt.total_seconds()
    
    return data

def insert_monk_data(data, num_monks=2):
        """
        Insert monk data based on the relative landing time.
        
        Args:
        data (pandas.DataFrame): The input DataFrame with relative_transit_time
        num_monks (int): Number of monk rows to insert (default: 2)
        
        Returns:
        pandas.DataFrame: DataFrame with inserted monk data
        """
        # Sort the data by relative_transit_time
        data = data.sort_values('relative_transit_time')
        
        # Calculate the average gap between landings
        avg_gap = data['relative_transit_time'].diff().mean()
        
        # Create a new DataFrame for monk data
        monk_data = pd.DataFrame({
            'callsign': [f'MONK{i+1}' for i in range(num_monks)],
            'wtc': ['M'] * num_monks,  # Assuming Medium wake turbulence category for monks
            'relative_transit_time': [
                data['relative_transit_time'].iloc[i * len(data) // (num_monks + 1)] + avg_gap/2
                for i in range(1, num_monks + 1)
            ]
        })
        
        # Concatenate the original data with monk data and sort again
        data = pd.concat([data, monk_data]).sort_values('relative_transit_time').reset_index(drop=True)
        
        return data

if __name__ == '__main__':
    # Load data
    df = pd.read_csv('first_three_weeks_august.csv')
    # Convert 'hour' column to datetime
    df['hour'] = pd.to_datetime(df['hour'])
    # Example usage:
    busiest_hour = get_busiest_hour(df, '2024-08-01', '2024-08-21')
    time = '2024-08-09 20:00:00+00:00' 
    print(busiest_hour)
    # Apply the function to add the new column
    hour_data = get_hour_data(df, busiest_hour)
    hour_data = add_relative_transit_time(hour_data)
    
    
    
    # hour_data = insert_monk_data(hour_data,num_monks=0)
    
    print(hour_data)
    # Model parameters
    planes = hour_data['callsign'].unique()
    n_aircraft = len(planes)
    E = dict(zip(range(n_aircraft), hour_data['relative_transit_time']-60))
    L = dict(zip(range(n_aircraft), hour_data['relative_transit_time']+1800))
    T = dict(zip(range(n_aircraft), hour_data['relative_transit_time']))
    sizes = dict(zip(range(n_aircraft), hour_data['wtc']))
    mean_transit_time = sum(T.values()) / len(T)
    # mean_transit_time = hour_data['transit_time'].mean()
    c = dict(zip(range(n_aircraft), hour_data['cost']*hour_data['transit_time_difference']))
    # Create and solve the model
    # Assuming you've already created and solved the model
    feats = hour_data[['lat', 'lon', 'velocity', 'heading', 'vertrate', 'baroaltitude', 'wind_score', 
                'dangerous_phenom_score', 'precip_score', 'vis_ceiling_score', 'freezing_score', 
                'aircraft_count_1h', 'expected_transit_time', 'cost']].to_numpy()

    costs = hour_data['cost'] * hour_data['transit_time_difference']
    costs = costs.to_numpy()
    model = ASPmodel(n_aircraft, E, L, sizes, T)
    # model.setObj(costs)  # Set the objective function
    # model.solve()  # Solve the model

    # # Perform post-analysis
    # analysis = model.post_analysis()

    # # Print the results
    # print("\nDetailed Comparison:")
    # print(analysis['summary'])

    # print("\nSummary:")
    # print(f"Optimized total cost: {analysis['optimized_cost']:.2f}")
    # print(f"FCFS total cost: {analysis['fcfs_cost']:.2f}")
    # print(f"Cost improvement: {analysis['cost_improvement']}")
    # print(f"Optimized late landings: {analysis['opt_late_landings']}")
    # print(f"FCFS late landings: {analysis['fcfs_late_landings']}")
    # print(f"Late landings improvement: {analysis['late_landings_improvement']}")
    
    