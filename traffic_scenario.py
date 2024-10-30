import numpy as np
import ast
import pandas as pd
from pyepo_asp import ASPmodel
# from pyepo.data.dataset import optDataset
from sklearn.model_selection import train_test_split
# from torch.utils.data import DataLoader
import torch.nn as nn
# import pyepo
# import time
import torch
# Optionally, you can also visualize the learning curve
# from vis import visLearningCurve

#-------------------------filter--------------
def safe_eval_list(s):
    """Safely evaluate string representation of a list with detailed error handling."""
    try:
        if isinstance(s, str):
            evaluated = ast.literal_eval(s)
            if not isinstance(evaluated, (list, tuple)):
                print(f"Warning: Evaluated to non-list type: {type(evaluated)}")
                return None
            return evaluated
        elif isinstance(s, (list, tuple)):
            return s
        else:
            print(f"Warning: Unexpected type: {type(s)}")
            return None
    except Exception as e:
        print(f"Error parsing: {type(e).__name__}: {str(e)}")
        print(f"Value (first 100 chars): {str(s)[:100]}...")
        return None

# Add validation after conversion
def validate_converted_data(df, list_columns):
    for col in list_columns:
        print(f"\nValidating column: {col}")
        # Check for None values
        null_count = df[col].isna().sum()
        if null_count > 0:
            print(f"Found {null_count} None values in {col}")
        
        # Check types
        type_counts = df[col].apply(type).value_counts()
        print("Type distribution:", type_counts)
        
        # For non-None values, check if they're lists and their lengths
        list_lengths = df[col][df[col].notna()].apply(len)
        if not list_lengths.empty:
            print("Length statistics:")
            print(list_lengths.describe())
            if len(list_lengths.unique()) > 1:
                print("Warning: Inconsistent lengths detected!")
                print("Unique lengths:", sorted(list_lengths.unique()))

df = pd.read_csv('instances/traffic_instances_n15_t0.75_all.csv')

# Convert columns containing lists from string to actual lists
list_columns = ['costs', 'wtc', 'original_feats', 'T_mean', 'T', 'callsigns', 'transit_times', 'relative_transit_times', 'cost_transit_time_diff', 'feats','transit_time_difference']

# Convert and validate
for col in list_columns:
    print(f"\nConverting column: {col}")
    df[col] = df[col].apply(safe_eval_list)
    if col not in ['callsigns', 'wtc']:
        df[col] = df[col].apply(lambda x: [float(i) if isinstance(i, (int, float, str)) else i for i in x] if isinstance(x, list) else x)

# validate_converted_data(df, list_columns)

# Create numpy array for costs

# Find rows with inconsistent lengths
feats_lengths = []
for idx, row in df.iterrows():
    # flattened_feats = [item for sublist in row['feats'] for item in sublist]
    flattened_feats = [item for sublist in row['original_feats'] for item in sublist]
    # additional_features = [
    #     row['wind_score'],
    #     row['dangerous_phenom_score'],
    #     row['precip_score'],
    #     row['vis_ceiling_score'],
    #     row['freezing_score'],
    # ]
    additional_features = []
    # additional_features.extend(row['costs'])
    # additional_features.extend(row['T'])
    combined_features = flattened_feats + additional_features
    feats_lengths.append(len(combined_features))

# Convert to Series for easier analysis
lengths_series = pd.Series(feats_lengths, index=df.index)
most_common_length = lengths_series.mode()[0]  # Get the most common length

# Remove rows with different lengths
inconsistent_rows = lengths_series[lengths_series != most_common_length].index
print(f"Removing {len(inconsistent_rows)} rows with inconsistent lengths")
print(f"Row indices being removed: {inconsistent_rows.tolist()}")

# Remove the inconsistent rows
df = df.drop(inconsistent_rows)
# Verify the cleanup
feats_list = []
for idx, row in df.iterrows():
    flattened_feats = [item for sublist in row['original_feats'] for item in sublist]
    # additional_features = [
    #     row['wind_score'],
    #     row['dangerous_phenom_score'],
    #     row['precip_score'],
    #     row['vis_ceiling_score'],
    #     row['freezing_score'],
    # ]
    additional_features = []
    # additional_features.extend(row['costs'])
    # additional_features.extend(row['T'])
    combined_features = flattened_feats + additional_features
    feats_list.append(combined_features)

#-------------------------filter--------------
#-------------------------training------------
# train model
# def trainModel(reg, loss_func, method_name, num_epochs=20, lr=1e-2):
#     # set adam optimizer
#     optimizer = torch.optim.Adam(reg.parameters(), lr=lr)
#     # train mode
#     reg.train()
#     # init log
#     loss_log = []
#     loss_log_regret = [pyepo.metric.regret(reg, optmodel, loader_test)]
#     # init elpased time
#     elapsed = 0
#     for epoch in range(num_epochs):
#         # start timing
#         tick = time.time()
#         # load data
#         for i, data in enumerate(loader_train):
#             x, c, w, z = data
#             # cuda
#             if torch.cuda.is_available():
#                 x, c, w, z = x.cuda(), c.cuda(), w.cuda(), z.cuda()
#             # forward pass
#             cp = reg(x)
#             if method_name == "spo+":
#                 loss = loss_func(cp, c, w, z)
#             if method_name in ["ptb", "pfy", "imle", "aimle", "nce", "cmap"]:
#                 loss = loss_func(cp, w)
#             if method_name in ["dbb", "nid"]:
#                 loss = loss_func(cp, c, z)
#             if method_name == "ltr":
#                 loss = loss_func(cp, c)
#             # backward pass
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             # record time
#             tock = time.time()
#             elapsed += tock - tick
#             # log
#             loss_log.append(loss.item())
#         regret = pyepo.metric.regret(reg, optmodel, loader_test)
#         loss_log_regret.append(regret)
#         print("Epoch {:2},  Loss: {:9.4f},  Regret: {:7.4f}%".format(epoch+1, loss.item(), regret*100))
#     print("Total Elapsed Time: {:.2f} Sec.".format(elapsed))
#     return loss_log, loss_log_regret
#-------------------------training------------
#-------------------------model----------------
class SimpleRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        return self.linear(x)

# Multi-layer Perceptron (MLP) model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x
#-------------------------model----------------


# Convert to numpy array and verify shape
feats = np.array(feats_list)
# costs = np.array(df['cost_transit_time_diff'].tolist())
# costs = np.array(df['transit_time_difference'].tolist())
costs = np.array(df['transit_times'].tolist())
print(f"\nFinal costs array shape: {costs.shape}")
print(f"\nFinal feats array shape: {feats.shape}")
min_time_window_row = df.loc[df['time_window_hours'].idxmin()]
# median_time_window_row = df.loc[df['time_window_hours'].median()]

# Find the row with the maximum values for the specified columns
max_wind_row = df.loc[df['wind_score'].idxmax()]
max_vis_row = df.loc[df['vis_ceiling_score'].idxmax()]
max_prec_row = df.loc[df['precip_score'].idxmax()]
max_danger_row = df.loc[df['dangerous_phenom_score'].idxmax()]

# Print the rows
print("Row with max wind_score:")
print(max_wind_row.wind_score)

print("\nRow with max vis_ceiling_score:")
print(max_vis_row.vis_ceiling_score)

print("\nRow with max dangerous_phenom_score:")
print(max_danger_row.dangerous_phenom_score)


test_set_up = min_time_window_row
#number of aircraft, must be the same number as the costs
n_aircraft = len(df['transit_times'][0])
# n_aircraft = len(df['transit_time_difference'][0])
E = dict(zip(range(n_aircraft), [t - 60 for t in test_set_up['relative_transit_times']]))
L = dict(zip(range(n_aircraft), [t + 1800 for t in test_set_up['relative_transit_times']]))
T = dict(zip(range(n_aircraft), test_set_up['relative_transit_times']))
sizes = dict(zip(range(n_aircraft), test_set_up['wtc']))
optmodel = ASPmodel(n_aircraft, E, L, sizes, T)

# split train test data
x_train, x_test, c_train, c_test = train_test_split(feats, costs, test_size=0.2, random_state=42)
# get optDataset
# dataset_train = optDataset(optmodel, x_train, c_train)
# dataset_test = optDataset(optmodel, x_test, c_test)

# # set data loader
# batch_size = 32
# loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
# loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)


# Get input and output sizes from the dataset
input_size = x_train.shape[-1]
output_size = c_train.shape[-1]
print(x_train.shape[-1])
# Simple one-layer regression model


# Initialize the models
simple_regression = SimpleRegression(input_size, output_size)
mlp = MLP(input_size, 64, output_size)  # Using 64 as hidden size, can be adjusted
if torch.cuda.is_available():
    simple_regression = simple_regression.cuda()
    mlp = mlp.cuda()


# spop = pyepo.func.SPOPlus(optmodel, processes=2)
# loss_log, loss_log_regret = trainModel(simple_regression, loss_func=spop, method_name="spo+")

# # Save the log
# # np.save('loss_log.npy', np.array(loss_log))
# # np.save('loss_log_regret.npy', np.array(loss_log_regret))

# visLearningCurve(loss_log, loss_log_regret,method='spo',n_aircraft = n_aircraft)


