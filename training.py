import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import pandas as pd
import ast

def calculate_input_sizes(sample_row):
    """Calculate input sizes for each feature type from a sample row."""
    n_costs = len(safe_eval_list(sample_row['costs']))
    n_times = len(safe_eval_list(sample_row['T_mean']))
    n_weather = 5  # wind, dangerous_phenom, precip, vis_ceiling, freezing
    n_feats = len(safe_eval_list(sample_row['feats']))  # Changed this line

    input_sizes = {
        'costs': n_costs,
        'times': n_times,
        'weather': n_weather,
        'feats': n_feats
    }
    
    print(f"\nFeature Dimensions:")
    print(f"Costs: {n_costs}")
    print(f"Times: {n_times}")
    print(f"Weather: {n_weather}")
    print(f"Feats: {n_feats}")
    print(f"Total features: {sum(input_sizes.values())}")
    
    return input_sizes

def safe_eval_list(s):
    """Safely evaluate string representation of a list."""
    try:
        if isinstance(s, str):
            return ast.literal_eval(s)
        return s
    except:
        print(f"Error parsing: {s[:100]}...")  # Print first 100 chars
        return None

class FlightDataset(Dataset):
    def __init__(self, processed_df, scalers=None, is_training=True):
        """
        Convert processed DataFrame into flat tabular format with proper string parsing.
        """
        all_features = []
        all_targets = []
        
        if is_training:
            scalers = {
                'pca': StandardScaler(),
                'times': StandardScaler(),
                'costs': StandardScaler(),
                'weather': StandardScaler(),
                'feats': StandardScaler()  # Add this line
            }
        self.scalers = scalers
        
        for idx, row in processed_df.iterrows():
            try:
                # Parse lists from string representations
                costs = safe_eval_list(row['costs'])
                times = safe_eval_list(row['T'])
                cost_time_diff = safe_eval_list(row['cost_transit_time_diff'])
                
                # Verify all required data is available
                if any(x is None for x in [costs, times, cost_time_diff]):
                    print(f"Skipping row {idx}: Contains None values")
                    continue
                
                costs = np.array(costs, dtype=np.float32)
                times = np.array(times, dtype=np.float32)
                cost_time_diff = np.array(cost_time_diff, dtype=np.float32)
                feats = np.array(safe_eval_list(row['feats']), dtype=np.float32)
                
                if np.isnan(feats).any():
                    print(f"Skipping row {idx}: Contains NaN values in feats")
                    continue
                # Weather features (already in correct format)
                weather = np.array([
                    row['wind_score'],
                    row['dangerous_phenom_score'],
                    row['precip_score'],
                    row['vis_ceiling_score'],
                    row['freezing_score']
                ], dtype=np.float32)
                
                if any(np.isnan(x).any() for x in [costs, times, cost_time_diff, weather]):
                    print(f"Skipping row {idx}: Contains NaN values")
                    continue
                
                # Verify dimensions
                if len(costs) == 0 or len(times) == 0 or len(cost_time_diff) == 0:
                    print(f"Skipping row {idx}: Empty lists")
                    continue
                
                # Stack features
                features = np.concatenate([
                    costs,
                    times,
                    weather,
                ])
                
                # Scale features
                # Scale features
                if is_training:
                    costs = scalers['costs'].fit_transform(costs.reshape(-1, 1)).ravel()
                    times = scalers['times'].fit_transform(times.reshape(-1, 1)).ravel()
                    weather = scalers['weather'].fit_transform(weather.reshape(-1, 1)).ravel()
                    feats = scalers['feats'].fit_transform(feats.reshape(-1, 1)).ravel()
                else:
                    costs = scalers['costs'].transform(costs.reshape(-1, 1)).ravel()
                    times = scalers['times'].transform(times.reshape(-1, 1)).ravel()
                    weather = scalers['weather'].transform(weather.reshape(-1, 1)).ravel()
                    feats = scalers['feats'].transform(feats.reshape(-1, 1)).ravel()
                
                features = np.concatenate([costs, times, weather, feats])
                all_features.append(features)
                all_targets.append(cost_time_diff)
                
            except Exception as e:
                print(f"Error processing row {idx}: {str(e)}")
                continue
        
        if len(all_features) == 0:
            raise ValueError("No valid instances found after preprocessing. Check your data and preprocessing steps.")
        
        self.features = torch.tensor(np.array(all_features), dtype=torch.float32)
        self.targets = torch.tensor(np.array(all_targets), dtype=torch.float32)
        
        print(f"Successfully processed {len(self.features)} instances")
        print(f"Feature shape: {self.features.shape}")
        print(f"Target shape: {self.targets.shape}")
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'target': self.targets[idx]
        }

def validate_data(df):
    print("Validating data...")
    print(f"DataFrame shape: {df.shape}")
    print("\nColumn dtypes:")
    print(df.dtypes)
    print("\nSample data:")
    print(df.head())
    print("\nChecking for NaN values:")
    print(df.isna().sum())
    print("\nChecking list columns:")
    for col in ['costs', 'T', 'cost_transit_time_diff']:
        sample = safe_eval_list(df[col].iloc[0])
        print(f"{col}: {type(sample)}, length: {len(sample) if sample is not None else 'N/A'}")

class TabularFlightModel(nn.Module):
    def __init__(self, input_sizes, output_size, hidden_sizes=[512, 256, 128, 64]):
        super().__init__()
        print(f"Creating model with input sizes: {input_sizes}, output size: {output_size}")
        
        # Separate feature processing paths
        self.cost_path = nn.Sequential(
            nn.Linear(input_sizes['costs'], 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2)
        )
        
        self.time_path = nn.Sequential(
            nn.Linear(input_sizes['times'], 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2)
        )
        
        self.weather_path = nn.Sequential(
            nn.Linear(input_sizes['weather'], 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2)
        )
        
        self.feats_path = nn.Sequential(
            nn.Linear(input_sizes['feats'], 64),  # Changed this to 64
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2)
        )
        
        # Combined processing
        combined_input_size = 64 + 64 + 32 + 64 
        
        layers = []
        prev_size = combined_input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(0.3)
            ])
            prev_size = hidden_size
        
        # Final prediction layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.combined_layers = nn.Sequential(*layers)
    
    def forward(self, x):
        # Split input into different feature types
        costs = x[:, :self.cost_path[0].in_features]
        times = x[:, self.cost_path[0].in_features:self.cost_path[0].in_features + self.time_path[0].in_features]
        weather = x[:, self.cost_path[0].in_features + self.time_path[0].in_features:self.cost_path[0].in_features + self.time_path[0].in_features + self.weather_path[0].in_features]
        feats = x[:, self.cost_path[0].in_features + self.time_path[0].in_features + self.weather_path[0].in_features:]
        
        # Process each path
        cost_features = self.cost_path(costs)
        time_features = self.time_path(times)
        weather_features = self.weather_path(weather)
        feats_features = self.feats_path(feats)
        
        # Combine features
        combined = torch.cat([cost_features, time_features, weather_features, feats_features], dim=1)
        
        # Final processing
        return self.combined_layers(combined)

def prepare_data(processed_df, batch_size=32, train_split=0.8):
    """
    Prepare DataLoaders with proper string parsing.
    """
    print("\nData Preview:")
    print(f"Total rows: {len(processed_df)}")
    print("\nSample values from first row:")
    first_row = processed_df.iloc[0]
    print(f"Costs (first few): {str(first_row['costs'])[:100]}...")
    print(f"T (first few): {str(first_row['T'])[:100]}...")
    print(f"Cost-time diff (first few): {str(first_row['cost_transit_time_diff'])[:100]}...")
    
    try:
        # Create training dataset and get fitted scalers
        train_dataset = FlightDataset(processed_df.iloc[:int(len(processed_df)*train_split)], 
                                    is_training=True)
        
        # Create validation dataset using scalers from training
        val_dataset = FlightDataset(processed_df.iloc[int(len(processed_df)*train_split):], 
                                  scalers=train_dataset.scalers,
                                  is_training=False)
        
        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        return train_loader, val_loader, train_dataset.scalers
    
    except Exception as e:
        print(f"Error in data preparation: {e}")
        raise
def train_model(model, train_loader, val_loader, n_epochs=100, learning_rate=0.001):
    """
    Train the model with early stopping.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss = 0
        for batch in train_loader:
            features = batch['features'].to(device)
            targets = batch['target'].to(device)
            
            predictions = model(features)
            loss = criterion(predictions, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device)
                targets = batch['target'].to(device)
                
                predictions = model(features)
                val_loss += criterion(predictions, targets).item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
            
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{n_epochs}]')
            print(f'Train Loss: {avg_train_loss:.4f}')
            print(f'Val Loss: {avg_val_loss:.4f}')


def prepare_and_train(processed_df, batch_size=16, train_split=0.8, epochs=200):
    """
    Prepare data and train model with improved training process.
    """
    # Calculate input sizes and prepare data
    input_sizes = calculate_input_sizes(processed_df.iloc[0])
    train_loader, val_loader, scalers = prepare_data(processed_df, batch_size=batch_size)
    
    # Determine the output size from the target
    sample_batch = next(iter(train_loader))
    output_size = sample_batch['target'].shape[1]
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TabularFlightModel(input_sizes=input_sizes, output_size=output_size).to(device)
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                         factor=0.5, patience=10, 
                                                         verbose=True)
    
    # Training loop with early stopping
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    
    print("\nStarting training...")
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        for batch in train_loader:
            features = batch['features'].to(device)
            targets = batch['target'].to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device)
                targets = batch['target'].to(device)
                outputs = model(features)
                val_loss += criterion(outputs, targets).item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f'\nEarly stopping at epoch {epoch+1}')
            break
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}]')
            print(f'Training Loss: {avg_train_loss:.4f}')
            print(f'Validation Loss: {avg_val_loss:.4f}')
        
        # Print some predictions vs actual values every 50 epochs
        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                sample_batch = next(iter(val_loader))
                sample_features = sample_batch['features'].to(device)
                sample_targets = sample_batch['target'].cpu().numpy()[0]
                sample_predictions = model(sample_features).cpu().numpy()[0]
                
                print("\nSample Predictions vs Actual:")
                for i in range(5):  # Show first 5 values
                    print(f"Predicted: {sample_predictions[i]:.2f}, Actual: {sample_targets[i]:.2f}")
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pt'))
    
    return model, scalers, train_losses, val_losses

# Usage example
if __name__ == "__main__":
    try:
        processed_df = pd.read_csv('traffic_instances_fixed.csv')
        validate_data(processed_df)
        model, scalers, train_losses, val_losses = prepare_and_train(processed_df)
        
        # Plot training history
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.yscale('log')
        plt.show()
        
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
