import torch
import time
from torch.utils.data import DataLoader
import os
from datetime import datetime
import json
from vis import visLearningCurve
from pyepo.data.dataset import optDataset
import pyepo
from traffic_scenario import SimpleRegression,MLP,x_test,x_train,c_train,c_test,optmodel,n_aircraft
from pyepo.func import (
        # SPOPlus, 
        # perturbedFenchelYoung,
        # perturbedOpt,
        # implicitMLE, 
        blackboxOpt, 
        negativeIdentity,
        # contrastiveMAP,
        # listwiseLTR
    )
def create_experiment_dirs():
    """Create directories for storing experiment results"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = f'experiments_{timestamp}'
    dirs = {
        'base': base_dir,
        'logs': f'{base_dir}/logs',
        'figures': f'{base_dir}/figures',
        'models': f'{base_dir}/models'
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs

def run_experiment(model, method_name, loss_func, loader_train, loader_test, optmodel, 
                  experiment_dirs, model_name, num_epochs=50, lr=1e-2):
    """Run a single experiment with given configuration"""
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    
    # Initialize logs
    loss_log = []
    loss_log_regret = [pyepo.metric.regret(model, optmodel, loader_test)]
    elapsed = 0
    
    print(f"\nStarting experiment - Model: {model_name}, Method: {method_name}")
    
    for epoch in range(num_epochs):
        tick = time.time()
        
        for i, data in enumerate(loader_train):
            x, c, w, z = data
            if torch.cuda.is_available():
                x, c, w, z = x.cuda(), c.cuda(), w.cuda(), z.cuda()
            
            # Forward pass
            cp = model(x)
            
            # Calculate loss based on method
            if method_name == "spo+":
                loss = loss_func(cp, c, w, z)
            elif method_name in ["ptb", "pfy", "imle", "aimle", "nce", "cmap"]:
                loss = loss_func(cp, w)
            elif method_name in ["dbb", "nid"]:
                loss = loss_func(cp, c, z)
            elif method_name == "ltr":
                loss = loss_func(cp, c)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            tock = time.time()
            elapsed += tock - tick
            loss_log.append(loss.item())
        
        # Calculate and log regret
        regret = pyepo.metric.regret(model, optmodel, loader_test)
        loss_log_regret.append(regret)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}, Loss: {loss.item():9.4f}, Regret: {regret*100:7.4f}%")
    
    print(f"Total Elapsed Time: {elapsed:.2f} Sec.")
    
    # Save results
    experiment_results = {
        'model_name': model_name,
        'method_name': method_name,
        'loss_log': loss_log,
        'loss_log_regret': loss_log_regret,
        'elapsed_time': elapsed,
        'final_regret': regret,
        'hyperparameters': {
            'num_epochs': num_epochs,
            'learning_rate': lr,
            'batch_size': loader_train.batch_size
        }
    }
    
    # Save logs
    save_experiment_results(experiment_results, experiment_dirs)
    
    # Save model
    save_model(model, model_name, method_name, experiment_dirs)
    
    # Generate and save figure
    save_learning_curve(loss_log, loss_log_regret, model_name, method_name, 
                       experiment_dirs, n_aircraft)
    
    return experiment_results

def save_experiment_results(results, dirs):
    """Save experiment logs and results"""
    filename = f"{results['model_name']}_{results['method_name']}_results.json"
    filepath = os.path.join(dirs['logs'], filename)
    
    # Convert numpy arrays to lists for JSON serialization
    results['loss_log'] = [float(x) for x in results['loss_log']]
    results['loss_log_regret'] = [float(x) for x in results['loss_log_regret']]
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=4)

def save_model(model, model_name, method_name, dirs):
    """Save trained model"""
    filename = f"{model_name}_{method_name}_model.pth"
    filepath = os.path.join(dirs['models'], filename)
    torch.save(model.state_dict(), filepath)

def save_learning_curve(loss_log, loss_log_regret, model_name, method_name, dirs, n_aircraft):
    """Save learning curve visualization"""
    # Store current directory
    current_dir = os.getcwd()
    
    # Change to the figures directory temporarily
    os.chdir(dirs['figures'])
    
    # Call visLearningCurve with the correct arguments
    visLearningCurve(
        loss_log=loss_log,
        loss_log_regret=loss_log_regret,
        method=f"{model_name}_{method_name}",
        n_aircraft=n_aircraft
    )
    
    # The figure is already saved by visLearningCurve, but let's rename it to match our naming convention
    old_filename = f"{model_name}_{method_name}_{n_aircraft}.png"
    new_filename = f"{model_name}_{method_name}_learning_curve.png"
    
    # Rename if the file exists
    if os.path.exists(old_filename):
        os.rename(old_filename, new_filename)
    
    # Change back to original directory
    os.chdir(current_dir)

def run_pipeline(x_train, x_test, c_train, c_test, optmodel, n_aircraft, batch_size=32):
    """Main pipeline to run all experiments"""
    
    # Create experiment directories
    experiment_dirs = create_experiment_dirs()
    
    # Create datasets and dataloaders
    dataset_train = optDataset(optmodel, x_train, c_train)
    dataset_test = optDataset(optmodel, x_test, c_test)
    
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
    
    # Model configurations
    input_size = x_train.shape[-1]
    output_size = c_train.shape[-1]
    
    models = {
        'simple_regression': SimpleRegression(input_size, output_size),
        'mlp': MLP(input_size, 64, output_size)
    }
    
    # Training methods and their loss functions
    methods = {
            # 'spo+': SPOPlus(optmodel, processes=2),
            # 'pfy': perturbedFenchelYoung(
            #     optmodel,
            #     processes=2,
            #     sigma=1.0,  # noise standard deviation
            #     n_samples=10  # number of samples
            # ),
            # 'ptb': perturbedOpt(
            #     optmodel,
            #     processes=2,
            #     # sigma=1.0,  # noise standard deviation
            #     # n_samples=10 # number of samples
            # ),
            # 'imle': implicitMLE(
            #     optmodel,
            #     # n_samples=10, 
            #     # sigma=1.0, 
            #     # lambd=10, 
            #     processes=2 
            # ),
            # 'nce': NCE(
            #     optmodel, processes=2
            #     # solve_ratio=0.05, dataset=dataset_train  # number of samples
            # ),
            'dbb': blackboxOpt(
                optmodel,
                processes=2,
                # lambd=20  # temperature parameter
            ),
            'nid': negativeIdentity(
                optmodel,
                processes=2
            ),
            # 'cmap': contrastiveMAP(
            #     optmodel,
            #     processes=2,
            #     # solve_ratio=0.05, 
            #     # dataset=dataset_train
            # ),
            # 'ltr': listwiseLTR(
            #     optmodel,
            #     processes=2,
            #     # solve_ratio=0.05, 
            #     # dataset=dataset_train
            # )
        }
    
    # Move models to GPU if available
    if torch.cuda.is_available():
        for model in models.values():
            model.cuda()
    
    # Run experiments for each combination
    results = {}
    for model_name, model in models.items():
        results[model_name] = {}
        for method_name, loss_func in methods.items():
            print(f"\nRunning experiment with {model_name} and {method_name}")
            results[model_name][method_name] = run_experiment(
                model=model,
                method_name=method_name,
                loss_func=loss_func,
                loader_train=loader_train,
                loader_test=loader_test,
                optmodel=optmodel,
                experiment_dirs=experiment_dirs,
                model_name=model_name,
                num_epochs=20
            )
    
    return results, experiment_dirs

# Example usage
if __name__ == "__main__":
    # Your existing data preparation code here
    results, experiment_dirs = run_pipeline(x_train, x_test, c_train, c_test, 
                                         optmodel, n_aircraft)
    print(f"\nExperiments completed. Results saved in: {experiment_dirs['base']}")