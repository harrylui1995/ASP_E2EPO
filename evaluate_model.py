import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from traffic_scenario import SimpleRegression, MLP, x_test, c_test, optmodel

def evaluate_trained_model(model_path, x_test, c_test, optmodel, model_type='mlp'):
    """
    Load trained model and evaluate its predictions, comparing with true costs and FCFS
    
    Args:
        model_path (str): Path to the trained model
        x_test: Test input data
        c_test: Test cost data
        optmodel: Optimization model instance
        model_type (str): Type of model ('mlp' or 'simple_regression')
    """
    # Initialize model based on type
    input_size = x_test.shape[-1]
    output_size = c_test.shape[-1]
    
    if model_type == 'mlp':
        model = MLP(input_size, 64, output_size)
    else:  # simple_regression
        model = SimpleRegression(input_size, output_size)
    
    # Load trained weights
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Generate predictions
    with torch.no_grad():
        predicted_costs = model(torch.FloatTensor(x_test))
    
    results = []
    # For each test instance
    for i in range(len(x_test)):
        # Get the predicted and true costs for this instance
        instance_pred_costs = predicted_costs[i].numpy()
        instance_true_costs = c_test[i]
        
        # Create copies of the optimization model for both scenarios
        pred_model = optmodel.copy()
        true_model = optmodel.copy()
        
        # Solve with predicted costs
        pred_model.setObj(instance_pred_costs)
        pred_sol, pred_obj = pred_model.solve()
        pred_analysis = pred_model.post_analysis()
        
        # Solve with true costs
        true_model.setObj(instance_true_costs)
        true_sol, true_obj = true_model.solve()
        true_analysis = true_model.post_analysis()
        
        # Calculate optimality gap
        if true_obj != 0:
            optimality_gap = ((pred_obj - true_obj) / true_obj) * 100
        else:
            optimality_gap = 0 if pred_obj == 0 else 100
        
        results.append({
            'Instance': i,
            'Predicted_Cost': pred_obj,
            'True_Cost': true_obj,
            'FCFS_Cost': true_analysis['fcfs_cost'],
            'Optimality_Gap(%)': optimality_gap
        })
    
    return pd.DataFrame(results)

def collect_scenario_results(scenarios):
    """
    Collect results from different scenarios for both methods
    """
    all_results = {}
    
    for scenario in scenarios:
        # Load MLP+SPO+ results
        mlp_path = f"{scenario}/models/mlp_spo+_model.pth"
        mlp_results = evaluate_trained_model(mlp_path, x_test, c_test, optmodel, model_type='mlp')
        
        # Load Linear Regression+Two-Stage results
        lr_path = f"{scenario}/models/simple_regression_2s_model.pth"
        lr_results = evaluate_trained_model(lr_path, x_test, c_test, optmodel, model_type='simple_regression')
        
        # Store results for this scenario
        all_results[scenario] = {
            'mlp_spo': mlp_results['Predicted_Cost'],
            'lr_two_stage': lr_results['Predicted_Cost'],
            'true_cost': mlp_results['True_Cost'],
            'fcfs_cost': mlp_results['FCFS_Cost']
        }
    
    return all_results

def plot_method_comparison(scenarios, save_path=None):
    """
    Create an enhanced error bar plot comparing MLP+SPO+ and Linear Regression+Two-Stage methods
    with improved styling and visualization.
    """
    # Set up plotting style
    plt.style.use('seaborn-v0_8')
    rc('font', family='serif', size=20)
    rc('text', usetex=True)
    rc('axes.spines', top=False, right=False)
    
    # Custom color palette
    COLORS = {
        'mlp': '#2E86C1',      # Deep blue
        'lr': '#28B463',       # Forest green
        'true': '#E74C3C',     # Red
        'fcfs': '#34495E'      # Dark gray
    }
    
    # Collect results
    results = collect_scenario_results(scenarios)
    
    # Prepare data for plotting
    n_scenarios = len(scenarios)
    x = np.arange(n_scenarios)
    width = 0.35
    
    # Calculate statistics
    stats = {
        'means_mlp': [],
        'means_lr': [],
        'means_true': [],
        'means_fcfs': [],
        'errs_mlp': [],
        'errs_lr': []
    }
    
    for scenario in scenarios:
        # Method predictions
        stats['means_mlp'].append(results[scenario]['mlp_spo'].mean())
        stats['means_lr'].append(results[scenario]['lr_two_stage'].mean())
        stats['errs_mlp'].append(results[scenario]['mlp_spo'].std() / np.sqrt(len(results[scenario]['mlp_spo'])))
        stats['errs_lr'].append(results[scenario]['lr_two_stage'].std() / np.sqrt(len(results[scenario]['lr_two_stage'])))
        
        # Reference values
        stats['means_true'].append(results[scenario]['true_cost'].mean())
        stats['means_fcfs'].append(results[scenario]['fcfs_cost'].mean())
    
    # Create figure with golden ratio
    fig, ax = plt.subplots(figsize=(12, 12 * 0.618))
    
    # Plot bars with enhanced styling
    rects1 = ax.bar(x - width/2, stats['means_mlp'], width, yerr=stats['errs_mlp'],
                   label='MLP + SPO+', color=COLORS['mlp'], alpha=0.8,
                   capsize=5, error_kw={'capthick': 2, 'ecolor': COLORS['mlp']})
    rects2 = ax.bar(x + width/2, stats['means_lr'], width, yerr=stats['errs_lr'],
                   label='Linear Regression + Two-Stage', color=COLORS['lr'], alpha=0.8,
                   capsize=5, error_kw={'capthick': 2, 'ecolor': COLORS['lr']})
    
    # Plot reference lines with enhanced styling
    ax.hlines(y=stats['means_true'], xmin=-width*2, xmax=len(scenarios)-1 + width*2, 
              linestyles='--', color=COLORS['true'], 
              label='Optimized True Cost', linewidth=2.5, alpha=0.9)
    ax.hlines(y=stats['means_fcfs'], xmin=-width*2, xmax=len(scenarios)-1 + width*2,
              linestyles='--', color=COLORS['fcfs'],
              label='FCFS Cost', linewidth=2.5, alpha=0.9)
    
    # Add text annotations with adjusted positions
    ax.text(len(scenarios)-1, stats['means_fcfs'][0], f'FCFS cost: {stats["means_fcfs"][0]:.1f}',
                ha='right', va='bottom',
                color=COLORS['fcfs'], alpha=0.9,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=2))
    ax.text(len(scenarios)-1, stats['means_true'][0], f'Opt true cost: {stats["means_true"][0]:.1f}',
                ha='right', va='top',
                color=COLORS['true'], alpha=0.9,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=2))
    
    # Customize plot appearance
    ax.set_ylabel('Mean Cost on Test Sets', fontsize=20)
    # ax.set_title('Performance Comparison Across Scenarios', fontsize=16, pad=20)
    ax.set_xticks(x)
    ax.set_yticks(ax.get_yticks())
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xticklabels([s.replace('_', ' ').title() for s in scenarios],
                       rotation=45, ha='right')
    
    # Enhanced legend with better positioning
    ax.legend(loc='best',
             ncol=2,
             fontsize=15,
             bbox_to_anchor=(0.5, 1.15))
    # frameon=True,borderaxespad=0.
    # bbox_to_anchor=(0.5, 1.15),
    # Add grid with improved styling
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, color='gray')
    ax.set_axisbelow(True)
    # Set axis limits with some padding
    ax.set_xlim(-width*2, len(scenarios)-1 + width*2)
    ymin = 0
    ymax = max(max(stats['means_mlp']), max(stats['means_lr']), max(stats['means_fcfs'])) * 1.05
    ax.set_ylim(ymin, ymax)
    # Add bar value labels with improved positioning
    def autolabel(rects, color):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(rect.get_x() + rect.get_width() / 2, height),
                       xytext=(0, 5),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       color=color,
                       bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1))
    
    autolabel(rects1, COLORS['mlp'])
    autolabel(rects2, COLORS['lr'])
    
    # Adjust layout
    plt.tight_layout()
    
    # Save with high quality if path provided
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
    
    plt.show()

def analyze_landing_patterns(results_df):
    """
    Analyze the patterns in landing times between predicted, true, and FCFS solutions
    """
    landing_patterns = []
    
    for idx, row in results_df.iterrows():
        pred_order = sorted(row['Pred_Landing_Times'].keys(), 
                          key=lambda x: row['Pred_Landing_Times'][x])
        true_order = sorted(row['True_Landing_Times'].keys(), 
                          key=lambda x: row['True_Landing_Times'][x])
        fcfs_order = sorted(row['FCFS_Landing_Times'].keys(), 
                          key=lambda x: row['FCFS_Landing_Times'][x])
        
        # Calculate order similarity
        pred_true_match = sum(p == t for p, t in zip(pred_order, true_order)) / len(pred_order)
        pred_fcfs_match = sum(p == f for p, f in zip(pred_order, fcfs_order)) / len(pred_order)
        
        landing_patterns.append({
            'Instance': idx,
            'Pred_True_Order_Match': pred_true_match,
            'Pred_FCFS_Order_Match': pred_fcfs_match,
            'Pred_Order': pred_order,
            'True_Order': true_order,
            'FCFS_Order': fcfs_order
        })
    
    return pd.DataFrame(landing_patterns)

if __name__ == "__main__":
    scenarios = [
        "max_vis",
        "max_prec",
        "max_wind",
        "max_danger",
        "min_time"
    ]
    
    plot_method_comparison(
        scenarios=scenarios,
        save_path="method_comparison.pdf"
    )
