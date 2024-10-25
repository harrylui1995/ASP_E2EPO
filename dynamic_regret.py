import numpy as np
import torch
from pyepo import EPO
from pyepo_asp import ASPmodel
from pyepo.data.dataset import optDataset
def create_opt_model(n_aircraft, transit_times, wtc):
    """
    Create an ASP optimization model with instance-specific parameters
    
    Args:
        n_aircraft (int): Number of aircraft
        transit_times (list): Transit times for each aircraft
        wtc (list): Wake turbulence categories
    """
    E = dict(zip(range(n_aircraft), [t - 60 for t in transit_times]))
    L = dict(zip(range(n_aircraft), [t + 1800 for t in transit_times]))
    T = dict(zip(range(n_aircraft), transit_times))
    sizes = dict(zip(range(n_aircraft), wtc))
    return ASPmodel(n_aircraft, E, L, sizes, T)

def dynamic_regret(predmodel, dataloader):
    """
    Evaluate model performance with normalized true regret for dynamic parameters
    
    Args:
        predmodel (nn): regression neural network for cost prediction
        dataloader (DataLoader): Torch dataloader from optDataSet containing instance-specific parameters
    Returns:
        float: true regret loss
    """
    predmodel.eval()
    loss = 0
    optsum = 0
    
    for data in dataloader:
        x, c, w, z = data
        # Get instance-specific parameters from the dataset
        transit_times = data[4] if len(data) > 4 else None
        wtc = data[5] if len(data) > 5 else None
        
        if transit_times is None or wtc is None:
            raise ValueError("Dataset must include transit_times and wtc parameters")
            
        if next(predmodel.parameters()).is_cuda:
            x, c, w, z = x.cuda(), c.cuda(), w.cuda(), z.cuda()
            transit_times = transit_times.cuda()
            wtc = wtc.cuda()
            
        with torch.no_grad():
            cp = predmodel(x).to("cpu").detach().numpy()
            
        for j in range(cp.shape[0]):
            # Create instance-specific optimization model
            instance_transit_times = transit_times[j].to("cpu").detach().numpy()
            instance_wtc = wtc[j].to("cpu").detach().numpy()
            n_aircraft = len(instance_transit_times)
            
            optmodel = create_opt_model(n_aircraft, instance_transit_times, instance_wtc)
            
            # Calculate regret for this instance
            loss += cal_dynamic_regret(
                optmodel,
                cp[j],
                c[j].to("cpu").detach().numpy(),
                z[j].item()
            )
            
        optsum += abs(z).sum().item()
    
    predmodel.train()
    return loss / (optsum + 1e-7)

def cal_dynamic_regret(optmodel, pred_cost, true_cost, true_obj):
    """
    Calculate normalized true regret for a single instance
    
    Args:
        optmodel (ASPmodel): optimization model with instance-specific parameters
        pred_cost (np.array): predicted costs
        true_cost (np.array): true costs
        true_obj (float): true optimal objective value
    Returns:
        float: true regret loss
    """
    optmodel.setObj(pred_cost)
    sol, _ = optmodel.solve()
    obj = np.dot(sol, true_cost)
    
    if optmodel.modelSense == EPO.MINIMIZE:
        loss = obj - true_obj
    if optmodel.modelSense == EPO.MAXIMIZE:
        loss = true_obj - obj
    
    return loss

# Example modification to the dataset class to include parameters
class DynamicOptDataset(optDataset):
    def __init__(self, x, c, transit_times, wtc):
        super().__init__(x, c)
        self.transit_times = transit_times
        self.wtc = wtc
    
    def __getitem__(self, idx):
        x = self.x[idx]
        c = self.c[idx]
        w = self.w[idx]
        z = self.z[idx]
        transit_times = self.transit_times[idx]
        wtc = self.wtc[idx]
        return x, c, w, z, transit_times, wtc