import gurobipy as gb
import pandas as pd

# Load data
df = pd.read_csv('Holding_STAR.csv')
df = df[(df['date'] == '2018-05-22')&(df['entry_hour'] == 19)]

# Model parameters
planes = df['Callsign'].unique()
E = dict(zip(planes, df['Actual']-300))
L = dict(zip(planes, df['Actual']+300))
T = dict(zip(planes, df['Actual']))
sizes = dict(zip(planes, df['WTC']))

# Wake separation function
def WVseparation(A1, A2):
    #Unit: seconds
    if A1 == 'J' and A2 == 'H':
        return 120
    elif A1 == 'J' and A2 == 'M':
        return 180
    elif A1 == 'J' and A2 == 'L':
        return 240
    elif A1 == 'H' and A2 == 'H':
        return 120
    elif A1 == 'H' and A2 == 'M':
        return 120
    elif A1 == 'H' and A2 == 'L':
        return 180
    elif A1 == 'M' and A2 == 'L':
        return 180
    else:
        return 120

# Separation times
S = {(i,j): WVseparation(sizes[i], sizes[j]) for i in planes for j in planes if i != j}

# Cost parameter c^*_i (to be predicted)
# For now, we'll initialize it with a placeholder value
c_star = {i: 1 for i in planes}  # Replace this with your prediction model later

# FCFS Simulation Function
def fcfs_simulation(planes, E, T, S, c_star):
    sorted_planes = sorted(planes, key=lambda x: E[x])
    landing_times = {}
    total_cost = 0
    current_time = min(E.values())

    for plane in sorted_planes:
        landing_time = max(current_time, E[plane])
        landing_times[plane] = landing_time
        if landing_time > T[plane]:
            total_cost += c_star[plane]
        if plane != sorted_planes[-1]:
            current_time = landing_time + S[plane, sorted_planes[sorted_planes.index(plane) + 1]]

    return landing_times, total_cost

# Create model
m = gb.Model("Aircraft Landing - Beasley's Approach with Time Windows")

# Decision variables
x = m.addVars(planes, name="x")  # Landing time
delta = m.addVars(planes, planes, vtype=gb.GRB.BINARY, name="delta")  # Ordering variables
y = m.addVars(planes, vtype=gb.GRB.BINARY, name="y")  # Binary variable for landing after target time

# Objective: Minimize weighted sum of delays
m.setObjective(gb.quicksum(c_star[i] * y[i] for i in planes), gb.GRB.MINIMIZE)

# Constraints
m.addConstrs((x[i] >= E[i] for i in planes), "earliest_landing_time")
m.addConstrs((x[i] <= L[i] for i in planes), "latest_landing_time")

bigM = max(L.values()) + max(S.values())
m.addConstrs((x[i] - T[i] <= bigM * y[i] for i in planes), "y_definition_1")
m.addConstrs((x[i] - T[i] >= -bigM * (1 - y[i]) for i in planes), "y_definition_2")

m.addConstrs((delta[i,j] + delta[j,i] == 1 for i in planes for j in planes if i != j), "delta_sum")

m.addConstrs((x[j] - x[i] >= S[i,j] - bigM*delta[j,i] for i in planes for j in planes if i != j), "separation")

# Optimize
m.optimize()

# FCFS Simulation
fcfs_landing_times, fcfs_total_cost = fcfs_simulation(planes, E, T, S, c_star)

# Print solution and comparison
if m.status == gb.GRB.OPTIMAL:
    print('\nOptimized Solution:')
    print('Optimal Objective Value:', m.objVal)
    for i in planes:
        print(f"Plane {i}: Landing time = {x[i].x:.2f}, Landed after target = {'Yes' if y[i].x > 0.5 else 'No'}, Cost = {c_star[i]}")
        print(f"    Time window: [{E[i]}, {L[i]}], Target time: {T[i]}")
    
    print('\nFCFS Solution:')
    print('Total Cost:', fcfs_total_cost)
    for i in planes:
        print(f"Plane {i}: Landing time = {fcfs_landing_times[i]:.2f}, Landed after target = {'Yes' if fcfs_landing_times[i] > T[i] else 'No'}, Cost = {c_star[i]}")
        print(f"    Time window: [{E[i]}, {L[i]}], Target time: {T[i]}")
    
    print('\nComparison:')
    print(f"Optimized total cost: {m.objVal}")
    print(f"FCFS total cost: {fcfs_total_cost}")
    
    if fcfs_total_cost == 0 and m.objVal == 0:
        print("Both optimized and FCFS solutions have zero cost.")
    elif fcfs_total_cost == 0:
        print("FCFS cost is zero, while optimized solution has a cost.")
    elif m.objVal == 0:
        print("Optimized solution has zero cost, while FCFS has a cost.")
    else:
        improvement = (fcfs_total_cost - m.objVal) / fcfs_total_cost * 100
        print(f"Improvement: {improvement:.2f}%")

elif m.status == gb.GRB.INFEASIBLE:
    print("Model is infeasible")
    m.computeIIS()
    m.write("model.ilp")
    print("IIS written to file 'model.ilp'")
    for c in m.getConstrs():
        if c.IISConstr:
            print(f"Constraint {c.ConstrName} is in the IIS")
else:
    print(f"Optimization was stopped with status {m.status}")

# Print some statistics
print(f"\nNumber of planes: {len(planes)}")
print(f"Earliest arrival time: {min(E.values())}")
print(f"Latest arrival time: {max(L.values())}")
print(f"Minimum separation time: {min(S.values())}")
print(f"Maximum separation time: {max(S.values())}")