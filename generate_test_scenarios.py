import torch
import os

# Parameters
num_agents_list = [20, 50, 100, 125]
num_scenarios = 100
area_size = 4.0  
car_radius = 0.05   # Example car radius

def generate_scenarios(num_agents, num_scenarios, area_size, car_radius, mode='test'):
    scenarios = []
    for _ in range(num_scenarios):
        side_length = area_size
        states = torch.zeros(num_agents, 2)
        goals = torch.zeros(num_agents, 2)
        
        # Generate positions of agents
        i = 0
        while i < num_agents:
            candidate = torch.rand(2) * side_length
            dist_min = torch.norm(states - candidate, dim=1).min()
            if dist_min <= car_radius * 4:
                continue
            states[i] = candidate
            i += 1

        # Generate goals of agents
        i = 0
        while i < num_agents:
            candidate = torch.rand(2) * side_length
            dist_min = torch.norm(goals - candidate, dim=1).min()
            if dist_min <= car_radius * 4:
                continue
            goals[i] = candidate
            i += 1
        
        scenarios.append((states, goals))
    
    return scenarios

def save_scenarios(scenarios, num_agents, directory='scenarios'):
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    for idx, (states, goals) in enumerate(scenarios):
        filename = os.path.join(directory, f'scenario_{num_agents}_agents_{idx}.pt')
        torch.save({'states': states, 'goals': goals}, filename)

def main():
    for num_agents in num_agents_list:
        scenarios = generate_scenarios(num_agents, num_scenarios, area_size, car_radius)
        save_scenarios(scenarios, num_agents)

if __name__ == "__main__":
    main()
