import numpy as np 

# R matrix
R = np.matrix([[-1, -1, -1, -1, 0, -1],
               [-1, -1, -1, 0, -1, 100],
               [-1, -1, -1, 0, -1, -1],
               [-1, 0, 0, -1, 0, -1],
               [-1, 0, 0, -1, -1, 100],
               [-1, 0, -1, -1, 0, 100],])

# Q matrix
Q = np.matrix(np.zeros((6,6)))

# Gamma (learning parameter)
gamma = 0.8

# Initial state (usually to be choose at random)
Initial_state = 1

# This function returns all available actions in the state given as an argument
def available_actions(state):
    current_state_row = R[state,]
    av_act = np.where(current_state_row>=0)[1]
    return av_act

# Get available actions in the current state
available_act = available_actions(Initial_state)

# This funciton choose at random whic action to be performed within the range of all the available actions.
def sample_next_action(available_actions_range):
    next_actions = int(np.random.choice(available_actions_range,1))
    return next_actions

# Sample next action to be performed
action = sample_next_action(available_act)

# This function updates the Q maritx according to the path selected and the Q learning algorithm
def update(current_state, action, gamma):
    max_index = np.where(Q[action,]==np.max(Q[action,]))[1]

    if max_index.shape[0]>1:
        max_index = int(np.random.choice(max_index, size=1))
    else:
        max_index = int(max_index)
    max_value = Q[action, max_index]

    # Q learning formula
    Q[current_state, action] = R[current_state,action] + gamma*max_value

# Update Q matrix
update(Initial_state,action,gamma)

# -------------------------------------------------------------------------------------------------

# Training
# Train over 10000 iterations. (Re-iterate the process above)
for i in range(10000):
    current_state = np.random.randint(0, int(Q.shape[0]))
    available_act = available_actions(current_state)
    action = sample_next_action(available_act)
    update(current_state, action, gamma)

# Normalize the Trained Q matrix
print(Q / np.max(Q) * 100)

# -------------------------------------------------------------------------------------------------

# Testing

# Goal state = 5
# Best sequence path starting from 2->2,3,1,5
current_state = 2
steps = [current_state]

while current_state!=5:

    next_step_index = np.where(Q[current_state,] == np.max(Q[current_state, ]))[1]

    if next_step_index.shape[0]>1:
        next_step_index = int(np.random.choice(next_step_index, size=1))
    else:
        next_step_index = int(next_step_index)
    
    steps.append(next_step_index)
    current_state = next_step_index

# Print selected sequence of steps
print(steps)

