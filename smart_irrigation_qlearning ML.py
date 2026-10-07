import random
import numpy as np

# ---------- 1. STATES & ACTIONS ----------
SOIL = ["Dry", "Wet"]
TEMP = ["Normal", "High"]
RAIN = ["No", "Yes"]
ACTIONS = ["DON'T WATER", "WATER", "MORE WATER"]

def state_index(soil, temp, rain):
    return soil * 4 + temp * 2 + rain          # 0..7

def decode(s):
    return s // 4, (s // 2) % 2, s % 2

N_STATES, N_ACTIONS = 8, 3

# ---------- 2. REWARD FUNCTION ----------
def get_reward(state, action):
    soil, temp, rain = decode(state)
    if rain == 1:                               # raining
        return {0: 10, 1: -10, 2: -15}[action]
    if soil == 1:                               # wet soil, no rain
        return {0: 10, 1: -5, 2: -10}[action]
    if temp == 1:                               # dry + hot
        return {0: -10, 1: 5, 2: 10}[action]
    return {0: -8, 1: 10, 2: -3}[action]        # dry + normal temp

# ---------- 3. TRAINING ----------
alpha, gamma = 0.1, 0.9                         # learning rate, discount
epsilon, eps_min, eps_decay = 1.0, 0.05, 0.995
episodes = 3000

Q = np.zeros((N_STATES, N_ACTIONS))

for ep in range(episodes):
    state = random.randint(0, N_STATES - 1)     # random weather condition
    for _ in range(10):
        # epsilon-greedy
        if random.random() < epsilon:
            action = random.randint(0, N_ACTIONS - 1)
        else:
            action = int(np.argmax(Q[state]))

        reward = get_reward(state, action)
        next_state = random.randint(0, N_STATES - 1)

        # Q-learning update rule
        Q[state, action] += alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state, action]
        )
        state = next_state
    epsilon = max(eps_min, epsilon * eps_decay)

# ---------- 4. LEARNED POLICY ----------
print("\nLearned Policy")
print("-" * 55)
for s in range(N_STATES):
    soil, temp, rain = decode(s)
    best = ACTIONS[int(np.argmax(Q[s]))]
    print(f"Soil={SOIL[soil]:<4} Temp={TEMP[temp]:<6} Rain={RAIN[rain]:<3} -> {best}")

# ---------- 5. TRY YOUR OWN INPUT ----------
print("\nTest the system (Ctrl+C to quit)")
while True:
    try:
        soil = int(input("Soil (0=Dry, 1=Wet): "))
        temp = int(input("Temp (0=Normal, 1=High): "))
        rain = int(input("Rain (0=No, 1=Yes): "))
        s = state_index(soil, temp, rain)
        print("Decision:", ACTIONS[int(np.argmax(Q[s]))], "\n")
    except (KeyboardInterrupt, EOFError):
        break
