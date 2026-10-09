"""
Smart Irrigation System using Q-Learning
----------------------------------------
State  : (soil, temperature, rain)  -> 3 x 2 x 2 = 12 states
Action : 0 = Don't water, 1 = Water, 2 = More water
Goal   : keep soil moisture healthy while using as little water as possible.
Run    : python irrigation_qlearning.py
"""
import random
import numpy as np

SOIL = ["Dry", "OK", "Wet"]
TEMP = ["Normal", "High"]
RAIN = ["No", "Yes"]
ACTIONS = ["Don't water", "Water", "More water"]

# Q-Learning hyper-parameters
ALPHA = 0.1        # learning rate
GAMMA = 0.9        # discount factor
EPS_START = 1.0    # exploration at start
EPS_MIN = 0.05     # exploration at end
EPISODES = 1000
STEPS = 30         # days per episode

random.seed(42)
np.random.seed(42)


# ---------------- Environment (simulated field) ----------------
def bucket(level):
    """Soil moisture level 0..5 -> Dry(0) / OK(1) / Wet(2)."""
    return 0 if level < 2 else 1 if level < 4 else 2


def state_index(soil, temp, rain):
    return soil * 4 + temp * 2 + rain


def new_weather():
    return {"temp": int(random.random() < 0.5), "rain": int(random.random() < 0.2)}


def env_step(level, action, weather):
    """Return (next_moisture_level, reward)."""
    change = 0
    if random.random() < (0.8 if weather["temp"] else 0.35):   # evaporation
        change -= 1
    if weather["rain"] and random.random() < 0.9:               # rain adds water
        change += 1
    change += action                                            # irrigation adds water
    new_level = max(0, min(5, level + change))

    if 2 <= new_level <= 3:
        reward = 10                                  # healthy soil
    elif new_level < 2:
        reward = -15 if new_level == 0 else -10      # too dry
    else:
        reward = -10 if new_level == 5 else -5       # too wet

    reward -= action                                 # cost of water used
    if weather["rain"] and action > 0:
        reward -= 5                                  # watering in rain = waste
    return new_level, reward


# ---------------- Q-Learning agent ----------------
Q = np.zeros((12, 3))


def best_action(s):
    return int(np.argmax(Q[s]))


def train():
    eps = EPS_START
    decay = (EPS_MIN / EPS_START) ** (1 / EPISODES)
    history = []
    for _ in range(EPISODES):
        level = random.randint(0, 5)
        w = new_weather()
        total = 0
        for _ in range(STEPS):
            s = state_index(bucket(level), w["temp"], w["rain"])
            # epsilon-greedy: explore or exploit
            a = random.randint(0, 2) if random.random() < eps else best_action(s)
            new_level, r = env_step(level, a, w)
            nw = new_weather()
            s2 = state_index(bucket(new_level), nw["temp"], nw["rain"])
            # Q-Learning update rule
            Q[s, a] += ALPHA * (r + GAMMA * Q[s2].max() - Q[s, a])
            level, w, total = new_level, nw, total + r
        history.append(total)
        eps = max(EPS_MIN, eps * decay)
    return history


# ---------------- Testing ----------------
def simulate(policy, weathers):
    level, water, dry, wet, ok = 3, 0, 0, 0, 0
    for w in weathers:
        a = policy(level, w)
        water += a
        level, _ = env_step(level, a, w)
        if level < 2: dry += 1
        elif level > 3: wet += 1
        else: ok += 1
    return water, ok, dry, wet


if __name__ == "__main__":
    hist = train()
    print(f"Trained for {EPISODES} episodes. "
          f"Avg reward first 50: {np.mean(hist[:50]):.1f} | last 50: {np.mean(hist[-50:]):.1f}\n")

    print(f"{'Soil':<5} {'Temp':<7} {'Rain':<5} -> Learned action")
    for s in range(3):
        for t in range(2):
            for r in range(2):
                i = state_index(s, t, r)
                print(f"{SOIL[s]:<5} {TEMP[t]:<7} {RAIN[r]:<5} -> {ACTIONS[best_action(i)]:<12} Q={np.round(Q[i], 1)}")

    weathers = [new_weather() for _ in range(100)]
    agent = simulate(lambda l, w: best_action(state_index(bucket(l), w["temp"], w["rain"])), weathers)
    base = simulate(lambda l, w: 1, weathers)
    print("\n100-day test        water  healthy  dry  wet")
    print(f"Q-Learning agent    {agent[0]:>5}  {agent[1]:>7}  {agent[2]:>3}  {agent[3]:>3}")
    print(f"Always water        {base[0]:>5}  {base[1]:>7}  {base[2]:>3}  {base[3]:>3}")
    print(f"Water saved: {round((1 - agent[0] / base[0]) * 100)}%")
