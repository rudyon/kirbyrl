import argparse
from pyboy import PyBoy
from pyboy.utils import WindowEvent
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from collections import deque
import random
import os

parser = argparse.ArgumentParser()
parser.add_argument("-w", "--window", type=str, default="SDL2", choices=["SDL2", "OpenGL", "GLFW", "null"])
parser.add_argument("-m", "--model", type=str)
args = parser.parse_args()

ACTIONS = [
    (WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.RELEASE_ARROW_RIGHT),
    (WindowEvent.PRESS_ARROW_LEFT,  WindowEvent.RELEASE_ARROW_LEFT),
    (WindowEvent.PRESS_ARROW_DOWN,  WindowEvent.RELEASE_ARROW_DOWN),
    (WindowEvent.PRESS_ARROW_UP,    WindowEvent.RELEASE_ARROW_UP),
    (WindowEvent.PRESS_BUTTON_A,    WindowEvent.RELEASE_BUTTON_A),
    (WindowEvent.PRESS_BUTTON_B,    WindowEvent.RELEASE_BUTTON_B),
]

def get_frame():
    img = pyboy.screen.image.convert('L').resize((84, 84))
    return np.array(img, dtype=np.uint8)

class Environment():
    def __init__(self):
        self.frame_stack = deque(maxlen=4)
        self.stuck_count = 0

    def step(self, action):
        prev_score = pyboy.memory[0xD08B]
        prev_x = pyboy.memory[0xD053]
        press, release = ACTIONS[action]
        if press == WindowEvent.PRESS_BUTTON_A:
            pyboy.send_input(press)
            pyboy.tick(12)
            pyboy.send_input(release)
            pyboy.tick(4)
        else:
            pyboy.send_input(press)
            pyboy.tick(4)
            if release is not None:
                pyboy.send_input(release)
            pyboy.tick(4)

        frame = get_frame()
        self.frame_stack.append(frame)
        state = np.stack(list(self.frame_stack), axis=0)

        curr_x = pyboy.memory[0xD053]
        curr_score = pyboy.memory[0xD08B]
        delta = curr_x - prev_x
        delta_score = curr_score - prev_score

        if delta < -50: delta = -1
        if delta > 50:  delta = 1

        reward = delta_score / 20
        reward += max(-1, min(1, delta))
        if delta < 0:  reward -= 2
        if delta == 0: reward -= 0.1

        done = pyboy.memory[0xD086] < 6
        if done: reward -= 10

        if curr_x == prev_x:
            self.stuck_count += 1
        else:
            self.stuck_count = 0

        if self.stuck_count > 30:  reward -= 1
        if self.stuck_count > 3000: done = True

        reward = max(-10, min(10, reward))
        return state, reward, done

    def reset(self):
        with open("save.state", 'rb') as s:
            pyboy.load_state(s)
        self.frame_stack.clear()
        self.stuck_count = 0
        frame = get_frame()
        for _ in range(4):
            self.frame_stack.append(frame)
        return np.stack(list(self.frame_stack), axis=0)

class NeuralNetwork(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        return self.network(x)

class Buffer():
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, size):
        batch = random.sample(self.buffer, size)
        states, acts, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states), dtype=torch.float32) / 255.0,
            torch.tensor(acts),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32) / 255.0,
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)

pyboy = PyBoy("kirby.gb", sound_emulated=False, window=args.window)
pyboy.set_emulation_speed(0)
assert pyboy.cartridge_title == "KIRBY DREAM LAN"

pyboy.tick(500)
pyboy.send_input(WindowEvent.PRESS_BUTTON_START)
pyboy.tick(500)
pyboy.send_input(WindowEvent.PRESS_BUTTON_START)
pyboy.tick(500)
with open("save.state", 'wb') as s:
    pyboy.save_state(s)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

env = Environment()
state = env.reset()
model = NeuralNetwork(num_actions=len(ACTIONS)).to(device)
target_model = NeuralNetwork(num_actions=len(ACTIONS)).to(device)
target_model.load_state_dict(model.state_dict())
target_model.eval()

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
buffer = Buffer()
batch_size = 32
epsilon = 1.0
epsilon_decay = 0.9999
epsilon_min = 0.05
step = 0

if args.model and os.path.isfile(args.model):
    ckpt = torch.load(args.model, map_location=device)
    model.load_state_dict(ckpt['model'])
    target_model.load_state_dict(ckpt['target_model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    step = ckpt['step']
    epsilon = max(ckpt['epsilon'], 0.3)
    print(f"[INFO] Loaded checkpoint. Step {step}, epsilon {epsilon:.3f}")

episode_reward = 0
episode_steps = 0
episode = 0

while True:
    if random.random() < epsilon:
        action = random.randint(0, len(ACTIONS) - 1)
    else:
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device) / 255.0
            action = torch.argmax(model(state_t)).item()

    next_state, reward, done = env.step(action)
    episode_reward += reward
    episode_steps += 1

    buffer.push(state.copy(), action, reward, next_state.copy(), done)
    state = next_state

    if done:
        episode += 1
        print(f"[INFO] Episode {episode} | Steps {episode_steps} | Reward {episode_reward:.2f} | Epsilon {epsilon:.3f}")
        episode_reward = 0
        episode_steps = 0
        state = env.reset()

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if len(buffer) < 1000:
        if step % 100 == 0:
            print(f"[INFO] Filling buffer {len(buffer)}/1000.")
        step += 1
        continue

    states, acts, rews, next_states, dones = buffer.sample(batch_size)
    states, acts, rews, next_states, dones = (
        states.to(device), acts.to(device), rews.to(device),
        next_states.to(device), dones.to(device)
    )

    predicted_q = model(states).gather(1, acts.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_q = target_model(next_states).max(1).values
        target_q = rews + 0.99 * next_q * (1 - dones)

    loss = F.mse_loss(predicted_q, target_q)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
    optimizer.step()

    if step % 1000 == 0:
        target_model.load_state_dict(model.state_dict())

    if step % 5000 == 0:
        torch.save({
            'model': model.state_dict(),
            'target_model': target_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'step': step,
            'epsilon': epsilon,
        }, "model.pt")
        print(f"[INFO] Saved model at step {step}.")

    if step % 100 == 0:
        print(f"[INFO] Step {step} | Epsilon {epsilon:.3f} | Loss {loss.item():.4f} | Buffer {len(buffer)}")

    step += 1