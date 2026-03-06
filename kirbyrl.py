import argparse

parser = argparse.ArgumentParser(description="An RL agent that plays Kirby's Dream Land")
parser.add_argument("-w", "--window", type=str, default="SDL2", choices=["SDL2", "OpenGL", "GLFW", "null"])
parser.add_argument("-m", "--model", type=str, help="Model to load.")

args = parser.parse_args()

from pyboy import *
from pyboy.utils import WindowEvent
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from collections import deque
import random
import os

def get_frame():
    img = pyboy.screen.image.convert('L')
    img = img.resize((84, 84))
    tensor = torch.tensor(np.array(img), dtype=torch.uint8)
    return tensor

class Enviroment():
    def __init__(self):
        self.frame_stack = deque(maxlen=4)
        self.stuck_count = 0

    def step(self, action):
        prev_score = pyboy.memory[0xD08B]
        prev_x = pyboy.memory[0xD053]
        press, release = action
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
        new_frame = get_frame()
        self.frame_stack.append(new_frame)
        state = torch.stack(list(self.frame_stack), dim=0).float() / 255.0
        curr_x = pyboy.memory[0xD053]
        curr_score = pyboy.memory[0xD08B]
        delta = curr_x - prev_x
        delta_score = curr_score - prev_score

        if delta < -50:
            delta = -1
        if delta > 50:
            delta = 1

        reward = delta_score/20
        reward += max(-1, min(1, delta))
        if delta < 0:
            reward -= 2
        if delta == 0:
            reward -= 0.1
        done = pyboy.memory[0xD086] < 6
        if done:
            reward -= 10

        if curr_x == prev_x:
            self.stuck_count += 1
        else:
            self.stuck_count = 0
        
        if self.stuck_count > 30:
            reward -= 1
        if self.stuck_count > 5000:
            done = True

        reward = max(-1, min(1, reward))
        return state, reward, done
    
    def reset(self):
        with open("save.state", 'rb') as s:
            pyboy.load_state(s)
        self.frame_stack.clear()
        obs = get_frame()
        for _ in range(4):
            self.frame_stack.append(obs)
        return torch.stack(list(self.frame_stack), dim=0).float() / 255.0

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

    def forward(self, inputs):
        logits = self.network(inputs)
        return logits

class Buffer():
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.frames = deque(maxlen=capacity + 4)
        self.transitions = deque(maxlen=capacity)
        self.frame_count = 0

    def push(self, frame, action, reward, done):
        self.frames.append(frame)
        self.transitions.append((self.frame_count, action, reward, done))
        self.frame_count += 1

    def _get_stack(self, frame_idx):
        start = self.frame_count - len(self.frames)
        local_idx = frame_idx - start
        local_idx = max(0, min(local_idx, len(self.frames) - 1))
        stack = [self.frames[max(0, local_idx - 3 + i)] for i in range(4)]
        return torch.stack(stack).float() / 255.0

    def sample(self, size):
        # reconstruct 4-frame stacks from indices on the fly
        valid = [t for t in self.transitions if t[0] >= 3]
        batch = random.sample(valid, size)
        states, acts, rewards, next_states, dones = [], [], [], [], []
        for frame_idx, action, reward, done in batch:
            states.append(self._get_stack(frame_idx))
            next_states.append(self._get_stack(frame_idx + 1))
            acts.append(action)
            rewards.append(reward)
            dones.append(done)
        return states, acts, rewards, next_states, dones

    def __len__(self):
        return len(self.transitions)

pyboy = PyBoy("kirby.gb", sound_emulated=False, window=args.window)
pyboy.set_emulation_speed(0)
assert pyboy.cartridge_title == "KIRBY DREAM LAN"

actions = [
    (WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.RELEASE_ARROW_RIGHT),
    (WindowEvent.PRESS_ARROW_LEFT,  WindowEvent.RELEASE_ARROW_LEFT),
    (WindowEvent.PRESS_ARROW_DOWN,  WindowEvent.RELEASE_ARROW_DOWN),
    (WindowEvent.PRESS_ARROW_UP,    WindowEvent.RELEASE_ARROW_UP),
    (WindowEvent.PRESS_BUTTON_A,    WindowEvent.RELEASE_BUTTON_A),
    (WindowEvent.PRESS_BUTTON_B,    WindowEvent.RELEASE_BUTTON_B)
]

pyboy.tick(500)
pyboy.send_input(WindowEvent.PRESS_BUTTON_START)
pyboy.tick(500)
pyboy.send_input(WindowEvent.PRESS_BUTTON_START)
pyboy.tick(500)
with open("save.state", 'wb') as s:
    pyboy.save_state(s)

if torch.cuda.is_available():
    print(f"[INFO] FOUND GPU: {torch.cuda.get_device_name(0)}!")
else:
    print("[WARNING] NO GPU FOUND. WILL RUN ON CPU!")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pyboy.set_emulation_speed(0)
env = Enviroment()
state = env.reset()
model = NeuralNetwork(num_actions=len(actions)).to(device)
target_model = NeuralNetwork(num_actions=len(actions)).to(device)
target_model.load_state_dict(model.state_dict())
target_model.eval()

batch_size = 32

epsilon = 1.0
epsilon_decay = 0.999
epsilon_min = 0.05
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
buffer = Buffer()
step = 0

if args.model != None:
    if os.path.isfile(args.model):
        loaded_model = torch.load(args.model, map_location=torch.device(device))
        model.load_state_dict(loaded_model['model'])
        target_model.load_state_dict(loaded_model['target_model'])
        optimizer.load_state_dict(loaded_model['optimizer'])
        step = loaded_model['step']
        epsilon = loaded_model['epsilon']
        epsilon = max(epsilon, 0.3)
        print(f"[INFO] Pretrained model loaded. Epsilon {epsilon}, step {step}.")
    else:
        print("[ERROR] Pretrained model not found in location!")

# episode stats
episode_reward = 0
episode_steps = 0
episode = 0

while True:
    # epsilon greedy
    if random.random() < epsilon:
        action = random.randint(0, len(actions)-1)
    else:
        with torch.no_grad():
            q_values = model(state.unsqueeze(0).to(device))
            action = torch.argmax(q_values).item()

    next_state, reward, done = env.step(actions[action])

    # update episode stats
    episode_reward += reward
    episode_steps += 1

    # store experience in buffer and update the state
    buffer.push(get_frame(), action, reward, done)
    state = next_state

    if done:
        episode += 1
        print(f"[INFO] Episode {episode} | Steps {episode_steps} | Reward {episode_reward:.2f} | Epsilon {epsilon:.3f}")
        episode_reward = 0
        episode_steps = 0
        state = env.reset().to(device)

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # check if buffer has enough to train
    if len(buffer) < 1000:
        if step % 100 == 0:
            print(f"[INFO] Filling buffer {len(buffer)}/1000.")
        step += 1
        continue

    states, acts, rewards, next_states, dones = buffer.sample(batch_size)

    states = torch.stack(states).to(device)
    next_states = torch.stack(next_states).to(device)
    acts = torch.tensor(acts).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    dones = torch.tensor(dones, dtype=torch.float32).to(device)
    
    predicted_q = model(states).gather(1, acts.unsqueeze(1)).squeeze(1) 

    with torch.no_grad():
        next_q = target_model(next_states).max(1).values
        target_q = rewards + 0.99 * next_q * (1 - dones)

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
        print(f"[INFO] Saved model at step {step} to ./model.pt.")

    if step % 100 == 0:
        print(f"[INFO] Step {step} | Epsilon {epsilon:.3f} | Loss {loss.item():.4f} | Buffer {len(buffer)}")
    

    step += 1