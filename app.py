import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import pygame
import math
import time

# Initialize Pygame
pygame.init()
WIDTH, HEIGHT = 600, 400
WINDOW = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Astro Blaster AI - Smart Movement")
FONT = pygame.font.SysFont("comicsans", 30)
SMALL_FONT = pygame.font.SysFont("comicsans", 20)

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
SILVER = (192, 192, 192)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GRAY = (128, 128, 128)


# Step 1: Define the environment
class AstroBlasterEnv(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Discrete(5)  # 0=Stay, 1=Up, 2=Down, 3=Left, 4=Right+Shoot
        self.observation_space = gym.spaces.Box(low=0, high=max(WIDTH, HEIGHT), shape=(6,),
                                                dtype=np.float32)  # ship_x, ship_y, ast_x, ast_y, enemy_x, enemy_y
        self.reset()
        self.feedback = ""
        self.asteroid_speed = 4
        self.enemy_speed = 3
        self.lasers = []

    def reset(self):
        self.ship_x, self.ship_y = 100, HEIGHT // 2
        self.asteroid_x, self.asteroid_y = WIDTH, random.randint(50, HEIGHT - 50)
        self.enemy_x, self.enemy_y = WIDTH, random.randint(50, HEIGHT - 50)
        self.score = 0
        self.feedback = "Lock and load!"
        self.lasers = []
        self.asteroid_angle = 0
        return np.array([self.ship_x, self.ship_y, self.asteroid_x, self.asteroid_y, self.enemy_x, self.enemy_y],
                        dtype=np.float32)

    def step(self, action):
        # Move ship with smarter logic
        if action == 1 and self.ship_y > 20:  # Up
            self.ship_y -= 20
        elif action == 2 and self.ship_y < HEIGHT - 20:  # Down
            self.ship_y += 20
        elif action == 3 and self.ship_x > 20:  # Left
            self.ship_x -= 20
        elif action == 4:  # Right+Shoot - only move right if safe
            if self.ship_x < WIDTH - 20 and self.ship_x < min(self.asteroid_x,
                                                              self.enemy_x) - 100:  # Stay left of threats
                self.ship_x += 20
            self.lasers.append([self.ship_x + 40, self.ship_y])

        # Move lasers
        self.lasers = [[lx + 10, ly] for lx, ly in self.lasers if lx < WIDTH]

        # Move asteroid and enemy
        self.asteroid_x -= self.asteroid_speed
        self.enemy_x -= self.enemy_speed
        self.enemy_y += math.sin(self.enemy_x / 50) * 5  # Wavy enemy path
        self.asteroid_angle += 5  # Spin asteroid

        # Collision and reward
        done = False
        reward = 0
        ship_rect = pygame.Rect(self.ship_x - 20, self.ship_y - 20, 40, 40)
        ast_rect = pygame.Rect(self.asteroid_x - 20, self.asteroid_y - 20, 40, 40)
        enemy_rect = pygame.Rect(self.enemy_x - 15, self.enemy_y - 15, 30, 30)

        # Check hits
        for laser in self.lasers[:]:
            laser_rect = pygame.Rect(laser[0], laser[1] - 2, 10, 4)
            if laser_rect.colliderect(ast_rect):
                reward += 2
                self.score += 2
                self.asteroid_x = WIDTH
                self.asteroid_y = random.randint(50, HEIGHT - 50)
                self.lasers.remove(laser)
                self.feedback = random.choice(["Asteroid blasted!", f"Score: {self.score}!"])
            elif laser_rect.colliderect(enemy_rect):
                reward += 5
                self.score += 5
                self.enemy_x = WIDTH
                self.enemy_y = random.randint(50, HEIGHT - 50)
                self.lasers.remove(laser)
                self.feedback = random.choice(["Enemy down!", "Space ace!"])

        # Check collisions with penalty for moving too close
        if ship_rect.colliderect(ast_rect) or ship_rect.colliderect(enemy_rect):
            reward = -5  # Harsher penalty
            done = True
            self.feedback = random.choice(["Ship wrecked!", "Asteroid hit!", "Enemy got you!"])
        elif self.ship_x > min(self.asteroid_x, self.enemy_x) - 50:  # Small penalty for getting too close
            reward -= 0.1

        # Reset objects
        if self.asteroid_x < -20:
            self.asteroid_x = WIDTH
            self.asteroid_y = random.randint(50, HEIGHT - 50)
        if self.enemy_x < -20:
            self.enemy_x = WIDTH
            self.enemy_y = random.randint(50, HEIGHT - 50)

        state = np.array([self.ship_x, self.ship_y, self.asteroid_x, self.asteroid_y, self.enemy_x, self.enemy_y],
                         dtype=np.float32)
        return state, reward, done, {}

    def render(self, action, epsilon):
        WINDOW.fill(BLACK)
        # Draw Ship
        pygame.draw.polygon(WINDOW, SILVER, [(self.ship_x, self.ship_y), (self.ship_x + 40, self.ship_y - 20),
                                             (self.ship_x + 40, self.ship_y + 20)])
        # Draw Lasers
        for lx, ly in self.lasers:
            pygame.draw.rect(WINDOW, BLUE, (lx, ly - 2, 10, 4))
        # Draw Asteroid (rotating)
        pygame.draw.circle(WINDOW, GRAY, (int(self.asteroid_x), int(self.asteroid_y)), 20)
        pygame.draw.line(WINDOW, WHITE, (self.asteroid_x, self.asteroid_y),
                         (self.asteroid_x + 15 * math.cos(math.radians(self.asteroid_angle)),
                          self.asteroid_y + 15 * math.sin(math.radians(self.asteroid_angle))), 2)
        # Draw Enemy
        pygame.draw.polygon(WINDOW, RED, [(self.enemy_x, self.enemy_y), (self.enemy_x + 30, self.enemy_y - 15),
                                          (self.enemy_x + 30, self.enemy_y + 15)])
        # UI
        score_text = FONT.render(f"Score: {self.score}", True, WHITE)
        epsilon_text = SMALL_FONT.render(f"Epsilon: {epsilon:.3f}", True, WHITE)
        WINDOW.blit(score_text, (10, 10))
        WINDOW.blit(epsilon_text, (10, 40))
        actions = ["Stay", "Up", "Down", "Left", "Right+Shoot"]
        action_text = SMALL_FONT.render(f"Action: {actions[action]}", True, WHITE)
        WINDOW.blit(action_text, (10, 60))
        feedback_text = FONT.render(self.feedback, True, WHITE)
        WINDOW.blit(feedback_text, (WIDTH // 2 - feedback_text.get_width() // 2, HEIGHT // 2))
        pygame.display.flip()
        time.sleep(0.03)


# Step 2: Define the DQN model
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)


# Step 3: DRL Agent with Target Network
class DQNAgent:
    def __init__(self, env):
        self.env = env
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.memory = deque(maxlen=10000)
        self.batch_size = 128
        self.model = DQN(6, 5)
        self.target_model = DQN(6, 5)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0005)
        self.loss_fn = nn.MSELoss()
        self.update_target_every = 10
        self.episode_count = 0

    def act(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)

        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]
        targets = rewards + (1 - dones) * self.gamma * next_q_values

        loss = self.loss_fn(q_values, targets.detach())
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())


# Step 4: Train
def train_agent(episodes=100):
    env = AstroBlasterEnv()
    agent = DQNAgent(env)
    scores = []
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        agent.episode_count += 1
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return agent, scores
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            env.render(action, agent.epsilon)
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            state = next_state
            total_reward += reward
        scores.append(env.score)
        avg_score = np.mean(scores[-10:]) if len(scores) >= 10 else np.mean(scores)
        print(
            f"Episode {episode + 1}/{episodes}, Score: {env.score}, Avg Score (last 10): {avg_score:.2f}, Epsilon: {agent.epsilon:.3f}")
        if agent.episode_count % agent.update_target_every == 0:
            agent.update_target()
    torch.save(agent.model.state_dict(), "dqn_astro_smart_move.pth")
    print("Model saved as 'dqn_astro_smart_move.pth'")
    return agent, scores


# Step 5: Test
def test_agent(agent, episodes=5):
    env = AstroBlasterEnv()
    agent.epsilon = 0.0
    for episode in range(episodes):
        state = env.reset()
        done = False
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            action = agent.act(state)
            state, _, done, _ = env.step(action)
            env.render(action, agent.epsilon)
        print(f"Test Episode {episode + 1}, Score: {env.score}")


# Step 6: Run
if __name__ == "__main__":
    agent, scores = train_agent(episodes=100)
    print("\nTesting the trained agent:")
    test_agent(agent)
    pygame.quit()
