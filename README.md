# Deep Reinforcement Learning Project
# Astro Blaster AI: DRL Space Shooter

A deep reinforcement learning project where an AI-controlled spaceship dodges enemies and blasts asteroids in a dynamic 2D space game.



https://github.com/user-attachments/assets/54b10e5b-2305-40e0-81be-ff7ee042ea07



## Overview
This project uses Deep Q-Learning (DQN) with a target network to train an agent that navigates a spaceship, avoiding obstacles and scoring points by shooting asteroids (+2) and enemies (+5), with a -5 penalty for collisions.

Trains over 100 episodes by default or tests the trained model after completion.
Requirements:
Python 3.7+
CPU-only (tested on Ryzen 5 5500U)


# Tech Stack
Python: Core programming language
PyTorch: DRL framework for DQN and neural networks
NumPy: Math and array operations
Gym: Reinforcement learning environment
Pygame: Graphics and game rendering
Results
Trained over 100 episodes, achieving an average score of 25.3.
Astro Blaster AI is a Deep Reinforcement Learning (DRL) project where an intelligent agent learns to navigate a spaceship in a dynamic 2D space shooter game. Built with Deep Q-Learning (DQN) and a target network, the agent dodges obstacles, shoots asteroids and enemies, and maximizes its score. The game features a real-time Pygame interface with animated sprites, making it both a technical AI showcase and a visually engaging experience.

This project demonstrates advanced AI techniques applied to a fun, interactive environment, achieving an average score of 25.3 after training over 100 episodes on CPU.

Features
DRL Agent: Uses DQN with a target network to optimize decision-making (stay, move up/down/left, or shoot+move right).
Reward System: Earns +2 for hitting asteroids, +5 for enemies, and -5 for collisions, with a -0.1 penalty for risky proximity.
Custom Environment: Built with Gym, featuring a spaceship, spinning asteroids, and wavy-moving enemies.
Real-Time UI: Rendered via Pygame with animated sprites, lasers, and on-screen feedback (e.g., "Asteroid blasted!").
Training & Testing: Trained for 100 episodes, with a saved model (dqn_astro_smart_move.pth) for reuse.
Technologies Used
Python: Core programming language.
PyTorch: Framework for building and training the DQN neural network.
NumPy: Efficient state and data handling.
Gym: Custom environment for reinforcement learning.
Pygame: Game visualization and UI rendering.


Install Dependencies: Ensure Python 3.6+ is installed, then run:

pip install -r requirements.txt
Requirements.txt 

gym==0.21.0
numpy==1.23.5
torch==1.13.1
pygame==2.1.2

# Run the Game:
Training Mode: Launches the agent training for 100 episodes, rendering the game and saving the model as dqn_astro_smart_move.pth.

python astro_blaster_ai.py
# Testing Mode: After training, the script automatically tests the agent for 5 episodes with exploration disabled (epsilon = 0).
Controls: Close the Pygame window to exit at any time.
Sample Output:
Episode 100/100, Score: 28, Avg Score (last 10): 25.3, Epsilon: 0.010
Model saved as 'dqn_astro_smart_move.pth'
Testing the trained agent:
Test Episode 1, Score: 30

# Future Improvements
Add difficulty scaling (e.g., faster asteroids, more enemies).
Implement Double DQN or Prioritized Experience Replay for better performance.
Deploy a pre-trained model demo online (e.g., via Streamlit or a web app).
Record and include a gameplay video in the README.
Contributing
Feel free to fork this repo, submit pull requests, or open issues for bugs and feature suggestions!

# Author
👤 Shubham Vishwakarma
