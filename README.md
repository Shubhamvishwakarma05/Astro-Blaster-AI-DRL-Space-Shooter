# Deep Reinforcement Learning Project
# Astro Blaster AI: DRL Space Shooter

A deep reinforcement learning project where an AI-controlled spaceship dodges enemies and blasts asteroids in a dynamic 2D space game.




https://github.com/user-attachments/assets/177613f6-e0d4-4982-898d-ee64ed781b27




## Overview



This project uses Deep Q-Learning (DQN) with a target network to train an agent that navigates a spaceship, avoiding obstacles and scoring points by shooting asteroids (+2) and enemies (+5), with a -5 penalty for collisions.

## How to Run
1. **Install Dependencies**:
   ```bash
   pip install gym torch numpy pygame
Run the Game:
python app.py

Trains over 100 episodes by default or tests the trained model after completion.
Requirements:
Python 3.7+
CPU-only (tested on Ryzen 5 5500U)
Gameplay Demo
Watch the video (Replace with your YouTube link after uploading)

Tech Stack
Python: Core programming language
PyTorch: DRL framework for DQN and neural networks
NumPy: Math and array operations
Gym: Reinforcement learning environment
Pygame: Graphics and game rendering
Results
Trained over 100 episodes, achieving an average score of 25.3.
Optimized for smart dodging and precise shooting using a 128-64-32 neural network.
Files
app3.py: Main game and training script
dqn_astro_smart_move.pth (optional): Pre-trained model weights
Credits
Built by [Your Name] as a showcase of DRL and game development skills.

### How to Use It
1. **Create the File**:
   - Open a text editor (Notepad, VS Code, PyCharm), paste this in, and save as `README.md` in `E:\PycharmProjects\AstroBlasterAI`.
2. **Add Video Link**:
   - After recording and uploading your gameplay to YouTube (per the deployment steps), replace `https://youtu.be/xyz123` with your actual link.
3. **Push to GitHub**:
   - From your project folder:
     ```bash
     git add README.md
     git commit -m "Added README with project details"
     git push
Or if it’s your first push, follow the full setup from the deployment method.
