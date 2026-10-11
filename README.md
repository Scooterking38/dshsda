# dshsda

A Python-based simulation of a card game using reinforcement learning techniques.

## Architecture Overview

The `dshsda` repository is structured to simulate a card game environment where players interact with the game through defined cards and actions. The core components are:

- **Card Definitions**: Located within the `main.py` file, these define various types of cards such as treasures, victory cards, and actions.
- **Player Class**: Handles player attributes like money, buys, actions, and card draws.
- **Game Environment**: Utilizes the Gymnasium library to create a custom environment for the card game.
- **Reinforcement Learning Agent**: Employs Stable Baselines3's PPO algorithm to learn optimal strategies through interactions with the game environment.

## Tech Stack

- **Programming Language**: Python
- **Libraries and Frameworks**:
  - `numpy`: For numerical operations.
  - `gymnasium`: A toolkit for developing and comparing reinforcement learning algorithms.
  - `torch`: PyTorch library, used by Stable Baselines3 for deep learning models.
  - `stable_baselines3`: Provides implementations of reinforcement learning algorithms like PPO.

## Getting Started / Setup Guide

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/dshsda.git
   cd dshsda
   ```

2. **Install Dependencies**:
   Ensure you have Python and pip installed, then install the required packages:
   ```bash
   pip install numpy gymnasium torch stable_baselines3
   ```

3. **Run the Simulation**:
   Execute the main script to start the card game simulation with a reinforcement learning agent:
   ```bash
   python main.py
   ```

This setup will initialize the game environment, train the PPO agent, and run simulations to demonstrate the learned strategies. Adjustments can be made to the card definitions or training parameters within `main.py` for customization.