import os
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

# -------------------------
# Card Definitions
# -------------------------
class Card:
    def __init__(self, name, cost, types, value=0, vp=0, effect=None):
        self.name, self.cost, self.types = name, cost, types
        self.value, self.vp, self.effect = value, vp, effect

# Effect Helpers
def add_stats(p, cards=0, actions=0, buys=0, money=0):
    if cards > 0: p.draw(cards)
    p.actions += actions
    p.buys += buys
    p.money_bonus += money

BASE_CARDS = [
    # Treasures & Victory
    Card('Copper', 0, ['treasure'], value=1),
    Card('Silver', 3, ['treasure'], value=2),
    Card('Gold', 6, ['treasure'], value=3),
    Card('Estate', 2, ['victory'], vp=1),
    Card('Duchy', 5, ['victory'], vp=3),
    Card('Province', 8, ['victory'], vp=6),
    # Actions
    Card('Village', 3, ['action'], effect=lambda p, g: add_stats(p, cards=1, actions=2)),
    Card('Smithy', 4, ['action'], effect=lambda p, g: add_stats(p, cards=3)),
    Card('Market', 5, ['action'], effect=lambda p, g: add_stats(p, cards=1, actions=1, buys=1, money=1)),
    Card('Laboratory', 5, ['action'], effect=lambda p, g: add_stats(p, cards=2, actions=1)),
    Card('Festival', 5, ['action'], effect=lambda p, g: add_stats(p, actions=2, buys=1, money=2)),
    Card('Poacher', 4, ['action'], effect=lambda p, g: add_stats(p, cards=1, actions=1, money=1)),
    Card('Moneylender', 4, ['action'], effect=lambda p, g: p.trash_copper()),
    Card('Chapel', 2, ['action'], effect=lambda p, g: p.chapel_trash()),
    Card('Workshop', 3, ['action'], effect=lambda p, g: p.gain_card(4)), # Simplified
]

CARD_IDX = {c.name: i for i, c in enumerate(BASE_CARDS)}
NUM_CARDS = len(BASE_CARDS)

# -------------------------
# Player Logic
# -------------------------
class Player:
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.deck = [CARD_IDX['Copper']] * 7 + [CARD_IDX['Estate']] * 3
        random.shuffle(self.deck)
        self.hand, self.discard, self.played = [], [], []
        self.actions, self.buys, self.money_bonus = 1, 1, 0

    def draw(self, n=1):
        for _ in range(n):
            if not self.deck:
                if not self.discard: break
                self.deck, self.discard = self.discard, []
                random.shuffle(self.deck)
            if self.deck: self.hand.append(self.deck.pop())

    def get_total_money(self):
        return sum(BASE_CARDS[c].value for c in self.hand if 'treasure' in BASE_CARDS[c].types) + self.money_bonus

    def total_vp(self):
        return sum(BASE_CARDS[c].vp for c in (self.hand + self.deck + self.discard + self.played))

    def trash_copper(self):
        if CARD_IDX['Copper'] in self.hand:
            self.hand.remove(CARD_IDX['Copper'])
            self.money_bonus += 3
    
    def chapel_trash(self):
        # AI strategy: Trash up to 4 Estates/Coppers
        trashed = 0
        for target in ['Estate', 'Copper']:
            while trashed < 4 and CARD_IDX[target] in self.hand:
                self.hand.remove(CARD_IDX[target])
                trashed += 1

    def gain_card(self, max_cost):
        # Placeholder for workshop logic
        pass

# -------------------------
# Environment
# -------------------------
class DominionEnv(gym.Env):
    def __init__(self, verbose=False):
        super().__init__()
        self.action_space = spaces.Discrete(NUM_CARDS + 1)
        self.observation_space = spaces.Box(low=0, high=100, shape=(NUM_CARDS*3 + 4,), dtype=np.int32)
        self.verbose = verbose

    def _get_obs(self):
        p = self.players[self.current_player_idx]
        h = [p.hand.count(i) for i in range(NUM_CARDS)]
        s = [self.supply.get(BASE_CARDS[i].name, 0) for i in range(NUM_CARDS)]
        pl = [p.played.count(i) for i in range(NUM_CARDS)]
        return np.array(h + s + pl + [p.actions, p.buys, p.get_total_money(), self.phase], dtype=np.int32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.players = [Player("Agent 1"), Player("Agent 2")]
        self.supply = {c.name: 10 for c in BASE_CARDS}
        for v in ['Estate', 'Duchy', 'Province']: self.supply[v] = 8
        self.turn_count, self.phase, self.current_player_idx = 0, 0, 0
        self.players[0].draw(5)
        return self._get_obs(), {}

    def step(self, action):
        p = self.players[self.current_player_idx]
        reward = 0

        # Action == NUM_CARDS is "End Phase"
        if action == NUM_CARDS:
            if self.phase == 0: self.phase = 1
            else: self._switch_player()
            return self._get_obs(), 0, False, False, {}

        if self.phase == 0: # Action Phase
            if action in p.hand and 'action' in BASE_CARDS[action].types and p.actions > 0:
                p.actions -= 1
                p.hand.remove(action)
                p.played.append(action)
                if BASE_CARDS[action].effect: BASE_CARDS[action].effect(p, self)
                reward += 0.1
            else:
                self.phase = 1 # Auto-transition on illegal or non-action

        elif self.phase == 1: # Buy Phase
            card = BASE_CARDS[action]
            if p.get_total_money() >= card.cost and self.supply[card.name] > 0 and p.buys > 0:
                p.discard.append(action)
                self.supply[card.name] -= 1
                p.buys -= 1
                if card.name == 'Province': reward += 5.0
                elif card.name == 'Gold': reward += 2.0
            else:
                reward -= 0.01
                self._switch_player()

        done = self.supply['Province'] == 0 or list(self.supply.values()).count(0) >= 3 or self.turn_count >= 60
        if done:
            score_diff = p.total_vp() - self.players[1-self.current_player_idx].total_vp()
            reward += score_diff

        return self._get_obs(), reward, done, False, {}

    def _switch_player(self):
        p = self.players[self.current_player_idx]
        p.discard.extend(p.hand + p.played)
        p.hand, p.played, p.actions, p.buys, p.money_bonus = [], [], 1, 1, 0
        self.current_player_idx = 1 - self.current_player_idx
        self.players[self.current_player_idx].draw(5)
        self.phase = 0
        if self.current_player_idx == 0: self.turn_count += 1

def train():
    # Use a small step count for GitHub Actions demonstration
    env = DominionEnv()
    model = PPO("MlpPolicy", env, verbose=1, n_steps=1024, batch_size=64)
    model.learn(total_timesteps=50000)
    model.save("dominion_selfplay_ai")

if __name__ == "__main__":
    train()
