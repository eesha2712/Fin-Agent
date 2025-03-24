import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

class PortfolioAgent:
    def __init__(self, state_size, action_size):
        self.model = self.build_model(state_size, action_size)
    
    def build_model(self, state_size, action_size):
        model = tf.keras.Sequential([
            layers.Dense(64, activation='relu', input_dim=state_size),
            layers.Dense(32, activation='relu'),
            layers.Dense(action_size, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def train(self, state, action, reward, next_state):
        target = reward + 0.95 * np.max(self.model.predict(next_state))
        target_f = self.model.predict(state)
        target_f[0][action] = target
        self.model.fit(state, target_f, epochs=1, verbose=0)

if __name__ == "__main__":
    state_size = 1
    action_size = 3
    agent = PortfolioAgent(state_size, action_size)
    
    state = np.random.rand(1, state_size)
    action = np.random.randint(0, action_size)
    reward = np.random.rand()
    next_state = np.random.rand(1, state_size)
    
    agent.train(state, action, reward, next_state)
