import ray
import numpy as np
import matplotlib.pyplot as plt

ray.init()


@ray.remote
class QLearningAgent:
    def __init__(self, agent_id, num_actions):
        self.agent_id = agent_id
        self.num_actions = num_actions
        self.q_values = np.zeros(num_actions)
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.1

    def choose_action(self):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.num_actions)
        else:
            return np.argmax(self.q_values)

    def update_q_value(self, action, reward):
        self.q_values[action] += self.learning_rate * (reward - self.q_values[action])

    def get_agent_id(self):
        return self.agent_id


@ray.remote
class Environment:
    def __init__(self, num_agents, num_actions):
        self.num_agents = num_agents
        self.agents = [QLearningAgent.remote(i, num_actions) for i in range(num_agents)]
        self.num_actions = num_actions

    def interaction(self):
        results = []
        for agent in self.agents:
            agent_id = ray.get(agent.get_agent_id.remote())
            action = ray.get(agent.choose_action.remote())
            action_result, action_description = self.perform_action(action)
            ray.get(agent.update_q_value.remote(action, action_result))
            results.append((agent_id, action_result, action_description))
        return results

    def perform_action(self, action):
        action_types = ['move', 'accelerate', 'decelerate']
        chosen_action = action_types[action]

        action_description = {
            'move': 'Рухається',
            'accelerate': 'Прискорюється',
            'decelerate': 'Сповільнюється'
        }

        if chosen_action == 'move':
            return np.random.uniform(0.5, 2.0), action_description[chosen_action]
        elif chosen_action == 'accelerate':
            return np.random.uniform(0.75, 2.0), action_description[chosen_action]
        elif chosen_action == 'decelerate':
            return np.random.uniform(0.25, 1.0), action_description[chosen_action]


NUM_AGENTS = 10
NUM_ACTIONS = 3
NUM_EPOCHS = 200

env = Environment.remote(NUM_AGENTS, NUM_ACTIONS)

epoch_mean_values = []
smoothed_values = []

for epoch in range(NUM_EPOCHS):
    all_results = ray.get([env.interaction.remote() for _ in range(10)])

    epoch_mean = np.mean([np.mean([result[1] for result in epoch_results]) for epoch_results in all_results])
    epoch_mean_values.append(epoch_mean)

    # Додаємо середнє значення для плавності графіка
    smoothed_values.append(np.mean(epoch_mean_values[-5:]))

    print(f"Епоха {epoch + 1}: Середнє значення дій - {epoch_mean}")

plt.plot(range(1, NUM_EPOCHS + 1), smoothed_values, marker='o')
plt.xlabel('Епохи')
plt.ylabel('Середнє значення дій')
plt.title('Розвиток агентів протягом епох')
plt.grid(True)
plt.show()

ray.shutdown()