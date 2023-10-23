from abc import ABC, abstractmethod
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm

# Initialize the logging
logging.basicConfig()
logger = logging.getLogger("MAB Application")

# Create a console handler with a higher log level and set a custom formatter
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(ch)

class Bandit(ABC):
    """ """
    def __init__(self, p):
        self.p = p
        self.p_estimate = 0
        self.N = 0
        self.r_estimate = 0

    def __repr__(self):
        return f'An Arm with {self.p} Win Rate'

    @abstractmethod
    def pull(self):
        """ """
        pass

    @abstractmethod
    def update(self, x):
        """

        :param x: 

        """
        pass

    @abstractmethod
    def experiment(self, bandit_rewards, N):
        """

        :param bandit_rewards: 
        :param N: 

        """
        pass

    def report(self, N, results, algorithm="Epsilon Greedy"):
        """

        :param N: 
        :param results: 
        :param algorithm:  (Default value = "Epsilon Greedy")

        """
        bandits, reward_per_bandit, bandit_selected, num_times_explored, num_times_exploited, num_optimal, cumulative_average, cumulative_sum, cumulative_regret = results
        
        df = pd.DataFrame({
            'Bandit': [bandit for bandit in bandit_selected],
            'Reward': [reward for reward in reward_per_bandit],
            'Algorithm': algorithm
        })

        df.to_csv(f'{algorithm}.csv', index=False)

        df1 = pd.DataFrame({
            'Bandit': [bandit for bandit in bandits],
            'Reward': [p.p_estimate for p in bandits],
            'Algorithm': algorithm
        })

        df1.to_csv(f'{algorithm}Final.csv', index=False)

        for b in range(len(bandits)):
            print(f'Bandit with True Win Rate {bandits[b].p} - Pulled {bandits[b].N} times - Estimated average reward - {round(bandits[b].p_estimate, 4)} - Estimated average regret - {round(bandits[b].r_estimate, 4)}')
            print("--------------------------------------------------")

        print(f"Cumulative Reward : {sum(reward_per_bandit)}", end='\n')
        print(f"Cumulative Regret : {cumulative_regret[-1]}", end='\n')
              
        if algorithm == 'EpsilonGreedy':
            print(f"Percent optimal : {round((float(num_optimal) / N), 4)}")

class Visualization:
    """ """
    def plot1(self, N, results, algorithm='EpsilonGreedy'):
        """

        :param N: 
        :param results: 
        :param algorithm:  (Default value = 'EpsilonGreedy')

        """
        cumulative_average = results[6]
        bandits = results[0]

        # Linear plot
        plt.plot(cumulative_average, label='Cumulative Average')
        plt.plot(np.ones(N) * max([b.p for b in bandits]), label='Optimal')
        plt.legend()
        plt.title(f"Win Rate Convergence for {algorithm} - Linear")
        plt.xlabel("Number of Trials")
        plt.ylabel("Estimated Reward")
        plt.show()

        # Log plot
        plt.plot(cumulative_average, label='Cumulative Average')
        plt.plot(np.ones(N) * max([b.p for b in bandits]), label='Optimal')
        plt.legend()
        plt.title(f"Win Rate Convergence for {algorithm} - Log")
        plt.xlabel("Number of Trials")
        plt.ylabel("Estimated Reward")
        plt.xscale("log")
        plt.show()

    def plot2(self, results_greedy, results_thompson):
        """

        :param results_greedy: 
        :param results_thompson: 

        """
        cumulative_rewards_eps = results_greedy[7]
        cumulative_rewards_th = results_thompson[0]
        cumulative_regret_eps = results_greedy[8]
        cumulative_regret_th = results_thompson[2]

        # Cumulative Reward
        plt.plot(cumulative_rewards_eps, label='Epsilon-Greedy')
        plt.plot(cumulative_rewards_th, label='Thompson Sampling')
        plt.legend()
        plt.title("Cumulative Reward Comparison")
        plt.xlabel("Number of Trials")
        plt.ylabel("Cumulative Reward")
        plt.show()

        # Cumulative Regret
        plt.plot(cumulative_regret_eps, label='Epsilon-Greedy')
        plt.plot(cumulative_regret_th, label='Thompson Sampling')
        plt.legend()
        plt.title("Cumulative Regret Comparison")
        plt.xlabel("Number of Trials")
        plt.ylabel("Cumulative Regret")
        plt.show()

class EpsilonGreedy(Bandit):
    """ """
    def pull(self):
        """ """
        return np.random.random() < self.p

    def update(self, x):
        """

        :param x: 

        """
        self.N += 1.
        self.p_estimate = (1 - 1.0/self.N) * self.p_estimate + 1.0/ self.N * x
        self.r_estimate = self.p - self.p_estimate

    def experiment(self, bandit_rewards, N, t):
        """

        :param bandit_rewards: 
        :param N: 
        :param t: 

        """
        bandits = [EpsilonGreedy(p) for p in bandit_rewards]
        probs = np.array(bandit_rewards)
        num_times_explored = 0
        num_times_exploited = 0
        num_optimal = 0
        optimal = np.argmax(probs)
        eps = 1/t

        reward_per_bandit = np.empty(N, dtype=float)
        bandit_selected = np.empty(N)

        for i in range(N):
            p = np.random.random()

            if p < eps:
                j = np.random.choice(len(bandits))
                num_times_explored += 1
            else:
                j = np.argmax([b.p_estimate for b in bandits])
                num_times_exploited += 1

            if j == optimal:
                num_optimal += 1

            x = bandits[j].pull()
            bandits[j].update(x)
            reward_per_bandit[i] = x
            bandit_selected[i] = j
            t += 1
            eps = 1/t

        cumulative_average = np.cumsum(reward_per_bandit) / (np.arange(N) + 1)
        cumulative_sum = np.cumsum(reward_per_bandit)

        cumulative_regret = np.zeros(N)
        for i in range(len(reward_per_bandit)):
            cumulative_regret[i] = N * max(probs) - cumulative_sum[i]

        return bandits, reward_per_bandit, bandit_selected, num_times_explored, num_times_exploited, num_optimal, cumulative_average, cumulative_sum, cumulative_regret

class ThompsonSampling(Bandit):
    """ """
    def __init__(self, true_mean):
        self.true_mean = true_mean
        self.m = 0
        self.lambda_ = 1
        self.tau = 1
        self.N = 0

    def pull(self):
        """ """
        return np.random.randn() / np.sqrt(self.tau) + self.true_mean

    def sample(self):
        """ """
        return np.random.randn() / np.sqrt(self.lambda_) + self.m

    def update(self, x):
        """

        :param x: 

        """
        self.p_estimate = (self.tau * x + self.lambda_ * self.p_estimate) / (self.tau + self.lambda_)
        self.lambda_ += self.tau
        self.N += 1
        self.r_estimate = self.p - self.p_estimate

    def plot(self, bandits, trial):
        """

        :param bandits: 
        :param trial: 

        """
        x = np.linspace(-3, 6, 200)
        for b in bandits:
            y = norm.pdf(x, b.m, np.sqrt(1. / b.lambda_))
            plt.plot(x, y, label=f"real mean: {b.true_mean:.4f}, num plays: {b.N}")
            plt.title("Bandit distributions after {} trials".format(trial))
        plt.legend()
        plt.show()

    def experiment(self, bandit_rewards, N):
        """

        :param bandit_rewards: 
        :param N: 

        """
        bandits = [ThompsonSampling(p) for p in bandit_rewards]
        probs = np.array(bandit_rewards)
        sample_points = [5, 10, 20, 50, 100, 200, 500, 1000, 1500, 1999]
        reward_per_bandit = np.empty(N)
        bandit_selected = np.empty(N)

        for i in range(N):
            j = np.argmax([b.sample() for b in bandits])

            if i in sample_points:
                self.plot(bandits, i)

            x = bandits[j].pull()
            reward_per_bandit[i] = x
            bandit_selected[i] = j

        cumulative_average = np.cumsum(reward_per_bandit) / (np.arange(N) + 1)
        cumulative_sum = np.cumsum(reward_per_bandit)
        cumulative_regret = np.empty(N)

        for i in range(len(reward_per_bandit)):
            cumulative_regret[i] = N * max(probs) - cumulative_sum[i]

        return cumulative_sum, cumulative_average, cumulative_regret, bandits, bandit_selected, reward_per_bandit

def comparison(N, epsilon_greedy_results, thompson_sampling_results):
    """

    :param N: 
    :param epsilon_greedy_results: 
    :param thompson_sampling_results: 

    """
    epsilon_greedy_cumulative = epsilon_greedy_results[7]
    thompson_sampling_cumulative = thompson_sampling_results[0]

    plt.figure(figsize=(8, 6))

    plt.plot(epsilon_greedy_cumulative, label='Epsilon Greedy')
    plt.plot(thompson_sampling_cumulative, label='Thompson Sampling')

    plt.legend()
    plt.title(f"Comparison of Cumulative Rewards")
    plt.xlabel("Number of Trials")
    plt.ylabel("Cumulative Reward")

    plt.tight_layout()
    plt.show()

# Add missing import for logging.Formatter
from logging import Formatter as CustomFormatter