from abc import ABC, abstractmethod
from logs import *
import numpy as np
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt

logging.basicConfig
logger = logging.getLogger("MAB Application")


# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

ch.setFormatter(CustomFormatter())

logger.addHandler(ch)



class Bandit(ABC):
    ##==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##

    def __init__(self, p):
        self.p = p
        self.p_estimate = 0 #estimate of average reward
        self.N = 0
        self.r_estimate = 0 #estimate of average regret

    def __repr__(self):
        return f'An Arm with {self.p} Win Rate'

    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def experiment(self):
        pass

    def report(self, N, results, algorithm = "Epsilon Greedy"):
        if algorithm == 'EpsilonGreedy':
            cumulative_reward_average, cumulative_reward,  cumulative_regret, bandits, chosen_bandit, reward, count_suboptimal, num_times_exploited, num_times_explored = results 
        else:
            cumulative_reward_average, cumulative_reward,  cumulative_regret, bandits, chosen_bandit, reward  = results 
        
        # Save experiment data to a CSV file
        df = pd.DataFrame({
            'Bandit': [b for b in chosen_bandit],
            'Reward': [r for r in reward],
            'Algorithm': algorithm
        })

        df.to_csv(f'{algorithm}.csv', index=False)

        # Save Final Results to a CSV file
        df1 = pd.DataFrame({
            'Bandit': [b for b in bandits],
            'Reward': [p.p_estimate for p in bandits],
            'Algorithm': algorithm
        })

        df1.to_csv(f'{algorithm}Final.csv', index=False)

        for b in range(len(bandits)):
            print(f'Bandit with True Win Rate {bandits[b].p} - Pulled {bandits[b].N} times - Estimated average reward - {round(bandits[b].p_estimate, 4)} - Estimated average regret - {round(bandits[b].r_estimate, 4)}')
            print("--------------------------------------------------")
        
        
        print(f"Cumulative Reward : {sum(reward)}", end = "\n")

        
        
        
        print(f"Cumulative Regret : {cumulative_regret[-1]}", end = "\n")
                      
        if algorithm == 'EpsilonGreedy':                            
            print(f"Percent suboptimal : {round((float(count_suboptimal) / N), 4)}")
            print("# of explored: {}".format(num_times_explored))
            print("# of exploited: {}".format(num_times_exploited))



#--------------------------------------#

class Visualization:
    def plot1(self, N, results, algorithm='EpsilonGreedy'):        
        cumulative_reward_average = results[0]
        bandits = results[3]
        
        ## lin
        plt.plot(cumulative_reward_average, label='Cumulative Average Reward')
        plt.plot(np.ones(N) * max([b.p for b in bandits]), label='Optimal Reward')
        plt.legend()
        plt.title(f"Win Rate Convergence for {algorithm} - Linear Scale")
        plt.xlabel("Number of Trials")
        plt.ylabel("Estimated Reward")
        plt.show()

        ## log
        plt.plot(cumulative_reward_average, label='Cumulative Average Reward')
        plt.plot(np.ones(N) * max([b.p for b in bandits]), label='Optimal Reward')
        plt.legend()
        plt.title(f"Win Rate Convergence for {algorithm} - Log Scale")
        plt.xlabel("Number of Trials")
        plt.ylabel("Estimated Reward")
        plt.xscale("log")
        plt.show()

    def plot2(self, results_eg, results_ts):
        cumulative_rewards_eps = results_eg[1]
        cumulative_rewards_th = results_ts[1]
        cumulative_regret_eps = results_eg[2]
        cumulative_regret_th = results_ts[2]

        ## Reward
        plt.plot(cumulative_rewards_eps, label='Epsilon-Greedy')
        plt.plot(cumulative_rewards_th, label='Thompson Sampling')
        plt.legend()
        plt.title("Cumulative Reward Comparison")
        plt.xlabel("Number of Trials")
        plt.ylabel("Cumulative Reward")
        plt.show()

        ## Regret
        plt.plot(cumulative_regret_eps, label='Epsilon-Greedy')
        plt.plot(cumulative_regret_th, label='Thompson Sampling')
        plt.legend()
        plt.title("Cumulative Regret Comparison")
        plt.xlabel("Number of Trials")
        plt.ylabel("Cumulative Regret")
        plt.show()

class EpsilonGreedy(Bandit):
    def __init__(self, p):
        super().__init__(p)

    def pull(self):
        return np.random.randn() + self.p

    def update(self, x):
        self.N += 1.
        self.p_estimate = (1 - 1.0/self.N) * self.p_estimate + 1.0/ self.N * x
        self.r_estimate = self.p - self.p_estimate


    def experiment(self, BANDIT_REWARDS, N, t = 1):
        bandits = [EpsilonGreedy(p) for p in BANDIT_REWARDS]
        means = np.array(BANDIT_REWARDS)
        true_best = np.argmax(means)  
        count_suboptimal = 0
        EPS = 1/t

        reward = np.empty(N)
        chosen_bandit = np.empty(N)
        num_times_explored = 0
        num_times_exploited = 0

        for i in range(N):
            p = np.random.random()
            
            if p < EPS:
                num_times_explored += 1
                j = np.random.choice(len(bandits))
            else:
                num_times_exploited += 1
                j = np.argmax([b.p_estimate for b in bandits])

            x = bandits[j].pull()
            
            bandits[j].update(x)
    

            if j != true_best:
                count_suboptimal += 1
            
            reward[i] = x
            chosen_bandit[i] = j
            
            t+=1
            EPS = 1/t

        cumulative_reward_average = np.cumsum(reward) / (np.arange(N) + 1)
        cumulative_reward = np.cumsum(reward)
        
        cumulative_regret = np.empty(N)
        for i in range(len(reward)):
            cumulative_regret[i] = N*max(means) - cumulative_reward[i]

        return cumulative_reward_average, cumulative_reward,  cumulative_regret, bandits, chosen_bandit, reward, count_suboptimal, num_times_exploited, num_times_explored


class ThompsonSampling(Bandit):
    
    def __init__(self, p):
        super().__init__(p)
        self.lambda_ = 1
        self.tau = 1


    def pull(self):
        return np.random.randn() / np.sqrt(self.tau) + self.p
    
    def sample(self):
        return np.random.randn() / np.sqrt(self.lambda_) + self.p_estimate
    
    def update(self, x):
        self.p_estimate = (self.tau * x + self.lambda_ * self.p_estimate) / (self.tau + self.lambda_)
        self.lambda_ += self.tau
        self.N += 1
        self.r_estimate = self.p - self.p_estimate
        
    def plot(self, bandits, trial):
        x = np.linspace(-3, 6, 200)
        for b in bandits:
            y = norm.pdf(x, b.p_estimate, np.sqrt(1. / b.lambda_))
            plt.plot(x, y, label=f"real mean: {b.p:.4f}, num plays: {b.N}")
            plt.title("Bandit distributions after {} trials".format(trial))
        plt.legend()
        plt.show()

    def experiment(self, BANDIT_REWARDS, N):
        
        bandits = [ThompsonSampling(m) for m in BANDIT_REWARDS]

        sample_points = [5, 20, 50,100,200,500,1000,1999, 5000,10000, 19999]
        reward = np.empty(N)
        chosen_bandit = np.empty(N)
        
        for i in range(N):
            j = np.argmax([b.sample() for b in bandits])

            if i in sample_points:
                self.plot(bandits, i)

            x = bandits[j].pull()

            bandits[j].update(x)

            reward[i] = x
            chosen_bandit[i] = j

        cumulative_reward_average = np.cumsum(reward) / (np.arange(N) + 1)
        cumulative_reward = np.cumsum(reward)
        
        cumulative_regret = np.empty(N)
        
        for i in range(len(reward)):
            cumulative_regret[i] = N*max([b.p for b in bandits]) - cumulative_reward[i]


        return cumulative_reward_average, cumulative_reward,  cumulative_regret, bandits, chosen_bandit, reward 
 



def comparison(N, results_eg, results_ts):
    # think of a way to compare the performances of the two algorithms VISUALLY 
    cumulative_reward_average_eg = results_eg[0]
    cumulative_reward_average_ts = results_ts[0]
    bandits_eg = results_eg[3]
    reward_eg = results_eg[5]
    reward_ts = results_ts[5]
    regret_eg = results_eg[2][-1]
    regret_ts = results_ts[2][-1]

    
    print(f"Total Reward Epsilon Greedy : {sum(reward_eg)}")
    print(f"Total Reward Thomspon Sampling : {sum(reward_ts)}")
        
    print(" ")
        
    print(f"Total Regret Epsilon Greedy : {regret_eg}")
    print(f"Total Regret Thomspon Sampling : {regret_ts}")
        

    plt.figure(figsize=(12, 5))

    ## log
    plt.subplot(1, 2, 2)
    plt.plot(cumulative_reward_average_eg, label='Cumulative Average Reward Epsilon Greedy')
    plt.plot(cumulative_reward_average_ts, label='Cumulative Average Reward Thompson Sampling')
    plt.plot(np.ones(N) * max([b.p for b in bandits_eg]), label='Optimal Reward')
    plt.legend()
    plt.title(f"Comparison of Win Rate Convergence  - Log Scale")
    plt.xlabel("Number of Trials")
    plt.ylabel("Estimated Reward")
    plt.xscale("log")
    plt.tight_layout()
    plt.show()
    

    ## lin
    plt.subplot(1, 2, 1)
    plt.plot(cumulative_reward_average_eg, label='Cumulative Average Reward Epsilon Greedy')
    plt.plot(cumulative_reward_average_ts, label='Cumulative Average Reward Thompson Sampling')
    plt.plot(np.ones(N) * max([b.p for b in bandits_eg]), label='Optimal Reward')
    plt.legend()
    plt.title(f"Comparison of Win Rate Convergence  - Linear Scale")
    plt.xlabel("Number of Trials")
    plt.ylabel("Estimated Reward")




    