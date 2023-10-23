# ab_test
incomplete

Please note that this code is incomplete and does not include a report file. It is also not functioning correctly.


Project Overview
This project aims to explore and compare two popular multi-armed bandit algorithms: Epsilon Greedy and Thompson Sampling. Multi-armed bandit problems involve finding a balance between exploration (trying different arms) and exploitation (choosing the best-known arm). These algorithms are commonly used in various applications, including online advertising and recommendation systems.


Project Structure
bandit.py
This Python file contains the core implementations for the project:

Create a Bandit Class: In this file, an abstract class is defined to represent the bandit arms. The bandit arms are used in both the Epsilon Greedy and Thompson Sampling algorithms.

Create EpsilonGreedy() and ThompsonSampling() Classes and Methods: The file defines the classes and methods for the Epsilon Greedy and Thompson Sampling algorithms. These classes are inherited from the Bandit class.

Epsilon Greedy
The Epsilon Greedy class is implemented with a decay in epsilon over time (1/t).
The experiment is designed using the Epsilon Greedy algorithm.
Thompson Sampling
The Thompson Sampling class is implemented with known precision.
The experiment is designed using the Thompson Sampling algorithm.