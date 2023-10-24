# ab_test
A/B Testing with Epsilon Greedy and Thompson Sampling
Project Summary

This project focuses on exploring and comparing two widely-used multi-armed bandit algorithms: Epsilon Greedy and Thompson Sampling. Multi-armed bandit problems revolve around balancing the exploration of different arms with the exploitation of the best-known arm. These algorithms find applications in various domains, such as online advertising and recommendation systems.

Bandit Class Creation: We start by developing a class to represent the bandit arms, which are essential for both Epsilon Greedy and Thompson Sampling algorithms.

EpsilonGreedy() and ThompsonSampling() Class and Method Implementation: The project involves creating classes and methods for the Epsilon Greedy and Thompson Sampling algorithms. Both of these classes inherit from the Bandit class.

Epsilon Greedy:

We implement an Epsilon Greedy class that progressively reduces epsilon as a function of time (1/t).
We design an experiment using the Epsilon Greedy algorithm.
Thompson Sampling:

We build a Thompson Sampling class with a known precision.
The experiment is designed based on the Thompson Sampling algorithm.
Project Structure

Bandit.py: This file defines an abstract class representing the bandit arms, along with the implementations of the Epsilon Greedy and Thompson Sampling algorithms. It also includes functions for visualizing the results of the experiments.
Report.ipynb: Checks the flow of the algo.