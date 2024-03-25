# Travelling Salesman Problem with Time Windows <a name="tsptw"></a>

## Table of Contents
* [Travelling Salesman Problem with Time Windows](#tsptw)
	* [Problem Description](#description)
	* [Contributors](#contributors)
	* [References](#references)


## Problem Description <a name="description"></a>

A delivery driver picks up goods at the warehouse (point 0) and needs to deliver goods to N customers 1,2,…, N. Customer i is located at point i and has a delivery request in the time interval from e(i) to l(i) and takes d(i) units of time (s) to deliver. It is known that t(i, j) is the travel time from point i to point j. The delivery driver departs from the warehouse at time t0, calculate the delivery route for the salesman so that the total travel time is the shortest. ​

Each solution is represented by a permutation s[1], s[2], . . ., s[N] of 1, 2, . . ., N.​

### Input
    * Line 1: N
    * Lines 1 + i (i = 1,...,N): e(i), l(i), d(i) of each customer
    * Lines i + N + 2 (i = 0,...,N): the i-th row of the travel time matrix t(i,j)

### Output
    A delivery path consisting of all points except point 0

## Contributors <a name="contributors"></a>
We want to thank the following contributors for their valuable contributions to this project:
- [Decent-Cypher](https://github.com/Decent-Cypher): Test generator, integer programming, Beam ACO
- [ankhanhtran02](https://github.com/ankhanhtran02): Constraint programming, insertion heuristic, general VNS
- [FieryPhoenix](https://github.com/bananagobananza): Tabu Search
- [Vu](https://github.com/bluff-king): Genetic algorithm

We would like to express our gratitude towards [trinhminh11](https://github.com/trinhminh11) for his insertion heuristic idea, from which we took inspiration and added some modifications.

## References <a name="references"></a>
1. da Silva, R.F. and Urrutia, S. (2010). A General VNS heuristic for the traveling salesman problem with time windows. Discrete Optimization, 7(4), pp.203–211. doi:[https://doi.org/10.1016/j.disopt.2010.04.002](https://doi.org/10.1016/j.disopt.2010.04.002).​
2. López-Ibáñez, M. and Blum, C. (2010). Beam-ACO for the travelling salesman problem with time windows. Computers & Operations Research, 37(9), pp.1570–1583. doi:[https://doi.org/10.1016/j.cor.2009.11.015](https://doi.org/10.1016/j.cor.2009.11.015).

