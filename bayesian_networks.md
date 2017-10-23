# Bayesian Networks

## Computational Examples

Dataset:

A | B | C
--|---|--
T | F | T
T | T | F
T | F | T
F | T | T
F | T | T
F | F | T
F | F | F
F | F | T

$P(A = T) = \frac{3}{8}$

$P(B = T) = \frac{3}{8}$

$P(A = T \mid B = T) = \frac{1}{3}$

$P(B = T \mid A = T) = \frac{1}{3}$

The variables $A$ and $B$ are dependent since $P(A = T \mid B = T) \neq P(A =
T)$. That is, knowing that $B$ occurs affects the probability of $A$ occuring.

The goal is, given a dataset, to create a Bayesian network describing the
dependency relations.

## Score-Based Approach

This is very similar to genetic algorithms.

1. Generate and/or modify Bayesian networks.

2. Evaluate quality of networks using some "score" metric.

3. Repeat steps 1 and 2 with variations on the "best" network.

### Minimum Description Length (MDL)

This is one possible way to score a network.

Suppose that we have $M$ datapoints on a graph $G$ with $N$ attributes. Then,
the MDL is defined as $$L(G) = \frac{1}{M} \log \prod_{k = 1}^M P(D_{k} \mid G)
= \frac{1}{M} \sum_{j = 1}^N \sum_{k = 1}^M \log(P(X_j = d_{kj} \mid C(X_j)),$$
where $X_j$ is the $j$th variable, $d_{kj}$ is the $j$th attribute of the $k$th
datapoint $C(X_j)$ is the conditioning set of $X_j$, or the set of parents of
$X_j$. 

For example (because we desperately need one), consider the dataset from above,
with the graph $G$ described by $A \to C$ and $B \to C$. Then,
\begin{align*}
    L(G) &= \frac{1}{8} \sum_{k = 1}^8 \sum_{j = 1}^3 \log P(X_k \mid C(X_k)) \\
         &= \frac{1}{8} [3\log P(A = T) + 5 \log P(A = F) + \dots]
\end{align*}

## Independence Tests

Formally, two variables $X$ and $Y$ are independent iff $P(X \mid Y) = P(X)$.
When sampling values of $X$ and $Y$, we may never be sure if they are _actually_
independent, since we will never have the full picture. Patterson introduced us
to the $\chi^2$ (chi-squared) test for independence.

Null hypothesis ($H_0$): the variables $X$ and $Y$ are independent given $Z$.

$\chi^2$ statistic: square of actual minus expected, divided by expected. (?)

Example: Suppose that 100 people took a course. Of those, 50 passed the course.
Of the original 100, 50 passed the first exam, and only 25 of those that passed
the exam passed the course. Is passing the course independent of the first exam?

However (and I quote), of 50 passing the first exam, 35 passed the course. Are
they independent? (What?)

$$\chi^2 = \frac{(35 - 25)^2}{25} = \frac{100}{25} = 4.$$ Roughly, this means
that this is unlikely.

## PC Algorithm

1. Begin with a complete, undirected graph.

2. (Remove direct indepencies.) For all pairs of variables $(X, Y)$, if $X$ and
$Y$ are independent under some independence test, remove the $X \to Y$ edge.

3. (Remove indirect independencies.) For all pairs $(X, Y)$, and for all $Z$
that are adjacent to $X$ _or_ $Y$, if $X$ and $Y$ are independent given $Z$
under some independence test, remove the $X \to Z$ edge.

4. Repeat step 3 for all sets $Z$ of size 2, 3, ..., until out of sets. That is,
check if $X$ and $Y$ are independent given $Z_1$ and $Z_2$ for all adjacent
$Z_1$ and $Z_2$, and so on, for all sizes of the set $Z$.

5. We now have an undirected skeleton. If $(X, Y)$ are _both_ adjacent to $Z$,
then check if $X$ and $Y$ are independent given $Z$. If they are dependent,
then orient edges as $X \to Z \leftarrow Y$.

6. Repeat step 5 until all pairs are tested.

7. If $A \to B$, $B --- C$, the pair $(A, C)$ is not adjacent, and $C \not\to
B$, then set $B \to C$.

8. If $A --- B$ and there's a _directed_ path from $A$ to $B$, then set $A \to
B$.

9. Repeat steps 7, 8 until there is no change. Orient remaining edges randomly.

## Hidden Markov Models

### Forward Algorithm

Goal: Estimate $P(\text{state})$ given a set of observations and the current
model.

Using the Law of Total Probability, $$P(O) = \sum_{r = 1}^R P(O \mid \Omega_r)
P(\Omega_r),$$ where $\Omega_r$ is a possible sequence of states. Since
$\Omega_r$ represents a sequence of states, we will assume that $$P(\Omega_r) =
\prod_{t = 1}^T a_{\Omega_{r, t} \Omega_{r, t + 1}},$$ where $T$ is the number
of states in each $\Omega_r$ and $a_{ij}$ are the transition probabilities of
the Markov chain. We will also assume that $$P(O \mid \Omega_r) = \prod_{t =
1}^T b_{\Omega_{r, t}} (O_k),$$ where $b_{i}(o)$ is the probability of emitting
observation $o$ given that we were in state $i$. Thus, $$P(O) = \sum_{r = 1}^R
\prod_{t = 1}^T b_{\Omega_{r, t}}(O_t) a_{\Omega_{r, t} \Omega_{r, t + 1}}.$$

Runtime: $O(T N^T)$.

Forward Trellis: $R$ and $T$ are usually _really_ big, and that means that $O(T
N^T)$ is hard. The "forward trellis" avoids this by focusing only the the
observations seen and the most likely steps. This gets us down to $O(T N^2)$.

### Baum-Welch (Foward-Backward) Algorithm

Goal: Learn transition and observation probabilities.

Issues:

- Unsupervised learning.

- NP-Complete (almost certainly exponential)

Variables:

- $\beta_i(t) =$ probability that we are in state $i$ at time $t$.

- $\pi_i =$ probability of state $i$ in the initial distribution.

- $a_{ij} =$ probability of moving from state $i$ to state $j$.

- $b_i(o_k) =$ probability of observing $o_k$ after being in state $i$.

See page 340 for actual definitions.

Algorithm:

- Initialize $\pi$ to uniform distribution and $a_{ij}$, $b_i(o_k)$ to be random
  probabilities.

- While the algorithm has not converged:

    1. E-step: calculate $\alpha$ and $\beta$. For each $o_t$, $i$, $j$, compute
    $\xi_{i, j, k}$.

    2. M-step:
