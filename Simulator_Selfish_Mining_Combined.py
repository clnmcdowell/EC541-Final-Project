"""
Selfish Mining Simulation Part-B

We tried to Reproduce results from the paper "Majority is not Enough: Bitcoin Mining
is Vulnerable", analyzing the analytical and simulation results for Selfish Mining compared
to Honest Mining for γ values of 0 and 1, with 95% confidence intervals for simulation results.

Useful Parameters Considered:
-------------
- Analytical revenue Equation from the paper.
- Simulation of DTMC.
- Confidence intervals for simulation mean revenues.

Author: Colin Mcdowell & Harshvardhan Shukla
"""

# Import necessary libraries
import math
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(0)

# Simulation parameters
ALPHA_VALUES = [0.10, 0.20, 0.25, 0.30, 0.33333, 0.40, 0.45, 0.475]
GAMMA_VALUES = [0.0, 1.0]  # Values of γ: tie-breaking parameter
NUM_ITER = 100  # Number of simulation batches
BURN_IN = 10_000  # Steps for system to reach steady-state
COUNT_STEPS = 100_000  # Steps to measure statistics
CONF_LEVEL = 0.05  # Confidence level for intervals (95%)


def analytical_Rpool(alpha: float, gamma: float) -> float:
    """
    Calculate analytical selfish mining revenue using the equation from the paper.

    Args:
        alpha (float): Fraction of mining power controlled by selfish miners.
        gamma (float): Fraction of honest miners choosing the selfish miner's block during tie.

    Returns:
        float: Analytical revenue for selfish mining pool.
    """
    num = alpha * (1 - alpha) ** 2 * (4 * alpha + gamma * (1 - 2 * alpha)) - alpha ** 3
    den = 1 - alpha * (1 + (2 - alpha) * alpha)
    return num / den


def run_trace(alpha: float, gamma: float, burn_in: int, count: int) -> float:
    """
    This function runs a single DTMC simulation trace for selfish mining.

    State Definitions:
        curr_state: Integer lead of the selfish pool.
        flag: Boolean flag distinguishing states 0 (single chain) and 0' (tie situation).

    Revenue Rules (Directly taken from the paper):
        - Neither miners collect revenue from state 0 to 1 transition.
        - Honest miners collect revenue only at state 0 (lead=0).
        - Pool collects additional revenues at certain transitions explicitly listed below.

    Args:
        alpha (float): Pool mining power fraction.
        gamma (float): Tie-breaking probability in favor of pool.
        burn_in (int): Steps for steady-state convergence.
        count (int): Steps counted for revenue measurement.

    Returns:
        float: Relative pool revenue from simulation trace.
    """
    curr_state, flag = 0, False
    r_pool = r_other = 0

    for step in range(burn_in + count):
        self_mined = np.random.random() < alpha  # Coin toss to decide who mines next block

        # Handle tie situation (state 0')
        if flag:
            if self_mined:
                if step >= burn_in: r_pool += 2  # Pool wins the race
            else:
                if np.random.random() < gamma:
                    if step >= burn_in:
                        r_pool += 1  # Honest miners build on pool's block
                        r_other += 1
                else:
                    if step >= burn_in:
                        r_other += 2  # Honest miners build on honest miners' block
            curr_state, flag = 0, False
            continue

        # Handle standard DTMC transitions
        if curr_state == 0:
            if self_mined:
                curr_state = 1  # Pool creates secret block (no revenue yet)
            else:
                if step >= burn_in:
                    r_other += 1  # Honest miners extend public chain
        elif curr_state == 1:
            if self_mined:
                curr_state = 2  # Pool secretly advances further
            else:
                curr_state, flag = 0, True  # Honest miners catch up, causing tie
        else:
            if self_mined:
                curr_state += 1  # Pool continues building secret lead
            else:
                if curr_state == 2:
                    if step >= burn_in:
                        r_pool += 2  # Pool publishes both hidden blocks immediately
                    curr_state, flag = 0, False
                else:
                    if step >= burn_in:
                        r_pool += 1  # Pool publishes one hidden block, maintains lead of one
                    curr_state -= 1

    total = r_pool + r_other
    return r_pool / total if total else 0.0


# Run Monte Carlo simulations for all gamma values
results = {}
for gamma in GAMMA_VALUES:
    sims = np.zeros((NUM_ITER, len(ALPHA_VALUES)))
    for j, alpha in enumerate(ALPHA_VALUES):
        for i in range(NUM_ITER):
            sims[i, j] = run_trace(alpha, gamma, BURN_IN, COUNT_STEPS)

    # Calculate mean, standard deviation, and confidence intervals
    mean = sims.mean(axis=0)
    std = sims.std(axis=0, ddof=1)
    t_factor = stats.t.ppf(1 - CONF_LEVEL / 2, NUM_ITER - 1)
    err = t_factor * std / math.sqrt(NUM_ITER)
    ana = np.array([analytical_Rpool(a, gamma) for a in ALPHA_VALUES])

    results[gamma] = {'mean': mean, 'err': err, 'ana': ana}

# Display simulation and analytical results neatly formatted
for gamma in GAMMA_VALUES:
    print(f"\nStatistical results for Gamma = {gamma:.3f}")
    for alpha, ana_val, sim_mean, sim_err in zip(ALPHA_VALUES, results[gamma]['ana'],
                                                 results[gamma]['mean'], results[gamma]['err']):
        print(f"At Alpha {alpha:.3f}:")
        print(f"Analytical Relative pool revenue is {ana_val:.3f}")
        print(f"Sample Mean Relative pool revenue is {sim_mean:.3f} with error {sim_err:.3f}")

# Generate comparison plot for gamma=0 and gamma=1
plt.figure(figsize=(7, 5))

# Honest mining baseline (45-degree line)
plt.plot(ALPHA_VALUES, ALPHA_VALUES, label='Honest Mining', color='gray', linestyle='--', linewidth=1)

# Plot analytical and simulation results
styles = {
    0.0: {'color': 'red', 'marker': 'o', 'linestyle': ':'},
    1.0: {'color': 'blue', 'marker': 's', 'linestyle': '-'}
}

for gamma in GAMMA_VALUES:
    plt.plot(ALPHA_VALUES, results[gamma]['ana'], label=f'γ={gamma} Analytical',
             color=styles[gamma]['color'], linestyle=styles[gamma]['linestyle'])
    plt.scatter(ALPHA_VALUES, results[gamma]['mean'], marker=styles[gamma]['marker'],
                color=styles[gamma]['color'], edgecolors='k', s=70, label=f'γ={gamma} Simulation')

plt.title('Comparison of Analytical vs. Simulation Results (γ = 0 and γ = 1)')
plt.xlabel('Pool size (α)')
plt.ylabel('Relative Pool Revenue (Rpool)')
plt.grid(True, linestyle=':')
plt.xlim(0.09, 0.5)
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.savefig('results_gamma0_and_1.png', dpi=300)
plt.show()

print("\nCombined figure saved as 'results_gamma0_and_1.png'.")
