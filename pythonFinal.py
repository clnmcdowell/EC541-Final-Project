
import math
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

ALPHA_VALUES = [0.10, 0.20, 0.25, 0.30, 0.33333, 0.40, 0.45, 0.475]
GAMMA        = 0.5
NUM_ITER     = 100
BURN_IN      = 10_000
COUNT_STEPS  = 100_000
CONF_LEVEL   = 0.05
np.random.seed(0)


def analytical_Rpool(alpha: float, gamma: float) -> float:
    num = alpha * (1 - alpha)**2 * (4*alpha + gamma*(1-2*alpha)) - alpha**3
    den = 1 - alpha * (1 + (2-alpha)*alpha)
    return num / den


def run_trace(alpha: float, gamma: float, burn_in: int, count: int) -> float:
    curr_state, flag = 0, False
    r_pool = r_other = 0

    for step in range(burn_in + count):
        self_mined = np.random.random() < alpha

        # 0′ state
        if flag:
            if self_mined:                       # pool wins race
                if step >= burn_in: r_pool += 2
            else:                                # honest wins race
                if np.random.random() < gamma:
                    if step >= burn_in: r_pool += 1; r_other += 1
                else:
                    if step >= burn_in: r_other += 2
            curr_state, flag = 0, False
            continue

        # regular states
        if curr_state == 0:
            if self_mined:
                curr_state = 1
            else:
                if step >= burn_in: r_other += 1
        elif curr_state == 1:
            if self_mined:
                curr_state = 2
            else:
                curr_state, flag = 0, True
        else:  # curr_state ≥ 2
            if self_mined:
                curr_state += 1
            else:
                if curr_state == 2:
                    if step >= burn_in: r_pool += 2
                    curr_state, flag = 0, False
                else:
                    if step >= burn_in: r_pool += 1
                    curr_state -= 1

    total = r_pool + r_other
    return r_pool / total if total else 0.0


sims = np.zeros((NUM_ITER, len(ALPHA_VALUES)))
for j, a in enumerate(ALPHA_VALUES):
    for i in range(NUM_ITER):
        sims[i, j] = run_trace(a, GAMMA, BURN_IN, COUNT_STEPS)
mean = sims.mean(axis=0)
std  = sims.std(axis=0, ddof=1)
err  = stats.t.ppf(1 - CONF_LEVEL/2, df=NUM_ITER-1) * std / math.sqrt(NUM_ITER)
ana  = np.array([analytical_Rpool(a, GAMMA) for a in ALPHA_VALUES])


print("An example of the output of the simulator for the case γ = 0.5 is shown below."
      " Note that in this case, selfish mining becomes preferable over honest "
      "mining once α > 0.250.")
print(f"Statistical results for Gamma = {GAMMA:.3f}")
for a, an, m, e in zip(ALPHA_VALUES, ana, mean, err):
    print(f"At Alpha {a:5.3f}:")
    print(f"Analytical Relative pool revenue is {an:6.3f}")
    print(f"Sample Mean Relative pool revenue is {m:6.3f} with error {e:6.3f}.")


plt.figure(figsize=(6.0, 4.0))
plt.plot(ALPHA_VALUES, ALPHA_VALUES, label='Honest Mining', color='tab:blue')
plt.plot(ALPHA_VALUES, ana, label='Analysis', color='tab:orange')
plt.scatter(ALPHA_VALUES, mean, marker='x', color='tab:green', s=70,
            label='Simulation')
plt.title('Comparison of analytical and simulation results for Gamma = 0.500')
plt.xlabel('Pool size (Alpha)')
plt.ylabel('Relative Pool Revenue (Rpool)')
plt.ylim(0, 1); plt.xlim(0.09, 0.5)
plt.grid(True, linestyle=':'); plt.legend()
plt.tight_layout(); plt.savefig('results_gamma0p5.png', dpi=300); plt.show()
print("\nFigure saved as 'results_gamma0p5.png'.")
