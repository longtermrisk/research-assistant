# squiggpy

A *lean* Monte‑Carlo‑based library for back‑of‑the‑envelope probabilistic modelling.

---

## Quick‑start

### Installation

```bash
pip install matplotlib           # required for plotting
pip install scipy                # optional – smoother KDEs in plot_kde()
# then drop `squiggpy.py` into your project or `pip install -e .` if packaged
```

### Five‑minute "NYC taxis" example

```python
from squiggpy import Normal, Uniform, maximum, Mixture

# --- core assumptions ------------------------------------------------------
ny_population        = Uniform(8_000_000, 9_000_000)
locals_per_taxi      = 10 ** Normal(3, 1)          # log‑normal

peak_tourists        = maximum(10_000, Normal(1_000_000, 500_000))
taxis_per_tourist    = Uniform(0.001, 0.2)

# --- arithmetic ------------------------------------------------------------
locals_taxis         = ny_population / locals_per_taxi
visitor_taxis        = peak_tourists / taxis_per_tourist

nyc_taxis            = locals_taxis + visitor_taxis

# --- results ---------------------------------------------------------------
print(nyc_taxis.summary())    # mean, std, percentiles, mode
nyc_taxis.plot_hist()         # PDF

# --- scenario modelling ----------------------------------------------------
p_extinction         = 0.08                        # existential risk
world_pop_no_cat     = Normal(20e9, 5e9)
world_pop_3025       = Mixture({
    p_extinction: 0,
    1-p_extinction: world_pop_no_cat,
})
```

---

## High‑level API

| Distribution | Constructor | Notes |
|--------------|-------------|-------|
| **`Normal(μ, σ)`** | Gaussian | Log-normal patterns: use `10 ** Normal(...)` |
| **`Uniform(a, b)`** | Flat between *a* and *b* | Continuous, inclusive bounds |
| **`LogUniform(a, b)`** | Log-uniform between *a* and *b* | Requires *a*, *b* > 0 |
| **`Exp(*, scale=None, rate=None)`** | Exponential | Supply **either** `scale = θ` (mean) **or** `rate = λ = 1/θ` |
| **`Beta(α, β)`** | Beta | Support (0 , 1); α, β > 0 |
| **`Gamma(k, θ)`** | Gamma | Shape *k*, scale θ (mean = *k θ*) |
| **`Poisson(λ)`** | Poisson | Integer counts of events |
| **`Deterministic(c)`** | Constant | Still composable in arithmetic |
| **`Mixture({wᵢ: dᵢ, …})`** | Scenario-weighted mix | **Weights may themselves be distributions**; auto-normalised |


### Combination helpers

* **Arithmetic operators**: `+  -  *  /  **` work element‑wise on samples.
* **`clamp_min(threshold, dist)` / `clamp_max`** – *censor* the distribution.
* **`truncate(dist, lower=?, upper=?)`** – *truncate* & renormalise.
* **`maximum(a, b)` / `minimum(a, b)`** – quick element‑wise max/min.

### Correlated variables
```python
from squiggpy import MultiNormal
u, v, w = MultiNormal(
    mean=[0, 1, 5],
    std=[2, 4, 6],
    corr=[[1, 0, 0.5],
          [0, 1, 0.5],
          [0.5, 0.5, 1]],
)
np.corrcoef([u.samples, v.samples, w.samples])

# array([[1.        , 0.00529269, 0.50592624],
#        [0.00529269, 1.        , 0.50353151],
#        [0.50592624, 0.50353151, 1.        ]])
```

### Quick stats & plots

```python
rv.mean()                # arithmetic mean
rv.std()                 # sample std‑dev
rv.percentile([5, 95])   # any percentiles
rv.summary()             # dict – mean, std, percentiles, mode

rv.plot_hist(bins=60)    # histogram / PDF
rv.plot_cdf()            # empirical CDF
rv.plot_kde()            # smooth PDF (SciPy)
```

---

## Modelling strategies

### 1. **Top‑down decomposition**
Start with coarse scenarios, then refine each into sub‑questions until every leaf is "something I can assign a distribution to".
Use `placeholders()` in top-level models when they depend on complex values.

`Mixture` is tailor‑made for the scenario layer; arithmetic + base distributions handle the leaves.

### 2. **Model uncertainty via *ensembles***
Build multiple independent decompositions (different *modelling frames*) and combine the answers in a second‑level `Mixture` to reflect your uncertainty about which frame is right.

### 3. **Layered approximation**
If individual sub‑components are still hard, replace them with **deterministic placeholders** first, check order‑of‑magnitude, then promote to proper distributions once you know which pieces matter most.

### 4. **Sensitivity checks**
Because everything is Monte‑Carlo, inspect tail behaviour: plot on log‑x, look at 5th / 95th percentiles, or vary sample size to ensure convergence.

### 5. **Censor vs. truncate**
When applying hard bounds ask: *"Does the probability mass below the bound vanish or merely move?"*  Use `clamp_*` to *move* it (censoring), `truncate` to *discard & renormalise* (conditioning).

### 6. **Priors first**
For long‑term forecasts (e.g., world population in 3025) start with simple priors (Uniform ranges, wide Normals), then adjust by conditioning on landmark events or expert judgment.

### 7. **Stress‑testing improbable events**
For catastrophe tails, create dedicated `Mixture` branches with explicit small weights – this prevents rare outcomes from being "washed out" in arithmetic composites.

Feel free to propose or implement other heuristics – *squiggpy* is intentionally hackable.
