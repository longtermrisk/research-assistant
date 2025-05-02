"""squiggpy - a *lean* Monte-Carlo toolkit for probabilistic back-of-the-envelope models
======================================================================================
Self-contained single file. Drop it next to your notebook and `import squiggpy`.

Key features
------------
* **Scalar & Distributional Parameters**: Most distribution parameters (e.g., `mean`, `std`, `low`, `high`)
  can themselves be Distributions.
* **Primitive distributions**: Normal, Uniform, LogUniform, Beta, Gamma, Poisson, Exp, Deterministic.
* **Scenario Mixture** - branch-weighted distributions (weights can also be distributions).
* **Censor / Truncate** - `clamp_min`, `clamp_max`, `truncate`.
* **Quick stats & plots** - `mean`, `std`, `summary`, `plot_hist`, `plot_cdf`, `plot_kde`.
* **Gaussian-copula correlation** via `MultiNormal`.

Dependencies: NumPy. Matplotlib & SciPy are optional (plots / KDE / copula).
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple, Union

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - plotting optional
    plt = None  # type: ignore

try:
    from scipy.stats import gaussian_kde, norm  # type: ignore
except ImportError:  # pragma: no cover - KDE & copula optional
    gaussian_kde = None  # type: ignore
    norm = None  # type: ignore

# -----------------------------------------------------------------------------
# Public interface
# -----------------------------------------------------------------------------
__all__ = [
    "Distribution",
    "Normal",
    "Uniform",
    "LogUniform",
    "Beta",
    "Gamma",
    "Poisson",
    "Exp",
    "Deterministic",
    "Mixture",
    "maximum",
    "minimum",
    "clamp_min",
    "clamp_max",
    "truncate",
    "MultiNormal", # Added MultiNormal to __all__
]

Number = Union[int, float] # Keep scalar type hint simple
DistOrNum = Union[Number, "Distribution"]
_DEFAULT_SAMPLES = 1_000_000 # Reduced default for faster examples during dev

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _ensure_dist(x: DistOrNum, *, n: int | None = None) -> "Distribution":
    """Converts input to a Distribution, propagating sample size `n`."""
    if isinstance(x, Distribution):
        # If sizes differ, this might indicate an issue, but we let resize handle it later.
        # A more robust system might enforce consistent n values.
        # For now, ensure the returned dist knows about the target 'n' if provided.
        if n is not None and x._size != n:
             # Return a *new* distribution linked to the original sampler but with the new size?
             # Or just trust resizing? Let's trust resizing for now.
             # The main place n matters is setting the *initial* size.
             pass # Keep original dist object
        return x
    # If x is scalar, create a Deterministic distribution with the specified size n
    return Deterministic(x, n=n) # Pass n here

# -----------------------------------------------------------------------------
# Core class
# -----------------------------------------------------------------------------

class Distribution:
    """Lazy 1-D random variable backed by Monte-Carlo samples."""

    __slots__ = ("_sampler", "_cache", "_size", "_params") # Added _params for potential introspection

    def __init__(self, sampler: Callable[[int], np.ndarray], *, n: int | None = None, params: dict | None = None):
        """
        Initializes a Distribution.

        Args:
            sampler: A function that takes an integer k and returns k samples.
            n: The default number of samples to generate.
            params: Optional dictionary storing parameters (can be Distributions).
        """
        self._sampler = sampler
        self._cache: np.ndarray | None = None
        self._size: int = n or _DEFAULT_SAMPLES
        self._params = params or {} # Store parameters if provided

    # ---------------- sampling & stats ----------------
    @property
    def samples(self) -> np.ndarray:
        """Generates and caches Monte Carlo samples on first access."""
        if self._cache is None:
            # Generate samples using the _sampler defined by the subclass
            # The sampler itself now handles getting samples from parameter distributions
            self._cache = np.asarray(self._sampler(self._size), dtype=float).ravel()
            if len(self._cache) != self._size:
                 # This shouldn't happen if samplers respect k, but worth a check
                 raise RuntimeError(f"Sampler generated {len(self._cache)} samples, expected {self._size}")
        return self._cache

    def mean(self) -> float:  # noqa: D401
        """Calculates the mean of the samples."""
        return float(np.mean(self.samples))

    def std(self) -> float:
        """Calculates the standard deviation of the samples."""
        return float(np.std(self.samples, ddof=1))

    def percentile(self, q: Union[int, float, Sequence[int | float]]) -> np.ndarray:
         """Calculates percentiles of the samples."""
         # Convert q to list if scalar for consistent output type with np.percentile
         qs = q if isinstance(q, (list, tuple, np.ndarray)) else [q]
         result = np.percentile(self.samples, qs)
         return result if isinstance(q, (list, tuple, np.ndarray)) else result[0] # Return scalar if input was scalar


    def mode(self, *, bins: int = 100) -> float:
        """Estimates the mode from a histogram of the samples."""
        # Handle case with zero std deviation (Deterministic)
        if self.std() == 0:
            return self.mean()
        # Handle case where all samples might be identical (though std should be 0)
        unique_samples = np.unique(self.samples)
        if len(unique_samples) == 1:
            return unique_samples[0]

        try:
             hist, edges = np.histogram(self.samples, bins=bins)
             idx = np.argmax(hist)
             # Ensure idx+1 is within bounds for edges
             if idx + 1 < len(edges):
                 return float((edges[idx] + edges[idx + 1]) / 2)
             else: # Handle edge case where max is in the last bin
                 return float(edges[idx])
        except Exception: # Catch potential errors with histogramming (e.g., bad data)
             # Fallback to mean if mode calculation fails
             return self.mean()


    def summary(self, percentiles: Iterable[int | float] = (5, 50, 95)) -> Dict[str, Any]:
        """Provides a summary dictionary of key statistics."""
        p_list = list(percentiles)
        p_values = self.percentile(p_list)
        return {
            "mean": self.mean(),
            "std": self.std(),
            "percentiles": {f"{p}%": float(v) for p, v in zip(p_list, p_values)},
            "mode": self.mode(),
        }

    # ---------------- plotting (optional) --------------
    def _need_mpl(self):
        if plt is None:
            raise RuntimeError("matplotlib is required - `pip install matplotlib`.")

    def plot_hist(self, *, bins: int = 50, density: bool = True,
              log_x: bool = False, log_y: bool = False,
              ax=None, title=None, **kw):
        """Plots a histogram of the samples."""
        self._need_mpl()
        ax = ax or plt.gca()

        samples_to_plot = self.samples
        # Avoid issues with log scale if samples are non-positive
        if log_x:
            samples_to_plot = samples_to_plot[samples_to_plot > 0]
            if len(samples_to_plot) == 0:
                print("Warning: No positive samples to plot on log scale.")
                return ax # Avoid plotting if no data
            # Create log-spaced bins, handle identical values
            min_sample, max_sample = samples_to_plot.min(), samples_to_plot.max()
            if min_sample == max_sample or min_sample <= 0:
                 bins_actual = bins # Fallback if log fails
            else:
                 bins_actual = np.logspace(np.log10(min_sample), np.log10(max_sample), bins + 1)
            ax.set_xscale('log')
        else:
            bins_actual = bins

        ax.hist(samples_to_plot, bins=bins_actual, density=density, **kw)

        if log_y:
            ax.set_yscale('log')

        ax.set_title(title or "Histogram")
        ax.set_ylabel("Density" if density else "Frequency")
        return ax


    def plot_cdf(self, *, ax=None, **kw):
        """Plots the empirical Cumulative Distribution Function (CDF)."""
        self._need_mpl()
        ax = ax or plt.gca()
        x = np.sort(self.samples)
        y = np.linspace(0, 1, len(x), endpoint=False)
        ax.step(x, y, where="post", **kw)
        ax.set_ylabel("CDF")
        ax.set_title("Empirical CDF")
        ax.set_ylim(0, 1) # Ensure CDF y-axis is 0 to 1
        return ax

    def plot_kde(self, *, ax=None, bw_method="scott", num: int = 200, **kw):
        """Plots the Kernel Density Estimate (KDE)."""
        if gaussian_kde is None:
             print("SciPy not found, falling back to histogram plot for KDE.")
             return self.plot_hist(ax=ax, density=True, title="Histogram (KDE fallback)", **kw)

        self._need_mpl()
        ax = ax or plt.gca()
        samples = self.samples
        # Avoid KDE errors with constant data
        if self.std() < 1e-9: # Check for near-zero std dev
             print("Warning: Data seems constant, plotting histogram instead of KDE.")
             return self.plot_hist(ax=ax, density=True, title="Histogram (KDE fallback)", **kw)

        try:
            kde = gaussian_kde(samples, bw_method=bw_method)
            xmin, xmax = samples.min(), samples.max()
            # Add a small margin unless min/max are equal
            margin = (xmax - xmin) * 0.1 if xmax > xmin else 1.0
            xs = np.linspace(xmin - margin, xmax + margin, num)
            ax.plot(xs, kde(xs), **kw)
            ax.set_ylabel("Density")
            ax.set_title("KDE")
            ax.set_ylim(bottom=0) # Density shouldn't be negative
        except Exception as e:
             print(f"KDE calculation failed: {e}. Plotting histogram instead.")
             return self.plot_hist(ax=ax, density=True, title="Histogram (KDE fallback)", **kw)
        return ax


    # ---------------- arithmetic ----------------------
    def _get_samples_resized(self, target_size: int) -> np.ndarray:
        """Internal helper to get samples, resizing if needed."""
        s = self.samples # Trigger sample generation/retrieval
        if len(s) == target_size:
            return s
        # Resize carefully: np.resize repeats if target > source size.
        # This is often the desired behavior in broadcasting-like scenarios.
        return np.resize(s, target_size)


    def _bin_op(self, other: DistOrNum, op: Callable[[np.ndarray, np.ndarray], np.ndarray], *, refl=False):
        """Performs a binary operation between self and other (scalar or Distribution)."""
        # Determine the size for the operation. Usually the size of self.
        # If 'other' is a Distribution with a *different* size, this might be ambiguous.
        # Let's default to self._size, but a warning or error might be warranted in a production system.
        target_size = self._size
        other_d = _ensure_dist(other, n=target_size) # Ensure other is Dist, propagating size

        # Define the sampler for the resulting distribution
        def sampler(k: int):
            # Get k samples from self and other, ensuring they match size k
            # Use the internal resizing helper
            self_samples_k = self._get_samples_resized(k)
            other_samples_k = other_d._get_samples_resized(k)

            # Apply the operation element-wise
            if refl:
                return op(other_samples_k, self_samples_k)
            else:
                return op(self_samples_k, other_samples_k)

        # Create the new distribution resulting from the operation
        # Pass the calculated size 'target_size'
        return Distribution(sampler, n=target_size)


    def __add__(self, o):  return self._bin_op(o, np.add)
    def __radd__(self, o): return self._bin_op(o, np.add, refl=True)
    def __sub__(self, o):  return self._bin_op(o, np.subtract)
    def __rsub__(self, o): return self._bin_op(o, np.subtract, refl=True)
    def __mul__(self, o):  return self._bin_op(o, np.multiply)
    def __rmul__(self, o): return self._bin_op(o, np.multiply, refl=True)
    def __truediv__(self, o):  return self._bin_op(o, np.divide)
    def __rtruediv__(self, o): return self._bin_op(o, np.divide, refl=True)
    def __pow__(self, o):  return self._bin_op(o, np.power)
    def __rpow__(self, o): return self._bin_op(o, np.power, refl=True)

    # ---------------- unary arithmetic --------------------
    def _unary_op(self, op: Callable[[np.ndarray], np.ndarray]):
        """Return a Distribution whose samples are op(self.samples)."""
        def sampler(k: int):
            self_samples_k = self._get_samples_resized(k)
            return op(self_samples_k)
        return Distribution(sampler, n=self._size)

    def __neg__(self):          #  −X
        return self._unary_op(np.negative)

    def __pos__(self):          #  +X   (mostly for symmetry)
        return self # No change needed

    def log(self):
        """Return Distribution for log(self)."""
        return self._unary_op(np.log)

    def exp(self):
        """Return Distribution for exp(self)."""
        return self._unary_op(np.exp)


    # comparisons on means - handy for Python conditionals
    # Note: Comparing distributions is complex. This compares *means only*.
    def __lt__(self, other: DistOrNum): return self.mean() < _ensure_dist(other).mean()
    def __le__(self, other: DistOrNum): return self.mean() <= _ensure_dist(other).mean()
    def __gt__(self, other: DistOrNum): return self.mean() > _ensure_dist(other).mean()
    def __ge__(self, other: DistOrNum): return self.mean() >= _ensure_dist(other).mean()

    def __repr__(self):  # pragma: no cover
        # Attempt to show parameters if they exist and are simple
        param_str = ""
        if self._params:
            simple_params = {}
            for k, v in self._params.items():
                if isinstance(v, Deterministic):
                    simple_params[k] = f"{v.mean():.3g}"
                elif isinstance(v, Distribution):
                     simple_params[k] = f"Dist(μ={v.mean():.2g},σ={v.std():.2g})" # Show mean/std of param dist
                else: # Should be scalar if not Dist
                    simple_params[k] = f"{v:.3g}"
            param_str = f" ({', '.join(f'{k}={v}' for k,v in simple_params.items())})"

        # Use stats even if params aren't simple
        return f"<{self.__class__.__name__}{param_str} n={self._size} mean={self.mean():.3g} sd={self.std():.3g}>"

# -----------------------------------------------------------------------------
# Primitive distributions - MODIFIED FOR DISTRIBUTIONAL PARAMETERS
# -----------------------------------------------------------------------------
class Normal(Distribution):
    def __init__(self, mean: DistOrNum, std: DistOrNum, *, n: int | None = None):
        n_eff = n or _DEFAULT_SAMPLES # Determine effective N early
        mean_d = _ensure_dist(mean, n=n_eff)
        std_d = _ensure_dist(std, n=n_eff)

        # Check if std deviation distribution can be negative (problematic)
        # Simple check on mean/percentiles - not foolproof for complex distributions
        try:
             if std_d.percentile(0) < 0:
                  print(f"Warning: Standard deviation distribution for Normal seems to allow negative values (min sample={std_d.percentile(0):.3g}). This might lead to errors or unexpected behavior.")
        except: # Handle cases where percentile might fail
             pass

        def sampler(k: int):
            mean_samples = mean_d._get_samples_resized(k)
            std_samples = std_d._get_samples_resized(k)
            # Clamp sampled std dev >= 0 before passing to np.random.normal
            std_samples_clamped = np.maximum(std_samples, 0)
            if np.any(std_samples < 0):
                 # This warning occurs at sampling time if negative stds were actually generated
                 # print(f"Warning: Clamped {np.sum(std_samples < 0)} negative std dev samples to 0.")
                 pass # Reduce verbosity, warning in __init__ is usually enough
            return np.random.normal(mean_samples, std_samples_clamped, size=k)

        super().__init__(sampler, n=n_eff, params={'mean': mean_d, 'std': std_d})

class Uniform(Distribution):
    def __init__(self, low: DistOrNum, high: DistOrNum, *, n: int | None = None):
        n_eff = n or _DEFAULT_SAMPLES
        low_d = _ensure_dist(low, n=n_eff)
        high_d = _ensure_dist(high, n=n_eff)

        # Basic check: does high mean seem lower than low mean?
        try:
            if high_d.mean() < low_d.mean():
                print(f"Warning: Mean of 'high' distribution ({high_d.mean():.3g}) is less than mean of 'low' distribution ({low_d.mean():.3g}) for Uniform.")
        except: pass

        def sampler(k: int):
            low_samples = low_d._get_samples_resized(k)
            high_samples = high_d._get_samples_resized(k)
            # Ensure high >= low element-wise for sampling
            # If high < low, sample from (high, low) instead? Or clamp?
            # Let's swap them element-wise to ensure a valid range.
            low_eff = np.minimum(low_samples, high_samples)
            high_eff = np.maximum(low_samples, high_samples)
            # Generate uniform(0, 1) and scale/shift
            return low_eff + (high_eff - low_eff) * np.random.rand(k)
            # Alt: np.random.uniform(low_eff, high_eff, size=k) # numpy >= 1.17 handles array args well

        super().__init__(sampler, n=n_eff, params={'low': low_d, 'high': high_d})

class LogUniform(Distribution):
    def __init__(self, low: DistOrNum, high: DistOrNum, *, n: int | None = None):
        n_eff = n or _DEFAULT_SAMPLES
        low_d = _ensure_dist(low, n=n_eff)
        high_d = _ensure_dist(high, n=n_eff)

        # Check if bounds distributions can be non-positive
        try:
            if low_d.percentile(0) <= 0 or high_d.percentile(0) <= 0:
                 print(f"Warning: LogUniform bounds distributions seem to allow non-positive values (low min={low_d.percentile(0):.3g}, high min={high_d.percentile(0):.3g}). This will cause errors during sampling.")
        except: pass
        # Basic check: does high mean seem lower than low mean?
        try:
             if high_d.mean() < low_d.mean():
                 print(f"Warning: Mean of 'high' distribution ({high_d.mean():.3g}) is less than mean of 'low' distribution ({low_d.mean():.3g}) for LogUniform.")
        except: pass


        def sampler(k: int):
            low_samples = low_d._get_samples_resized(k)
            high_samples = high_d._get_samples_resized(k)

            # Ensure low and high are positive element-wise
            if np.any(low_samples <= 0) or np.any(high_samples <= 0):
                 raise ValueError("LogUniform requires positive bound samples. Check input distributions.")

             # Ensure high >= low element-wise
            low_eff = np.minimum(low_samples, high_samples)
            high_eff = np.maximum(low_samples, high_samples)

            # Sample uniformly in log space, then exponentiate
            log_low = np.log(low_eff)
            log_high = np.log(high_eff)
            log_samples = log_low + (log_high - log_low) * np.random.rand(k)
            # Alt: np.random.uniform(log_low, log_high, size=k)
            return np.exp(log_samples)

        super().__init__(sampler, n=n_eff, params={'low': low_d, 'high': high_d})

class Beta(Distribution):
    def __init__(self, a: DistOrNum, b: DistOrNum, *, n: int | None = None):
        n_eff = n or _DEFAULT_SAMPLES
        a_d = _ensure_dist(a, n=n_eff)
        b_d = _ensure_dist(b, n=n_eff)

        # Check if params can be non-positive
        try:
             if a_d.percentile(0) <= 0 or b_d.percentile(0) <= 0:
                  print(f"Warning: Beta parameters a or b seem to allow non-positive values (a min={a_d.percentile(0):.3g}, b min={b_d.percentile(0):.3g}). This will cause errors.")
        except: pass

        def sampler(k: int):
            a_samples = a_d._get_samples_resized(k)
            b_samples = b_d._get_samples_resized(k)
            # Ensure params > 0 element-wise
            if np.any(a_samples <= 0) or np.any(b_samples <= 0):
                 raise ValueError("Beta parameters 'a' and 'b' must have positive samples.")
            return np.random.beta(a_samples, b_samples, size=k)

        super().__init__(sampler, n=n_eff, params={'a': a_d, 'b': b_d})

class Gamma(Distribution):
    # Note: NumPy uses shape/scale parameterization. scale = 1/rate. Mean = shape*scale. Var = shape*scale^2.
    def __init__(self, shape: DistOrNum, scale: DistOrNum = 1.0, *, n: int | None = None):
        n_eff = n or _DEFAULT_SAMPLES
        shape_d = _ensure_dist(shape, n=n_eff)
        scale_d = _ensure_dist(scale, n=n_eff)

        # Check if params can be non-positive
        try:
            if shape_d.percentile(0) <= 0 or scale_d.percentile(0) <= 0:
                 print(f"Warning: Gamma parameters shape or scale seem to allow non-positive values (shape min={shape_d.percentile(0):.3g}, scale min={scale_d.percentile(0):.3g}). This will cause errors.")
        except: pass

        def sampler(k: int):
            shape_samples = shape_d._get_samples_resized(k)
            scale_samples = scale_d._get_samples_resized(k)
             # Ensure params > 0 element-wise
            if np.any(shape_samples <= 0) or np.any(scale_samples <= 0):
                 raise ValueError("Gamma parameters 'shape' and 'scale' must have positive samples.")
            return np.random.gamma(shape_samples, scale_samples, size=k)

        super().__init__(sampler, n=n_eff, params={'shape': shape_d, 'scale': scale_d})

class Poisson(Distribution):
    # Parameter lambda (rate) > 0
    def __init__(self, lam: DistOrNum, *, n: int | None = None):
        n_eff = n or _DEFAULT_SAMPLES
        lam_d = _ensure_dist(lam, n=n_eff)

        # Check if lam can be non-positive
        try:
             if lam_d.percentile(0) <= 0:
                  print(f"Warning: Poisson parameter lambda seems to allow non-positive values (min={lam_d.percentile(0):.3g}). This will cause errors.")
        except: pass

        def sampler(k: int):
            lam_samples = lam_d._get_samples_resized(k)
             # Ensure lam > 0 element-wise
            if np.any(lam_samples <= 0):
                 raise ValueError("Poisson parameter 'lam' must have positive samples.")
            return np.random.poisson(lam_samples, size=k)

        super().__init__(sampler, n=n_eff, params={'lam': lam_d})


class Exp(Distribution):
    """Exponential distribution with rate (lambda) or scale (theta = 1/lambda) parameter."""
    def __init__(self, *, rate: DistOrNum | None = None, scale: DistOrNum | None = None,
                 n: int | None = None):
        n_eff = n or _DEFAULT_SAMPLES

        if (rate is None) == (scale is None):
            raise ValueError("Specify exactly one of `rate` or `scale` for Exp distribution.")

        params = {}
        if scale is not None:
            scale_d = _ensure_dist(scale, n=n_eff)
            params['scale'] = scale_d
            # Check scale > 0
            try:
                 if scale_d.percentile(0) <= 0:
                      print(f"Warning: Exp scale parameter seems to allow non-positive values (min={scale_d.percentile(0):.3g}). This will cause errors.")
            except: pass
            def sampler(k: int):
                scale_samples = scale_d._get_samples_resized(k)
                if np.any(scale_samples <= 0):
                     raise ValueError("Exponential 'scale' parameter must have positive samples.")
                # numpy uses scale (theta) = 1 / lambda
                return np.random.exponential(scale_samples, size=k)
        else: # rate is not None
            rate_d = _ensure_dist(rate, n=n_eff)
            params['rate'] = rate_d
             # Check rate > 0
            try:
                 if rate_d.percentile(0) <= 0:
                      print(f"Warning: Exp rate parameter seems to allow non-positive values (min={rate_d.percentile(0):.3g}). This will cause errors.")
            except: pass
            def sampler(k: int):
                rate_samples = rate_d._get_samples_resized(k)
                if np.any(rate_samples <= 0):
                     raise ValueError("Exponential 'rate' parameter must have positive samples.")
                # numpy uses scale = 1 / rate
                scale_samples = 1.0 / rate_samples
                return np.random.exponential(scale_samples, size=k)

        super().__init__(sampler, n=n_eff, params=params)

class Deterministic(Distribution):
    """Represents a fixed, single value as a Distribution."""
    def __init__(self, value: Number, *, n: int | None = None):
        # Note: value parameter remains scalar for Deterministic.
        # A "Deterministic distribution of a distribution" doesn't fit the model well.
        v = float(value)
        n_eff = n or _DEFAULT_SAMPLES
        # Sampler simply returns an array filled with the value
        super().__init__(lambda k: np.full(k, v, dtype=float), n=n_eff, params={'value': v})
        self._v = v # Cache the value for direct access in overrides

    # Override stats for efficiency
    def mean(self) -> float: return self._v
    def std(self) -> float: return 0.0
    def mode(self, **_) -> float: return self._v
    def percentile(self, q: Union[int, float, Sequence[int | float]]) -> np.ndarray:
         # Return the constant value, matching shape of q if it's an array
         if isinstance(q, (list, tuple, np.ndarray)):
             return np.full(len(q), self._v)
         else:
             return np.array(self._v) # Return scalar-like array


# ------------------------------------------------------------------
# Multivariate Normal helper (family of correlated Normal Distributions)
# ------------------------------------------------------------------
def MultiNormal(
    mean: Sequence[DistOrNum], # Allow distributions for mean
    std: Sequence[DistOrNum],  # Allow distributions for std
    corr: Sequence[Sequence[float]], # Correlation matrix remains fixed floats
    *,
    n: int | None = None,
) -> List[Distribution]: # Changed return type hint to List[Distribution]
    """
    Generates correlated Normal distributions using Gaussian Copula + Normal marginals.
    Marginal means and std devs can now be specified as Distributions.

    Args:
        mean: Sequence of means (scalar or Distribution).
        std: Sequence of standard deviations (scalar or Distribution).
        corr: The correlation matrix (floats).
        n: Number of samples.

    Returns:
        A list of correlated Distribution objects, each behaving as the target Normal marginal.

    Example:
        mean_x = Uniform(0, 1)
        std_z = Uniform(5, 6)
        x, y, z = MultiNormal(
            mean=[mean_x, 1, 5],
            std=[2, 4, std_z],
            corr=[[1, 0, 0.5], [0, 1, -0.2], [0.5, -0.2, 1]],
        )

    Note: This implementation uses the Gaussian Copula method. It generates
    correlated standard normal samples first, then transforms them using the
    inverse CDF (quantile function) of the target marginal distributions.
    If mean/std are distributions, this happens sample-by-sample.
    """
    if norm is None:
         raise ImportError("MultiNormal requires SciPy (`pip install scipy`).")

    num_vars = len(mean)
    if len(std) != num_vars or np.shape(corr) != (num_vars, num_vars):
        raise ValueError("Mismatch in dimensions of mean, std, or corr matrix.")

    n_eff = n or _DEFAULT_SAMPLES

    # Ensure all means/stds are distributions
    mean_dists = [_ensure_dist(m, n=n_eff) for m in mean]
    std_dists = [_ensure_dist(s, n=n_eff) for s in std]

    # --- Gaussian Copula Setup ---
    corr_matrix = np.asarray(corr, float)
    try:
        # Cholesky decomposition of the correlation matrix
        cholesky_L = np.linalg.cholesky(corr_matrix)
    except np.linalg.LinAlgError:
        raise ValueError("Correlation matrix must be positive semi-definite for Cholesky decomposition.")

    # Cache for the underlying standard normal correlated samples
    _std_norm_correlated_cache: np.ndarray | None = None

    def get_std_norm_correlated(k: int) -> np.ndarray:
        """Generates or retrieves cache of k standard normal correlated samples."""
        nonlocal _std_norm_correlated_cache
        if _std_norm_correlated_cache is None or _std_norm_correlated_cache.shape[0] < k:
             # 1. Generate independent standard normal samples
             independent_norm = np.random.standard_normal(size=(k, num_vars))
             # 2. Induce correlation using Cholesky decomposition
             correlated_norm = independent_norm @ cholesky_L.T
             _std_norm_correlated_cache = correlated_norm # Store cache
        # Return the first k samples from cache
        return _std_norm_correlated_cache[:k, :]

    # --- Create Marginal Distributions ---
    marginal_distributions: List[Distribution] = [] # Return type is list of base Distribution
    for i in range(num_vars):
        mean_dist_i = mean_dists[i]
        std_dist_i = std_dists[i]

        def marginal_sampler(k: int, index: int = i):
             # 1. Get correlated standard normal samples (U(0,1) equivalent via CDF)
             std_norm_samples = get_std_norm_correlated(k)[:, index] # Shape (k,)

             # 2. Convert to uniform samples (Probability Integral Transform)
             uniform_samples = norm.cdf(std_norm_samples) # Shape (k,)

             # 3. Get samples for this marginal's mean and std
             mean_samples_i = mean_dist_i._get_samples_resized(k) # Shape (k,)
             std_samples_i = std_dist_i._get_samples_resized(k) # Shape (k,)
             std_samples_i_clamped = np.maximum(std_samples_i, 0) # Ensure non-negative std

             # 4. Transform uniform samples to the target Normal marginal
             #    using the inverse CDF (quantile function, norm.ppf)
             #    with the *sampled* mean and std for this marginal.
             # This is the key step: norm.ppf(uniform, loc=mean, scale=std)
             return norm.ppf(uniform_samples, loc=mean_samples_i, scale=std_samples_i_clamped)

        # Create the final Normal distribution object for this marginal
        # It uses the specific sampler derived from the copula
        # We pass the original parameter distributions for potential introspection
        marginal_dist = Distribution(marginal_sampler, n=n_eff, params={'mean': mean_dist_i, 'std': std_dist_i})
        # Optionally, we could subclass Normal again, but a generic Distribution works.
        # Set class name for better repr - REMOVED TO FIX TypeError
        # marginal_dist.__class__ = Normal # REMOVED
        marginal_distributions.append(marginal_dist)

    return marginal_distributions
# -----------------------------------------------------------------
# Scenario-weighted mixture with stochastic weights
# -----------------------------------------------------------------
from itertools import chain # Keep this import local to where it's used

class Mixture(Distribution):
    """
    Creates a mixture distribution from weighted components.
    Weights and components can be scalars or Distributions.

    Usage: Mixture({weight1: component1, weight2: component2, ...})

    For each Monte Carlo sample:
    1. Samples are drawn from each weight distribution (w_i).
    2. These sampled weights are renormalized to sum to 1.
    3. A component (j) is chosen based on the renormalized weights (P(j) = w_j / sum(w_i)).
    4. A sample is drawn from the chosen component distribution (d_j).
    """
    def __init__(
        self,
        branches: Dict[DistOrNum, DistOrNum],
        *,
        n: int | None = None,
    ):
        if not branches:
            raise ValueError("Mixture requires at least one branch.")

        # Determine the size for the mixture distribution
        # Consider sizes of all input distributions (weights and components)
        all_input_dists = []
        processed_branches = {}
        temp_n = n or _DEFAULT_SAMPLES # Start with default or user N

        for w, d in branches.items():
             w_dist = _ensure_dist(w, n=temp_n) # Tentatively use temp_n
             d_dist = _ensure_dist(d, n=temp_n) # Tentatively use temp_n
             all_input_dists.extend([w_dist, d_dist])
             processed_branches[w_dist] = d_dist # Store dists

        # Use the maximum size found among all input distributions, or the provided n
        n_eff = n or max(d._size for d in all_input_dists) if all_input_dists else _DEFAULT_SAMPLES

        # Now ensure all distributions are using the final effective size n_eff
        # (This might involve creating new Deterministic nodes if needed)
        final_branches = {}
        w_dists_final = []
        comp_dists_final = []
        for w_dist_orig, d_dist_orig in processed_branches.items():
             w_dist = _ensure_dist(w_dist_orig, n=n_eff) # Ensure final size
             d_dist = _ensure_dist(d_dist_orig, n=n_eff) # Ensure final size
             final_branches[w_dist] = d_dist # Keep track for repr maybe?
             w_dists_final.append(w_dist)
             comp_dists_final.append(d_dist)


        # Define the sampler using the finalized distributions
        def sampler(k: int) -> np.ndarray:
            # 1. Get samples for all weights
            #    Use _get_samples_resized which handles cache and size adaptation
            w_samples_k = np.vstack([wd._get_samples_resized(k) for wd in w_dists_final]) # Shape (num_branches, k)

            # 2. Clean-up weights: force non-negative, handle all-zero case, renormalize
            w_samples_k = np.maximum(w_samples_k, 0.0) # Clip negative weights to 0
            col_sum = w_samples_k.sum(axis=0, keepdims=True)
            # Identify columns where sum is close to zero
            zero_cols = np.isclose(col_sum, 0.0).flatten() # Flatten for 1D indexing

            # If a column sum is zero, distribute probability equally among branches for that sample
            if np.any(zero_cols):
                num_branches = w_samples_k.shape[0]
                w_samples_k[:, zero_cols] = 1.0 / num_branches # Assign uniform weight
                col_sum[:, zero_cols] = 1.0 # Update sum to 1 to avoid division by zero

            # Normalize weights (where sum > 0)
            valid_cols = ~zero_cols
            if np.any(valid_cols):
                w_samples_k[:, valid_cols] /= col_sum[:, valid_cols]

            # Weights `w_samples_k` should now be normalized per column (sample)

            # 3. Choose branches based on weights
            #    Generate uniform random numbers for selection
            u = np.random.rand(k)
            # Calculate cumulative probabilities along the branches axis
            cum_probs = np.cumsum(w_samples_k, axis=0) # Shape (num_branches, k)
            # Find the index of the first branch where cumulative prob exceeds u
            # `argmax` finds the first True. If all are False (u > 1, shouldn't happen), it returns 0.
            # Need to handle edge case u near 1 correctly. Add small epsilon?
            # Or simpler: `(u > cum_probs).sum(axis=0)` counts how many ceilings `u` is *above*.
            # Example: u=0.7, cum=[0.2, 0.6, 1.0] -> (T, T, F).sum() = 2. Correct index.
            # Example: u=0.1, cum=[0.2, 0.6, 1.0] -> (F, F, F).sum() = 0. Correct index.
            branch_indices = (u > cum_probs).sum(axis=0) # Shape (k,) -> indices from 0 to num_branches-1

            # 4. Draw samples from the selected components
            out = np.empty(k, dtype=float)
            # Get component samples resized to k *once* per component if needed
            comp_samples_k = [cd._get_samples_resized(k) for cd in comp_dists_final]

            for j, comp_samples_j in enumerate(comp_samples_k):
                mask = (branch_indices == j) # Mask for samples belonging to branch j
                if np.any(mask):
                    # Use the corresponding samples from the j-th component distribution
                    out[mask] = comp_samples_j[mask]
            return out

        # Initialize the base Distribution class
        # Store original branches for repr, but use finalized ones in sampler
        super().__init__(sampler, n=n_eff)
        self._branches_repr = branches # Store original for repr

    def __repr__(self):  # pragma: no cover
        # Try to create a more informative repr for Mixture
        branch_reprs = []
        for w, d in self._branches_repr.items():
             w_repr = f"{w:.3g}" if isinstance(w, Number) else f"Dist(μ={w.mean():.2g})"
             d_repr = f"{d:.3g}" if isinstance(d, Number) else f"Dist(μ={d.mean():.2g})"
             branch_reprs.append(f"{w_repr}: {d_repr}")
        # Limit number of branches shown for brevity
        max_show = 3
        branches_str = ", ".join(branch_reprs[:max_show])
        if len(branch_reprs) > max_show:
             branches_str += ", ..."

        return f"<Mixture branches=[{branches_str}] n={self._size} mean={self.mean():.3g} sd={self.std():.3g}>"


# -----------------------------------------------------------------------------
# Element-wise extrema & conditioning (Thresholds remain scalar Numbers for now)
# -----------------------------------------------------------------------------

def _pairwise(u: DistOrNum, v: DistOrNum, fn: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> Distribution:
    """Helper for element-wise binary functions like max, min."""
    # Determine size based on inputs
    u_dist = _ensure_dist(u) # Uses its own default N or pre-existing N
    v_dist = _ensure_dist(v, n=u_dist._size) # Try to match u's size
    # If u was scalar, v determines size
    n_eff = v_dist._size if isinstance(u, Number) else u_dist._size

    # Ensure both are using the effective size n_eff
    u_d = _ensure_dist(u, n=n_eff)
    v_d = _ensure_dist(v, n=n_eff)

    # Use the _bin_op logic defined in Distribution class
    # Need to access it via one of the Distribution instances
    return u_d._bin_op(v_d, fn)


def maximum(u: DistOrNum, v: DistOrNum) -> Distribution:
    """Element-wise maximum of two distributions or scalars."""
    return _pairwise(u, v, np.maximum)

def minimum(u: DistOrNum, v: DistOrNum) -> Distribution:
    """Element-wise minimum of two distributions or scalars."""
    return _pairwise(u, v, np.minimum)

def clamp_min(threshold: Number, dist: DistOrNum) -> Distribution:
     """Clamps the minimum value of a distribution element-wise."""
     # `threshold` remains a scalar Number here
     return maximum(threshold, dist)

def clamp_max(threshold: Number, dist: DistOrNum) -> Distribution:
     """Clamps the maximum value of a distribution element-wise."""
     # `threshold` remains a scalar Number here
     return minimum(threshold, dist)


def truncate(dist: DistOrNum, *,
             lower: Number | None = None,
             upper: Number | None = None,
             n: int | None = None,
             batch: int = 4096) -> Distribution:
    """
    Truncates a distribution by resampling until samples fall within bounds.

    Args:
        dist: The distribution or scalar to truncate.
        lower: The lower bound (scalar).
        upper: The upper bound (scalar).
        n: Number of samples for the truncated distribution. If None, uses dist's size.
        batch: How many samples to generate per resampling attempt.

    Returns:
        A new Distribution containing only samples within the bounds.

    Note: Bounds `lower` and `upper` remain scalar numbers. Truncating with
          distributional bounds would require a different approach (likely rejection
          sampling comparing element-wise).
    """
    base = _ensure_dist(dist, n=n) # Ensure base is a distribution, propagate n
    n_eff = n or base._size # Final size

    if lower is None and upper is None:
        return base # No truncation needed

    # Validate bounds relative to each other
    if lower is not None and upper is not None and upper < lower:
         raise ValueError(f"Truncation upper bound {upper} cannot be less than lower bound {lower}.")

    def sampler(k: int):
        # Ensure k is at least 1 for logic below
        if k <= 0: return np.array([], dtype=float)

        out_samples = np.empty(k, dtype=float)
        filled_count = 0
        safety_max_iters = 1000 # Prevent infinite loops if bounds are impossible
        iters = 0

        # Access the *original* sampler of the base distribution
        base_sampler = base._sampler

        while filled_count < k and iters < safety_max_iters:
            # Generate a batch of fresh samples from the original distribution
            # Estimate how many more needed, maybe add buffer?
            needed = k - filled_count
            # Request slightly more to improve efficiency? Max batch size?
            request_size = max(batch, needed * 2) # Heuristic: generate more than needed
            fresh = np.asarray(base_sampler(request_size), dtype=float).ravel()

            # Apply filters based on bounds
            mask = np.ones_like(fresh, dtype=bool)
            if lower is not None:
                mask &= (fresh >= lower)
            if upper is not None:
                mask &= (fresh <= upper)

            accepted = fresh[mask]
            num_accepted = len(accepted)

            if num_accepted > 0:
                take = min(num_accepted, k - filled_count) # How many we can actually use
                out_samples[filled_count : filled_count + take] = accepted[:take]
                filled_count += take

            iters += 1

        if filled_count < k:
            # This might happen if bounds are very restrictive or impossible
            # for the base distribution.
            print(f"Warning: Truncation could only generate {filled_count}/{k} samples within bounds "
                   f"({lower=}, {upper=}) after {iters} iterations. "
                   f"Check if bounds are compatible with the base distribution '{base.__class__.__name__}'.")
            # Return the partially filled array (or raise error?)
            return out_samples[:filled_count]

        return out_samples

    # Create the new truncated distribution
    return Distribution(sampler, n=n_eff)
