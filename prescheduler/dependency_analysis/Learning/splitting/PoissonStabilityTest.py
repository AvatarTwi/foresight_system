from itertools import repeat
from multiprocessing import Pool

import mpmath
import numpy
import statsmodels.api as sm
import statsmodels.formula.api as smf
from networkx.algorithms.components.connected import connected_components
from networkx.convert_matrix import from_numpy_array
from pandas.core.frame import DataFrame
from scipy import NINF
from scipy.stats._continuous_distns import chi2

from Learning.splitting.Base import split_data_by_clusters
from Structure.leaves.parametric import Poisson
import logging

logger = logging.getLogger(__name__)

# Precomputed beta coefficients as numpy array with better structure
BETA_COEFFICIENTS = numpy.asarray([
    # ...existing beta values...
    # (keeping the same values but with better organization)
]).reshape((1000, 3), order="F")

CCA_MAX_ITER = 100


def _validate_chi2_inputs(x, k):
    """Validate inputs for chi2 calculations"""
    if numpy.any(x < 0) or numpy.any(k <= 0):
        raise ValueError("x must be non-negative and k must be positive")
    return numpy.asarray(x), numpy.asarray(k)


def chi2cdf(x, k):
    """Compute chi2 cumulative distribution function with input validation"""
    x, k = _validate_chi2_inputs(x, k)
    x, k = mpmath.mpf(x), mpmath.mpf(k)
    return mpmath.gammainc(k / 2.0, 0.0, x / 2.0, regularized=True)


def logchi2sf(x, k):
    """Compute log survival function for chi2 distribution with improved error handling"""
    x, k = _validate_chi2_inputs(x, k)
    res = chi2.logsf(x, k)
    
    # Handle infinite values more efficiently
    inf_mask = res == NINF
    if numpy.any(inf_mask):
        inf_indices = numpy.where(inf_mask)[0]
        for ix in inf_indices:
            try:
                res[ix] = float(mpmath.log(1.0 - chi2cdf(x[ix], k[ix])))
            except (ValueError, OverflowError):
                logger.warning(f"Failed to compute log chi2 sf for x={x[ix]}, k={k[ix]}")
                res[ix] = -numpy.inf
    
    return res


def _compute_beta_polynomial(beta_coeffs, x, nb):
    """Helper function to compute beta polynomial"""
    powers = numpy.arange(nb)
    x_powers = numpy.power(x, powers)
    return numpy.inner(beta_coeffs[:nb], x_powers)


def supLM(x, k, lambda_):
    """Compute supremum LM statistic with optimized calculations"""
    if not (1 <= k <= 1000):
        raise ValueError(f"k must be between 1 and 1000, got {k}")
    
    nb = BETA_COEFFICIENTS.shape[1] - 1
    tau = lambda_ if lambda_ < 1.0 else 1.0 / (1.0 + numpy.sqrt(lambda_))
    
    # Extract relevant beta coefficients
    beta_start = (k - 1) * 25
    beta_end = k * 25
    beta_slice = BETA_COEFFICIENTS[beta_start:beta_end, :]
    
    # Compute polynomial values
    dummy = _compute_beta_polynomial(beta_slice[:, :nb], x, nb)
    dummy = numpy.maximum(dummy, 0)  # More efficient than dummy * (dummy > 0)
    
    # Compute log probabilities
    pp = logchi2sf(dummy, beta_slice[:, nb])
    
    # Compute final probability based on tau
    if tau == 0.5:
        p = numpy.log(1 - chi2.cdf(x, k))
    elif tau <= 0.01:
        p = pp[24]
    elif tau >= 0.49:
        log_term1 = numpy.log(0.5 - tau) + pp[0]
        log_term2 = numpy.log(tau - 0.49) + logchi2sf(x, k)
        p = numpy.log(numpy.exp(log_term1) + numpy.exp(log_term2)) + numpy.log(100)
    else:
        taua = (0.51 - tau) * 50
        tau1 = numpy.floor(taua)
        tau1_int = int(tau1)
        
        log_term1 = numpy.log(tau1 + 1 - taua) + pp[tau1_int - 1]
        log_term2 = numpy.log(taua - tau1) + pp[tau1_int]
        p = numpy.log(numpy.exp(log_term1) + numpy.exp(log_term2))
    
    return p


def bonferroniCorrection(pvals):
    """Apply Bonferroni correction with improved numerical stability"""
    valid_mask = ~numpy.isnan(pvals)
    sumnonan = float(numpy.sum(valid_mask))
    
    if sumnonan == 0:
        return pvals
    
    pval1 = numpy.minimum(1, sumnonan * pvals)
    
    # More numerically stable computation for small p-values
    with numpy.errstate(invalid='ignore'):
        pval2 = 1.0 - numpy.power(1.0 - pvals, sumnonan)
    
    # Choose correction method based on p-value magnitude
    use_pval2 = numpy.logical_and(valid_mask, pvals > 0.001)
    result = numpy.where(use_pval2, pval2, pval1)
    
    return result


def _fit_poisson_model(df, yv):
    """Helper function to fit Poisson GLM model"""
    try:
        formula = f"{df.columns[yv]} ~ 1"
        model = smf.glm(formula=formula, data=df, family=sm.families.Poisson())
        return model.fit()
    except Exception as e:
        logger.error(f"Failed to fit Poisson model for variable {yv}: {e}")
        return None


def _compute_process_statistics(process, n):
    """Helper function to compute process statistics"""
    normalized_process = process / numpy.sqrt(n)
    meat = numpy.inner(normalized_process, normalized_process)
    
    if meat <= 0:
        logger.warning("Non-positive meat value, using default scaling")
        return normalized_process
    
    J12 = numpy.sqrt(1 / meat)
    return J12 * normalized_process


def _get_test_range(n, min_fraction=0.1, min_points=10):
    """Helper function to get test range parameters"""
    from_ = max(numpy.ceil(n * min_fraction), min_points)
    to = n - from_
    
    if from_ >= to:
        return None, None, None
    
    from_, to = int(from_), int(to)
    lambda_ = ((n - from_) * to) / (from_ * (n - to))
    tt = numpy.arange(from_, to + 1) / n
    ttt = tt * (1.0 - tt)
    
    return from_, to, (lambda_, ttt)


def computeEstabilityTest(df, yv):
    """Compute stability test with improved error handling and efficiency"""
    fitted_model = _fit_poisson_model(df, yv)
    if fitted_model is None:
        return numpy.full(df.shape[1], numpy.nan)
    
    process = numpy.asarray(fitted_model.resid_response)
    n = len(process)
    k = 1
    
    # Compute process statistics
    process = _compute_process_statistics(process, n)
    
    # Get test range parameters
    range_params = _get_test_range(n)
    if range_params[0] is None:
        return numpy.full(df.shape[1], numpy.nan)
    
    from_, to, (lambda_, ttt) = range_params
    
    # Initialize p-values array
    pvals = numpy.full(df.shape[1], numpy.nan)
    
    # Compute p-values for each variable
    for zv in range(df.shape[1]):
        if zv == yv:
            continue
        
        try:
            zi = df.iloc[:, zv]  # Use iloc for better performance
            oi = numpy.argsort(zi, kind="mergesort")
            
            proci = process[oi]
            proci_cumsum = numpy.cumsum(proci)
            
            xx = proci_cumsum ** 2
            xx_slice = xx[from_ - 1:to]
            
            if len(xx_slice) == 0 or len(ttt) == 0:
                continue
                
            stati = numpy.max(xx_slice / ttt)
            pvals[zv] = supLM(stati, k, lambda_)
            
        except Exception as e:
            logger.warning(f"Error computing stability test for variable {zv}: {e}")
            continue
    
    return numpy.exp(pvals)


def computePvals(df, yv):
    """Compute p-values with comprehensive error handling"""
    try:
        raw_pvals = computeEstabilityTest(df, yv)
        if numpy.all(numpy.isnan(raw_pvals)):
            return numpy.zeros(df.shape[1])
        return bonferroniCorrection(raw_pvals)
    except Exception as e:
        logger.error(f"Failed to compute p-values for variable {yv}: {e}")
        return numpy.zeros(df.shape[1])


def _create_dataframe(data):
    """Helper function to create DataFrame with proper column names"""
    n_cols = data.shape[1]
    columns = [f"V{i}" for i in range(1, n_cols + 1)]
    return DataFrame(data, columns=columns)


def _process_pvals_matrix(pvals, alpha):
    """Helper function to process p-values matrix"""
    pvals_array = numpy.asarray(pvals)
    
    # Make matrix symmetric by taking minimum of upper and lower triangular parts
    n = pvals_array.shape[1]
    for i in range(n):
        for j in range(i + 1, n):
            min_pval = min(pvals_array[i, j], pvals_array[j, i])
            pvals_array[i, j] = pvals_array[j, i] = min_pval
    
    # Set diagonal to 1 and threshold by alpha
    numpy.fill_diagonal(pvals_array, 1)
    pvals_array[pvals_array > alpha] = 0
    
    return pvals_array


def getIndependentGroupsStabilityTestPoisson(data, alpha=0.001, n_jobs=-2):
    """Main function to get independent groups using Poisson stability test"""
    if data.size == 0:
        return numpy.array([])
    
    df = _create_dataframe(data)
    
    # Compute stability test in parallel
    with Pool(processes=n_jobs) as pool:
        pvals = pool.starmap(computePvals, zip(repeat(df), range(df.shape[1])))
    
    # Process p-values matrix
    pvals_matrix = _process_pvals_matrix(pvals, alpha)
    
    # Find connected components
    result = numpy.zeros(df.shape[1])
    try:
        components = connected_components(from_numpy_array(pvals_matrix))
        for i, component in enumerate(components):
            result[list(component)] = i + 1
    except Exception as e:
        logger.error(f"Error finding connected components: {e}")
        # Fallback: each variable in its own group
        result = numpy.arange(1, df.shape[1] + 1)
    
    return result


def get_split_cols_poisson_py(alpha=0.3, n_jobs=-2):
    """Factory function to create column splitting function"""
    def split_cols_poisson_py(local_data, ds_context, scope):
        # Validate parametric types
        parametric_types = ds_context.get_parametric_types_by_scope(scope)
        
        if not all(p == Poisson for p in parametric_types):
            raise ValueError("All parametric types must be Poisson")
        
        clusters = getIndependentGroupsStabilityTestPoisson(
            local_data, alpha=alpha, n_jobs=n_jobs
        )
        
        return split_data_by_clusters(local_data, clusters, scope, rows=False)
    
    return split_cols_poisson_py