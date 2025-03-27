# API Reference

## `__init__.py`

## `analyzer.py`

### `Analyzer`
```python
No docstring provided.
```

### `IsolationForest`
```python
Isolation Forest Algorithm.

Return the anomaly score of each sample using the IsolationForest algorithm

The IsolationForest 'isolates' observations by randomly selecting a feature
and then randomly selecting a split value between the maximum and minimum
values of the selected feature.

Since recursive partitioning can be represented by a tree structure, the
number of splittings required to isolate a sample is equivalent to the path
length from the root node to the terminating node.

This path length, averaged over a forest of such random trees, is a
measure of normality and our decision function.

Random partitioning produces noticeably shorter paths for anomalies.
Hence, when a forest of random trees collectively produce shorter path
lengths for particular samples, they are highly likely to be anomalies.

Read more in the :ref:`User Guide <isolation_forest>`.

.. versionadded:: 0.18

Parameters
----------
n_estimators : int, default=100
    The number of base estimators in the ensemble.

max_samples : "auto", int or float, default="auto"
    The number of samples to draw from X to train each base estimator.
        - If int, then draw `max_samples` samples.
        - If float, then draw `max_samples * X.shape[0]` samples.
        - If "auto", then `max_samples=min(256, n_samples)`.

    If max_samples is larger than the number of samples provided,
    all samples will be used for all trees (no sampling).

contamination : 'auto' or float, default='auto'
    The amount of contamination of the data set, i.e. the proportion
    of outliers in the data set. Used when fitting to define the threshold
    on the scores of the samples.

        - If 'auto', the threshold is determined as in the
          original paper.
        - If float, the contamination should be in the range (0, 0.5].

    .. versionchanged:: 0.22
       The default value of ``contamination`` changed from 0.1
       to ``'auto'``.

max_features : int or float, default=1.0
    The number of features to draw from X to train each base estimator.

        - If int, then draw `max_features` features.
        - If float, then draw `max(1, int(max_features * n_features_in_))` features.

bootstrap : bool, default=False
    If True, individual trees are fit on random subsets of the training
    data sampled with replacement. If False, sampling without replacement
    is performed.

n_jobs : int, default=None
    The number of jobs to run in parallel for both :meth:`fit` and
    :meth:`predict`. ``None`` means 1 unless in a
    :obj:`joblib.parallel_backend` context. ``-1`` means using all
    processors. See :term:`Glossary <n_jobs>` for more details.

random_state : int, RandomState instance or None, default=None
    Controls the pseudo-randomness of the selection of the feature
    and split values for each branching step and each tree in the forest.

    Pass an int for reproducible results across multiple function calls.
    See :term:`Glossary <random_state>`.

verbose : int, default=0
    Controls the verbosity of the tree building process.

warm_start : bool, default=False
    When set to ``True``, reuse the solution of the previous call to fit
    and add more estimators to the ensemble, otherwise, just fit a whole
    new forest. See :term:`the Glossary <warm_start>`.

    .. versionadded:: 0.21

Attributes
----------
base_estimator_ : ExtraTreeRegressor instance
    The child estimator template used to create the collection of
    fitted sub-estimators.

estimators_ : list of ExtraTreeRegressor instances
    The collection of fitted sub-estimators.

estimators_features_ : list of ndarray
    The subset of drawn features for each base estimator.

estimators_samples_ : list of ndarray
    The subset of drawn samples (i.e., the in-bag samples) for each base
    estimator.

max_samples_ : int
    The actual number of samples.

offset_ : float
    Offset used to define the decision function from the raw scores. We
    have the relation: ``decision_function = score_samples - offset_``.
    ``offset_`` is defined as follows. When the contamination parameter is
    set to "auto", the offset is equal to -0.5 as the scores of inliers are
    close to 0 and the scores of outliers are close to -1. When a
    contamination parameter different than "auto" is provided, the offset
    is defined in such a way we obtain the expected number of outliers
    (samples with decision function < 0) in training.

    .. versionadded:: 0.20

n_features_ : int
    The number of features when ``fit`` is performed.

    .. deprecated:: 1.0
        Attribute `n_features_` was deprecated in version 1.0 and will be
        removed in 1.2. Use `n_features_in_` instead.

n_features_in_ : int
    Number of features seen during :term:`fit`.

    .. versionadded:: 0.24

feature_names_in_ : ndarray of shape (`n_features_in_`,)
    Names of features seen during :term:`fit`. Defined only when `X`
    has feature names that are all strings.

    .. versionadded:: 1.0

See Also
--------
sklearn.covariance.EllipticEnvelope : An object for detecting outliers in a
    Gaussian distributed dataset.
sklearn.svm.OneClassSVM : Unsupervised Outlier Detection.
    Estimate the support of a high-dimensional distribution.
    The implementation is based on libsvm.
sklearn.neighbors.LocalOutlierFactor : Unsupervised Outlier Detection
    using Local Outlier Factor (LOF).

Notes
-----
The implementation is based on an ensemble of ExtraTreeRegressor. The
maximum depth of each tree is set to ``ceil(log_2(n))`` where
:math:`n` is the number of samples used to build the tree
(see (Liu et al., 2008) for more details).

References
----------
.. [1] Liu, Fei Tony, Ting, Kai Ming and Zhou, Zhi-Hua. "Isolation forest."
       Data Mining, 2008. ICDM'08. Eighth IEEE International Conference on.
.. [2] Liu, Fei Tony, Ting, Kai Ming and Zhou, Zhi-Hua. "Isolation-based
       anomaly detection." ACM Transactions on Knowledge Discovery from
       Data (TKDD) 6.1 (2012): 3.

Examples
--------
>>> from sklearn.ensemble import IsolationForest
>>> X = [[-1.1], [0.3], [0.5], [100]]
>>> clf = IsolationForest(random_state=0).fit(X)
>>> clf.predict([[0.1], [0], [90]])
array([ 1,  1, -1])
```

### `SelectKBest`
```python
Select features according to the k highest scores.

Read more in the :ref:`User Guide <univariate_feature_selection>`.

Parameters
----------
score_func : callable, default=f_classif
    Function taking two arrays X and y, and returning a pair of arrays
    (scores, pvalues) or a single array with scores.
    Default is f_classif (see below "See Also"). The default function only
    works with classification tasks.

    .. versionadded:: 0.18

k : int or "all", default=10
    Number of top features to select.
    The "all" option bypasses selection, for use in a parameter search.

Attributes
----------
scores_ : array-like of shape (n_features,)
    Scores of features.

pvalues_ : array-like of shape (n_features,)
    p-values of feature scores, None if `score_func` returned only scores.

n_features_in_ : int
    Number of features seen during :term:`fit`.

    .. versionadded:: 0.24

feature_names_in_ : ndarray of shape (`n_features_in_`,)
    Names of features seen during :term:`fit`. Defined only when `X`
    has feature names that are all strings.

    .. versionadded:: 1.0

See Also
--------
f_classif: ANOVA F-value between label/feature for classification tasks.
mutual_info_classif: Mutual information for a discrete target.
chi2: Chi-squared stats of non-negative features for classification tasks.
f_regression: F-value between label/feature for regression tasks.
mutual_info_regression: Mutual information for a continuous target.
SelectPercentile: Select features based on percentile of the highest
    scores.
SelectFpr : Select features based on a false positive rate test.
SelectFdr : Select features based on an estimated false discovery rate.
SelectFwe : Select features based on family-wise error rate.
GenericUnivariateSelect : Univariate feature selector with configurable
    mode.

Notes
-----
Ties between features with equal scores will be broken in an unspecified
way.

Examples
--------
>>> from sklearn.datasets import load_digits
>>> from sklearn.feature_selection import SelectKBest, chi2
>>> X, y = load_digits(return_X_y=True)
>>> X.shape
(1797, 64)
>>> X_new = SelectKBest(chi2, k=20).fit_transform(X, y)
>>> X_new.shape
(1797, 20)
```

### `f_classif`
```python
Compute the ANOVA F-value for the provided sample.

Read more in the :ref:`User Guide <univariate_feature_selection>`.

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    The set of regressors that will be tested sequentially.

y : ndarray of shape (n_samples,)
    The target vector.

Returns
-------
f_statistic : ndarray of shape (n_features,)
    F-statistic for each feature.

p_values : ndarray of shape (n_features,)
    P-values associated with the F-statistic.

See Also
--------
chi2 : Chi-squared stats of non-negative features for classification tasks.
f_regression : F-value between label/feature for regression tasks.
```

### `mutual_info_classif`
```python
Estimate mutual information for a discrete target variable.

Mutual information (MI) [1]_ between two random variables is a non-negative
value, which measures the dependency between the variables. It is equal
to zero if and only if two random variables are independent, and higher
values mean higher dependency.

The function relies on nonparametric methods based on entropy estimation
from k-nearest neighbors distances as described in [2]_ and [3]_. Both
methods are based on the idea originally proposed in [4]_.

It can be used for univariate features selection, read more in the
:ref:`User Guide <univariate_feature_selection>`.

Parameters
----------
X : array-like or sparse matrix, shape (n_samples, n_features)
    Feature matrix.

y : array-like of shape (n_samples,)
    Target vector.

discrete_features : {'auto', bool, array-like}, default='auto'
    If bool, then determines whether to consider all features discrete
    or continuous. If array, then it should be either a boolean mask
    with shape (n_features,) or array with indices of discrete features.
    If 'auto', it is assigned to False for dense `X` and to True for
    sparse `X`.

n_neighbors : int, default=3
    Number of neighbors to use for MI estimation for continuous variables,
    see [2]_ and [3]_. Higher values reduce variance of the estimation, but
    could introduce a bias.

copy : bool, default=True
    Whether to make a copy of the given data. If set to False, the initial
    data will be overwritten.

random_state : int, RandomState instance or None, default=None
    Determines random number generation for adding small noise to
    continuous variables in order to remove repeated values.
    Pass an int for reproducible results across multiple function calls.
    See :term:`Glossary <random_state>`.

Returns
-------
mi : ndarray, shape (n_features,)
    Estimated mutual information between each feature and the target.

Notes
-----
1. The term "discrete features" is used instead of naming them
   "categorical", because it describes the essence more accurately.
   For example, pixel intensities of an image are discrete features
   (but hardly categorical) and you will get better results if mark them
   as such. Also note, that treating a continuous variable as discrete and
   vice versa will usually give incorrect results, so be attentive about
   that.
2. True mutual information can't be negative. If its estimate turns out
   to be negative, it is replaced by zero.

References
----------
.. [1] `Mutual Information
       <https://en.wikipedia.org/wiki/Mutual_information>`_
       on Wikipedia.
.. [2] A. Kraskov, H. Stogbauer and P. Grassberger, "Estimating mutual
       information". Phys. Rev. E 69, 2004.
.. [3] B. C. Ross "Mutual Information between Discrete and Continuous
       Data Sets". PLoS ONE 9(2), 2014.
.. [4] L. F. Kozachenko, N. N. Leonenko, "Sample Estimate of the Entropy
       of a Random Vector:, Probl. Peredachi Inf., 23:2 (1987), 9-16
```

### `pearsonr`
```python
Pearson correlation coefficient and p-value for testing non-correlation.

The Pearson correlation coefficient [1]_ measures the linear relationship
between two datasets. Like other correlation
coefficients, this one varies between -1 and +1 with 0 implying no
correlation. Correlations of -1 or +1 imply an exact linear relationship.
Positive correlations imply that as x increases, so does y. Negative
correlations imply that as x increases, y decreases.

This function also performs a test of the null hypothesis that the
distributions underlying the samples are uncorrelated and normally
distributed. (See Kowalski [3]_
for a discussion of the effects of non-normality of the input on the
distribution of the correlation coefficient.)
The p-value roughly indicates the probability of an uncorrelated system
producing datasets that have a Pearson correlation at least as extreme
as the one computed from these datasets.

Parameters
----------
x : (N,) array_like
    Input array.
y : (N,) array_like
    Input array.
alternative : {'two-sided', 'greater', 'less'}, optional
    Defines the alternative hypothesis. Default is 'two-sided'.
    The following options are available:

    * 'two-sided': the correlation is nonzero
    * 'less': the correlation is negative (less than zero)
    * 'greater':  the correlation is positive (greater than zero)

    .. versionadded:: 1.9.0

Returns
-------
result : `~scipy.stats._result_classes.PearsonRResult`
    An object with the following attributes:

    statistic : float
        Pearson product-moment correlation coefficient.
    pvalue : float
        The p-value associated with the chosen alternative.

    The object has the following method:

    confidence_interval(confidence_level=0.95)
        This method computes the confidence interval of the correlation
        coefficient `statistic` for the given confidence level.
        The confidence interval is returned in a ``namedtuple`` with
        fields `low` and `high`.  See the Notes for more details.

Warns
-----
`~scipy.stats.ConstantInputWarning`
    Raised if an input is a constant array.  The correlation coefficient
    is not defined in this case, so ``np.nan`` is returned.

`~scipy.stats.NearConstantInputWarning`
    Raised if an input is "nearly" constant.  The array ``x`` is considered
    nearly constant if ``norm(x - mean(x)) < 1e-13 * abs(mean(x))``.
    Numerical errors in the calculation ``x - mean(x)`` in this case might
    result in an inaccurate calculation of r.

See Also
--------
spearmanr : Spearman rank-order correlation coefficient.
kendalltau : Kendall's tau, a correlation measure for ordinal data.

Notes
-----
The correlation coefficient is calculated as follows:

.. math::

    r = \frac{\sum (x - m_x) (y - m_y)}
             {\sqrt{\sum (x - m_x)^2 \sum (y - m_y)^2}}

where :math:`m_x` is the mean of the vector x and :math:`m_y` is
the mean of the vector y.

Under the assumption that x and y are drawn from
independent normal distributions (so the population correlation coefficient
is 0), the probability density function of the sample correlation
coefficient r is ([1]_, [2]_):

.. math::
    f(r) = \frac{{(1-r^2)}^{n/2-2}}{\mathrm{B}(\frac{1}{2},\frac{n}{2}-1)}

where n is the number of samples, and B is the beta function.  This
is sometimes referred to as the exact distribution of r.  This is
the distribution that is used in `pearsonr` to compute the p-value.
The distribution is a beta distribution on the interval [-1, 1],
with equal shape parameters a = b = n/2 - 1.  In terms of SciPy's
implementation of the beta distribution, the distribution of r is::

    dist = scipy.stats.beta(n/2 - 1, n/2 - 1, loc=-1, scale=2)

The default p-value returned by `pearsonr` is a two-sided p-value. For a
given sample with correlation coefficient r, the p-value is
the probability that abs(r') of a random sample x' and y' drawn from
the population with zero correlation would be greater than or equal
to abs(r). In terms of the object ``dist`` shown above, the p-value
for a given r and length n can be computed as::

    p = 2*dist.cdf(-abs(r))

When n is 2, the above continuous distribution is not well-defined.
One can interpret the limit of the beta distribution as the shape
parameters a and b approach a = b = 0 as a discrete distribution with
equal probability masses at r = 1 and r = -1.  More directly, one
can observe that, given the data x = [x1, x2] and y = [y1, y2], and
assuming x1 != x2 and y1 != y2, the only possible values for r are 1
and -1.  Because abs(r') for any sample x' and y' with length 2 will
be 1, the two-sided p-value for a sample of length 2 is always 1.

For backwards compatibility, the object that is returned also behaves
like a tuple of length two that holds the statistic and the p-value.

References
----------
.. [1] "Pearson correlation coefficient", Wikipedia,
       https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
.. [2] Student, "Probable error of a correlation coefficient",
       Biometrika, Volume 6, Issue 2-3, 1 September 1908, pp. 302-310.
.. [3] C. J. Kowalski, "On the Effects of Non-Normality on the Distribution
       of the Sample Product-Moment Correlation Coefficient"
       Journal of the Royal Statistical Society. Series C (Applied
       Statistics), Vol. 21, No. 1 (1972), pp. 1-12.

Examples
--------
>>> from scipy import stats
>>> res = stats.pearsonr([1, 2, 3, 4, 5], [10, 9, 2.5, 6, 4])
>>> res
PearsonRResult(statistic=-0.7426106572325056, pvalue=0.15055580885344558)
>>> res.confidence_interval()
ConfidenceInterval(low=-0.9816918044786463, high=0.40501116769030976)

There is a linear dependence between x and y if y = a + b*x + e, where
a,b are constants and e is a random error term, assumed to be independent
of x. For simplicity, assume that x is standard normal, a=0, b=1 and let
e follow a normal distribution with mean zero and standard deviation s>0.

>>> rng = np.random.default_rng()
>>> s = 0.5
>>> x = stats.norm.rvs(size=500, random_state=rng)
>>> e = stats.norm.rvs(scale=s, size=500, random_state=rng)
>>> y = x + e
>>> stats.pearsonr(x, y).statistic
0.9001942438244763

This should be close to the exact value given by

>>> 1/np.sqrt(1 + s**2)
0.8944271909999159

For s=0.5, we observe a high level of correlation. In general, a large
variance of the noise reduces the correlation, while the correlation
approaches one as the variance of the error goes to zero.

It is important to keep in mind that no correlation does not imply
independence unless (x, y) is jointly normal. Correlation can even be zero
when there is a very simple dependence structure: if X follows a
standard normal distribution, let y = abs(x). Note that the correlation
between x and y is zero. Indeed, since the expectation of x is zero,
cov(x, y) = E[x*y]. By definition, this equals E[x*abs(x)] which is zero
by symmetry. The following lines of code illustrate this observation:

>>> y = np.abs(x)
>>> stats.pearsonr(x, y)
PearsonRResult(statistic=-0.05444919272687482, pvalue=0.22422294836207743)

A non-zero correlation coefficient can be misleading. For example, if X has
a standard normal distribution, define y = x if x < 0 and y = 0 otherwise.
A simple calculation shows that corr(x, y) = sqrt(2/Pi) = 0.797...,
implying a high level of correlation:

>>> y = np.where(x < 0, x, 0)
>>> stats.pearsonr(x, y)
PearsonRResult(statistic=0.861985781588, pvalue=4.813432002751103e-149)

This is unintuitive since there is no dependence of x and y if x is larger
than zero which happens in about half of the cases if we sample x and y.
```

### `ttest_ind`
```python
Calculate the T-test for the means of *two independent* samples of scores.

This is a test for the null hypothesis that 2 independent samples
have identical average (expected) values. This test assumes that the
populations have identical variances by default.

Parameters
----------
a, b : array_like
    The arrays must have the same shape, except in the dimension
    corresponding to `axis` (the first, by default).
axis : int or None, optional
    Axis along which to compute test. If None, compute over the whole
    arrays, `a`, and `b`.
equal_var : bool, optional
    If True (default), perform a standard independent 2 sample test
    that assumes equal population variances [1]_.
    If False, perform Welch's t-test, which does not assume equal
    population variance [2]_.

    .. versionadded:: 0.11.0

nan_policy : {'propagate', 'raise', 'omit'}, optional
    Defines how to handle when input contains nan.
    The following options are available (default is 'propagate'):

      * 'propagate': returns nan
      * 'raise': throws an error
      * 'omit': performs the calculations ignoring nan values

    The 'omit' option is not currently available for permutation tests or
    one-sided asympyotic tests.

permutations : non-negative int, np.inf, or None (default), optional
    If 0 or None (default), use the t-distribution to calculate p-values.
    Otherwise, `permutations` is  the number of random permutations that
    will be used to estimate p-values using a permutation test. If
    `permutations` equals or exceeds the number of distinct partitions of
    the pooled data, an exact test is performed instead (i.e. each
    distinct partition is used exactly once). See Notes for details.

    .. versionadded:: 1.7.0

random_state : {None, int, `numpy.random.Generator`,
        `numpy.random.RandomState`}, optional

    If `seed` is None (or `np.random`), the `numpy.random.RandomState`
    singleton is used.
    If `seed` is an int, a new ``RandomState`` instance is used,
    seeded with `seed`.
    If `seed` is already a ``Generator`` or ``RandomState`` instance then
    that instance is used.

    Pseudorandom number generator state used to generate permutations
    (used only when `permutations` is not None).

    .. versionadded:: 1.7.0

alternative : {'two-sided', 'less', 'greater'}, optional
    Defines the alternative hypothesis.
    The following options are available (default is 'two-sided'):

    * 'two-sided': the means of the distributions underlying the samples
      are unequal.
    * 'less': the mean of the distribution underlying the first sample
      is less than the mean of the distribution underlying the second
      sample.
    * 'greater': the mean of the distribution underlying the first
      sample is greater than the mean of the distribution underlying
      the second sample.

    .. versionadded:: 1.6.0

trim : float, optional
    If nonzero, performs a trimmed (Yuen's) t-test.
    Defines the fraction of elements to be trimmed from each end of the
    input samples. If 0 (default), no elements will be trimmed from either
    side. The number of trimmed elements from each tail is the floor of the
    trim times the number of elements. Valid range is [0, .5).

    .. versionadded:: 1.7

Returns
-------
statistic : float or array
    The calculated t-statistic.
pvalue : float or array
    The p-value.

Notes
-----
Suppose we observe two independent samples, e.g. flower petal lengths, and
we are considering whether the two samples were drawn from the same
population (e.g. the same species of flower or two species with similar
petal characteristics) or two different populations.

The t-test quantifies the difference between the arithmetic means
of the two samples. The p-value quantifies the probability of observing
as or more extreme values assuming the null hypothesis, that the
samples are drawn from populations with the same population means, is true.
A p-value larger than a chosen threshold (e.g. 5% or 1%) indicates that
our observation is not so unlikely to have occurred by chance. Therefore,
we do not reject the null hypothesis of equal population means.
If the p-value is smaller than our threshold, then we have evidence
against the null hypothesis of equal population means.

By default, the p-value is determined by comparing the t-statistic of the
observed data against a theoretical t-distribution.
When ``1 < permutations < binom(n, k)``, where

* ``k`` is the number of observations in `a`,
* ``n`` is the total number of observations in `a` and `b`, and
* ``binom(n, k)`` is the binomial coefficient (``n`` choose ``k``),

the data are pooled (concatenated), randomly assigned to either group `a`
or `b`, and the t-statistic is calculated. This process is performed
repeatedly (`permutation` times), generating a distribution of the
t-statistic under the null hypothesis, and the t-statistic of the observed
data is compared to this distribution to determine the p-value.
Specifically, the p-value reported is the "achieved significance level"
(ASL) as defined in 4.4 of [3]_. Note that there are other ways of
estimating p-values using randomized permutation tests; for other
options, see the more general `permutation_test`.

When ``permutations >= binom(n, k)``, an exact test is performed: the data
are partitioned between the groups in each distinct way exactly once.

The permutation test can be computationally expensive and not necessarily
more accurate than the analytical test, but it does not make strong
assumptions about the shape of the underlying distribution.

Use of trimming is commonly referred to as the trimmed t-test. At times
called Yuen's t-test, this is an extension of Welch's t-test, with the
difference being the use of winsorized means in calculation of the variance
and the trimmed sample size in calculation of the statistic. Trimming is
recommended if the underlying distribution is long-tailed or contaminated
with outliers [4]_.

The statistic is calculated as ``(np.mean(a) - np.mean(b))/se``, where
``se`` is the standard error. Therefore, the statistic will be positive
when the sample mean of `a` is greater than the sample mean of `b` and
negative when the sample mean of `a` is less than the sample mean of
`b`.

References
----------
.. [1] https://en.wikipedia.org/wiki/T-test#Independent_two-sample_t-test

.. [2] https://en.wikipedia.org/wiki/Welch%27s_t-test

.. [3] B. Efron and T. Hastie. Computer Age Statistical Inference. (2016).

.. [4] Yuen, Karen K. "The Two-Sample Trimmed t for Unequal Population
       Variances." Biometrika, vol. 61, no. 1, 1974, pp. 165-170. JSTOR,
       www.jstor.org/stable/2334299. Accessed 30 Mar. 2021.

.. [5] Yuen, Karen K., and W. J. Dixon. "The Approximate Behaviour and
       Performance of the Two-Sample Trimmed t." Biometrika, vol. 60,
       no. 2, 1973, pp. 369-374. JSTOR, www.jstor.org/stable/2334550.
       Accessed 30 Mar. 2021.

Examples
--------
>>> from scipy import stats
>>> rng = np.random.default_rng()

Test with sample with identical means:

>>> rvs1 = stats.norm.rvs(loc=5, scale=10, size=500, random_state=rng)
>>> rvs2 = stats.norm.rvs(loc=5, scale=10, size=500, random_state=rng)
>>> stats.ttest_ind(rvs1, rvs2)
Ttest_indResult(statistic=-0.4390847099199348, pvalue=0.6606952038870015)
>>> stats.ttest_ind(rvs1, rvs2, equal_var=False)
Ttest_indResult(statistic=-0.4390847099199348, pvalue=0.6606952553131064)

`ttest_ind` underestimates p for unequal variances:

>>> rvs3 = stats.norm.rvs(loc=5, scale=20, size=500, random_state=rng)
>>> stats.ttest_ind(rvs1, rvs3)
Ttest_indResult(statistic=-1.6370984482905417, pvalue=0.1019251574705033)
>>> stats.ttest_ind(rvs1, rvs3, equal_var=False)
Ttest_indResult(statistic=-1.637098448290542, pvalue=0.10202110497954867)

When ``n1 != n2``, the equal variance t-statistic is no longer equal to the
unequal variance t-statistic:

>>> rvs4 = stats.norm.rvs(loc=5, scale=20, size=100, random_state=rng)
>>> stats.ttest_ind(rvs1, rvs4)
Ttest_indResult(statistic=-1.9481646859513422, pvalue=0.05186270935842703)
>>> stats.ttest_ind(rvs1, rvs4, equal_var=False)
Ttest_indResult(statistic=-1.3146566100751664, pvalue=0.1913495266513811)

T-test with different means, variance, and n:

>>> rvs5 = stats.norm.rvs(loc=8, scale=20, size=100, random_state=rng)
>>> stats.ttest_ind(rvs1, rvs5)
Ttest_indResult(statistic=-2.8415950600298774, pvalue=0.0046418707568707885)
>>> stats.ttest_ind(rvs1, rvs5, equal_var=False)
Ttest_indResult(statistic=-1.8686598649188084, pvalue=0.06434714193919686)

When performing a permutation test, more permutations typically yields
more accurate results. Use a ``np.random.Generator`` to ensure
reproducibility:

>>> stats.ttest_ind(rvs1, rvs5, permutations=10000,
...                 random_state=rng)
Ttest_indResult(statistic=-2.8415950600298774, pvalue=0.0052994700529947)

Take these two samples, one of which has an extreme tail.

>>> a = (56, 128.6, 12, 123.8, 64.34, 78, 763.3)
>>> b = (1.1, 2.9, 4.2)

Use the `trim` keyword to perform a trimmed (Yuen) t-test. For example,
using 20% trimming, ``trim=.2``, the test will reduce the impact of one
(``np.floor(trim*len(a))``) element from each tail of sample `a`. It will
have no effect on sample `b` because ``np.floor(trim*len(b))`` is 0.

>>> stats.ttest_ind(a, b, trim=.2)
Ttest_indResult(statistic=3.4463884028073513,
                pvalue=0.01369338726499547)
```

## `auto_optimizer.py`

> Error loading `auto_optimizer`: No module named 'optuna'

## `data_handler.py`

### `DataHandler`
```python
No docstring provided.
```

## `imputation_engine.py`

> Error loading `imputation_engine`: No module named 'tensorflow'

## `memory.py`

## `ml_selector.py`

> Error loading `ml_selector`: No module named 'tpot'

## `model_explainer.py`

> Error loading `model_explainer`: /home/sandbox/.local/lib/python3.11/site-packages/nvidia/cublas/lib/libcublas.so.11: failed to map segment from shared object

## `reporter.py`

### `Environment`
```python
The core component of Jinja is the `Environment`.  It contains
important shared variables like configuration, filters, tests,
globals and others.  Instances of this class may be modified if
they are not shared and if no template was loaded so far.
Modifications on environments after the first template was loaded
will lead to surprising effects and undefined behavior.

Here are the possible initialization parameters:

    `block_start_string`
        The string marking the beginning of a block.  Defaults to ``'{%'``.

    `block_end_string`
        The string marking the end of a block.  Defaults to ``'%}'``.

    `variable_start_string`
        The string marking the beginning of a print statement.
        Defaults to ``'{{'``.

    `variable_end_string`
        The string marking the end of a print statement.  Defaults to
        ``'}}'``.

    `comment_start_string`
        The string marking the beginning of a comment.  Defaults to ``'{#'``.

    `comment_end_string`
        The string marking the end of a comment.  Defaults to ``'#}'``.

    `line_statement_prefix`
        If given and a string, this will be used as prefix for line based
        statements.  See also :ref:`line-statements`.

    `line_comment_prefix`
        If given and a string, this will be used as prefix for line based
        comments.  See also :ref:`line-statements`.

        .. versionadded:: 2.2

    `trim_blocks`
        If this is set to ``True`` the first newline after a block is
        removed (block, not variable tag!).  Defaults to `False`.

    `lstrip_blocks`
        If this is set to ``True`` leading spaces and tabs are stripped
        from the start of a line to a block.  Defaults to `False`.

    `newline_sequence`
        The sequence that starts a newline.  Must be one of ``'\r'``,
        ``'\n'`` or ``'\r\n'``.  The default is ``'\n'`` which is a
        useful default for Linux and OS X systems as well as web
        applications.

    `keep_trailing_newline`
        Preserve the trailing newline when rendering templates.
        The default is ``False``, which causes a single newline,
        if present, to be stripped from the end of the template.

        .. versionadded:: 2.7

    `extensions`
        List of Jinja extensions to use.  This can either be import paths
        as strings or extension classes.  For more information have a
        look at :ref:`the extensions documentation <jinja-extensions>`.

    `optimized`
        should the optimizer be enabled?  Default is ``True``.

    `undefined`
        :class:`Undefined` or a subclass of it that is used to represent
        undefined values in the template.

    `finalize`
        A callable that can be used to process the result of a variable
        expression before it is output.  For example one can convert
        ``None`` implicitly into an empty string here.

    `autoescape`
        If set to ``True`` the XML/HTML autoescaping feature is enabled by
        default.  For more details about autoescaping see
        :class:`~markupsafe.Markup`.  As of Jinja 2.4 this can also
        be a callable that is passed the template name and has to
        return ``True`` or ``False`` depending on autoescape should be
        enabled by default.

        .. versionchanged:: 2.4
           `autoescape` can now be a function

    `loader`
        The template loader for this environment.

    `cache_size`
        The size of the cache.  Per default this is ``400`` which means
        that if more than 400 templates are loaded the loader will clean
        out the least recently used template.  If the cache size is set to
        ``0`` templates are recompiled all the time, if the cache size is
        ``-1`` the cache will not be cleaned.

        .. versionchanged:: 2.8
           The cache size was increased to 400 from a low 50.

    `auto_reload`
        Some loaders load templates from locations where the template
        sources may change (ie: file system or database).  If
        ``auto_reload`` is set to ``True`` (default) every time a template is
        requested the loader checks if the source changed and if yes, it
        will reload the template.  For higher performance it's possible to
        disable that.

    `bytecode_cache`
        If set to a bytecode cache object, this object will provide a
        cache for the internal Jinja bytecode so that templates don't
        have to be parsed if they were not changed.

        See :ref:`bytecode-cache` for more information.

    `enable_async`
        If set to true this enables async template execution which
        allows using async functions and generators.
```

### `FileSystemLoader`
```python
Load templates from a directory in the file system.

The path can be relative or absolute. Relative paths are relative to
the current working directory.

.. code-block:: python

    loader = FileSystemLoader("templates")

A list of paths can be given. The directories will be searched in
order, stopping at the first matching template.

.. code-block:: python

    loader = FileSystemLoader(["/override/templates", "/default/templates"])

:param searchpath: A path, or list of paths, to the directory that
    contains the templates.
:param encoding: Use this encoding to read the text from template
    files.
:param followlinks: Follow symbolic links in the path.

.. versionchanged:: 2.8
    Added the ``followlinks`` parameter.
```

### `Reporter`
```python
No docstring provided.
```

### `classification_report`
```python
Build a text report showing the main classification metrics.

Read more in the :ref:`User Guide <classification_report>`.

Parameters
----------
y_true : 1d array-like, or label indicator array / sparse matrix
    Ground truth (correct) target values.

y_pred : 1d array-like, or label indicator array / sparse matrix
    Estimated targets as returned by a classifier.

labels : array-like of shape (n_labels,), default=None
    Optional list of label indices to include in the report.

target_names : list of str of shape (n_labels,), default=None
    Optional display names matching the labels (same order).

sample_weight : array-like of shape (n_samples,), default=None
    Sample weights.

digits : int, default=2
    Number of digits for formatting output floating point values.
    When ``output_dict`` is ``True``, this will be ignored and the
    returned values will not be rounded.

output_dict : bool, default=False
    If True, return output as dict.

    .. versionadded:: 0.20

zero_division : "warn", 0 or 1, default="warn"
    Sets the value to return when there is a zero division. If set to
    "warn", this acts as 0, but warnings are also raised.

Returns
-------
report : str or dict
    Text summary of the precision, recall, F1 score for each class.
    Dictionary returned if output_dict is True. Dictionary has the
    following structure::

        {'label 1': {'precision':0.5,
                     'recall':1.0,
                     'f1-score':0.67,
                     'support':1},
         'label 2': { ... },
          ...
        }

    The reported averages include macro average (averaging the unweighted
    mean per label), weighted average (averaging the support-weighted mean
    per label), and sample average (only for multilabel classification).
    Micro average (averaging the total true positives, false negatives and
    false positives) is only shown for multi-label or multi-class
    with a subset of classes, because it corresponds to accuracy
    otherwise and would be the same for all metrics.
    See also :func:`precision_recall_fscore_support` for more details
    on averages.

    Note that in binary classification, recall of the positive class
    is also known as "sensitivity"; recall of the negative class is
    "specificity".

See Also
--------
precision_recall_fscore_support: Compute precision, recall, F-measure and
    support for each class.
confusion_matrix: Compute confusion matrix to evaluate the accuracy of a
    classification.
multilabel_confusion_matrix: Compute a confusion matrix for each class or sample.

Examples
--------
>>> from sklearn.metrics import classification_report
>>> y_true = [0, 1, 2, 2, 2]
>>> y_pred = [0, 0, 2, 2, 1]
>>> target_names = ['class 0', 'class 1', 'class 2']
>>> print(classification_report(y_true, y_pred, target_names=target_names))
              precision    recall  f1-score   support
<BLANKLINE>
     class 0       0.50      1.00      0.67         1
     class 1       0.00      0.00      0.00         1
     class 2       1.00      0.67      0.80         3
<BLANKLINE>
    accuracy                           0.60         5
   macro avg       0.50      0.56      0.49         5
weighted avg       0.70      0.60      0.61         5
<BLANKLINE>
>>> y_pred = [1, 1, 0]
>>> y_true = [1, 1, 1]
>>> print(classification_report(y_true, y_pred, labels=[1, 2, 3]))
              precision    recall  f1-score   support
<BLANKLINE>
           1       1.00      0.67      0.80         3
           2       0.00      0.00      0.00         0
           3       0.00      0.00      0.00         0
<BLANKLINE>
   micro avg       1.00      0.67      0.80         3
   macro avg       0.33      0.22      0.27         3
weighted avg       1.00      0.67      0.80         3
<BLANKLINE>
```

### `confusion_matrix`
```python
Compute confusion matrix to evaluate the accuracy of a classification.

By definition a confusion matrix :math:`C` is such that :math:`C_{i, j}`
is equal to the number of observations known to be in group :math:`i` and
predicted to be in group :math:`j`.

Thus in binary classification, the count of true negatives is
:math:`C_{0,0}`, false negatives is :math:`C_{1,0}`, true positives is
:math:`C_{1,1}` and false positives is :math:`C_{0,1}`.

Read more in the :ref:`User Guide <confusion_matrix>`.

Parameters
----------
y_true : array-like of shape (n_samples,)
    Ground truth (correct) target values.

y_pred : array-like of shape (n_samples,)
    Estimated targets as returned by a classifier.

labels : array-like of shape (n_classes), default=None
    List of labels to index the matrix. This may be used to reorder
    or select a subset of labels.
    If ``None`` is given, those that appear at least once
    in ``y_true`` or ``y_pred`` are used in sorted order.

sample_weight : array-like of shape (n_samples,), default=None
    Sample weights.

    .. versionadded:: 0.18

normalize : {'true', 'pred', 'all'}, default=None
    Normalizes confusion matrix over the true (rows), predicted (columns)
    conditions or all the population. If None, confusion matrix will not be
    normalized.

Returns
-------
C : ndarray of shape (n_classes, n_classes)
    Confusion matrix whose i-th row and j-th
    column entry indicates the number of
    samples with true label being i-th class
    and predicted label being j-th class.

See Also
--------
ConfusionMatrixDisplay.from_estimator : Plot the confusion matrix
    given an estimator, the data, and the label.
ConfusionMatrixDisplay.from_predictions : Plot the confusion matrix
    given the true and predicted labels.
ConfusionMatrixDisplay : Confusion Matrix visualization.

References
----------
.. [1] `Wikipedia entry for the Confusion matrix
       <https://en.wikipedia.org/wiki/Confusion_matrix>`_
       (Wikipedia and other references may use a different
       convention for axes).

Examples
--------
>>> from sklearn.metrics import confusion_matrix
>>> y_true = [2, 0, 2, 2, 0, 1]
>>> y_pred = [0, 0, 2, 2, 0, 2]
>>> confusion_matrix(y_true, y_pred)
array([[2, 0, 0],
       [0, 0, 1],
       [1, 0, 2]])

>>> y_true = ["cat", "ant", "cat", "cat", "ant", "bird"]
>>> y_pred = ["ant", "ant", "cat", "cat", "ant", "cat"]
>>> confusion_matrix(y_true, y_pred, labels=["ant", "bird", "cat"])
array([[2, 0, 0],
       [0, 0, 1],
       [1, 0, 2]])

In the binary case, we can extract true positives, etc as follows:

>>> tn, fp, fn, tp = confusion_matrix([0, 1, 0, 1], [1, 1, 1, 0]).ravel()
>>> (tn, fp, fn, tp)
(0, 2, 1, 1)
```

## `visualizer.py`

### `PCA`
```python
Principal component analysis (PCA).

Linear dimensionality reduction using Singular Value Decomposition of the
data to project it to a lower dimensional space. The input data is centered
but not scaled for each feature before applying the SVD.

It uses the LAPACK implementation of the full SVD or a randomized truncated
SVD by the method of Halko et al. 2009, depending on the shape of the input
data and the number of components to extract.

It can also use the scipy.sparse.linalg ARPACK implementation of the
truncated SVD.

Notice that this class does not support sparse input. See
:class:`TruncatedSVD` for an alternative with sparse data.

Read more in the :ref:`User Guide <PCA>`.

Parameters
----------
n_components : int, float or 'mle', default=None
    Number of components to keep.
    if n_components is not set all components are kept::

        n_components == min(n_samples, n_features)

    If ``n_components == 'mle'`` and ``svd_solver == 'full'``, Minka's
    MLE is used to guess the dimension. Use of ``n_components == 'mle'``
    will interpret ``svd_solver == 'auto'`` as ``svd_solver == 'full'``.

    If ``0 < n_components < 1`` and ``svd_solver == 'full'``, select the
    number of components such that the amount of variance that needs to be
    explained is greater than the percentage specified by n_components.

    If ``svd_solver == 'arpack'``, the number of components must be
    strictly less than the minimum of n_features and n_samples.

    Hence, the None case results in::

        n_components == min(n_samples, n_features) - 1

copy : bool, default=True
    If False, data passed to fit are overwritten and running
    fit(X).transform(X) will not yield the expected results,
    use fit_transform(X) instead.

whiten : bool, default=False
    When True (False by default) the `components_` vectors are multiplied
    by the square root of n_samples and then divided by the singular values
    to ensure uncorrelated outputs with unit component-wise variances.

    Whitening will remove some information from the transformed signal
    (the relative variance scales of the components) but can sometime
    improve the predictive accuracy of the downstream estimators by
    making their data respect some hard-wired assumptions.

svd_solver : {'auto', 'full', 'arpack', 'randomized'}, default='auto'
    If auto :
        The solver is selected by a default policy based on `X.shape` and
        `n_components`: if the input data is larger than 500x500 and the
        number of components to extract is lower than 80% of the smallest
        dimension of the data, then the more efficient 'randomized'
        method is enabled. Otherwise the exact full SVD is computed and
        optionally truncated afterwards.
    If full :
        run exact full SVD calling the standard LAPACK solver via
        `scipy.linalg.svd` and select the components by postprocessing
    If arpack :
        run SVD truncated to n_components calling ARPACK solver via
        `scipy.sparse.linalg.svds`. It requires strictly
        0 < n_components < min(X.shape)
    If randomized :
        run randomized SVD by the method of Halko et al.

    .. versionadded:: 0.18.0

tol : float, default=0.0
    Tolerance for singular values computed by svd_solver == 'arpack'.
    Must be of range [0.0, infinity).

    .. versionadded:: 0.18.0

iterated_power : int or 'auto', default='auto'
    Number of iterations for the power method computed by
    svd_solver == 'randomized'.
    Must be of range [0, infinity).

    .. versionadded:: 0.18.0

n_oversamples : int, default=10
    This parameter is only relevant when `svd_solver="randomized"`.
    It corresponds to the additional number of random vectors to sample the
    range of `X` so as to ensure proper conditioning. See
    :func:`~sklearn.utils.extmath.randomized_svd` for more details.

    .. versionadded:: 1.1

power_iteration_normalizer : {'auto', 'QR', 'LU', 'none'}, default='auto'
    Power iteration normalizer for randomized SVD solver.
    Not used by ARPACK. See :func:`~sklearn.utils.extmath.randomized_svd`
    for more details.

    .. versionadded:: 1.1

random_state : int, RandomState instance or None, default=None
    Used when the 'arpack' or 'randomized' solvers are used. Pass an int
    for reproducible results across multiple function calls.
    See :term:`Glossary <random_state>`.

    .. versionadded:: 0.18.0

Attributes
----------
components_ : ndarray of shape (n_components, n_features)
    Principal axes in feature space, representing the directions of
    maximum variance in the data. Equivalently, the right singular
    vectors of the centered input data, parallel to its eigenvectors.
    The components are sorted by ``explained_variance_``.

explained_variance_ : ndarray of shape (n_components,)
    The amount of variance explained by each of the selected components.
    The variance estimation uses `n_samples - 1` degrees of freedom.

    Equal to n_components largest eigenvalues
    of the covariance matrix of X.

    .. versionadded:: 0.18

explained_variance_ratio_ : ndarray of shape (n_components,)
    Percentage of variance explained by each of the selected components.

    If ``n_components`` is not set then all components are stored and the
    sum of the ratios is equal to 1.0.

singular_values_ : ndarray of shape (n_components,)
    The singular values corresponding to each of the selected components.
    The singular values are equal to the 2-norms of the ``n_components``
    variables in the lower-dimensional space.

    .. versionadded:: 0.19

mean_ : ndarray of shape (n_features,)
    Per-feature empirical mean, estimated from the training set.

    Equal to `X.mean(axis=0)`.

n_components_ : int
    The estimated number of components. When n_components is set
    to 'mle' or a number between 0 and 1 (with svd_solver == 'full') this
    number is estimated from input data. Otherwise it equals the parameter
    n_components, or the lesser value of n_features and n_samples
    if n_components is None.

n_features_ : int
    Number of features in the training data.

n_samples_ : int
    Number of samples in the training data.

noise_variance_ : float
    The estimated noise covariance following the Probabilistic PCA model
    from Tipping and Bishop 1999. See "Pattern Recognition and
    Machine Learning" by C. Bishop, 12.2.1 p. 574 or
    http://www.miketipping.com/papers/met-mppca.pdf. It is required to
    compute the estimated data covariance and score samples.

    Equal to the average of (min(n_features, n_samples) - n_components)
    smallest eigenvalues of the covariance matrix of X.

n_features_in_ : int
    Number of features seen during :term:`fit`.

    .. versionadded:: 0.24

feature_names_in_ : ndarray of shape (`n_features_in_`,)
    Names of features seen during :term:`fit`. Defined only when `X`
    has feature names that are all strings.

    .. versionadded:: 1.0

See Also
--------
KernelPCA : Kernel Principal Component Analysis.
SparsePCA : Sparse Principal Component Analysis.
TruncatedSVD : Dimensionality reduction using truncated SVD.
IncrementalPCA : Incremental Principal Component Analysis.

References
----------
For n_components == 'mle', this class uses the method from:
`Minka, T. P.. "Automatic choice of dimensionality for PCA".
In NIPS, pp. 598-604 <https://tminka.github.io/papers/pca/minka-pca.pdf>`_

Implements the probabilistic PCA model from:
`Tipping, M. E., and Bishop, C. M. (1999). "Probabilistic principal
component analysis". Journal of the Royal Statistical Society:
Series B (Statistical Methodology), 61(3), 611-622.
<http://www.miketipping.com/papers/met-mppca.pdf>`_
via the score and score_samples methods.

For svd_solver == 'arpack', refer to `scipy.sparse.linalg.svds`.

For svd_solver == 'randomized', see:
:doi:`Halko, N., Martinsson, P. G., and Tropp, J. A. (2011).
"Finding structure with randomness: Probabilistic algorithms for
constructing approximate matrix decompositions".
SIAM review, 53(2), 217-288.
<10.1137/090771806>`
and also
:doi:`Martinsson, P. G., Rokhlin, V., and Tygert, M. (2011).
"A randomized algorithm for the decomposition of matrices".
Applied and Computational Harmonic Analysis, 30(1), 47-68.
<10.1016/j.acha.2010.02.003>`

Examples
--------
>>> import numpy as np
>>> from sklearn.decomposition import PCA
>>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
>>> pca = PCA(n_components=2)
>>> pca.fit(X)
PCA(n_components=2)
>>> print(pca.explained_variance_ratio_)
[0.9924... 0.0075...]
>>> print(pca.singular_values_)
[6.30061... 0.54980...]

>>> pca = PCA(n_components=2, svd_solver='full')
>>> pca.fit(X)
PCA(n_components=2, svd_solver='full')
>>> print(pca.explained_variance_ratio_)
[0.9924... 0.00755...]
>>> print(pca.singular_values_)
[6.30061... 0.54980...]

>>> pca = PCA(n_components=1, svd_solver='arpack')
>>> pca.fit(X)
PCA(n_components=1, svd_solver='arpack')
>>> print(pca.explained_variance_ratio_)
[0.99244...]
>>> print(pca.singular_values_)
[6.30061...]
```

### `TSNE`
```python
T-distributed Stochastic Neighbor Embedding.

t-SNE [1] is a tool to visualize high-dimensional data. It converts
similarities between data points to joint probabilities and tries
to minimize the Kullback-Leibler divergence between the joint
probabilities of the low-dimensional embedding and the
high-dimensional data. t-SNE has a cost function that is not convex,
i.e. with different initializations we can get different results.

It is highly recommended to use another dimensionality reduction
method (e.g. PCA for dense data or TruncatedSVD for sparse data)
to reduce the number of dimensions to a reasonable amount (e.g. 50)
if the number of features is very high. This will suppress some
noise and speed up the computation of pairwise distances between
samples. For more tips see Laurens van der Maaten's FAQ [2].

Read more in the :ref:`User Guide <t_sne>`.

Parameters
----------
n_components : int, default=2
    Dimension of the embedded space.

perplexity : float, default=30.0
    The perplexity is related to the number of nearest neighbors that
    is used in other manifold learning algorithms. Larger datasets
    usually require a larger perplexity. Consider selecting a value
    between 5 and 50. Different values can result in significantly
    different results. The perplexity must be less that the number
    of samples.

early_exaggeration : float, default=12.0
    Controls how tight natural clusters in the original space are in
    the embedded space and how much space will be between them. For
    larger values, the space between natural clusters will be larger
    in the embedded space. Again, the choice of this parameter is not
    very critical. If the cost function increases during initial
    optimization, the early exaggeration factor or the learning rate
    might be too high.

learning_rate : float or 'auto', default=200.0
    The learning rate for t-SNE is usually in the range [10.0, 1000.0]. If
    the learning rate is too high, the data may look like a 'ball' with any
    point approximately equidistant from its nearest neighbours. If the
    learning rate is too low, most points may look compressed in a dense
    cloud with few outliers. If the cost function gets stuck in a bad local
    minimum increasing the learning rate may help.
    Note that many other t-SNE implementations (bhtsne, FIt-SNE, openTSNE,
    etc.) use a definition of learning_rate that is 4 times smaller than
    ours. So our learning_rate=200 corresponds to learning_rate=800 in
    those other implementations. The 'auto' option sets the learning_rate
    to `max(N / early_exaggeration / 4, 50)` where N is the sample size,
    following [4] and [5]. This will become default in 1.2.

n_iter : int, default=1000
    Maximum number of iterations for the optimization. Should be at
    least 250.

n_iter_without_progress : int, default=300
    Maximum number of iterations without progress before we abort the
    optimization, used after 250 initial iterations with early
    exaggeration. Note that progress is only checked every 50 iterations so
    this value is rounded to the next multiple of 50.

    .. versionadded:: 0.17
       parameter *n_iter_without_progress* to control stopping criteria.

min_grad_norm : float, default=1e-7
    If the gradient norm is below this threshold, the optimization will
    be stopped.

metric : str or callable, default='euclidean'
    The metric to use when calculating distance between instances in a
    feature array. If metric is a string, it must be one of the options
    allowed by scipy.spatial.distance.pdist for its metric parameter, or
    a metric listed in pairwise.PAIRWISE_DISTANCE_FUNCTIONS.
    If metric is "precomputed", X is assumed to be a distance matrix.
    Alternatively, if metric is a callable function, it is called on each
    pair of instances (rows) and the resulting value recorded. The callable
    should take two arrays from X as input and return a value indicating
    the distance between them. The default is "euclidean" which is
    interpreted as squared euclidean distance.

metric_params : dict, default=None
    Additional keyword arguments for the metric function.

    .. versionadded:: 1.1

init : {'random', 'pca'} or ndarray of shape (n_samples, n_components),             default='random'
    Initialization of embedding. Possible options are 'random', 'pca',
    and a numpy array of shape (n_samples, n_components).
    PCA initialization cannot be used with precomputed distances and is
    usually more globally stable than random initialization. `init='pca'`
    will become default in 1.2.

verbose : int, default=0
    Verbosity level.

random_state : int, RandomState instance or None, default=None
    Determines the random number generator. Pass an int for reproducible
    results across multiple function calls. Note that different
    initializations might result in different local minima of the cost
    function. See :term:`Glossary <random_state>`.

method : str, default='barnes_hut'
    By default the gradient calculation algorithm uses Barnes-Hut
    approximation running in O(NlogN) time. method='exact'
    will run on the slower, but exact, algorithm in O(N^2) time. The
    exact algorithm should be used when nearest-neighbor errors need
    to be better than 3%. However, the exact method cannot scale to
    millions of examples.

    .. versionadded:: 0.17
       Approximate optimization *method* via the Barnes-Hut.

angle : float, default=0.5
    Only used if method='barnes_hut'
    This is the trade-off between speed and accuracy for Barnes-Hut T-SNE.
    'angle' is the angular size (referred to as theta in [3]) of a distant
    node as measured from a point. If this size is below 'angle' then it is
    used as a summary node of all points contained within it.
    This method is not very sensitive to changes in this parameter
    in the range of 0.2 - 0.8. Angle less than 0.2 has quickly increasing
    computation time and angle greater 0.8 has quickly increasing error.

n_jobs : int, default=None
    The number of parallel jobs to run for neighbors search. This parameter
    has no impact when ``metric="precomputed"`` or
    (``metric="euclidean"`` and ``method="exact"``).
    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.

    .. versionadded:: 0.22

square_distances : True, default='deprecated'
    This parameter has no effect since distance values are always squared
    since 1.1.

    .. deprecated:: 1.1
         `square_distances` has no effect from 1.1 and will be removed in
         1.3.

Attributes
----------
embedding_ : array-like of shape (n_samples, n_components)
    Stores the embedding vectors.

kl_divergence_ : float
    Kullback-Leibler divergence after optimization.

n_features_in_ : int
    Number of features seen during :term:`fit`.

    .. versionadded:: 0.24

feature_names_in_ : ndarray of shape (`n_features_in_`,)
    Names of features seen during :term:`fit`. Defined only when `X`
    has feature names that are all strings.

    .. versionadded:: 1.0

n_iter_ : int
    Number of iterations run.

See Also
--------
sklearn.decomposition.PCA : Principal component analysis that is a linear
    dimensionality reduction method.
sklearn.decomposition.KernelPCA : Non-linear dimensionality reduction using
    kernels and PCA.
MDS : Manifold learning using multidimensional scaling.
Isomap : Manifold learning based on Isometric Mapping.
LocallyLinearEmbedding : Manifold learning using Locally Linear Embedding.
SpectralEmbedding : Spectral embedding for non-linear dimensionality.

References
----------

[1] van der Maaten, L.J.P.; Hinton, G.E. Visualizing High-Dimensional Data
    Using t-SNE. Journal of Machine Learning Research 9:2579-2605, 2008.

[2] van der Maaten, L.J.P. t-Distributed Stochastic Neighbor Embedding
    https://lvdmaaten.github.io/tsne/

[3] L.J.P. van der Maaten. Accelerating t-SNE using Tree-Based Algorithms.
    Journal of Machine Learning Research 15(Oct):3221-3245, 2014.
    https://lvdmaaten.github.io/publications/papers/JMLR_2014.pdf

[4] Belkina, A. C., Ciccolella, C. O., Anno, R., Halpert, R., Spidlen, J.,
    & Snyder-Cappione, J. E. (2019). Automated optimized parameters for
    T-distributed stochastic neighbor embedding improve visualization
    and analysis of large datasets. Nature Communications, 10(1), 1-12.

[5] Kobak, D., & Berens, P. (2019). The art of using t-SNE for single-cell
    transcriptomics. Nature Communications, 10(1), 1-14.

Examples
--------
>>> import numpy as np
>>> from sklearn.manifold import TSNE
>>> X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
>>> X_embedded = TSNE(n_components=2, learning_rate='auto',
...                   init='random', perplexity=3).fit_transform(X)
>>> X_embedded.shape
(4, 2)
```

### `Visualizer`
```python
No docstring provided.
```

