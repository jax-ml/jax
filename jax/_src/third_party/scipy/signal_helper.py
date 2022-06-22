"""Utility functions adopted from scipy.signal."""

import scipy.signal as osp_signal
import warnings

from jax._src.numpy import lax_numpy as jnp


def _triage_segments(window, nperseg, input_length, dtype):
  """
  Parses window and nperseg arguments for spectrogram and _spectral_helper.
  This is a helper function, not meant to be called externally.

  Args:
    window : string, tuple, or ndarray
      If window is specified by a string or tuple and nperseg is not
      specified, nperseg is set to the default of 256 and returns a window of
      that length.
      If instead the window is array_like and nperseg is not specified, then
      nperseg is set to the length of the window. A ValueError is raised if
      the user supplies both an array_like window and a value for nperseg but
      nperseg does not equal the length of the window.
    nperseg : int
      Length of each segment
    input_length: int
      Length of input signal, i.e. x.shape[-1]. Used to test for errors.
    dtype: dtype for window if specified as a string or tuple. Not referenced
      if window is an array.

  Returns:
    win : ndarray
      window. If function was called with string or tuple than this will hold
      the actual array used as a window.
    nperseg : int
      Length of each segment. If window is str or tuple, nperseg is set to
      256. If window is array_like, nperseg is set to the length of the window.
  """
  if isinstance(window, (str, tuple)):
    if nperseg is None:
      nperseg = 256
    if nperseg > input_length:
      warnings.warn(f'nperseg = {nperseg} is greater than input length '
                    f' = {input_length}, using nperseg = {nperseg}')
      nperseg = input_length
    win = jnp.array(osp_signal.get_window(window, nperseg), dtype=dtype)
  else:
    win = jnp.asarray(window)
    if win.ndim != 1:
      raise ValueError('window must be 1-D')
    if input_length < win.size:
      raise ValueError('window is longer than input signal')
    if nperseg is None:
      nperseg = win.size
    elif nperseg != win.size:
      raise ValueError("value specified for nperseg is different from length of window")
  return win, nperseg


def _median_bias(n):
  """
  Returns the bias of the median of a set of periodograms relative to
  the mean. See Appendix B from [1]_ for details.

  Args:
   n : int
      Numbers of periodograms being averaged.

  Returns:
    bias : float
      Calculated bias.

  References:
  .. [1] B. Allen, W.G. Anderson, P.R. Brady, D.A. Brown, J.D.E. Creighton.
          "FINDCHIRP: an algorithm for detection of gravitational waves from
          inspiraling compact binaries", Physical Review D 85, 2012,
          :arxiv:`gr-qc/0509116`
  """
  ii_2 = jnp.arange(2., n, 2)
  return 1 + jnp.sum(1. / (ii_2 + 1) - 1. / ii_2)
