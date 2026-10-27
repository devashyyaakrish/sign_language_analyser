"""
hmm.py — Hidden Markov Model with Viterbi decoding for temporal smoothing
of ASL sign predictions.

The 29 ASL classes are treated as hidden states. The ANN's softmax output
at each frame provides the emission probabilities. A "sticky" transition
matrix (high self-transition probability) penalises rapid flickering between
classes, producing smoother and more accurate letter commits.

Covers: HMM, transition matrices, forward algorithm (with log-scaling),
Viterbi decoding, sum-product on a chain.
"""

import numpy as np
from typing import Optional


class ASLHiddenMarkovModel:
    """
    Sliding-window HMM decoder for real-time ASL recognition.
    
    Parameters
    ----------
    num_classes : int
        Number of hidden states (ASL sign classes).
    self_transition_prob : float
        Probability of staying in the same state (stickiness).
        The remaining probability is spread uniformly across other states.
    window_size : int
        Number of recent frames to keep in the sliding window for Viterbi.
    """

    def __init__(
        self,
        num_classes: int = 29,
        self_transition_prob: float = 0.85,
        window_size: int = 10,
    ):
        self.num_classes = num_classes
        self.window_size = window_size

        # --- Transition matrix A (num_classes × num_classes) ---
        # A[i, j] = P(state_t = j | state_{t-1} = i)
        # "Sticky" diagonal: high probability of staying in the same state.
        off_diag = (1.0 - self_transition_prob) / (num_classes - 1)
        self.log_A = np.full(
            (num_classes, num_classes),
            np.log(off_diag + 1e-12),
            dtype=np.float64,
        )
        np.fill_diagonal(self.log_A, np.log(self_transition_prob))

        # --- Initial state distribution (uniform) ---
        self.log_pi = np.full(num_classes, np.log(1.0 / num_classes), dtype=np.float64)

        # --- Sliding window of emission log-probabilities ---
        self._emission_buffer: list[np.ndarray] = []

        # --- Cache of the last decoded state ---
        self._last_state: Optional[int] = None

    def reset(self):
        """Clear the observation buffer and cached state."""
        self._emission_buffer.clear()
        self._last_state = None

    def observe(self, softmax_probs: np.ndarray) -> int:
        """
        Feed one frame's softmax output and return the Viterbi-decoded
        best state for the current time step.

        Parameters
        ----------
        softmax_probs : np.ndarray, shape (num_classes,)
            The ANN's softmax probability vector for this frame.

        Returns
        -------
        int
            Index of the most likely hidden state at the current frame.
        """
        # Clamp to avoid log(0)
        probs = np.clip(softmax_probs.astype(np.float64), 1e-12, 1.0)
        log_emission = np.log(probs)

        self._emission_buffer.append(log_emission)

        # Trim to sliding window
        if len(self._emission_buffer) > self.window_size:
            self._emission_buffer = self._emission_buffer[-self.window_size:]

        # Run Viterbi over the current window
        best_state = self._viterbi()
        self._last_state = best_state
        return best_state

    def _viterbi(self) -> int:
        """
        Run the Viterbi algorithm (max-product on HMM chain) over the
        buffered emission log-probs.

        Uses log-space throughout to avoid numerical underflow (this replaces
        the traditional "scaling factors" approach with an equivalent and
        numerically cleaner log-domain formulation).

        Returns
        -------
        int
            The decoded state at the last time step of the window.
        """
        T = len(self._emission_buffer)
        N = self.num_classes

        # delta[t, j] = max log-prob of the best path ending in state j at time t
        delta = np.zeros((T, N), dtype=np.float64)
        # psi[t, j] = argmax predecessor state for backtracking
        psi = np.zeros((T, N), dtype=np.int32)

        # Initialisation: t = 0
        delta[0] = self.log_pi + self._emission_buffer[0]

        # Recursion: t = 1 .. T-1
        for t in range(1, T):
            for j in range(N):
                # delta[t-1, :] + log_A[:, j] → best predecessor for state j
                scores = delta[t - 1] + self.log_A[:, j]
                psi[t, j] = int(np.argmax(scores))
                delta[t, j] = scores[psi[t, j]] + self._emission_buffer[t][j]

        # Termination: best state at last time step
        best_last = int(np.argmax(delta[T - 1]))
        return best_last

    @property
    def last_state(self) -> Optional[int]:
        """The most recently decoded state index, or None if no observations yet."""
        return self._last_state

    @property
    def buffer_length(self) -> int:
        """Current number of frames in the sliding window."""
        return len(self._emission_buffer)
