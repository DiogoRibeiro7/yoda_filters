import numpy as np


def _bilinear_transform(p: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform bilinear transform from analog to digital domain.

    Args:
        p (np.ndarray): Analog filter poles.
        fs (float): Sampling frequency.

    Returns:
        tuple[np.ndarray, np.ndarray]: Digital filter zeros and poles.

    """
    z = (1 + p) / (1 - p)
    p = np.exp(2j * np.pi * fs / 2 * np.imag(p))
    return z, p


def kalman_filter(
    initial_state: np.ndarray,
    initial_covariance: np.ndarray,
    measurements: np.ndarray,
    transition_matrix: np.ndarray,
    observation_matrix: np.ndarray,
    process_noise_covariance: np.ndarray,
    measurement_noise_covariance: np.ndarray,
) -> np.ndarray:
    """
    Kalman filter implementation.

    Args:
        initial_state (np.ndarray): Initial state estimate.
        initial_covariance (np.ndarray): Initial covariance estimate.
        measurements (np.ndarray): Array of measurements.
        transition_matrix (np.ndarray): State transition matrix.
        observation_matrix (np.ndarray): Observation matrix.
        process_noise_covariance (np.ndarray): Process noise covariance.
        measurement_noise_covariance (np.ndarray): Measurement noise covariance.

    Returns:
        np.ndarray: Filtered state estimates.

    Raises:
        ValueError: If the dimensions of the input matrices are incompatible.

    Example:
        # Initialize matrices
        initial_state = np.array([[0], [0]])  # Initial state estimate
        initial_covariance = np.eye(2)  # Initial covariance estimate
        measurements = np.array([[1, 0], [0, 1], [1, 1]])  # Measurements
        transition_matrix = np.eye(2)  # State transition matrix
        observation_matrix = np.eye(2)  # Observation matrix
        process_noise_covariance = 0.1 * np.eye(2)  # Process noise covariance
        measurement_noise_covariance = 0.1 * np.eye(2)  # Measurement noise covariance

        # Apply Kalman filter
        filtered_states = kalman_filter(
            initial_state,
            initial_covariance,
            measurements,
            transition_matrix,
            observation_matrix,
            process_noise_covariance,
            measurement_noise_covariance,
        )

        print(filtered_states)

    """

    # Check input dimensions
    n_states = initial_state.shape[0]
    if initial_state.shape != (n_states, 1):
        raise ValueError("Initial state must be a column vector.")
    if initial_covariance.shape != (n_states, n_states):
        raise ValueError("Initial covariance must be a square matrix.")
    if measurements.shape[1] != n_states:
        raise ValueError("Measurement matrix dimensions are incompatible.")
    if transition_matrix.shape != (n_states, n_states):
        raise ValueError("Transition matrix dimensions are incompatible.")
    if observation_matrix.shape[1] != n_states:
        raise ValueError("Observation matrix dimensions are incompatible.")
    if (
        process_noise_covariance.shape != (n_states, n_states)
        or measurement_noise_covariance.shape != (n_states, n_states)
    ):
        raise ValueError(
            "Noise covariance matrix dimensions are incompatible.")

    # Initialize variables
    filtered_state_estimates = []
    filtered_covariance_estimates = []

    state_estimate = initial_state
    covariance_estimate = initial_covariance

    # Kalman filter loop
    for measurement in measurements:
        # Predict step
        predicted_state = np.dot(transition_matrix, state_estimate)
        predicted_covariance = (
            np.dot(np.dot(transition_matrix, covariance_estimate),
                   transition_matrix.T)
            + process_noise_covariance
        )

        # Update step
        innovation = measurement - np.dot(observation_matrix, predicted_state)
        innovation_covariance = (
            np.dot(np.dot(observation_matrix, predicted_covariance),
                   observation_matrix.T)
            + measurement_noise_covariance
        )
        kalman_gain = np.dot(
            np.dot(predicted_covariance, observation_matrix.T),
            np.linalg.inv(innovation_covariance),
        )

        state_estimate = predicted_state + np.dot(kalman_gain, innovation)
        covariance_estimate = np.dot(
            np.eye(n_states) - np.dot(kalman_gain, observation_matrix),
            predicted_covariance,
        )

        filtered_state_estimates.append(state_estimate)
        filtered_covariance_estimates.append(covariance_estimate)

    return np.array(filtered_state_estimates)


def chebyshev_filter(
    data: np.ndarray,
    cutoff_freq: float,
    sampling_freq: float,
    order: int,
    rp: float = 1,
    btype: str = 'lowpass',
) -> np.ndarray:
    """
    Chebyshev filter implementation.

    Args:
        data (np.ndarray): Input data to be filtered.
        cutoff_freq (float): Cutoff frequency of the filter.
        sampling_freq (float): Sampling frequency of the data.
        order (int): Order of the filter.
        rp (float): Passband ripple in decibels. Default is 1 dB.
        btype (str): Type of the filter. Default is 'lowpass'.

    Returns:
        np.ndarray: Filtered data.

    Raises:
        ValueError: If the filter type is not supported.

    Example:
        # Generate example data
        t = np.linspace(0, 1, num=1000)
        data = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 10 * t)

        # Apply Chebyshev filter
        cutoff_freq = 8
        sampling_freq = 100
        order = 4
        rp = 1
        filtered_data = chebyshev_filter(data, cutoff_freq, sampling_freq, order, rp)

        print(filtered_data)

    """
    # Check filter type
    if btype not in ['lowpass', 'highpass', 'bandpass', 'bandstop']:
        raise ValueError(
            "Unsupported filter type. Supported types are 'lowpass', 'highpass', 'bandpass', 'bandstop'.")

    # Normalize frequencies
    nyquist_freq = 0.5 * sampling_freq
    cutoff_norm = cutoff_freq / nyquist_freq

    # Calculate analog filter poles and zeros
    theta = np.pi * cutoff_norm
    e = np.exp(1j * np.pi * np.arange(1, 2 * order + 1) / (2 * order))
    p = np.cos(theta) + np.sin(theta) * e * 1j

    # Warp poles and zeros from analog to digital domain
    z, p = _bilinear_transform(p, sampling_freq)

    # Calculate transfer function coefficients
    b, a = _zp2tf(z, p, rp, btype)

    # Apply filter using difference equation
    filtered_data = np.zeros_like(data)
    for i in range(len(data)):
        filtered_data[i] = b[0] * data[i]
        for j in range(1, len(b)):
            if i - j >= 0:
                filtered_data[i] += b[j] * data[i - j]
        for j in range(1, len(a)):
            if i - j >= 0:
                filtered_data[i] -= a[j] * filtered_data[i - j]

    return filtered_data


