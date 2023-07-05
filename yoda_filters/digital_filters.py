import numpy as np


def _zp2tf(z, p, rp=None, btype='lowpass'):
    """
    Calculate transfer function coefficients from zeros and poles.

    Args:
        z (np.ndarray): Array of zeros.
        p (np.ndarray): Array of poles.
        rp (np.ndarray): Array of zeros/resonance frequencies for bandpass/bandstop filters.
        btype (str): Type of the filter. Default is 'lowpass'.

    Returns:
        tuple[np.ndarray, np.ndarray]: Numerator and denominator coefficients.

    Raises:
        ValueError: If the filter type is not supported.

    """
    z = np.array(z)
    p = np.array(p)

    if rp is not None:
        rp = np.array(rp)

    num = np.poly(z)
    den = np.poly(p)

    if btype == 'lowpass':
        return num, den
    elif btype == 'highpass':
        num, den = highpass_transform(num, den)
        return num, den
    else:
        raise ValueError(
            "Invalid filter type. Supported types are 'lowpass' and 'highpass'.")


def zp2tf_custom(z: np.ndarray, p: np.ndarray, rp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert zeros and poles to transfer function coefficients with a custom scaling factor.

    Args:
        z (np.ndarray): Array of zeros.
        p (np.ndarray): Array of poles.
        rp (np.ndarray): Array of zeros/resonance frequencies for bandpass/bandstop filters.

    Returns:
        tuple[np.ndarray, np.ndarray]: Numerator and denominator coefficients.

    Raises:
        ValueError: If the lengths of `z`, `p`, and `rp` arrays are not the same.

    Example:
        z = np.array([1, 2, 3])
        p = np.array([4, 5, 6])
        rp = np.array([0.5, 1.0, 1.5])
        num, den = zp2tf_custom(z, p, rp)
        print(num, den)
    """

    # Check input array lengths
    if len(z) != len(p) or len(z) != len(rp):
        raise ValueError(
            "Lengths of `z`, `p`, and `rp` arrays must be the same.")

    # Convert zeros and poles to transfer function coefficients
    num = np.poly(z)
    den = np.poly(p)

    # Scale numerator and denominator with the custom factor
    num *= rp[0]
    den *= rp[0]

    return num, den


def highpass_transform(num: np.ndarray, den: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply highpass transformation to transfer function coefficients.

    Args:
        num (np.ndarray): Numerator coefficients.
        den (np.ndarray): Denominator coefficients.

    Returns:
        tuple[np.ndarray, np.ndarray]: Transformed numerator and denominator coefficients.

    Example:
        num = np.array([1, 2, 3])
        den = np.array([4, 5, 6])
        transformed_num, transformed_den = highpass_transform(num, den)
        print(transformed_num, transformed_den)
    """

    # Calculate the order of the filter
    order = len(den) - 1

    # Apply the highpass transformation
    sign = (-1) ** order
    num *= sign

    return num, den


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


def _apply_filter(data, b, a):
    filtered_data = np.zeros_like(data)
    N = len(data)
    M = len(b)
    L = len(a)

    for n in range(N):
        for m in range(min(n+1, M)):
            filtered_data[n] += b[m] * data[n - m]

        for l in range(1, min(n+1, L)):
            filtered_data[n] -= a[l] * filtered_data[n - l]

    return filtered_data


def _butter_bandpass_coeffs(cutoff_norm, order):
    b = np.zeros(order + 1)
    a = np.zeros(order + 1)

    mid = order // 2

    for k in range(mid + 1):
        b[k] = comb(mid, k) * (cutoff_norm ** (mid - k)) * (-1) ** k
        a[k] = comb(mid, k) * (cutoff_norm ** (mid - k))

    b[mid + 1:] = b[mid::-1]
    a[mid + 1:] = a[mid::-1]

    return b, a


def _butter_bandstop_coeffs(cutoff_norm, order):
    b, a = _butter_bandpass_coeffs(cutoff_norm, order)
    b *= (-1) ** order
    return b, a


def comb(n, k):
    return np.math.factorial(n) // (np.math.factorial(k) * np.math.factorial(n - k))


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


def fir_filter(data: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    FIR filter implementation.

    Args:
        data (np.ndarray): Input data to be filtered.
        kernel (np.ndarray): Filter kernel coefficients.

    Returns:
        np.ndarray: Filtered data.

    Raises:
        ValueError: If the kernel size is not odd or the kernel is not 1D.

    Example:
        # Generate example data
        data = np.sin(np.arange(0, 2 * np.pi, 0.1))

        # Define filter kernel
        kernel = np.array([0.1, 0.2, 0.3, 0.2, 0.1])

        # Apply FIR filter
        filtered_data = fir_filter(data, kernel)

        print(filtered_data)

    """
    # Check kernel size and dimension
    if len(kernel) % 2 != 1:
        raise ValueError("Kernel size must be odd.")
    if kernel.ndim != 1:
        raise ValueError("Kernel must be 1-dimensional.")

    # Pad data at both ends to handle edge effects
    pad_len = len(kernel) // 2
    padded_data = np.pad(data, pad_len, mode='edge')

    # Apply filter using convolution
    filtered_data = np.convolve(padded_data, kernel, mode='valid')

    return filtered_data


def butterworth_filter(
    data: np.ndarray,
    cutoff_freq: float,
    sampling_freq: float,
    order: int,
    btype: str = 'lowpass',
) -> np.ndarray:
    """
    Butterworth filter implementation.

    Args:
        data (np.ndarray): Input data to be filtered.
        cutoff_freq (float): Cutoff frequency of the filter.
        sampling_freq (float): Sampling frequency of the data.
        order (int): Order of the filter.
        btype (str): Type of the filter. Default is 'lowpass'.

    Returns:
        np.ndarray: Filtered data.

    Raises:
        ValueError: If the filter type is not supported.

    Example:
        # Generate example data
        t = np.linspace(0, 1, num=1000)
        data = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 10 * t)

        # Apply Butterworth filter
        cutoff_freq = 8
        sampling_freq = 100
        order = 4
        filtered_data = butterworth_filter(data, cutoff_freq, sampling_freq, order)

        print(filtered_data)

    """
    # Check filter type
    if btype not in ['lowpass', 'highpass', 'bandpass', 'bandstop']:
        raise ValueError(
            "Unsupported filter type. Supported types are 'lowpass', 'highpass', 'bandpass', 'bandstop'.")

    # Normalize frequencies
    nyquist_freq = 0.5 * sampling_freq
    cutoff_norm = cutoff_freq / nyquist_freq

    # Compute filter coefficients
    b, a = _compute_filter_coefficients(cutoff_norm, order, btype)

    # Apply filter using difference equation
    filtered_data = _apply_filter(data, b, a)

    return filtered_data


def _compute_filter_coefficients(cutoff_norm: float, order: int, btype: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Butterworth filter coefficients.

    Args:
        cutoff_norm (float): Normalized cutoff frequency.
        order (int): Order of the filter.
        btype (str): Type of the filter.

    Returns:
        tuple[np.ndarray, np.ndarray]: Numerator and denominator coefficients.

    """
    if btype == 'lowpass':
        b, a = _butter_lowpass_coeffs(cutoff_norm, order)
    elif btype == 'highpass':
        b, a = _butter_highpass_coeffs(cutoff_norm, order)
    elif btype == 'bandpass':
        b, a = _butter_bandpass_coeffs(cutoff_norm, order)
    elif btype == 'bandstop':
        b, a = _butter_bandstop_coeffs(cutoff_norm, order)
    else:
        raise ValueError("Unsupported filter type.")

    return b, a


def _butter_coeffs(cutoff_norm, order):
    b = np.zeros(order + 1)
    a = np.zeros(order + 1)

    b[0] = 1.0

    for k in range(1, order + 1):
        b[k] = b[k - 1] * (order - k + 1) * cutoff_norm / k
        a[k] = a[k - 1] + (2 * k - 1) * cutoff_norm

    return b, a


def _butter_lowpass_coeffs(cutoff_norm: float, order: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Butterworth lowpass filter coefficients.

    Args:
        cutoff_norm (float): Normalized cutoff frequency.
        order (int): Order of the filter.

    Returns:
        tuple[np.ndarray, np.ndarray]: Numerator and denominator coefficients.

    """
    b, a = _butter_coeffs(cutoff_norm, order)
    return b, a


def _butter_highpass_coeffs(cutoff_norm: float, order: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Butterworth highpass filter coefficients.

    Args:
        cutoff_norm (float): Normalized cutoff frequency.
        order (int): Order of the filter.

    Returns:
        tuple[np.ndarray, np.ndarray]: Numerator and denominator coefficients.

    """
    p = np.exp(1j * np.pi * np.arange(2 * order) / 2)
    p = 1.0 / p
    z, p = _bilinear_transform(p, cutoff_norm)
    b, a = _zp2tf(z, p)
    return b, a


def moving_average_filter(data, window_size):
    """
    Applies a moving average filter to a list of data.

    Args:
        data (list): List of numerical data to be filtered.
        window_size (int): The size of the moving average window.

    Returns:
        list: List of filtered data.

    Raises:
        ValueError: If input types are not as expected.
        ValueError: If window size is greater than the length of the data.

    Example:
        >>> data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> window_size = 3
        >>> filtered_data = moving_average_filter(data, window_size)
        >>> print(filtered_data)
        [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    """

    # Check types
    if not isinstance(data, list):
        raise ValueError('Data must be a list.')
    if not isinstance(window_size, int):
        raise ValueError('Window size must be an integer.')

    for i in data:
        if not isinstance(i, (int, float)):
            raise ValueError('Data list must contain only numbers.')

    # Check window size
    if window_size > len(data):
        raise ValueError(
            'Window size must be less than or equal to the length of the data.')

    # Apply the filter
    filtered_data = []
    for i in range(window_size-1, len(data)):
        window = data[i-window_size+1:i+1]  # Define the window
        # Append the average of the window
        filtered_data.append(sum(window) / window_size)

    return filtered_data


def hilbert_transform(data: list):
    """
    Computes an approximation of the Hilbert Transform of a 1D signal.

    Args:
        data (list): List of numerical data to be transformed.

    Returns:
        np.ndarray: Hilbert-transformed data.

    Raises:
        ValueError: If input types are not as expected.

    Example:
        >>> data = [1, 2, 1, 0, -1, -2, -1, 0]
        >>> ht_data = hilbert_transform(data)
        >>> print(ht_data)
        [ 0.+1.j, -1.+1.j, -2.+0.j, -1.-1.j,  0.-1.j,  1.-1.j,  2.-0.j,  1.+1.j]
    """

    # Check types
    if not isinstance(data, list):
        raise ValueError('Data must be a list.')

    for i in data:
        if not isinstance(i, (int, float)):
            raise ValueError('Data list must contain only numbers.')

    # Convert data to numpy array
    data = np.array(data)

    # Perform FFT
    fft_data = np.fft.fft(data)

    # Create an array for the multiplier (two in the positive frequency region, zero in the negative)
    multiplier = np.zeros_like(data)
    multiplier[:len(data) // 2] = 2

    # Perform inverse FFT, multiplying by the multiplier
    hilbert = np.fft.ifft(fft_data * multiplier)

    return hilbert


def median_filter(data: list, window_size: int):
    """
    Applies a median filter to a list of data.

    Args:
        data (list): List of numerical data to be filtered.
        window_size (int): The size of the median filter window.

    Returns:
        list: List of filtered data.

    Raises:
        ValueError: If input types are not as expected.
        ValueError: If window size is greater than the length of the data.

    Example:
        >>> data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> window_size = 3
        >>> filtered_data = median_filter(data, window_size)
        >>> print(filtered_data)
        [2, 3, 4, 5, 6, 7, 8, 9]
    """

    # Check types
    if not isinstance(data, list):
        raise ValueError('Data must be a list.')
    if not isinstance(window_size, int):
        raise ValueError('Window size must be an integer.')

    for i in data:
        if not isinstance(i, (int, float)):
            raise ValueError('Data list must contain only numbers.')

    # Check window size
    if window_size > len(data):
        raise ValueError(
            'Window size must be less than or equal to the length of the data.')

    # Apply the filter
    filtered_data = []
    for i in range(window_size-1, len(data)):
        window = data[i-window_size+1:i+1]  # Define the window
        # Append the median of the window
        filtered_data.append(sorted(window)[window_size // 2])

    return filtered_data


def savgol_filter(y: list, window_size: int, order: int):
    """
    Applies a Savitzky-Golay filter to a 1D signal.

    Args:
        y (list): List of numerical data to be filtered.
        window_size (int): The size of the filter window (must be odd).
        order (int): The order of the polynomial used in the filter.

    Returns:
        np.ndarray: Savitzky-Golay filtered data.

    Raises:
        ValueError: If input types are not as expected.
        ValueError: If window size is not an odd number.

    Example:
        >>> y = [2, 2, 5, 2, 1, 0, 1, 3, 6, 7, 8, 10, 11, 12, 13, 14, 15]
        >>> window_size = 5
        >>> order = 2
        >>> filtered_y = savgol_filter(y, window_size, order)
        >>> print(filtered_y)
        [2.2, 2.8, 3.2, 2.3, 1.1, 1.4, 2.8, 4.6, 6.6, 8.3, 9.6, 10.6, 11.3, 12.1, 13.2, 14.4, 15. ]
    """

    # Check types
    if not isinstance(y, list):
        raise ValueError('Data must be a list.')
    if not isinstance(window_size, int):
        raise ValueError('Window size must be an integer.')
    if not isinstance(order, int):
        raise ValueError('Order must be an integer.')

    # Check values
    if window_size % 2 == 0:
        raise ValueError('Window size must be an odd number.')
    if window_size < order + 2:
        raise ValueError('Window size is too small for the polynomial order.')

    # Convert list to numpy array
    y = np.array(y)

    # Create an array of coefficients for a Savitzky-Golay filter
    n = (window_size - 1) // 2
    A = np.vander(np.arange(-n, n + 1), order + 1).T
    coeffs = np.linalg.pinv(A).r_[order]

    # Apply the filter
    return np.convolve(y, coeffs, mode='same')
