# yoda_filters

![](create_me_a_baby.png)

This documentation provides an overview and usage examples for various signal processing filters implemented in Python.

## Installation

To use the signal processing filters package, follow these steps:

1. Clone the Git repository to your local machine:
<pre><code>
git clone git@github.com:DiogoRibeiro7/yoda_filters.git
</code></pre>

2. Navigate to the project directory:
<pre><code>
cd yoda_filters
</code></pre>

3. Install the required dependencies using Poetry:
<pre><code>
poetry install
</code></pre>

This will create a virtual environment and install all the necessary dependencies specified in the `pyproject.toml` file.

4. Activate the virtual environment:
<pre><code>
poetry shell
</code></pre>

This will activate the virtual environment, allowing you to use the installed package.

## Usage Examples

For each filter, usage examples are provided in the docstrings of the respective functions. You can refer to those examples to see how to use each filter function with different parameter settings.

Please note that these implementations are provided as examples and may not cover all possible variations or edge cases. It's recommended to consult relevant documentation and references for a comprehensive understanding of signal processing filters.

## Butterworth Filter

The Butterworth filter is a type of infinite impulse response (IIR) filter that provides a maximally flat frequency response in the passband. It is commonly used for applications such as lowpass, highpass, bandpass, and bandstop filtering. The `butterworth_filter` function implements the Butterworth filter and allows you to specify the filter type, cutoff frequency, sampling frequency, and filter order.

## Band-Stop Filter (Notch Filter)

The band-stop filter, also known as the notch filter, is used to suppress or reject a specific range of frequencies while allowing other frequencies to pass through. The `band_stop_filter` function implements the band-stop filter and allows you to specify the lower and higher cutoff frequencies, sampling frequency, and filter order.

## FIR Filter

The Finite Impulse Response (FIR) filter is a type of digital filter with a finite impulse response. It can provide linear phase characteristics and is commonly used for applications such as lowpass, highpass, bandpass, and bandstop filtering. The `fir_filter` function implements the FIR filter and allows you to specify the filter type, cutoff frequency, sampling frequency, filter order, and window function.

## IIR Filter

The Infinite Impulse Response (IIR) filter is a type of digital filter with an infinite impulse response. It can provide a sharper roll-off and better performance compared to FIR filters for certain applications. The `iir_filter` function implements the IIR filter and allows you to specify the filter type, cutoff frequency, sampling frequency, filter order, and filter design method.

## Usage Examples

For each filter, usage examples are provided in the docstrings of the respective functions. You can refer to those examples to see how to use each filter function with different parameter settings.

Please note that these implementations are provided as examples and may not cover all possible variations or edge cases. It's recommended to consult relevant documentation and references for a comprehensive understanding of signal processing filters.


