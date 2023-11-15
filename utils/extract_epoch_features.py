import numpy as np
import mne

BANDS = [(8, 13), (13, 18), (18, 25)]

def extract_epoch_features(epochs: mne.Epochs) -> np.ndarray:
    """
    Extracts various features from MNE epochs object across multiple frequency bands,
    and returns them as a numpy array.

    Parameters:
    epochs (mne.Epochs): The epochs object with EEG data.

    Returns:
    np.ndarray: An array containing 12 averaged features across epochs.
    """
   
    # Initialize list to collect features
    feature_values = []

    # Extract data from epochs
    data = epochs.get_data()
    sfreq = epochs.info['sfreq']

    # Calculate features for each band
    for fmin, fmax in BANDS:
        psds, freqs = mne.time_frequency.psd_array_multitaper(data, sfreq=sfreq, fmin=fmin, fmax=fmax, verbose=False)

        average_power = np.mean(psds, axis=(1, 2))
        sum_power = np.sum(psds, axis=(1, 2))
        peak_frequency = freqs[np.argmax(psds, axis=2)].mean(axis=1)

        feature_values.append(np.mean(average_power))
        feature_values.append(np.mean(sum_power))
        feature_values.append(np.mean(peak_frequency))

    # Overall features across all bands
    psds, _ = mne.time_frequency.psd_array_multitaper(data, sfreq=sfreq, fmin=8, fmax=25, verbose=False)
    overall_average_power = np.mean(psds)
    overall_sum_power = np.sum(psds)
    overall_std_dev_power = np.std(np.mean(psds, axis=(1, 2)))

    feature_values.extend([overall_average_power, overall_sum_power, overall_std_dev_power])

    return np.array(feature_values)