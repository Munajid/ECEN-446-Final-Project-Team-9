import numpy as np
import matplotlib.pyplot as plt

# Load BER data
data_ber = np.loadtxt('./model/BER(576_432)_BP(5_5)_model0.txt')
snr = data_ber[:, 0]
ber = data_ber[:, 1]
bler = data_ber[:, 2]

# Load residual noise data
data_noise = np.loadtxt('./model/residual_noise_property_netid0_model0.txt')
snr_noise = data_noise[:, 0]
residual_power = data_noise[:, 1]

# Plot BER vs SNR
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.semilogy(snr, ber, 'o-', label='BER')
plt.semilogy(snr, bler, 's-', label='BLER')
plt.xlabel('SNR (dB)')
plt.ylabel('Error Rate')
plt.title('BER/BLER vs SNR (corr_para=0.0)')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(snr_noise, residual_power, 'o-')
plt.xlabel('SNR (dB)')
plt.ylabel('Residual Noise Power')
plt.title('Residual Noise vs SNR')
plt.grid(True)

plt.tight_layout()
plt.savefig('results_plot.png', dpi=300)
plt.show()