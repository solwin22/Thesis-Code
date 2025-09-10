# generationsray.py — mixed-channel support + module-qualified import
import numpy as np
import globalparametersray as gp   # module import so tf.data never "loses" the symbol

payloadBits_per_OFDM = gp.payloadBits_per_OFDM
_CHANNEL_POOL = ['rayleigh', 'rician', 'wiener', 'awgn']

def _pick_channel(ch_type: str) -> str:
    if ch_type == 'mixed':
        return np.random.choice(_CHANNEL_POOL)
    return ch_type

def training_gen(bs, SNRdb=None, channel_type='rayleigh'):
    """Batches at a fixed SNR if SNRdb is given (e.g., 20 dB)."""
    gp.set_phase('train')
    while True:
        X = np.empty((bs, 2 * payloadBits_per_OFDM), dtype=np.float32)
        Y = np.empty((bs, payloadBits_per_OFDM),      dtype=np.float32)
        for i in range(bs):
            bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM,)).astype(np.float32)
            snr  = np.random.randint(5, 26) if SNRdb is None else SNRdb
            ch   = _pick_channel(channel_type)
            feats, _ = gp.ofdm_simulate(bits, snr, channel_type=ch)
            X[i, :] = feats
            Y[i, :] = bits
        yield (X, Y)

def validation_gen(bs, SNRdb=20, channel_type='rayleigh'):
    """Batches at a fixed SNR (default 20 dB)."""
    gp.set_phase('val')
    while True:
        X = np.empty((bs, 2 * payloadBits_per_OFDM), dtype=np.float32)
        Y = np.empty((bs, payloadBits_per_OFDM),      dtype=np.float32)
        for i in range(bs):
            bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM,)).astype(np.float32)
            ch   = _pick_channel(channel_type)
            feats, _ = gp.ofdm_simulate(bits, SNRdb, channel_type=ch)
            X[i, :] = feats
            Y[i, :] = bits
        yield (X, Y)
