# globalparametersray.py — 16-QAM, 32 pilots, AWGN + Wiener; exports ofdm_simulate
import os
import numpy as np

__all__ = [
    "payloadBits_per_OFDM",
    "ofdm_simulate",
    "set_phase",
    "Modulation",
    "K", "CP", "P", "pilotCarriers", "dataCarriers", "pilotValue",
    "OFDM_symbol", "IDFT", "addCP", "removeCP", "DFT", "channel", "generate_channel_response"
]

# ---------- OFDM grid ----------
K = 64
CP = K // 4
P  = 32                    # <<<<<< 32 pilot tones (was 64)
allCarriers = np.arange(K)

if P < K:
    pilotCarriers = allCarriers[:: K // P]
    pilotCarriers = pilotCarriers[:P]  # ensure exactly 32
    dataCarriers  = np.setdiff1d(allCarriers, pilotCarriers, assume_unique=True)
else:
    pilotCarriers = allCarriers
    dataCarriers  = np.array([], dtype=int)

# Keep labels length EXACTLY 128 to match your model
mu = 2
payloadBits_per_OFDM = K * mu  # 64*2 = 128 (detector target stays 128 bits)

# ---------- Phase flag for Wiener channel banks ----------
try:
    WIENER_TRAIN = np.load('/mnt/data/channel_train.npy', allow_pickle=True)
    WIENER_TEST  = np.load('/mnt/data/channel_test.npy',  allow_pickle=True)
except Exception:
    WIENER_TRAIN = None
    WIENER_TEST  = None

_PHASE = 'train'
def set_phase(phase: str):
    """phase in {'train','val','test'} controls which Wiener bank is used."""
    global _PHASE
    _PHASE = phase

# ---------- 16-QAM (Gray), unit avg power ----------
_QAM16_LEVELS = np.array([3.0, 1.0, -1.0, -3.0], dtype=np.float32)  # Gray per axis
_QAM16_SCALE  = 1.0 / np.sqrt(10.0)

def qam16_mod(bits: np.ndarray) -> np.ndarray:
    b = bits.astype(np.uint8).reshape(-1, 4)
    i_idx = (b[:, 0] << 1) | b[:, 1]
    q_idx = (b[:, 2] << 1) | b[:, 3]
    i = _QAM16_LEVELS[i_idx]
    q = _QAM16_LEVELS[q_idx]
    s = (i + 1j * q) * _QAM16_SCALE
    return s.astype(np.complex64)

# Always 16-QAM
def Modulation(bits: np.ndarray) -> np.ndarray:
    assert bits.size % 4 == 0, "16-QAM requires bit length multiple of 4"
    return qam16_mod(bits)

# ---------- Pilot values (16-QAM on P=32 tones) ----------
Pilot_file_name = f'Pilot_{P}'
num_pilot_syms = len(pilotCarriers)
pilot_bits_len = max(1, 4 * num_pilot_syms)   # 4 bits per 16-QAM symbol

if os.path.isfile(Pilot_file_name):
    bits_pilots = np.loadtxt(Pilot_file_name, delimiter=',').astype(int)
    if bits_pilots.size != pilot_bits_len or (bits_pilots.size % 4) != 0:
        bits_pilots = np.random.binomial(n=1, p=0.5, size=(pilot_bits_len,))
        np.savetxt(Pilot_file_name, bits_pilots, delimiter=',')
else:
    bits_pilots = np.random.binomial(n=1, p=0.5, size=(pilot_bits_len,))
    np.savetxt(Pilot_file_name, bits_pilots, delimiter=',')

pilotValue = Modulation(bits_pilots)   # shape == (len(pilotCarriers),)

# ---------- OFDM helpers ----------
def OFDM_symbol(Data):
    symbol = np.zeros(K, dtype=complex)
    if num_pilot_syms > 0:
        symbol[pilotCarriers] = pilotValue
    if dataCarriers.size > 0 and Data.size > 0:
        symbol[dataCarriers] = Data
    return symbol

def IDFT(OFDM_data):    return np.fft.ifft(OFDM_data)
def addCP(OFDM_time):   cp = OFDM_time[-CP:]; return np.hstack([cp, OFDM_time])
def removeCP(signal):   return signal[CP:(CP + K)]
def DFT(OFDM_RX):       return np.fft.fft(OFDM_RX)

# ---------- Channels (Rayleigh / Rician / Wiener / AWGN) ----------
def _normalize_h(h: np.ndarray) -> np.ndarray:
    return h.astype(np.complex64) / (np.sqrt(np.sum(np.abs(h)**2)) + 1e-9)

def generate_channel_response(channel_type):
    ct = (channel_type or "").lower().strip()
    if ct == "awgn":
        return np.array([1.0 + 0j], dtype=np.complex64)

    if ct == "rayleigh":
        h = (np.random.randn(8) + 1j * np.random.randn(8)) / np.sqrt(2.0)
        return _normalize_h(h)

    if ct == "rician":
        delays = np.array([0, 1, 2, 3, 5, 6, 8, 10])
        gains = np.exp(-0.1 * delays)
        phase = (np.random.randn(len(gains)) + 1j * np.random.randn(len(gains))) / np.sqrt(2.0)
        return _normalize_h(gains * phase)

    if ct in ("wiener", "weiner", "weinner"):
        bank = WIENER_TRAIN if _PHASE == "train" else (WIENER_TEST if WIENER_TEST is not None else WIENER_TRAIN)
        if bank is None or len(bank) == 0:
            h = (np.random.randn(8) + 1j * np.random.randn(8)) / np.sqrt(2.0)
            return _normalize_h(h)
        arr = np.asarray(bank)
        idx = np.random.randint(0, arr.shape[0])
        h = arr[idx]
        if h.ndim == 2 and h.shape[-1] == 2 and not np.iscomplexobj(h):
            h = h[..., 0].astype(np.float32) + 1j * h[..., 1].astype(np.float32)
        return _normalize_h(h)

    raise ValueError("Unknown channel type (use 'rayleigh', 'rician', 'wiener', or 'awgn')")

def channel(signal, channelResponse, SNRdb):
    convolved = np.convolve(signal, channelResponse)
    sig_power = np.mean(np.abs(convolved)**2) + 1e-12
    sigma2 = sig_power * 10 ** (-SNRdb / 10.0)
    noise = np.sqrt(sigma2 / 2.0) * (
        np.random.randn(*convolved.shape) + 1j * np.random.randn(*convolved.shape)
    )
    return convolved + noise

# ---------- End-to-end simulator ----------
def ofdm_simulate(codeword, SNRdb, channel_type='rayleigh'):
    """
    Returns:
      features: float32, shape (2 * payloadBits_per_OFDM,) == 256
                [Re/Im RX of (pilot+data) frame | Re/Im RX of codeword frame]
      chan_abs: float32, |h|
    """
    h = generate_channel_response(channel_type)

    # Frame 1: pilots + random 16-QAM data on remaining carriers
    if dataCarriers.size > 0:
        rand_bits_len = 4 * dataCarriers.size
        rand_bits = np.random.binomial(n=1, p=0.5, size=(rand_bits_len,))
        QAM_data = Modulation(rand_bits)
    else:
        QAM_data = np.array([], dtype=np.complex64)

    ofdm_grid = np.zeros(K, dtype=complex)
    if num_pilot_syms > 0:
        ofdm_grid[pilotCarriers] = pilotValue
    if dataCarriers.size > 0:
        ofdm_grid[dataCarriers] = QAM_data

    tx1 = IDFT(ofdm_grid)
    tx1_cp = addCP(tx1)
    rx1 = channel(tx1_cp, h, SNRdb)
    rx1_nocp = removeCP(rx1)
    RX1 = DFT(rx1_nocp)  # (K,)

    # Frame 2: codeword (128 bits) -> 32 16-QAM syms -> repeat each twice -> K=64
    cw_bits = np.asarray(codeword).astype(np.uint8)
    assert cw_bits.size == payloadBits_per_OFDM and (cw_bits.size % 4) == 0, \
        "codeword must be 128 bits (multiple of 4 for 16-QAM)"
    cw_syms_32 = Modulation(cw_bits)        # (32,)
    cw_syms_64 = np.repeat(cw_syms_32, 2)   # (64,)
    sym_grid = np.zeros(K, dtype=complex)
    sym_grid[np.arange(K)] = cw_syms_64

    tx2 = np.fft.ifft(sym_grid)
    tx2_cp = addCP(tx2)
    rx2 = channel(tx2_cp, h, SNRdb)
    rx2_nocp = removeCP(rx2)
    RX2 = DFT(rx2_nocp)  # (K,)

    feats = np.concatenate((
        np.concatenate((np.real(RX1), np.imag(RX1))),  # 128
        np.concatenate((np.real(RX2), np.imag(RX2)))   # 128
    )).astype(np.float32)

    return feats, np.abs(h).astype(np.float32)
