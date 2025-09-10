# ====================== universal_archA_prof_fixed.py ======================
#   * 16-QAM everywhere (pilots + payload), 32 pilots
#   * Training accuracy (detector bit_acc) at SNR=20 dB -> PNG + MAT
#   * Validation loss (mixed=20 dB here) -> PNG + MAT
#   * Per-channel validation loss figure -> PNG + MAT
#   * BER evaluation unchanged
#   * No architecture changes

import os, glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, BatchNormalization, concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, Callback
from tensorflow.keras import regularizers

REG = regularizers.l2(1e-4)

# -------------------------
# Output folder
# -------------------------
try:
    from google.colab import drive  # noqa
    drive.mount('/content/drive')
    OUTDIR = "/content/drive/MyDrive/universalA_results_prof_fixed"
except Exception:
    OUTDIR = "./universalA_results_prof_fixed"
os.makedirs(OUTDIR, exist_ok=True)
print("Saving artifacts to:", OUTDIR)

# -------------------------
# System / Modulation (16-QAM; pilots=32)
# -------------------------
K = 64
CP = 16
BITS_PER_SYMBOL = 4          # 16-QAM payload
N_PILOTS        = 32         # 32 pilots
CHANNELS        = ['rayleigh', 'rician', 'winnerii', 'awgn']
N_CLASSES       = len(CHANNELS)

# With 32 pilots, 32 data tones; 16-QAM => 4 bits/symbol -> 128 bits/OFDM
payloadBits_per_OFDM = (K - N_PILOTS) * BITS_PER_SYMBOL  # 128
IN_FEAT   = 2 * K                      # 128 (Re/Im across K)
CSI_LEN   = 12
IN_FEAT_CSI = IN_FEAT + 2 * CSI_LEN    # 128 + 24 = 152

# carriers
allCarriers   = np.arange(K)
step          = max(1, K // N_PILOTS)
pilotCarriers = allCarriers[:: step][:N_PILOTS]
dataCarriers  = np.setdiff1d(allCarriers, pilotCarriers, assume_unique=True)

# -------------------------
# 16-QAM mapper (Gray) + pilots
# -------------------------
_QAM16_LEVELS = np.array([3.0, 1.0, -1.0, -3.0], dtype=np.float32)
_QAM16_SCALE  = 1.0 / np.sqrt(10.0)

def bits_to_16qam(bits_vec):
    b = bits_vec.reshape(-1, 4).astype(np.int32)
    i_idx = (b[:,0] << 1) | b[:,1]
    q_idx = (b[:,2] << 1) | b[:,3]
    I = _QAM16_LEVELS[i_idx]
    Q = _QAM16_LEVELS[q_idx]
    return ((I + 1j*Q) * _QAM16_SCALE).astype(np.complex64)

def Modulation(bits_vec):  # 16-QAM everywhere
    return bits_to_16qam(bits_vec)

Pilot_file = f'Pilot_{N_PILOTS}_mu{BITS_PER_SYMBOL}.csv'
if os.path.isfile(Pilot_file):
    pilot_bits = np.loadtxt(Pilot_file, delimiter=',').astype(int)
    # ensure correct length / multiple of 4
    need = len(pilotCarriers) * BITS_PER_SYMBOL
    if pilot_bits.size != need or (pilot_bits.size % 4) != 0:
        pilot_bits = np.random.binomial(1, 0.5, size=(need,))
        np.savetxt(Pilot_file, pilot_bits, delimiter=',')
else:
    pilot_bits = np.random.binomial(1, 0.5, size=(len(pilotCarriers) * BITS_PER_SYMBOL,))
    np.savetxt(Pilot_file, pilot_bits, delimiter=',')
pilotValue = Modulation(pilot_bits)

# -------------------------
# OFDM helpers
# -------------------------
def IDFT(X):       return np.fft.ifft(X)
def DFT(x):        return np.fft.fft(x)
def addCP(x):      return np.hstack([x[-CP:], x])
def removeCP(x):   return x[CP:(CP + K)]

def awgn_after_channel(signal, h, SNRdb):
    y = np.convolve(signal, h)
    Ps = np.mean(np.abs(y)**2)
    sigma2 = Ps * 10 ** (-SNRdb / 10.0)
    n = np.sqrt(sigma2/2) * (np.random.randn(*y.shape) + 1j*np.random.randn(*y.shape))
    return y + n

# -------------------------
# WINNER II taps (train vs test)
# -------------------------
class WinnerIIBank:
    def __init__(self, path_glob, name):
        cands = sorted(glob.glob(path_glob))
        if not cands:
            raise FileNotFoundError(f"[winnerii] {name} taps file missing. Place a '{path_glob}' here.")
        self.path = cands[0]
        self.arr = np.load(self.path, allow_pickle=True, mmap_mode='r')
        if self.arr.ndim != 2 or self.arr.shape[1] < 1:
            raise ValueError(f"[winnerii] {name} file {self.path} must be 2-D (N, L_taps). Got {self.arr.shape}")
        subset = min(100000, self.arr.shape[0])
        mp = float(np.mean(np.abs(self.arr[:subset])**2))
        self.scale = 1.0 / np.sqrt(mp + 1e-12)
        print(f"[winnerii] Loaded {name}: {self.path}, shape={self.arr.shape}, mean_power~{mp:.6g}, scale={self.scale:.6g}")

    def sample(self):
        idx = np.random.randint(self.arr.shape[0])
        return (np.asarray(self.arr[idx], dtype=np.complex64).ravel() * self.scale)

WINNER_TRAIN = WinnerIIBank("channel_train.npy", "TRAIN")
WINNER_TEST  = WinnerIIBank("channel_test.npy",  "TEST")

# -------------------------
# Rayleigh / Rician generators
# -------------------------
def _exp_pdp(L, alpha=0.35):
    p = np.exp(-alpha * np.arange(L)).astype(np.float32)
    p /= (p.sum() + 1e-12)
    return p

def _rayleigh_taps(L, pdp=None):
    if pdp is None:
        pdp = np.ones(L, np.float32) / L
    g = (np.random.randn(L) + 1j*np.random.randn(L)) / np.sqrt(2.0)
    h = g * np.sqrt(pdp).astype(np.complex64)
    h /= np.sqrt(np.mean(np.abs(h)**2) + 1e-12)
    return h.astype(np.complex64)

def _rician_taps(L, K_lin, pdp_tail=None):
    assert L >= 2
    if pdp_tail is None:
        pdp_tail = _exp_pdp(L-1, alpha=0.35)
    P_los  = K_lin / (K_lin + 1.0)
    P_nlos = 1.0   / (K_lin + 1.0)
    phi = 2*np.pi*np.random.rand()
    h0  = np.sqrt(P_los) * np.exp(1j*phi)
    tail = _rayleigh_taps(L-1, pdp_tail) * np.sqrt(P_nlos)
    h = np.zeros(L, dtype=np.complex64)
    h[0]  = h0.astype(np.complex64)
    h[1:] = tail
    h /= np.sqrt(np.mean(np.abs(h)**2) + 1e-12)
    return h

RICIAN_K = 10.0  # linear K-factor

def generate_channel_response(channel_type, mode="train"):
    if channel_type == 'rayleigh':
        return _rayleigh_taps(L=4, pdp=_exp_pdp(4, alpha=0.35))
    if channel_type == 'rician':
        return _rician_taps(L=6, K_lin=RICIAN_K, pdp_tail=_exp_pdp(5, alpha=0.35))
    if channel_type == 'winnerii':
        return WINNER_TRAIN.sample() if mode != "eval" else WINNER_TEST.sample()
    if channel_type == 'awgn':
        return np.array([1.0 + 0j], dtype=np.complex64)
    raise ValueError(f"Unknown channel type: {channel_type}")

# -------------------------
# One RX feature (with CSI concat) + bits
# -------------------------
def make_rx_and_bits(SNRdb, channel_type='rayleigh', mode="train", return_bits=False):
    h = generate_channel_response(channel_type, mode=mode)
    # pad/truncate CSI taps to CSI_LEN
    if len(h) < CSI_LEN:
        h = np.pad(h, (0, CSI_LEN - len(h)), 'constant')
    elif len(h) > CSI_LEN:
        h = h[:CSI_LEN]

    bits = np.random.binomial(1, 0.5, size=(payloadBits_per_OFDM,))
    X = np.zeros(K, dtype=np.complex64)
    X[pilotCarriers] = pilotValue
    X[dataCarriers]  = Modulation(bits)  # 16-QAM payload

    y = awgn_after_channel(addCP(IDFT(X)), h, SNRdb)
    Y = DFT(removeCP(y))
    feat128 = np.concatenate([np.real(Y), np.imag(Y)], axis=0).astype(np.float32)
    feat128 /= np.sqrt(np.mean(feat128**2) + 1e-8)

    h_feat = np.concatenate([np.real(h), np.imag(h)], axis=0).astype(np.float32)
    full_feat = np.concatenate([feat128, h_feat], axis=0)
    return (full_feat, bits.astype(np.float32)) if return_bits else (full_feat, None)

# -------------------------
# Generators (SNR fixed at 20 dB for training/validation as requested)
# -------------------------
def _rand_ch():   return np.random.choice(CHANNELS)
def _rand_snr(s): return (np.random.randint(0, 21) if s is None else s)

def training_class_gen(bs, SNRdb=20):
    while True:
        X, Y = [], []
        for _ in range(bs):
            ch  = _rand_ch(); snr = _rand_snr(SNRdb)  # 20 dB
            feat, _ = make_rx_and_bits(snr, ch, mode="train", return_bits=False)
            X.append(feat)
            onehot = np.zeros((N_CLASSES,), np.float32); onehot[CHANNELS.index(ch)] = 1.0
            Y.append(onehot)
        yield np.asarray(X, np.float32), np.asarray(Y, np.float32)

def validation_class_gen(bs, SNRdb=20):
    while True:
        X, Y = [], []
        for _ in range(bs):
            ch  = _rand_ch(); snr = _rand_snr(SNRdb)  # 20 dB
            feat, _ = make_rx_and_bits(snr, ch, mode="train", return_bits=False)
            X.append(feat)
            onehot = np.zeros((N_CLASSES,), np.float32); onehot[CHANNELS.index(ch)] = 1.0
            Y.append(onehot)
        yield np.asarray(X, np.float32), np.asarray(Y, np.float32)

def training_gen(bs, SNRdb=20):
    while True:
        X, Y = [], []
        for _ in range(bs):
            ch  = _rand_ch(); snr = _rand_snr(SNRdb)  # 20 dB
            feat, bits = make_rx_and_bits(snr, ch, mode="train", return_bits=True)
            X.append(feat); Y.append(bits)
        yield np.asarray(X, np.float32), np.asarray(Y, np.float32)

def validation_gen(bs, SNRdb=20):
    while True:
        X, Y = [], []
        for _ in range(bs):
            ch  = _rand_ch(); snr = _rand_snr(SNRdb)  # 20 dB
            feat, bits = make_rx_and_bits(snr, ch, mode="train", return_bits=True)
            X.append(feat); Y.append(bits)
        yield np.asarray(X, np.float32), np.asarray(Y, np.float32)

def validation_gen_fixed(bs, SNRdb, channel_type):
    while True:
        X, Y = [], []
        for _ in range(bs):
            feat, bits = make_rx_and_bits(SNRdb, channel_type, mode="train", return_bits=True)
            X.append(feat); Y.append(bits)
        yield np.asarray(X, np.float32), np.asarray(Y, np.float32)

# -------------------------
# Metrics
# -------------------------
def bit_err(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return 1.0 - tf.reduce_mean(
        tf.reduce_mean(tf.cast(tf.equal(tf.sign(y_pred - 0.5), tf.sign(y_true - 0.5)), tf.float32), axis=1)
    )

def bit_acc(y_true, y_pred):   # training accuracy metric for detector
    return 1.0 - bit_err(y_true, y_pred)

# -------------------------
# Models (unchanged architecture)
# -------------------------
def build_classifier_exact():
    inp = Input(shape=(IN_FEAT_CSI,), name='input_bits')
    x = BatchNormalization()(inp)
    x = Dense(256, activation='relu', kernel_regularizer=REG)(x); x = Dropout(0.1)(x); x = BatchNormalization()(x)
    x = Dense(128, activation='relu', kernel_regularizer=REG)(x); x = Dropout(0.1)(x); x = BatchNormalization()(x)
    x = Dense(64,  activation='relu', kernel_regularizer=REG)(x); x = Dropout(0.1)(x); x = BatchNormalization()(x)
    x = Dense(32,  activation='relu', kernel_regularizer=REG)(x); x = Dropout(0.1)(x); x = BatchNormalization()(x)
    x = Dense(16,  activation='relu', kernel_regularizer=REG)(x); x = Dropout(0.1)(x); x = BatchNormalization()(x)
    out = Dense(N_CLASSES, activation='softmax', name='classes')(x)
    m = Model(inp, out, name='Classifier_EXACT')
    m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return m

def build_detector_exact(trained_classifier, n_hidden_1=32, n_hidden_2=16, n_hidden_3=16):
    trained_classifier.trainable = False
    inp = Input(shape=(IN_FEAT_CSI,), name='input_bits')
    classes = trained_classifier(inp, training=False)

    cc1 = concatenate([classes, inp]); t1 = BatchNormalization()(cc1)
    t1  = Dense(n_hidden_1*4, activation='relu', kernel_regularizer=REG)(t1)

    cc2 = concatenate([classes, t1]);  t2 = BatchNormalization()(cc2)
    t2  = Dense(n_hidden_2*4, activation='relu', kernel_regularizer=REG)(t2)

    cc3 = concatenate([classes, t2]);  t3 = BatchNormalization()(cc3)
    t3  = Dense(n_hidden_3*4, activation='relu', kernel_regularizer=REG)(t3)

    cc4 = concatenate([classes, t3]);  t4 = BatchNormalization()(cc4)
    out_bits = Dense(payloadBits_per_OFDM, activation='sigmoid', name='bits')(t4)

    m = Model(inp, out_bits, name='Detector_EXACT')
    m.compile(optimizer='adam', loss='mse', metrics=[bit_err, bit_acc])
    return m

# -------------------------
# Training knobs (yours)
# -------------------------
BATCH          = 1000
EPOCHS_CLS     = 2
EPOCHS_DET     = 2
STEPS_PER_EPOCH_CLS = 50
VAL_STEPS_CLS       = 25
STEPS_PER_EPOCH_DET = 50
VAL_STEPS_DET       = 25

TEST_SNRS    = [0, 5, 10, 15, 20]
TEST_SAMPLES = 100000

# -------------------------
# Per-channel validation loss callback
# -------------------------
class PerChannelValLoss(Callback):
    def __init__(self, model, val_sets: dict, steps_per_val: int = 5):
        super().__init__()
        self.model = model
        self.val_sets = val_sets
        self.history = {k: [] for k in val_sets.keys()}
    def on_epoch_end(self, epoch, logs=None):
        for name, (gen, steps) in self.val_sets.items():
            res = self.model.evaluate(gen, steps=steps, verbose=0)
            loss = float(res[0]) if isinstance(res, (list, tuple, np.ndarray)) else float(res)
            self.history[name].append(loss)

# -------------------------
# Train Classifier (SNR=20 dB)
# -------------------------
classifier = build_classifier_exact()
ckpt_cls = ModelCheckpoint(os.path.join(OUTDIR, 'cls_best.weights.h5'),
                           monitor='accuracy', mode='max',
                           save_best_only=True, save_weights_only=True, verbose=1)
classifier.fit(
    training_class_gen(BATCH, 20),
    steps_per_epoch=STEPS_PER_EPOCH_CLS,
    epochs=EPOCHS_CLS,
    validation_data=validation_class_gen(BATCH, 20),
    validation_steps=VAL_STEPS_CLS,
    callbacks=[ckpt_cls, ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)],
    verbose=2
)
classifier.load_weights(os.path.join(OUTDIR, 'cls_best.weights.h5'))

# -------------------------
# Train Detector (cascade) + record TRAINING ACCURACY & VAL LOSS
# -------------------------
detector = build_detector_exact(classifier, n_hidden_1=32, n_hidden_2=16, n_hidden_3=16)
ckpt_det = ModelCheckpoint(os.path.join(OUTDIR, 'det_best.weights.h5'),
                           monitor='val_bit_err', mode='min',
                           save_best_only=True, save_weights_only=True, verbose=1)

VAL_SNR_FOR_CURVES = 20
val_sets = {
    "Rayleigh": (validation_gen_fixed(6000, VAL_SNR_FOR_CURVES, 'rayleigh'), 5),
    "Rician"  : (validation_gen_fixed(6000, VAL_SNR_FOR_CURVES, 'rician'),   5),
    "WinnerII": (validation_gen_fixed(6000, VAL_SNR_FOR_CURVES, 'winnerii'), 5),
    "AWGN"    : (validation_gen_fixed(6000, VAL_SNR_FOR_CURVES, 'awgn'),     5),
}
pcvl_cb = PerChannelValLoss(detector, val_sets, steps_per_val=5)

hist_det = detector.fit(
    training_gen(BATCH, 20),
    steps_per_epoch=STEPS_PER_EPOCH_DET,
    epochs=EPOCHS_DET,
    validation_data=validation_gen(BATCH, 20),
    validation_steps=VAL_STEPS_DET,
    callbacks=[
        ckpt_det,
        ReduceLROnPlateau(monitor='val_bit_err', factor=0.5, patience=20, mode='min', verbose=1),
        EarlyStopping(monitor='val_bit_err', patience=60, restore_best_weights=True, mode='min', verbose=1),
        pcvl_cb,
    ],
    verbose=2
)
detector.load_weights(os.path.join(OUTDIR, 'det_best.weights.h5'))

# -------------------------
# Save & plot: TRAINING ACCURACY (detector) and VALIDATION LOSS (mixed@20dB)
# -------------------------
epochs_axis    = np.arange(1, len(hist_det.history['loss']) + 1, dtype=np.int32)
train_acc_curve = np.array(hist_det.history.get('bit_acc', []), dtype=np.float32)
val_loss_curve  = np.array(hist_det.history.get('val_loss', []), dtype=np.float32)

final_train_acc = float(train_acc_curve[-1]) if train_acc_curve.size else np.nan
best_train_acc  = float(np.max(train_acc_curve)) if train_acc_curve.size else np.nan
best_epoch      = int(1 + np.argmax(train_acc_curve)) if train_acc_curve.size else -1

# PNG: Training accuracy
plt.figure(figsize=(8,6))
plt.plot(epochs_axis, train_acc_curve, 'o-')
plt.xlabel('Epoch'); plt.ylabel('Training Accuracy (Detector bit-acc @ 20 dB)')
plt.title('Universal DNN — Training Accuracy vs Epoch (16-QAM, 32 pilots, SNR=20 dB)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'Universal_TrainingAccuracy_vs_Epoch.png'), dpi=300)
plt.close()

# PNG: Validation loss (mixed@20 dB)
plt.figure(figsize=(8,6))
plt.plot(epochs_axis, val_loss_curve, 's-')
plt.xlabel('Epoch'); plt.ylabel('Validation Loss (MSE, SNR=20 dB)')
plt.title('Universal DNN — Validation Loss vs Epoch (16-QAM, 32 pilots)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'Universal_ValLossMixed_vs_Epoch.png'), dpi=300)
plt.close()

# MAT: curves + summary
sio.savemat(os.path.join(OUTDIR, 'Universal_TrainAcc_ValLoss_Mixed.mat'),
            dict(
                epochs=epochs_axis,
                train_acc=train_acc_curve,
                val_loss_mixed=val_loss_curve,
                final_train_acc=final_train_acc,
                best_train_acc=best_train_acc,
                best_epoch=best_epoch,
                info="16QAM payload+pilots, 32 pilots, SNR=20 dB for training/validation"
            ))

# -------------------------
# Save & plot: VALIDATION LOSS per-channel (one figure + MAT)
# -------------------------
plt.figure(figsize=(10,7))
for name, losses in pcvl_cb.history.items():
    y = np.array(losses, dtype=np.float32)
    plt.plot(epochs_axis[:len(y)], y, '-o', label=name)
plt.xlabel('Epoch'); plt.ylabel('Validation Loss (MSE, SNR=20 dB)')
plt.title('Universal DNN — Validation Loss by Channel (16-QAM, 32 pilots)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'Universal_ValLoss_PerChannel.png'), dpi=300)
plt.close()

sio.savemat(os.path.join(OUTDIR, 'Universal_ValLoss_PerChannel.mat'),
            dict(epochs=epochs_axis,
                 **{f'val_loss_{k.lower()}': np.array(v, dtype=np.float32)
                    for k, v in pcvl_cb.history.items()}))

# -------------------------
# BER evaluation (unchanged)
# -------------------------
def compute_ber(model, channel, snr_db, total=TEST_SAMPLES):
    chunk = 2048
    n_done, bit_errs, n_bits = 0, 0, 0
    while n_done < total:
        bs = min(chunk, total - n_done)
        X = np.empty((bs, IN_FEAT_CSI), np.float32)
        Y = np.empty((bs, payloadBits_per_OFDM), np.float32)
        for i in range(bs):
            feat, bits = make_rx_and_bits(snr_db, channel, mode="eval", return_bits=True)
            X[i] = feat; Y[i] = bits
        y_prob = model.predict(X, verbose=0)
        y_hat  = (y_prob >= 0.5).astype(np.float32)
        bit_errs += np.sum(np.abs(Y - y_hat))
        n_bits   += Y.size
        n_done   += bs
    return bit_errs / n_bits

BER = {ch: [] for ch in CHANNELS}
for ch in CHANNELS:
    print(f"\n[Eval] Channel: {ch}")
    for snr in TEST_SNRS:
        ber = compute_ber(detector, ch, snr, total=TEST_SAMPLES)
        BER[ch].append(float(ber))
        print(f"SNR={snr:>2} dB -> BER={ber:.6f}")

mat_path = os.path.join(OUTDIR, 'BER_vs_SNR_ProfFixed.mat')
npz_path = os.path.join(OUTDIR, 'BER_vs_SNR_ProfFixed.npz')
sio.savemat(mat_path, {'SNRs': np.array(TEST_SNRS, float),
                       **{f'BER_{ch}': np.array(BER[ch], float) for ch in CHANNELS}})
np.savez(npz_path, SNRs=np.array(TEST_SNRS, float),
         **{f'ber_{ch}': np.array(BER[ch], float) for ch in CHANNELS})

plt.figure(figsize=(10,7))
marks = ['o-','s-','^-','d-']
for i, ch in enumerate(CHANNELS):
    plt.semilogy(TEST_SNRS, BER[ch], marks[i], label=ch)
plt.xlabel("SNR (dB)"); plt.ylabel("Bit Error Rate (BER)")
plt.title("UniversalDNN — BER vs SNR (16-QAM, 32 pilots)")
plt.grid(True, which='both', linestyle='--', alpha=0.7)
plt.legend(loc='lower left'); plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'BER_vs_SNR_All.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUTDIR, 'BER_vs_SNR_All.pdf'), dpi=300, bbox_inches='tight')
plt.close()

for ch in CHANNELS:
    plt.figure(figsize=(7,5))
    plt.semilogy(TEST_SNRS, BER[ch], 'o-')
    plt.xlabel("SNR (dB)"); plt.ylabel("Bit Error Rate (BER)")
    plt.title(f"BER vs SNR — {ch} (16-QAM, 32 pilots)")
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, f'BER_vs_SNR_{ch}.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTDIR, f'BER_vs_SNR_{ch}.pdf'), dpi=300, bbox_inches='tight')
    plt.close()

# MATLAB helper (for the BER mat)
mfile = os.path.join(OUTDIR, 'plot_ber_prof_fixed.m')
with open(mfile, 'w') as f:
    f.write(f"""\
function plot_ber_prof_fixed()
  S = load('{os.path.basename(mat_path)}'); SNRs = S.SNRs;
  chans = {{{', '.join(["'%s'"%c for c in CHANNELS])}}};
  ber = {{{', '.join(['S.BER_'+c for c in CHANNELS])}}};
  f = figure('Color','w'); ax = axes('Parent',f); hold(ax,'on'); grid(ax,'on'); set(ax,'YScale','log');
  xlabel(ax,'SNR (dB)'); ylabel(ax,'Bit Error Rate (BER)'); title(ax,'Architecture A — BER vs SNR (16-QAM, 32 pilots)');
  marks = {{'o-','s-','^-','d-'}};
  for i = 1:numel(chans)
    semilogy(ax, SNRs, ber{{i}}, marks{{i}}, 'LineWidth',1.5, 'MarkerSize',7, 'DisplayName', chans{{i}});
  end
  legend(ax,'Location','southwest'); box(ax,'on');
end
""")
print("Wrote MATLAB helper:", mfile)
# ====================== end of file ======================
