# mainray.py — Conventional DNN (mixed train/val) + TRAIN ACC / VAL LOSS saves
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, Callback
from globalparametersray import payloadBits_per_OFDM
from generationsray import training_gen, validation_gen

# --------- where to save ----------
OUTDIR = "./conventional_results"
os.makedirs(OUTDIR, exist_ok=True)

# --------- metrics ----------
def bit_err(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    err = 1 - tf.reduce_mean(
        tf.reduce_mean(
            tf.cast(tf.equal(tf.sign(y_pred - 0.5), tf.sign(y_true - 0.5)), tf.float32),
            axis=1
        )
    )
    return err

def bit_acc(y_true, y_pred):  # overall training accuracy (bit-wise)
    return 1.0 - bit_err(y_true, y_pred)

def build_model():
    input_layer = Input(shape=(payloadBits_per_OFDM * 2,))
    x = Dense(512, activation='relu')(input_layer)
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    output = Dense(payloadBits_per_OFDM, activation='sigmoid')(x)
    model = Model(input_layer, output)
    model.compile(optimizer='adam', loss='mse', metrics=[bit_err, bit_acc])
    return model

# ---- Per-channel validation loss curves ----
class MultiChannelValLoss(Callback):
    def __init__(self, val_sets: dict, steps_per_val: int = 10):
        super().__init__()
        self.val_sets = val_sets
        self.steps_per_val = steps_per_val
        self.history = {k: [] for k in val_sets.keys()}
    def on_epoch_end(self, epoch, logs=None):
        for name, ds in self.val_sets.items():
            res = self.model.evaluate(ds, steps=self.steps_per_val, verbose=0)  # [loss, bit_err, bit_acc]
            self.history[name].append(float(res[0]))  # store LOSS

# ---------- Training settings ----------
train_snr = 20  # fixed 20 dB scenario for training accuracy curve
print(f"\nTraining model at SNR = {train_snr} dB")
model = build_model()
checkpoint_path = os.path.join(OUTDIR, f'model_mixed_{train_snr}dB.weights.h5')

checkpoint = ModelCheckpoint(
    checkpoint_path,
    monitor='val_bit_err',
    save_best_only=True,
    save_weights_only=True,
    mode='min',
    verbose=1
)

callbacks = [
    checkpoint,
    ReduceLROnPlateau(monitor='val_bit_err', factor=0.5, patience=5, verbose=1, mode='min'),
    EarlyStopping(monitor='val_bit_err', patience=10, restore_best_weights=True, mode='min')
]

# ---------- Train/Validate on MIXED (but SNR fixed at 20 dB) ----------
train_dataset = tf.data.Dataset.from_generator(
    lambda: training_gen(100000, train_snr, channel_type='mixed'),  # 20 dB
    output_signature=(
        tf.TensorSpec(shape=(None, payloadBits_per_OFDM * 2), dtype=tf.float32),
        tf.TensorSpec(shape=(None, payloadBits_per_OFDM), dtype=tf.float32)
    )
)

val_dataset_mixed = tf.data.Dataset.from_generator(
    lambda: validation_gen(30000, train_snr, channel_type='mixed'),  # 20 dB
    output_signature=(
        tf.TensorSpec(shape=(None, payloadBits_per_OFDM * 2), dtype=tf.float32),
        tf.TensorSpec(shape=(None, payloadBits_per_OFDM), dtype=tf.float32)
    )
)

# Per-channel datasets (20 dB)
val_dataset_rayleigh = tf.data.Dataset.from_generator(
    lambda: validation_gen(30000, train_snr, channel_type='rayleigh'),
    output_signature=(
        tf.TensorSpec(shape=(None, payloadBits_per_OFDM * 2), dtype=tf.float32),
        tf.TensorSpec(shape=(None, payloadBits_per_OFDM), dtype=tf.float32)
    )
)
val_dataset_rician = tf.data.Dataset.from_generator(
    lambda: validation_gen(30000, train_snr, channel_type='rician'),
    output_signature=(
        tf.TensorSpec(shape=(None, payloadBits_per_OFDM * 2), dtype=tf.float32),
        tf.TensorSpec(shape=(None, payloadBits_per_OFDM), dtype=tf.float32)
    )
)
val_dataset_wiener = tf.data.Dataset.from_generator(
    lambda: validation_gen(30000, train_snr, channel_type='wiener'),
    output_signature=(
        tf.TensorSpec(shape=(None, payloadBits_per_OFDM * 2), dtype=tf.float32),
        tf.TensorSpec(shape=(None, payloadBits_per_OFDM), dtype=tf.float32)
    )
)
val_dataset_awgn = tf.data.Dataset.from_generator(
    lambda: validation_gen(30000, train_snr, channel_type='awgn'),
    output_signature=(
        tf.TensorSpec(shape=(None, payloadBits_per_OFDM * 2), dtype=tf.float32),
        tf.TensorSpec(shape=(None, payloadBits_per_OFDM), dtype=tf.float32)
    )
)

multi_val_cb = MultiChannelValLoss(
    val_sets={"Rayleigh": val_dataset_rayleigh, "Rician": val_dataset_rician,
              "Wiener": val_dataset_wiener, "AWGN": val_dataset_awgn},
    steps_per_val=10
)
callbacks.append(multi_val_cb)

# ---------- Train ----------
hist = model.fit(
    train_dataset,
    steps_per_epoch=64,
    epochs=20,
    validation_data=val_dataset_mixed,
    validation_steps=15,
    callbacks=callbacks,
    verbose=2
)

# ---------- Save training accuracy & validation loss curves ----------
epochs_axis = np.arange(1, len(hist.history['loss']) + 1)
train_acc_curve = np.array(hist.history.get('bit_acc', []), dtype=np.float32)
val_loss_curve  = np.array(hist.history.get('val_loss', []), dtype=np.float32)

plt.figure(figsize=(8,6))
plt.plot(epochs_axis, train_acc_curve, 'o-')
plt.xlabel('Epoch'); plt.ylabel('Training Accuracy (bit-acc, 20 dB)')
plt.title('Conventional DNN — Training Accuracy vs Epoch (20 dB)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'Conventional_TrainingAccuracy_vs_Epoch.png'), dpi=300)
plt.close()

plt.figure(figsize=(8,6))
plt.plot(epochs_axis, val_loss_curve, 's-')
plt.xlabel('Epoch'); plt.ylabel('Validation Loss (MSE, mixed @ 20 dB)')
plt.title('Conventional DNN — Validation Loss (Mixed) vs Epoch (20 dB)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'Conventional_ValLossMixed_vs_Epoch.png'), dpi=300)
plt.close()

sio.savemat(os.path.join(OUTDIR, 'Conventional_TrainAcc_ValLoss_Mixed.mat'),
            dict(epochs=epochs_axis, train_acc=train_acc_curve, val_loss_mixed=val_loss_curve))

# ---------- Per-channel validation loss figure ----------
plt.figure(figsize=(10, 7))
for name, losses in multi_val_cb.history.items():
    xs = np.arange(1, len(losses) + 1)
    plt.plot(xs, losses, '-o', label=name)
plt.xlabel('Epoch'); plt.ylabel('Validation Loss (MSE)')
plt.title('Validation Loss by Channel (20 dB)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'Conventional_ValLoss_PerChannel.png'), dpi=300)
plt.close()

sio.savemat(os.path.join(OUTDIR, 'Conventional_ValLoss_PerChannel.mat'),
            dict(epochs=np.arange(1, 1 + max(len(v) for v in multi_val_cb.history.values())),
                 **{f'val_loss_{k.lower()}': np.array(v, dtype=np.float32)
                    for k, v in multi_val_cb.history.items()}))

# ---------- Test on ALL FOUR channels ----------
model.load_weights(checkpoint_path)
test_SNRs = list(range(0, 21, 5))
channels = ['rayleigh', 'rician', 'wiener', 'awgn']
BER_results = {ch: [] for ch in channels}

for ch_type in channels:
    print(f"\nTesting on channel: {ch_type}")
    for snr in test_SNRs:
        y = model.evaluate(
            validation_gen(100000, snr, channel_type=ch_type),
            steps=1,
            verbose=0
        )
        BER_results[ch_type].append(y[1])  # bit_err
        print(f"SNR={snr} dB, BER={y[1]:.6f}")

mat_obj = {
    'SNRs': np.array(test_SNRs),
    'BER_rayleigh': np.array(BER_results['rayleigh']),
    'BER_rician':   np.array(BER_results['rician']),
    'BER_wiener':   np.array(BER_results['wiener']),
    'BER_awgn':     np.array(BER_results['awgn']),
}
sio.savemat(os.path.join(OUTDIR, 'BER_vs_SNR_allChannels.mat'), mat_obj)
sio.savemat('BER_vs_SNR_allChannels.mat', mat_obj)

plt.figure(figsize=(10, 7))
plt.semilogy(test_SNRs, BER_results['rayleigh'], 'o-', label='Rayleigh')
plt.semilogy(test_SNRs, BER_results['rician'],   's-', label='Rician')
plt.semilogy(test_SNRs, BER_results['wiener'],   '^-', label='Wiener')
plt.semilogy(test_SNRs, BER_results['awgn'],     'd-', label='AWGN')
plt.xlabel("SNR (dB)"); plt.ylabel("Bit Error Rate (BER)")
plt.title("BER vs SNR (Conventional DNN, train/val @ 20 dB)")
plt.grid(True, which='both', linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "BER_vs_SNR_Curve.png"), dpi=300)
plt.savefig("BER_vs_SNR_Curve.png", dpi=300)
plt.close()
