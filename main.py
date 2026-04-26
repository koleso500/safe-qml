import copy
import random
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pennylane as qml
import torch
import torch.nn as nn
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models, transforms

from safe.rge import compare_models_rge
from safe.rgr import compare_models_rgr, compare_models_rgr_adversarial, compare_models_rgr_spatial_images
from safe.rga import compare_models_rga
from safe.utils import (
    CroppedImage,
    crop_img,
    compute_gradcam_maps,
    precompute_patch_rankings,
    train_cam_model,
    extract_features_from_images,
    align_proba_to_class_order, show_heatmap_per_class, show_occlusions_same_idx,
)


# ----------------------------
# Reproducibility and paths
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

config_path = PROJECT_ROOT / 'config.yaml'
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

SEED = int(config['seed'])

DATA_DIR = PROJECT_ROOT / config['paths']['data_dir']
CSV_DIR  = PROJECT_ROOT / config['paths']['csv_dir']
FIG_DIR  = PROJECT_ROOT / config['paths']['fig_dir']

CSV_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

print('Dataset exists:', DATA_DIR.exists())
if not DATA_DIR.exists():
    raise FileNotFoundError(
        f'Dataset not found at {DATA_DIR}. '
        'Place the dataset to the configured data_dir.'
    )

IMG_SIZE = int(config['data']['img_size'])

N_SPLITS = int(config['cross_validation']['n_splits'])

EPOCHS = int(config['training']['epochs'])
BATCH_SIZE_TRAIN = int(config['training']['batch_size'])
LR = float(config['training']['learning_rate'])

BATCH_SIZE_IMAGES = int(config['data']['batch_size_images'])
BATCH_SIZE_SAFE = int(config['safe']['batch_size_safe'])

N_QUBITS = int(config['quantum']['n_qubits'])
N_LAYERS = int(config['quantum']['n_layers'])

N_SEGMENTS = int(config['safe']['rga']['n_segments'])
PATCH_SIZE = int(config['safe']['rge']['patch_size'])

CAM_EPOCHS = int(config['gradcam']['epochs'])
CAM_LR = float(config['gradcam']['learning_rate'])
CAM_BATCH_SIZE = int(config['gradcam']['batch_size'])

# SAFE grids
def grid(start: float, end: float, step: float):
    n = int(round((end - start) / step)) + 1
    return start + step * np.arange(n, dtype=float)

noise_levels = grid(
    float(config['safe']['rgr']['noise_start']),
    float(config['safe']['rgr']['noise_end']),
    float(config['safe']['rgr']['noise_step']),
)

removal_fractions = grid(
    float(config['safe']['rge']['removal_start']),
    float(config['safe']['rge']['removal_end']),
    float(config['safe']['rge']['removal_step']),
)

attack_strengths = grid(0.0, 0.3, 0.05)
spatial_strengths = np.array([0, 5, 10, 15, 20, 25])

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

def save_and_close(fig_path: Path, dpi: int = 300):
    plt.savefig(fig_path, dpi=dpi, bbox_inches='tight')
    plt.close()


# ----------------------------
# Dataset and transforms
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

dataset = CroppedImage(DATA_DIR, transform=transform, apply_crop=True)
loader = DataLoader(dataset, batch_size=BATCH_SIZE_IMAGES, shuffle=False)

class_names = dataset.classes
n_classes = len(class_names)


# -------------------------------------
# Test cropping (one sample per class)
# -------------------------------------
fig, axes = plt.subplots(n_classes, 2, figsize=(10, 5 * n_classes))

if n_classes == 1:
    axes = axes.reshape(1, -1)

for class_idx, class_name in enumerate(class_names):
    class_images = [s for s in dataset.dataset.samples if s[1] == class_idx]
    img_path, label = class_images[np.random.randint(len(class_images))]

    img_bgr = cv2.imread(img_path)
    cropped_bgr = crop_img(img_bgr)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)

    axes[class_idx, 0].imshow(img_rgb)
    axes[class_idx, 0].set_title(f'{class_name} - Original')
    axes[class_idx, 0].axis('off')

    axes[class_idx, 1].imshow(cropped_rgb)
    axes[class_idx, 1].set_title(f'{class_name} - Cropped')
    axes[class_idx, 1].axis('off')

plt.tight_layout()
save_and_close(FIG_DIR / 'cropping_examples.png')


# ----------------------------------------
# ResNet18 feature extraction and scaling
# ----------------------------------------
resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
resnet.fc = nn.Identity()
resnet = resnet.to(device).eval()

features, labels = [], []

with torch.no_grad():
    for x, y in loader:
        x = x.to(device)
        f = resnet(x)
        features.append(f.cpu())
        labels.append(y)

features = torch.cat(features).numpy()
labels = torch.cat(labels).numpy()

print('Feature shape:', features.shape)

scaler = StandardScaler()
x = scaler.fit_transform(features)
y = labels


# ----------------------------
# Models
# ----------------------------
def build_rf():
    return RandomForestClassifier(
        n_estimators=300,
        random_state=SEED,
        n_jobs=-1
    )

def build_svm():
    return SVC(
        kernel='rbf',
        C=10,
        gamma='scale',
        probability=True,
        random_state=SEED
    )

class MLPBaseline(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.out = nn.Linear(input_dim, num_classes)
    def forward(self, inputs):
        inputs = torch.functional.F.gelu(self.fc1(inputs))
        return self.out(inputs)

def build_mlp(input_dim, num_classes):
    return MLPBaseline(input_dim, num_classes).to(device)

def build_linear():
    return LogisticRegression(
        l1_ratio=0,
        C=1.0,
        solver='lbfgs',
        max_iter=500
    )

# ----------------------------
# VQC (Amplitude Encoding)
# ----------------------------
dev = qml.device('default.qubit', wires=N_QUBITS)

@qml.qnode(dev, interface='torch')
def vqc_circuit(inputs, weights):
    qml.AmplitudeEmbedding(inputs, wires=range(N_QUBITS), normalize=True)
    for _ in range(N_LAYERS):
        qml.templates.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

weight_shapes = {'weights': (N_LAYERS, N_QUBITS, 3)}

class QuantumClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.pre_pqc = nn.Linear(input_dim, input_dim)
        self.q_layer = qml.qnn.TorchLayer(vqc_circuit, weight_shapes)
        self.fc = nn.Linear(N_QUBITS, num_classes)

    def forward(self, inputs):
        inputs = self.pre_pqc(inputs)
        inputs = torch.functional.F.gelu(inputs)
        inputs = self.q_layer(inputs)
        return self.fc(inputs)


# ============================================================
# Cross-validation
# ============================================================
def compute_metrics(y_true, probs):
    preds = np.argmax(probs, axis=1)
    acc_value = accuracy_score(y_true, preds)
    f1_value  = f1_score(y_true, preds, average='macro')
    mse_value = mean_squared_error(
        np.eye(probs.shape[1])[y_true],
        probs
    )
    return acc_value, f1_value, mse_value

k_fold = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

results = {
    'Linear': {'acc': [], 'f1': [], 'mse': []},
    'RF': {'acc': [], 'f1': [], 'mse': []},
    'SVM': {'acc': [], 'f1': [], 'mse': []},
    'MLP': {'acc': [], 'f1': [], 'mse': []},
    'QML': {'acc': [], 'f1': [], 'mse': []},
}

linear_models, rf_models, svm_models, mlp_models, qml_models = [], [], [], [], []

in_dim = x.shape[1]

for fold, (train_idx, val_idx) in enumerate(k_fold.split(x, y), 1):
    print(f'\n====== Fold {fold} ======')

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    x_train, x_val = x[train_idx], x[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Linear
    linear = build_linear()
    linear.fit(x_train, y_train)

    probs_linear = linear.predict_proba(x_val)
    acc, f1, mse = compute_metrics(y_val, probs_linear)
    results['Linear']['acc'].append(acc)
    results['Linear']['f1'].append(f1)
    results['Linear']['mse'].append(mse)

    print(f'LIN | ACC={acc:.4f}  F1={f1:.4f}  MSE={mse:.6f}')
    linear_models.append(linear)

    # RF
    rf = build_rf()
    rf.fit(x_train, y_train)

    probs_rf = rf.predict_proba(x_val)
    acc, f1, mse = compute_metrics(y_val, probs_rf)
    results['RF']['acc'].append(acc)
    results['RF']['f1'].append(f1)
    results['RF']['mse'].append(mse)

    print(f'RF | ACC={acc:.4f}  F1={f1:.4f}  MSE={mse:.6f}')
    rf_models.append(rf)

    # SVM
    svm = build_svm()
    svm.fit(x_train, y_train)

    probs_svm = svm.predict_proba(x_val)
    acc, f1, mse = compute_metrics(y_val, probs_svm)
    results['SVM']['acc'].append(acc)
    results['SVM']['f1'].append(f1)
    results['SVM']['mse'].append(mse)

    print(f'SVM | ACC={acc:.4f}  F1={f1:.4f}  MSE={mse:.6f}')
    svm_models.append(svm)

    # Torch loaders for MLP and QML
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(x_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        ),
        batch_size=BATCH_SIZE_TRAIN,
        shuffle=True,
    )

    def train_torch_model(mod):
        opt = torch.optim.Adam(mod.parameters(), lr=LR)
        loss_fn = nn.CrossEntropyLoss()

        best_val = float('inf')
        best_state = None

        for epoch in range(EPOCHS):
            mod.train()
            running = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = loss_fn(mod(xb), yb)
                loss.backward()
                opt.step()
                running += loss.item()

            train_loss = running / max(len(train_loader), 1)

            mod.eval()
            with torch.no_grad():
                xv = torch.tensor(x_val, dtype=torch.float32).to(device)
                yv = torch.tensor(y_val, dtype=torch.long).to(device)
                val_loss = loss_fn(mod(xv), yv).item()

            if val_loss < best_val:
                best_val = val_loss
                best_state = copy.deepcopy(mod.state_dict())

            print(f'Epoch {epoch + 1:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')

        if best_state is not None:
            mod.load_state_dict(best_state)

        return mod

    # MLP
    mlp = build_mlp(in_dim, n_classes)
    mlp = train_torch_model(mlp)

    mlp.eval()
    with torch.no_grad():
        probs_mlp = torch.softmax(
            mlp(torch.tensor(x_val, dtype=torch.float32).to(device)),
            dim=1,
        ).cpu().numpy()

    acc, f1, mse = compute_metrics(y_val, probs_mlp)
    results['MLP']['acc'].append(acc)
    results['MLP']['f1'].append(f1)
    results['MLP']['mse'].append(mse)

    print(f'MLP | ACC={acc:.4f}  F1={f1:.4f}  MSE={mse:.6f}')
    mlp_models.append(copy.deepcopy(mlp))

    # QML
    qml_model = QuantumClassifier(in_dim, n_classes).to(device)
    qml_model = train_torch_model(qml_model)

    qml_model.eval()
    with torch.no_grad():
        probs_qml = torch.softmax(
            qml_model(torch.tensor(x_val, dtype=torch.float32).to(device)),
            dim=1,
        ).cpu().numpy()

    acc, f1, mse = compute_metrics(y_val, probs_qml)
    results['QML']['acc'].append(acc)
    results['QML']['f1'].append(f1)
    results['QML']['mse'].append(mse)

    print(f'QML | ACC={acc:.4f}  F1={f1:.4f}  MSE={mse:.6f}')
    qml_models.append(copy.deepcopy(qml_model))

print('\n================ OVERALL 5-FOLD ================')

for model, m in results.items():
    print(
        f"{model:>3} | "
        f"ACC={np.mean(m['acc']):.4f} "
        f"F1={np.mean(m['f1']):.4f} "
        f"MSE={np.mean(m['mse']):.6f}"
    )


# ----------------------------
# SAFE metrics (RGA/RGR/RGE)
# ----------------------------
MODEL_NAMES = ['Linear', 'RF', 'SVM', 'MLP', 'QML']

def init_safe_store(model_names):
    return {
        n: {
            'rga_curve': [],
            'rgr_curve': [],
            'rge_curve': [],
            'rgr_fgsm_curve': [],
            'rgr_spatial_curve': [],
            'rga_full': [],
            'aurga': [],
            'aurgr': [],
            'aurge': [],
            'aurgr_fgsm': [],
            'aurgr_spatial': [],
        }
        for n in model_names
    }

safe_store = init_safe_store(MODEL_NAMES)

class_order = np.unique(labels)

x_t = torch.tensor(x, dtype=torch.float32)
y_labels = labels

x_images = torch.stack([img for img, _ in dataset])
x_images_dataset = TensorDataset(x_images)

cam_model = train_cam_model(
    feature_extractor=resnet,
    images=x_images,
    labels=labels,
    scaler=scaler,
    n_classes=n_classes,
    device=device,
    epochs=CAM_EPOCHS,
    lr=CAM_LR,
    batch_size=CAM_BATCH_SIZE,
    verbose=True,
)

importance_maps = compute_gradcam_maps(
    images=x_images,
    cam_model=cam_model,
    device=device,
    batch_pred=CAM_BATCH_SIZE,
    verbose=True,
)

patch_rankings, patch_meta = precompute_patch_rankings(
    importance_maps=importance_maps,
    patch_size=PATCH_SIZE,
)

print('Total patches:', patch_meta['total_patches'])

# Heatmap and occlusions plots
idx = 0

show_heatmap_per_class(x_images, importance_maps, labels, class_names, n_classes,
                       save_path=FIG_DIR / 'gradcam_heatmap_per_class.png')

show_occlusions_same_idx(x_images, patch_rankings, patch_meta, idx=idx,
                         save_path=FIG_DIR / 'gradcam_occlusions.png')


# RGE preprocessing function
def preprocess(images):
    return extract_features_from_images(
        images,
        feature_extractor=resnet,
        pca=None,
        scaler=scaler,
        device=device,
        batch_size=BATCH_SIZE_IMAGES
    )

# Per-fold SAFE metrics
safe_fig_dir = FIG_DIR / 'safe_folds'
safe_fig_dir.mkdir(parents=True, exist_ok=True)

for fold, (linear, rf, svm, mlp, q_model) in enumerate(
        zip(linear_models, rf_models, svm_models, mlp_models, qml_models), 1):

    fold_dir = safe_fig_dir / f'fold_{fold:02d}'
    fold_dir.mkdir(parents=True, exist_ok=True)

    rga_path = fold_dir / 'rga.png'
    rgr_path = fold_dir / 'rgr.png'
    rge_path = fold_dir / 'rge.png'

    print(f'\n====== SAFE AI METRICS Fold {fold} ======')

    mlp.eval()
    q_model.eval()

    with torch.no_grad():
        prob_base = torch.softmax(mlp(x_t.to(device)), dim=1).cpu().numpy()
        prob_qml = torch.softmax(q_model(x_t.to(device)), dim=1).cpu().numpy()

    prob_lin = linear.predict_proba(x)
    prob_rf = rf.predict_proba(x)
    prob_svm = svm.predict_proba(x)

    prob_base = align_proba_to_class_order(prob_base, class_order, class_order)
    prob_qml = align_proba_to_class_order(prob_qml, class_order, class_order)
    prob_lin = align_proba_to_class_order(prob_lin, linear.classes_, class_order)
    prob_rf = align_proba_to_class_order(prob_rf, rf.classes_, class_order)
    prob_svm = align_proba_to_class_order(prob_svm, svm.classes_, class_order)

    # RGA
    models_rga = {
        'Linear': (prob_lin, class_order),
        'RF': (prob_rf, class_order),
        'SVM': (prob_svm, class_order),
        'MLP': (prob_base, class_order),
        'QML': (prob_qml, class_order),
    }

    results_rga = compare_models_rga(
        models_rga,
        y_labels=y_labels,
        n_segments=N_SEGMENTS,
        fig_size=(8, 6),
        verbose=True,
        save_path=rga_path
    )

    rga_dict = {m: float(results_rga[m]['rga_full']) for m in MODEL_NAMES}

    for m in MODEL_NAMES:
        safe_store[m]['rga_curve'].append(np.asarray(results_rga[m]['curve_model'], float))
        safe_store[m]['rga_full'].append(float(results_rga[m]['rga_full']))
        safe_store[m]['aurga'].append(float(results_rga[m]['aurga']))

    # RGR
    models_rgr = {
        'Linear': (linear, x, prob_lin, class_order, 'sklearn', None),
        'RF': (rf, x, prob_rf, class_order, 'sklearn', None),
        'SVM': (svm, x, prob_svm, class_order, 'sklearn', None),
        'MLP': (mlp, x_t, prob_base, class_order, 'pytorch', device),
        'QML': (q_model, x_t, prob_qml, class_order, 'pytorch', device),
    }

    results_rgr = compare_models_rgr(
        models_dict=models_rgr,
        noise_levels=noise_levels,
        class_order=class_order,
        rga_dict=rga_dict,
        fig_size=(8, 6),
        verbose=True,
        random_seed=SEED,
        save_path=rgr_path
    )

    for m in MODEL_NAMES:
        safe_store[m]['rgr_curve'].append(
            np.asarray(results_rgr[m]['rgr_rescaled'], float)
        )
        safe_store[m]['aurgr'].append(float(results_rgr[m]['aurgr']))

    models_rgr_fgsm = {
        'Linear': (linear, x, prob_lin, class_order, 'sklearn', None),
        'MLP': (mlp, x_t, prob_base, class_order, 'pytorch', device),
        'QML': (q_model, x_t, prob_qml, class_order, 'pytorch', device),
    }

    results_rgr_fgsm = compare_models_rgr_adversarial(
        models_dict=models_rgr_fgsm,
        attack_strengths=attack_strengths,
        class_order=class_order,
        y_true_dict={
            'Linear': y_labels,
            'MLP': y_labels,
            'QML': y_labels,
        },
        attack_name='fgsm',
        rga_dict={k: rga_dict[k] for k in models_rgr_fgsm},
        fig_size=(8, 6),
        verbose=True,
        base_attack_params={'norm': np.inf},
        save_path=fold_dir / 'rgr_fgsm.png'
    )

    _, spatial_idx = train_test_split(
        np.arange(len(y_labels)),
        test_size=100,
        stratify=y_labels,
        random_state=SEED + fold,
    )

    x_images_spatial = x_images[spatial_idx]
    y_spatial = y_labels[spatial_idx]

    prob_lin_spatial = prob_lin[spatial_idx]
    prob_rf_spatial = prob_rf[spatial_idx]
    prob_svm_spatial = prob_svm[spatial_idx]
    prob_base_spatial = prob_base[spatial_idx]
    prob_qml_spatial = prob_qml[spatial_idx]

    models_rgr_spatial = {
        'Linear': (linear, prob_lin_spatial, class_order, 'sklearn', None),
        'RF': (rf, prob_rf_spatial, class_order, 'sklearn', None),
        'SVM': (svm, prob_svm_spatial, class_order, 'sklearn', None),
        'MLP': (mlp, prob_base_spatial, class_order, 'pytorch', device),
        'QML': (q_model, prob_qml_spatial, class_order, 'pytorch', device),
    }

    results_rgr_spatial = compare_models_rgr_spatial_images(
        models_dict=models_rgr_spatial,
        images=x_images_spatial,
        y_true=y_spatial,
        attack_model=cam_model,
        preprocess_fn=preprocess,
        attack_strengths=spatial_strengths,
        class_order=class_order,
        rga_dict=rga_dict,
        device=device,
        fig_size=(8, 6),
        verbose=True,
        save_path=fold_dir / 'rgr_spatial.png',
        num_translations=3,
        num_rotations=3,
    )

    for m in MODEL_NAMES:
        if m in results_rgr_fgsm:
            safe_store[m]['rgr_fgsm_curve'].append(np.asarray(results_rgr_fgsm[m]['rgr_rescaled'], float))
            safe_store[m]['aurgr_fgsm'].append(float(results_rgr_fgsm[m]['aurgr']))
        else:
            safe_store[m]['rgr_fgsm_curve'].append(np.full(len(attack_strengths), np.nan))
            safe_store[m]['aurgr_fgsm'].append(np.nan)

        safe_store[m]['rgr_spatial_curve'].append(np.asarray(results_rgr_spatial[m]['rgr_rescaled'], float))
        safe_store[m]['aurgr_spatial'].append(float(results_rgr_spatial[m]['aurgr']))

    # RGE
    models_rge = {
        'Linear': (linear, preprocess, class_order, 'sklearn'),
        'RF': (rf, preprocess, class_order, 'sklearn'),
        'SVM': (svm, preprocess, class_order, 'sklearn'),
        'MLP': (mlp, preprocess, class_order, 'pytorch'),
        'QML': (q_model, preprocess, class_order, 'pytorch'),
    }

    results_rge = compare_models_rge(
        models_dict=models_rge,
        images_dataset=x_images_dataset,
        removal_fractions=removal_fractions,
        class_order=class_order,
        occlusion_method='gradcam_most',
        patch_size=PATCH_SIZE,
        batch_size=BATCH_SIZE_SAFE,
        class_weights=None,
        rga_dict=rga_dict,
        device=device,
        verbose=True,
        random_seed=SEED,
        patch_rankings=patch_rankings,
        patch_meta=patch_meta,
        save_path=rge_path,
        use_shared_feature_cache=True
    )

    for m in MODEL_NAMES:
        safe_store[m]['rge_curve'].append(np.asarray(results_rge[m]['rge_rescaled'], float))
        safe_store[m]['aurge'].append(float(results_rge[m]['aurge']))

    fold_rows = []
    for m in MODEL_NAMES:
        fold_rows.append({
            'fold': fold,
            'model': m,
            'RGA': float(results_rga[m]['rga_full']),
            'AURGA': float(results_rga[m]['aurga']),
            'AURGR': float(results_rgr[m]['aurgr']),
            'AURGR_FGSM': float(results_rgr_fgsm[m]['aurgr']) if m in results_rgr_fgsm else np.nan,
            'AURGR_SPATIAL': float(results_rgr_spatial[m]['aurgr']),
            'AURGE': float(results_rge[m]['aurge']),
        })

    df_fold = pd.DataFrame(fold_rows)
    df_fold.to_csv(fold_dir / 'safe_metrics_fold.csv', index=False)

# ---------------------------------------
# RGX plots, SAFE results and radar plot
# ---------------------------------------
def plot_mean(inputs, safe_dict, curve_key, title, x_label, y_label, save_path):
    plt.figure(figsize=(8, 6))
    for mod in MODEL_NAMES:
        curves = safe_dict[mod][curve_key]
        a = np.stack(curves, axis=0)
        mu = np.nanmean(a, axis=0)
        plt.plot(inputs, mu, label=mod)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(alpha=0.3, linestyle='--')
    plt.legend()
    plt.tight_layout()
    save_and_close(save_path)

l_rga = len(safe_store['MLP']['rga_curve'][0])
x_rga = np.linspace(0, 1, l_rga)

plot_mean(
    x_rga, safe_store, 'rga_curve',
    title='RGA (mean across 5 folds)',
    x_label='Fraction of Data Removed',
    y_label='RGA Score',
    save_path=FIG_DIR / 'rga_mean.png',
)

plot_mean(
    noise_levels * 100, safe_store, 'rgr_curve',
    title='RGR (mean across 5 folds)',
    x_label='Noise Standard Deviation (%)',
    y_label='RGR Score',
    save_path=FIG_DIR / 'rgr_mean.png',
)

plot_mean(
    attack_strengths, safe_store, 'rgr_fgsm_curve',
    title='FGSM RGR (mean across 5 folds)',
    x_label='Attack strength ε',
    y_label='RGR Score',
    save_path=FIG_DIR / 'rgr_fgsm_mean.png',
)

plot_mean(
    spatial_strengths, safe_store, 'rgr_spatial_curve',
    title='Spatial Attack RGR (mean across 5 folds)',
    x_label='Attack strength ε',
    y_label='RGR Score',
    save_path=FIG_DIR / 'rgr_spatial_mean.png',
)

plot_mean(
    removal_fractions * 100, safe_store, 'rge_curve',
    title='RGE (mean across 5 folds)',
    x_label='Occluded Image Area (%)',
    y_label='RGE Score',
    save_path=FIG_DIR / 'rge_mean.png'
)

def mean_std(vals):
    a = np.asarray(vals, float)
    return (
        float(a.mean()) if len(a) else float('nan'),
        float(a.std(ddof=1)) if len(a) > 1 else 0.0,
    )

print('\nSAFE summary (mean ± std across folds):')
for m in MODEL_NAMES:
    rga_mu, rga_sd = mean_std(safe_store[m]['rga_full'])
    aurga_mu, aurga_sd = mean_std(safe_store[m]['aurga'])
    aurgr_mu, aurgr_sd = mean_std(safe_store[m]['aurgr'])
    aurgr_fgsm_mu, aurgr_fgsm_sd = mean_std(safe_store[m]['aurgr_fgsm'])
    aurgr_spatial_mu, aurgr_spatial_sd = mean_std(safe_store[m]['aurgr_spatial'])
    aurge_mu, aurge_sd = mean_std(safe_store[m]['aurge'])

    print(
        f'{m:10s} | '
        f'RGA={rga_mu:.4f}±{rga_sd:.4f} | '
        f'AURGA={aurga_mu:.4f}±{aurga_sd:.4f} | '
        f'AURGR={aurgr_mu:.4f}±{aurgr_sd:.4f} | '
        f'AURGR_FGSM={aurgr_fgsm_mu:.4f}±{aurgr_fgsm_sd:.4f} | '
        f'AURGR_SPATIAL={aurgr_spatial_mu:.4f}±{aurgr_spatial_sd:.4f} | '
        f'AURGE={aurge_mu:.4f}±{aurge_sd:.4f}'
    )

# Summary table
rows_summary = []

for model_name in MODEL_NAMES:
    rga_mu, rga_sd = mean_std(safe_store[model_name]['rga_full'])
    aurga_mu, aurga_sd = mean_std(safe_store[model_name]['aurga'])
    aurgr_mu, aurgr_sd = mean_std(safe_store[model_name]['aurgr'])
    aurgr_fgsm_mu, aurgr_fgsm_sd = mean_std(safe_store[model_name]['aurgr_fgsm'])
    aurgr_spatial_mu, aurgr_spatial_sd = mean_std(safe_store[model_name]['aurgr_spatial'])
    aurge_mu, aurge_sd = mean_std(safe_store[model_name]['aurge'])

    # Predictive metrics (mean only)
    acc_mu, _ = mean_std(results[model_name]['acc'])
    f1_mu, _  = mean_std(results[model_name]['f1'])
    mse_mu, _ = mean_std(results[model_name]['mse'])

    rows_summary.append({
        'model': model_name,
        'RGA_mean': rga_mu,
        'RGA_std': rga_sd,
        'AURGA_mean': aurga_mu,
        'AURGA_std': aurga_sd,
        'AURGR_mean': aurgr_mu,
        'AURGR_std': aurgr_sd,
        'AURGR_FGSM_mean': aurgr_fgsm_mu,
        'AURGR_FGSM_std': aurgr_fgsm_sd,
        'AURGR_SPATIAL_mean': aurgr_spatial_mu,
        'AURGR_SPATIAL_std': aurgr_spatial_sd,
        'AURGE_mean': aurge_mu,
        'AURGE_std': aurge_sd,
        'ACC': acc_mu,
        'F1': f1_mu,
        'MSE': mse_mu,
    })

df_summary = pd.DataFrame(rows_summary)
csv_path = CSV_DIR / 'safe_summary_metrics.csv'
df_summary.to_csv(csv_path, index=False)
print(f'Summary is saved to: {csv_path}')

# Radar plot (Min-Max rescaling)
def radar_plot_minmax(df, metric, lab, save_path, title=None):
    r = df[['model'] + metric].copy()

    for col in metric:
        m_min = r[col].min()
        m_max = r[col].max()
        r[col] = 1.0 if (m_max - m_min) == 0 else (r[col] - m_min) / (m_max - m_min)

    angles = np.linspace(0, 2 * np.pi, len(lab), endpoint=False).tolist()
    angles += angles[:1]

    figure, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for _, row in r.iterrows():
        vals = [row[c] for c in metric]
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=2, label=row['model'])
        ax.fill(angles, vals, alpha=0.12)

    ax.set_thetagrids(np.degrees(angles[:-1]), lab)
    ax.tick_params(axis='x', pad=10)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    save_and_close(save_path)

rows = []
for model_name in MODEL_NAMES:
    rows.append({
        'model': model_name,
        'AURGA_mean': float(np.mean(safe_store[model_name]['aurga'])),
        'AURGE_mean': float(np.mean(safe_store[model_name]['aurge'])),
        'AURGR_mean': float(np.mean(safe_store[model_name]['aurgr'])),
        'AURGR_FGSM_mean': float(np.nanmean(safe_store[model_name]['aurgr_fgsm'])),
        'AURGR_SPATIAL_mean': float(np.nanmean(safe_store[model_name]['aurgr_spatial'])),
        'ACC': float(np.mean(results[model_name]['acc'])),
        'F1': float(np.mean(results[model_name]['f1']))
    })

df_radar = pd.DataFrame(rows)

metrics = ['AURGA_mean', 'AURGE_mean', 'AURGR_mean', 'AURGR_SPATIAL_mean', 'ACC', 'F1']
labels = ['AURGA', 'AURGE', 'AURGR', 'AURGR_Spatial', 'Accuracy', 'F1']

radar_plot_minmax(
    df_radar,
    metric=metrics,
    lab=labels,
    save_path=FIG_DIR / 'radar_minmax.png'
)
