import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from art.attacks.evasion import (
    FastGradientMethod,
    ProjectedGradientDescent,
    SquareAttack,
    HopSkipJump,
    SimBA,
    Wasserstein,
    SpatialTransformation
)
from art.estimators.classification import PyTorchClassifier, SklearnClassifier
from sklearn.metrics import auc
from typing import Literal
from safe.cramer import gini_via_lorenz, cvm1_concordance_weighted
from safe.utils import align_proba_to_class_order, ensure_prob_matrix


def rgr_cramer(pred, pred_pert):
    """
    RGR which compares original predictions with perturbed predictions.

    Parameters
    ----------
    pred : array-like
        Original predictions
    pred_pert : array-like
        Perturbed predictions

    Returns
    -------
    float
        RGR score
    """
    g = gini_via_lorenz(pred)
    if not np.isfinite(g) or g == 0:
        return np.nan
    cvm = cvm1_concordance_weighted(pred, pred_pert)
    if not np.isfinite(cvm):
        return np.nan
    return 1 - cvm / g


def rgr_cramer_multiclass(prob_original, prob_perturbed, class_order=None, class_weights=None, verbose=False):
    """
    Calculate RGR for multiclass classification.

    Parameters
    ----------
    prob_original : array-like, shape (n_samples, n_classes)
        Original predicted probabilities
    prob_perturbed : array-like, shape (n_samples, n_classes)
        Perturbed predicted probabilities
    class_order :
        Class order
    class_weights : array-like, optional
        Custom weights for each class. If None, uses uniform weighting.
    verbose : bool, optional
        Print detailed information

    Returns
    -------
    tuple
        (rgr_weighted, rgr_per_class, weights_used)
        - rgr_weighted: Overall weighted RGR score
        - rgr_per_class: RGR score for each class
        - weights_used: Weights used for each class
    """
    prob_original = np.asarray(prob_original)
    prob_perturbed = np.asarray(prob_perturbed)

    if class_order is not None:
        class_order = np.asarray(class_order)
        prob_original = ensure_prob_matrix(prob_original, class_order)
        prob_perturbed = ensure_prob_matrix(prob_perturbed, class_order)
    else:
        if prob_original.ndim != 2 or prob_perturbed.ndim != 2:
            raise ValueError('For 1D binary probabilities, pass class_order with 2 classes')

    n_samples, n_classes = prob_original.shape

    if prob_perturbed.shape != prob_original.shape:
        raise ValueError(
            f'Shape mismatch: prob_original {prob_original.shape} and prob_perturbed {prob_perturbed.shape}'
        )

    if class_weights is None:
        class_weights = np.ones(n_classes) / n_classes
    else:
        class_weights = np.asarray(class_weights)
        if len(class_weights) != n_classes:
            raise ValueError(
                f'Class_weights length {len(class_weights)} does not match n_classes {n_classes}'
            )

    rgrs = []

    for k in range(n_classes):
        pred_orig = prob_original[:, k]
        pred_pert = prob_perturbed[:, k]

        rgr_k = rgr_cramer(pred_orig, pred_pert)
        rgrs.append(rgr_k)

        if verbose:
            print(f'Class {k}: RGR = {rgr_k:.4f}')

    rgrs = np.array(rgrs)

    # Weighted average
    rgr_weighted = np.nansum(rgrs * class_weights) / np.nansum(class_weights)

    return rgr_weighted, rgrs, class_weights


def evaluate_rgr_multiclass_noise(model, x_data, prob_original, noise_levels,
                                   model_class_order, class_order,
                                   class_weights=None, model_type='sklearn',
                                   device=None, rga_full=None, model_name='Model',
                                   plot=True, fig_size=(10, 6), verbose=True,
                                   random_seed=None, save_path=None):
    """
    Evaluate RGR robustness for multiclass classification with noise perturbation.

    Parameters
    ----------
    model : sklearn model or PyTorch model
        Trained model to evaluate
    x_data : array-like or torch.Tensor
        Input features
    prob_original : array-like, shape (n_samples, n_classes)
        Original predicted probabilities (columns in model_class_order)
    noise_levels : array-like
        Standard deviations of Gaussian noise to test
    model_class_order : array-like
        Order of classes in model's output (e.g., model.classes_ for sklearn)
    class_order : array-like
        Target class order for alignment (shared across all models in comparison)
    class_weights : array-like, optional
        Custom weights for each class. If None, uses uniform weighting.
    model_type : {'sklearn', 'pytorch'}, optional
        Type of model
    device : torch.device, optional
        Device for PyTorch models
    rga_full : float, optional
        Full RGA score for rescaling. If None, no rescaling is applied.
    model_name : str, optional
        Name of model for display
    plot : bool, optional
        Whether to generate visualization
    fig_size : tuple, optional
        Figure size for plot
    verbose : bool, optional
        Print detailed results
    random_seed : int, optional
        Random seed for reproducibility
    save_path :
        Path for saving the plot

    Returns
    -------
    dict
        Dictionary containing:
        - 'rgr_scores': RGR scores at each noise level
        - 'rgr_rescaled': Rescaled RGR scores (if rga_full provided)
        - 'aurgr': Area under RGR curve
        - 'noise_levels': Noise levels tested
        - 'per_class_rgr': Per-class RGR at each noise level
        - 'class_order': Class order used
    """
    prob_original = np.asarray(prob_original)
    noise_levels = np.asarray(noise_levels)
    model_class_order = np.asarray(model_class_order)
    class_order = np.asarray(class_order)

    rng = np.random.default_rng(random_seed)

    prob_original_aligned = align_proba_to_class_order(
        prob_original, model_class_order, class_order
    )

    n_samples, n_classes = prob_original_aligned.shape

    if class_weights is None:
        class_weights = np.ones(n_classes) / n_classes

    if model_type == 'pytorch':
        if not isinstance(x_data, torch.Tensor):
            x_data = torch.tensor(x_data, dtype=torch.float32)

    rgr_scores = []
    per_class_rgr_list = []

    if verbose:
        print(f'RGR Evaluation: {model_name}')
        print(f'Testing {len(noise_levels)} noise levels')

    for i, sigma_val in enumerate(noise_levels):
        sigma = float(sigma_val.item() if hasattr(sigma_val, 'item') else sigma_val)

        if model_type == 'sklearn':
            # Add Gaussian noise to features
            noise = rng.normal(0, sigma, size=x_data.shape)
            x_noisy = x_data + noise

            # Get predictions and align to target class order
            prob_perturbed_raw = model.predict_proba(x_noisy)
            prob_perturbed = align_proba_to_class_order(
                prob_perturbed_raw, model_class_order, class_order
            )

        elif model_type == 'pytorch':
            noise = torch.randn_like(x_data) * sigma
            x_noisy = x_data + noise

            # Get predictions and align to target class order
            with torch.no_grad():
                logits = model(x_noisy.to(device))
                prob_perturbed_raw = torch.softmax(logits, dim=1).cpu().numpy()

            prob_perturbed = align_proba_to_class_order(
                prob_perturbed_raw, model_class_order, class_order
            )
        else:
            raise ValueError(f"model_type must be 'sklearn' or 'pytorch', got '{model_type}'")

        # Calculate RGR
        rgr_val, rgr_per_class, _ = rgr_cramer_multiclass(
            prob_original_aligned,
            prob_perturbed,
            class_order=class_order,
            class_weights=class_weights,
            verbose=False
        )

        rgr_scores.append(0.0 if np.isnan(rgr_val) else rgr_val)
        per_class_rgr_list.append(rgr_per_class)

        if verbose and i % max(1, len(noise_levels) // 5) == 0:
            print(f"σ = {sigma:.3f}: RGR = {rgr_scores[-1]:.4f}")

    rgr_scores = np.array(rgr_scores)
    per_class_rgr_list = np.array(per_class_rgr_list)

    if rga_full is not None:
        rgr_rescaled = rgr_scores * rga_full
    else:
        rgr_rescaled = rgr_scores

    max_noise = np.max(noise_levels)
    noise_norm = noise_levels / max_noise if max_noise > 0 else noise_levels

    aurgr = auc(noise_norm, rgr_rescaled)

    if verbose:
        print(f'AURGR: {aurgr:.4f}')

    if plot:
        plt.figure(figsize=fig_size)
        plt.plot(noise_levels * 100, rgr_rescaled, '-o', linewidth=2.5,
                 markersize=6, color='steelblue',
                 label=f'{model_name} (AURGR={aurgr:.3f})')
        plt.fill_between(noise_levels * 100, 0, rgr_rescaled,
                         alpha=0.2, color='steelblue')
        plt.xlabel('Noise Standard Deviation', fontsize=11, fontweight='bold')
        plt.ylabel('RGR Score', fontsize=11, fontweight='bold')
        plt.title(f'RGR Curve: {model_name}', fontsize=12, fontweight='bold')
        plt.grid(alpha=0.3, linestyle='--')
        plt.xlim([0, noise_levels[-1] * 100])
        plt.ylim([0, max(rgr_rescaled) * 1.1 if max(rgr_rescaled) > 0 else 1])
        plt.legend(fontsize=10)
        plt.tight_layout()
        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.close()

    return {
        'rgr_scores': rgr_scores,
        'rgr_rescaled': rgr_rescaled,
        'aurgr': aurgr,
        'noise_levels': noise_levels,
        'per_class_rgr': per_class_rgr_list,
        'class_order': class_order
    }


def compare_models_rgr(models_dict, noise_levels, class_order,
                        rga_dict=None, class_weights=None,
                        fig_size=(12, 6), verbose=True, random_seed=None, save_path=None):
    """
    Compare robustness of multiple models using RGR metrics.

    Parameters
    ----------
    models_dict : dict
        Dictionary mapping model names to tuples of:
        (model, x_data, prob_original, model_class_order, model_type, device)

        Example: {
            'RF': (rf_model, x_test, prob_rf, rf.classes_, 'sklearn', None),
            'VQC': (vqc_model, x_tensor, prob_vqc, np.array([0,1,2]), 'pytorch', device)
        }
    noise_levels : array-like
        Standard deviations of noise to test
    class_order : array-like
        Target class order for alignment (shared across all models)
    rga_dict : dict, optional
        Dictionary mapping model names to RGA scores for rescaling
    class_weights : array-like, optional
        Class weights for all models
    fig_size : tuple, optional
        Figure size for comparison plot
    verbose : bool, optional
        Print detailed results
    random_seed : int, optional
        Random seed for reproducibility
    save_path:
        Path for saving the plot

    Returns
    -------
    dict
        RGR evaluation results for all models
    """
    results = {}

    for model_name, model_config in models_dict.items():
        model, x_data, prob_original, model_class_order, model_type, device = model_config
        rga_full = rga_dict.get(model_name) if rga_dict else None

        if verbose:
            print(f'\nEvaluating {model_name}...')

        result = evaluate_rgr_multiclass_noise(
            model=model,
            x_data=x_data,
            prob_original=prob_original,
            noise_levels=noise_levels,
            model_class_order=model_class_order,
            class_order=class_order,
            class_weights=class_weights,
            model_type=model_type,
            device=device,
            rga_full=rga_full,
            model_name=model_name,
            plot=False,
            verbose=verbose,
            random_seed=random_seed,
            save_path=save_path
        )
        results[model_name] = result

    model_names = list(results.keys())
    aurgr_scores = np.array([results[name]['aurgr'] for name in model_names], dtype=float)

    plt.figure(figsize=fig_size)
    cmap = plt.get_cmap('tab10')
    colors = cmap(np.linspace(0, 1, len(results)))

    for (model_name, result), color in zip(results.items(), colors):
        plt.plot(
            result['noise_levels'] * 100,
            result['rgr_rescaled'],
            '-o',
            linewidth=2.5,
            markersize=5,
            color=color,
            label=f"{model_name} (AURGR={result['aurgr']:.3f})"
        )

    plt.xlabel('Noise Standard Deviation', fontsize=11, fontweight='bold')
    plt.ylabel('RGR Score', fontsize=11, fontweight='bold')
    plt.title('RGR Curves Comparison', fontsize=12, fontweight='bold')
    plt.grid(alpha=0.3, linestyle='--')
    plt.xlim([0, float(np.max(noise_levels)) * 100])
    plt.legend(fontsize=9)
    plt.tight_layout()
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close()

    if verbose:
        print('Robustness Comparison Summary')
        for name, score in zip(model_names, aurgr_scores):
            print(f'{name}: AURGR = {score:.4f}')

        if len(model_names) >= 2:
            best_idx = int(np.nanargmax(aurgr_scores))
            worst_idx = int(np.nanargmin(aurgr_scores))

            best = aurgr_scores[best_idx]
            worst = aurgr_scores[worst_idx]

            print(f'Best: {model_names[best_idx]} (AURGR={best:.4f})')
            print(f'Worst: {model_names[worst_idx]} (AURGR={worst:.4f})')

    return results


def _bound_values(x_data):
    """
    Estimate the minimum and maximum values in the dataset for ART
    """
    x_np = x_data.detach().cpu().numpy() if isinstance(x_data, torch.Tensor) else np.asarray(x_data)
    x_min = float(np.min(x_np))
    x_max = float(np.max(x_np))
    if not np.isfinite(x_min) or not np.isfinite(x_max):
        raise ValueError('Invalid values in x_data')
    if x_min == x_max:
        x_max = x_min + 1e-6
    bounds = (x_min, x_max)
    return bounds


def _art_classifier(
    model,
    x_data,
    nb_classes,
    model_type='sklearn',
    device=None,
    clip_values=None,
):
    """
    ART classifier
    """
    if clip_values is None:
        clip_values = _bound_values(x_data)

    if model_type == 'sklearn':
        return SklearnClassifier(model=model, clip_values=clip_values)

    if model_type == 'pytorch':
        if not isinstance(x_data, torch.Tensor):
            x_data = torch.tensor(x_data, dtype=torch.float32)

        input_shape = tuple(x_data.shape[1:])
        loss = nn.CrossEntropyLoss()

        device_type = 'gpu' if (
            device is not None and str(device).startswith('cuda') and torch.cuda.is_available()
        ) else 'cpu'

        return PyTorchClassifier(
            model=model,
            loss=loss,
            input_shape=input_shape,
            nb_classes=nb_classes,
            optimizer=None,
            clip_values=clip_values,
            device_type=device_type,
        )

    raise ValueError(f'Unsupported model_type: {model_type}')


def generate_adversarial_examples(
    model,
    x_data,
    y_labels,
    nb_classes,
    attack_name: Literal['fgsm', 'pgd', 'square', 'hsj', 'simba'] = 'fgsm',
    attack_params=None,
    model_type='sklearn',
    device=None,
    clip_values=None,
):
    """
    Generate adversarial examples with ART

    Parameters
    ----------
    model :
        Trained model
    x_data :
        Input features
    y_labels :
        Integer class labels
    nb_classes : int
        Number of classes
    attack_name : {'fgsm', 'pgd', 'square', 'hsj', 'simba'}
        Attack type
    attack_params : dict, optional
        ART attack parameters
    model_type : {'sklearn', 'pytorch'}
        Model type
    device :
        Torch device for pytorch models
    clip_values : tuple, optional
        (min, max) bounds for ART

    Returns
    -------
    np.ndarray
        Adversarial samples
    """
    if attack_params is None:
        attack_params = {}

    x_np = x_data.detach().cpu().numpy() if isinstance(x_data, torch.Tensor) else np.asarray(x_data)
    y_np = np.asarray(y_labels).astype(int)

    classifier = _art_classifier(
        model=model,
        x_data=x_np,
        nb_classes=nb_classes,
        model_type=model_type,
        device=device,
        clip_values=clip_values,
    )

    if attack_name == 'fgsm':
        attack = FastGradientMethod(estimator=classifier, **attack_params)
    elif attack_name == 'pgd':
        attack = ProjectedGradientDescent(estimator=classifier, **attack_params)
    elif attack_name == 'square':
        attack = SquareAttack(estimator=classifier, **attack_params)
    elif attack_name == 'hsj':
        attack = HopSkipJump(classifier=classifier, **attack_params)
    elif attack_name == 'simba':
        attack = SimBA(classifier=classifier, **attack_params)
    else:
        raise ValueError(f'Unsupported attack_name: {attack_name}')

    x_adv = attack.generate(x=x_np, y=y_np)
    return np.asarray(x_adv, dtype=np.float32)


def evaluate_rgr_multiclass_adversarial(
    model,
    x_data,
    prob_original,
    attack_strengths,
    model_class_order,
    class_order,
    y_true,
    attack_name: Literal['fgsm', 'pgd', 'square', 'hsj', 'simba'] = 'fgsm',
    base_attack_params=None,
    class_weights=None,
    model_type='sklearn',
    device=None,
    rga_full=None,
    model_name='Model',
    plot=True,
    fig_size=(10, 6),
    verbose=True,
    save_path=None,
):
    """
    Evaluate RGR robustness under adversarial perturbations
    """
    prob_original = np.asarray(prob_original)
    attack_strengths = np.asarray(attack_strengths, dtype=float)
    model_class_order = np.asarray(model_class_order)
    class_order = np.asarray(class_order)
    y_true = np.asarray(y_true).astype(int)

    prob_original_aligned = align_proba_to_class_order(
        prob_original, model_class_order, class_order
    )
    n_samples, n_classes = prob_original_aligned.shape

    if class_weights is None:
        class_weights = np.ones(n_classes) / n_classes

    adv_rgr_scores = []
    per_class_rgr_list = []

    if verbose:
        print(f'Adversarial RGR Evaluation: {model_name}')
        print(f'Attack: {attack_name}')
        print(f'Testing {len(attack_strengths)} attack strengths')

    for eps in attack_strengths:
        params = {} if base_attack_params is None else dict(base_attack_params)

        if attack_name in ['fgsm', 'pgd', 'square']:
            params['eps'] = float(eps)

        elif attack_name == 'simba':
            params['epsilon'] = float(eps)

        elif attack_name == 'hsj':
            params['max_iter'] = int(eps)

        if attack_name == 'pgd' and 'eps_step' not in params:
            params['eps_step'] = max(float(eps) / 4.0, 1e-4)

        x_adv = generate_adversarial_examples(
            model=model,
            x_data=x_data,
            y_labels=y_true,
            nb_classes=n_classes,
            attack_name=attack_name,
            attack_params=params,
            model_type=model_type,
            device=device,
        )

        if model_type == 'sklearn':
            prob_perturbed_raw = model.predict_proba(x_adv)
        elif model_type == 'pytorch':
            x_adv_tensor = torch.tensor(x_adv, dtype=torch.float32, device=device)
            with torch.no_grad():
                logits = model(x_adv_tensor)
                prob_perturbed_raw = torch.softmax(logits, dim=1).cpu().numpy()
        else:
            raise ValueError(f"model_type must be 'sklearn' or 'pytorch', got '{model_type}'")

        prob_perturbed = align_proba_to_class_order(
            prob_perturbed_raw, model_class_order, class_order
        )

        rgr_val, rgr_per_class, _ = rgr_cramer_multiclass(
            prob_original_aligned,
            prob_perturbed,
            class_order=class_order,
            class_weights=class_weights,
            verbose=False
        )

        adv_rgr_scores.append(0.0 if np.isnan(rgr_val) else rgr_val)
        per_class_rgr_list.append(rgr_per_class)

        if verbose:
            print(f'eps = {eps:.4f}: RGR = {adv_rgr_scores[-1]:.4f}')

    adv_rgr_scores = np.array(adv_rgr_scores)
    per_class_rgr_list = np.array(per_class_rgr_list)

    if rga_full is not None:
        rgr_rescaled = adv_rgr_scores * rga_full
    else:
        rgr_rescaled = adv_rgr_scores

    max_eps = np.max(attack_strengths)
    eps_norm = attack_strengths / max_eps if max_eps > 0 else attack_strengths
    aurgr = auc(eps_norm, rgr_rescaled)

    if verbose:
        print(f'Adversarial AURGR: {aurgr:.4f}')

    if plot:
        plt.figure(figsize=fig_size)
        plt.plot(
            attack_strengths,
            rgr_rescaled,
            '-o',
            linewidth=2.5,
            markersize=6,
            label=f'{model_name} ({attack_name.upper()}, AURGR={aurgr:.3f})'
        )
        plt.fill_between(attack_strengths, 0, rgr_rescaled, alpha=0.2)
        plt.xlabel('Attack strength ε', fontsize=11, fontweight='bold')
        plt.ylabel('RGR Score', fontsize=11, fontweight='bold')
        plt.title(f'Adversarial RGR Curve: {model_name}', fontsize=12, fontweight='bold')
        plt.grid(alpha=0.3, linestyle='--')
        plt.xlim([0, float(np.max(attack_strengths))])
        plt.ylim([0, max(rgr_rescaled) * 1.1 if max(rgr_rescaled) > 0 else 1])
        plt.legend(fontsize=10)
        plt.tight_layout()
        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    return {
        'attack_name': attack_name,
        'rgr_scores': adv_rgr_scores,
        'rgr_rescaled': rgr_rescaled,
        'aurgr': aurgr,
        'attack_strengths': attack_strengths,
        'per_class_rgr': per_class_rgr_list,
        'class_order': class_order,
    }


def compare_models_rgr_adversarial(
    models_dict,
    attack_strengths,
    class_order,
    y_true_dict,
    attack_name: Literal['fgsm', 'pgd', 'square', 'hsj', 'simba'] = 'fgsm',
    rga_dict=None,
    class_weights=None,
    fig_size=(12, 6),
    verbose=True,
    save_path=None,
    base_attack_params=None,
):
    """
    Compare robustness of multiple models under adversarial attacks
    """
    results = {}

    for model_name, model_config in models_dict.items():
        model, x_data, prob_original, model_class_order, model_type, device = model_config

        if model_name not in y_true_dict:
            raise ValueError(f'Missing true labels for model: {model_name}')

        y_true = y_true_dict[model_name]
        rga_full = rga_dict.get(model_name) if rga_dict else None

        if verbose:
            print(f'\nEvaluating {model_name} with {attack_name.upper()}...')

        result = evaluate_rgr_multiclass_adversarial(
            model=model,
            x_data=x_data,
            prob_original=prob_original,
            attack_strengths=attack_strengths,
            model_class_order=model_class_order,
            class_order=class_order,
            y_true=y_true,
            attack_name=attack_name,
            base_attack_params=base_attack_params,
            class_weights=class_weights,
            model_type=model_type,
            device=device,
            rga_full=rga_full,
            model_name=model_name,
            plot=False,
            verbose=verbose,
            save_path=save_path,
        )
        results[model_name] = result

    model_names = list(results.keys())
    aurgr_scores = np.array([results[name]['aurgr'] for name in model_names], dtype=float)

    plt.figure(figsize=fig_size)
    cmap = plt.get_cmap('tab10')
    colors = cmap(np.linspace(0, 1, len(results)))

    for (model_name, result), color in zip(results.items(), colors):
        plt.plot(
            result['attack_strengths'],
            result['rgr_rescaled'],
            '-o',
            linewidth=2.5,
            markersize=5,
            color=color,
            label=f"{model_name} (AURGR={result['aurgr']:.3f})"
        )

    plt.xlabel('Attack strength ε', fontsize=11, fontweight='bold')
    plt.ylabel('RGR Score', fontsize=11, fontweight='bold')
    plt.title(f'Adversarial RGR Curves Comparison ({attack_name.upper()})', fontsize=12, fontweight='bold')
    plt.grid(alpha=0.3, linestyle='--')
    plt.xlim([0, float(np.max(attack_strengths))])
    plt.legend(fontsize=9)
    plt.tight_layout()

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close()

    if verbose:
        print('Adversarial Robustness Comparison Summary')
        for name, score in zip(model_names, aurgr_scores):
            print(f'{name}: AURGR = {score:.4f}')

        if len(model_names) >= 2:
            best_idx = int(np.nanargmax(aurgr_scores))
            worst_idx = int(np.nanargmin(aurgr_scores))

            best = aurgr_scores[best_idx]
            worst = aurgr_scores[worst_idx]

            print(f'Best: {model_names[best_idx]} (AURGR={best:.4f})')
            print(f'Worst: {model_names[worst_idx]} (AURGR={worst:.4f})')

    return results


def compare_models_rgr_wasserstein_images(
    models_dict,
    images,
    y_true,
    attack_model,
    preprocess_fn,
    attack_strengths,
    class_order,
    rga_dict=None,
    class_weights=None,
    device=None,
    fig_size=(12, 6),
    verbose=True,
    save_path=None,
    max_iter=50,
    eps_step=0.01,
):
    """
    Image-level Wasserstein RGR.
    """

    attack_strengths = np.asarray(attack_strengths, dtype=float)
    y_true = np.asarray(y_true).astype(int)
    class_order = np.asarray(class_order)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    attack_model = attack_model.to(device)
    attack_model.eval()

    if isinstance(images, torch.Tensor):
        images_tensor = images.detach().cpu()
    else:
        images_tensor = torch.tensor(images, dtype=torch.float32)

    images_01 = torch.clamp((images_tensor + 1.0) / 2.0, 0.0, 1.0)

    class AttackWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            x_norm = x * 2.0 - 1.0
            return self.model(x_norm)

    wrapped_model = AttackWrapper(attack_model).to(device)
    loss = nn.CrossEntropyLoss()

    classifier = PyTorchClassifier(
        model=wrapped_model,
        loss=loss,
        input_shape=tuple(images_01.shape[1:]),
        nb_classes=len(class_order),
        optimizer=None,
        clip_values=(0.0, 1.0),
        device_type='gpu' if str(device).startswith('cuda') and torch.cuda.is_available() else 'cpu',
    )

    results = {
        name: {
            'attack_strengths': attack_strengths,
            'rgr_scores': [],
            'rgr_rescaled': [],
            'per_class_rgr': [],
            'class_order': class_order,
        }
        for name in models_dict
    }

    if verbose:
        print('Wasserstein image-level RGR')
        print(f'Testing {len(attack_strengths)} Wasserstein strengths')

    for eps in attack_strengths:
        if verbose:
            print(f'\nGenerating Wasserstein adversarial images: eps={eps:.4f}')

        attack = Wasserstein(
            estimator=classifier,
            eps=float(eps),
            eps_step=float(eps_step),
            max_iter=int(max_iter),
            targeted=False,
            verbose=False,
        )

        x_adv_01 = attack.generate(
            x=images_01.numpy().astype(np.float32),
            y=y_true,
        )

        x_adv_norm = torch.tensor(x_adv_01, dtype=torch.float32) * 2.0 - 1.0

        x_adv_features = preprocess_fn(x_adv_norm)

        for model_name, model_config in models_dict.items():
            model, prob_original, model_class_order, model_type, model_device = model_config

            prob_original_aligned = align_proba_to_class_order(
                prob_original,
                model_class_order,
                class_order,
            )

            if model_type == 'sklearn':
                prob_perturbed_raw = model.predict_proba(x_adv_features)

            elif model_type == 'pytorch':
                model.eval()
                use_device = model_device if model_device is not None else device
                x_adv_tensor = torch.tensor(x_adv_features, dtype=torch.float32, device=use_device)

                with torch.no_grad():
                    logits = model(x_adv_tensor)
                    prob_perturbed_raw = torch.softmax(logits, dim=1).cpu().numpy()

            else:
                raise ValueError(f"model_type must be 'sklearn' or 'pytorch', got '{model_type}'")

            prob_perturbed = align_proba_to_class_order(
                prob_perturbed_raw,
                model_class_order,
                class_order,
            )

            rgr_val, rgr_per_class, _ = rgr_cramer_multiclass(
                prob_original_aligned,
                prob_perturbed,
                class_order=class_order,
                class_weights=class_weights,
                verbose=False,
            )

            rgr_score = 0.0 if np.isnan(rgr_val) else float(rgr_val)
            results[model_name]['rgr_scores'].append(rgr_score)
            results[model_name]['per_class_rgr'].append(rgr_per_class)

            if verbose:
                print(f'{model_name}: RGR={rgr_score:.4f}')

    plt.figure(figsize=fig_size)

    for model_name, result in results.items():
        rgr_scores = np.asarray(result['rgr_scores'], dtype=float)

        rga_full = rga_dict.get(model_name) if rga_dict else None
        if rga_full is not None:
            rgr_rescaled = rgr_scores * rga_full
        else:
            rgr_rescaled = rgr_scores

        eps_norm = attack_strengths / np.max(attack_strengths) if np.max(attack_strengths) > 0 else attack_strengths
        aurgr = auc(eps_norm, rgr_rescaled)

        result['rgr_scores'] = rgr_scores.tolist()
        result['rgr_rescaled'] = rgr_rescaled.tolist()
        result['per_class_rgr'] = np.asarray(result['per_class_rgr']).tolist()
        result['aurgr'] = float(aurgr)
        result['class_order'] = class_order.tolist()

        plt.plot(
            attack_strengths,
            rgr_rescaled,
            '-o',
            linewidth=2.5,
            markersize=5,
            label=f'{model_name} (AURGR={aurgr:.3f})',
        )

    plt.xlabel('Wasserstein attack strength ε', fontsize=11, fontweight='bold')
    plt.ylabel('RGR Score', fontsize=11, fontweight='bold')
    plt.title('Image-level Wasserstein RGR Curves', fontsize=12, fontweight='bold')
    plt.grid(alpha=0.3, linestyle='--')
    plt.legend(fontsize=9)
    plt.tight_layout()

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close()

    if verbose:
        print('\nWasserstein Robustness Summary')
        for model_name, result in results.items():
            print(f"{model_name}: AURGR_WASSERSTEIN = {result['aurgr']:.4f}")

    return results


def compare_models_rgr_spatial_images(
    models_dict,
    images,
    y_true,
    attack_model,
    preprocess_fn,
    attack_strengths,
    class_order,
    rga_dict=None,
    class_weights=None,
    device=None,
    fig_size=(12, 6),
    verbose=True,
    save_path=None,
    num_translations=3,
    num_rotations=3,
):
    """
    Image-level spatial-transformation RGR.
    """

    attack_strengths = np.asarray(attack_strengths, dtype=float)
    y_true = np.asarray(y_true).astype(int)
    class_order = np.asarray(class_order)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    attack_model = attack_model.to(device)
    attack_model.eval()

    if isinstance(images, torch.Tensor):
        images_tensor = images.detach().cpu()
    else:
        images_tensor = torch.tensor(images, dtype=torch.float32)

    images_01 = torch.clamp((images_tensor + 1.0) / 2.0, 0.0, 1.0)

    class AttackWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            x_norm = x * 2.0 - 1.0
            return self.model(x_norm)

    wrapped_model = AttackWrapper(attack_model).to(device)
    loss = nn.CrossEntropyLoss()

    classifier = PyTorchClassifier(
        model=wrapped_model,
        loss=loss,
        input_shape=tuple(images_01.shape[1:]),
        nb_classes=len(class_order),
        optimizer=None,
        clip_values=(0.0, 1.0),
        device_type='gpu' if str(device).startswith('cuda') and torch.cuda.is_available() else 'cpu',
    )

    results: dict[str, dict] = {
        name: {
            'attack_strengths': attack_strengths.tolist(),
            'rgr_scores': [],
            'rgr_rescaled': [],
            'per_class_rgr': [],
            'class_order': class_order.tolist(),
            'aurgr': None,
        }
        for name in models_dict
    }

    if verbose:
        print('Spatial image-level RGR')
        print(f'Testing {len(attack_strengths)} spatial strengths')

    for strength in attack_strengths:
        if verbose:
            print(f'\nGenerating spatial adversarial images: strength={strength:.4f}')

        attack = SpatialTransformation(
            classifier=classifier,
            max_translation=float(strength),
            max_rotation=float(strength),
            num_translations=int(num_translations),
            num_rotations=int(num_rotations),
            verbose=False,
        )

        x_adv_01 = attack.generate(
            x=images_01.numpy().astype(np.float32),
            y=y_true,
        )

        x_adv_norm = torch.tensor(x_adv_01, dtype=torch.float32) * 2.0 - 1.0
        x_adv_features = preprocess_fn(x_adv_norm)

        for model_name, model_config in models_dict.items():
            model, prob_original, model_class_order, model_type, model_device = model_config

            prob_original_aligned = align_proba_to_class_order(
                prob_original,
                model_class_order,
                class_order,
            )

            if model_type == 'sklearn':
                prob_perturbed_raw = model.predict_proba(x_adv_features)

            elif model_type == 'pytorch':
                model.eval()
                use_device = model_device if model_device is not None else device
                x_adv_tensor = torch.tensor(
                    x_adv_features,
                    dtype=torch.float32,
                    device=use_device,
                )

                with torch.no_grad():
                    logits = model(x_adv_tensor)
                    prob_perturbed_raw = torch.softmax(logits, dim=1).cpu().numpy()

            else:
                raise ValueError(f"model_type must be 'sklearn' or 'pytorch', got '{model_type}'")

            prob_perturbed = align_proba_to_class_order(
                prob_perturbed_raw,
                model_class_order,
                class_order,
            )

            rgr_val, rgr_per_class, _ = rgr_cramer_multiclass(
                prob_original_aligned,
                prob_perturbed,
                class_order=class_order,
                class_weights=class_weights,
                verbose=False,
            )

            rgr_score = 0.0 if np.isnan(rgr_val) else float(rgr_val)
            results[model_name]['rgr_scores'].append(rgr_score)
            results[model_name]['per_class_rgr'].append(rgr_per_class.tolist())

            if verbose:
                print(f'{model_name}: RGR={rgr_score:.4f}')

    plt.figure(figsize=fig_size)

    for model_name, result in results.items():
        rgr_scores = np.asarray(result['rgr_scores'], dtype=float)

        rga_full = rga_dict.get(model_name) if rga_dict else None
        if rga_full is not None:
            rgr_rescaled = rgr_scores * rga_full
        else:
            rgr_rescaled = rgr_scores

        strength_norm = (
            attack_strengths / np.max(attack_strengths)
            if np.max(attack_strengths) > 0
            else attack_strengths
        )
        aurgr = auc(strength_norm, rgr_rescaled)

        result['rgr_scores'] = rgr_scores.tolist()
        result['rgr_rescaled'] = rgr_rescaled.tolist()
        result['aurgr'] = float(aurgr)

        plt.plot(
            attack_strengths,
            rgr_rescaled,
            '-o',
            linewidth=2.5,
            markersize=5,
            label=f'{model_name} (AURGR={aurgr:.3f})',
        )

    plt.xlabel('Spatial attack strength', fontsize=11, fontweight='bold')
    plt.ylabel('RGR Score', fontsize=11, fontweight='bold')
    plt.title('Image-level Spatial Transformation RGR Curves', fontsize=12, fontweight='bold')
    plt.grid(alpha=0.3, linestyle='--')
    plt.legend(fontsize=9)
    plt.tight_layout()

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close()

    if verbose:
        print('\nSpatial Transformation Robustness Summary')
        for model_name, result in results.items():
            print(f"{model_name}: AURGR_SPATIAL = {result['aurgr']:.4f}")

    return results