import numpy as np
import pandas as pd
import pickle
import yaml
import warnings
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import json
from copy import deepcopy

from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve, roc_curve
from catboost import CatBoostClassifier, Pool
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek

import sys
import torch

sys.path.append('utils')
from robust_data_preprocessor import RobustBankDataPreprocessor
from metrics_calculator import MetricsCalculator

warnings.filterwarnings('ignore')
np.random.seed(42)


@dataclass
class FoldResults:
    fold: int
    train_data: pd.DataFrame
    val_data: pd.DataFrame
    test_data: pd.DataFrame
    model: Any
    metrics: Dict[str, float]
    predictions: np.ndarray
    probabilities: np.ndarray
    feature_importances: pd.DataFrame
    training_time: float
    y_test_true: np.ndarray
    y_val_true: np.ndarray
    y_val_proba: np.ndarray
    test_predictions_df: Optional[pd.DataFrame] = None


class ImbalancedDataCatBoostTrainer:

    def __init__(self, folds_dir: str, config_path: str = 'configs/model_config.yaml'):
        self.folds_dir = Path(folds_dir)
        if not self.folds_dir.exists():
            raise ValueError(f"Folds directory not found: {folds_dir}")

        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.models = []
        self.ensemble_weights = []
        self.fold_results = []
        self.metrics_calculator = MetricsCalculator()

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.result_dir = Path(f"results/experiment_{self.timestamp}")
        self.result_dir.mkdir(parents=True, exist_ok=True)
        
        self.predictions_dir = self.result_dir / "predictions"
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir = self.result_dir / "metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.visualization_dir = self.result_dir / "visualizations"
        self.visualization_dir.mkdir(parents=True, exist_ok=True)

        self.model_dir = Path(f"models/experiment_{self.timestamp}")
        self.model_dir.mkdir(parents=True, exist_ok=True)

        print(f"Folds directory: {self.folds_dir}")
        print(f"Results will be saved to: {self.result_dir}")
        print(f"  - Predictions: {self.predictions_dir}")
        print(f"  - Metrics: {self.metrics_dir}")
        print(f"  - Visualizations: {self.visualization_dir}")
        print(f"Models will be saved to: {self.model_dir}")

    def load_fold_data(self, fold_num: int, target_col: str = 'y'):
        fold_dir = self.folds_dir / f"fold_{fold_num}"

        if not fold_dir.exists():
            raise ValueError(f"Fold {fold_num} directory not found: {fold_dir}")

        print(f"\nLoading data for fold {fold_num}...")

        train_data = pd.read_csv(fold_dir / "train_data.csv")
        val_data = pd.read_csv(fold_dir / "val_data.csv")
        test_data = pd.read_csv(fold_dir / "test_data.csv")

        print(f"Train data: {train_data.shape}")
        print(f"Val data: {val_data.shape}")
        print(f"Test data: {test_data.shape}")

        X_train = train_data.drop(columns=[target_col])
        y_train = train_data[target_col]

        X_val = val_data.drop(columns=[target_col])
        y_val = val_data[target_col]

        X_test = test_data.drop(columns=[target_col])
        y_test = test_data[target_col]

        pos_ratio = y_train.mean()
        print(f"\nClass distribution:")
        print(f"  Train: {pos_ratio:.3f} positive ({y_train.sum()}/{len(y_train)}) - Imbalance ratio: {1/pos_ratio:.1f}:1")
        print(f"  Val: {y_val.mean():.3f} positive ({y_val.sum()}/{len(y_val)})")
        print(f"  Test: {y_test.mean():.3f} positive ({y_test.sum()}/{len(y_test)})")

        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test,
            'train_data': train_data,
            'val_data': val_data,
            'test_data': test_data
        }

    def handle_class_imbalance(self, X_train, y_train, method='class_weights'):
        print(f"\nHandling class imbalance using method: {method}")

        if method == 'class_weights':
            class_counts = y_train.value_counts()
            total_samples = len(y_train)
            weight_for_0 = total_samples / (2 * class_counts[0])
            weight_for_1 = total_samples / (2 * class_counts[1])

            print(f"  Class weights: 0={weight_for_0:.2f}, 1={weight_for_1:.2f}")
            return X_train, y_train, {'scale_pos_weight': weight_for_1 / weight_for_0}

        elif method == 'smote':
            smote = SMOTE(random_state=42, sampling_strategy=0.5)
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
            print(f"  After SMOTE: {X_resampled.shape}, positive ratio: {y_resampled.mean():.3f}")
            return X_resampled, y_resampled, {}

        elif method == 'undersample':
            rus = RandomUnderSampler(random_state=42, sampling_strategy=0.5)
            X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
            print(f"  After undersampling: {X_resampled.shape}, positive ratio: {y_resampled.mean():.3f}")
            return X_resampled, y_resampled, {}

        elif method == 'smote_tomek':
            smote_tomek = SMOTETomek(random_state=42, sampling_strategy=0.5)
            X_resampled, y_resampled = smote_tomek.fit_resample(X_train, y_train)
            print(f"  After SMOTETomek: {X_resampled.shape}, positive ratio: {y_resampled.mean():.3f}")
            return X_resampled, y_resampled, {}

        else:
            return X_train, y_train, {}

    def preprocess_fold_data_safe(self, X_train, X_val, X_test, y_train, y_val, y_test):
        print("\nPreprocessing data (COMPLETELY SAFE - no data leakage)...")

        print("Fitting preprocessor on training data only...")
        preprocessor_train = RobustBankDataPreprocessor()
        train_combined = pd.concat([X_train, y_train], axis=1)
        X_train_processed, y_train_processed, feature_info = preprocessor_train.preprocess(
            train_combined, is_training=True
        )

        print("Transforming validation data using training fit...")
        val_combined = pd.concat([X_val, y_val], axis=1)
        X_val_processed, y_val_processed, _ = preprocessor_train.preprocess(
            val_combined, is_training=False
        )

        print("Transforming test data using training fit...")
        test_combined = pd.concat([X_test, y_test], axis=1)
        X_test_processed, y_test_processed, _ = preprocessor_train.preprocess(
            test_combined, is_training=False
        )

        print(f"Processed shapes - Train: {X_train_processed.shape}, Val: {X_val_processed.shape}, Test: {X_test_processed.shape}")

        return (X_train_processed, X_val_processed, X_test_processed,
                y_train_processed, y_val_processed, y_test_processed, feature_info)
    
    def get_imbalanced_optimized_hyperparameters(self, imbalance_ratio):
        if imbalance_ratio > 20:
            print(f"High imbalance detected: {imbalance_ratio:.1f}:1")
            scale_pos_weight = imbalance_ratio
            iterations = 1500
            depth = 8
        elif imbalance_ratio > 10:
            print(f"Moderate imbalance detected: {imbalance_ratio:.1f}:1")
            scale_pos_weight = imbalance_ratio * 0.8
            iterations = 1200
            depth = 10
        else:
            print(f"Mild imbalance detected: {imbalance_ratio:.1f}:1")
            scale_pos_weight = imbalance_ratio * 0.6
            iterations = 1000
            depth = 12

        optimized_configs = [
            {
                'name': 'Focal_Loss',
                'params': {
                    'iterations': iterations,
                    'depth': depth,
                    'learning_rate': 0.05,
                    'l2_leaf_reg': 3.0,
                    'random_strength': 1.0,
                    'bagging_temperature': 0.5,
                    'border_count': 128,
                    'grow_policy': 'SymmetricTree',
                    'min_data_in_leaf': 50,
                    'random_seed': 42,
                    'eval_metric': 'AUC',
                    'loss_function': 'Logloss',
                    'scale_pos_weight': scale_pos_weight,
                    'od_type': 'Iter',
                    'od_wait': 30,
                    'use_best_model': True,
                    'verbose': False
                }
            },
            {
                'name': 'Balanced_Weighted',
                'params': {
                    'iterations': iterations,
                    'depth': 10,
                    'learning_rate': 0.03,
                    'l2_leaf_reg': 5.0,
                    'random_strength': 2.0,
                    'bagging_temperature': 0.3,
                    'border_count': 100,
                    'grow_policy': 'Lossguide',
                    'min_data_in_leaf': 30,
                    'random_seed': 43,
                    'eval_metric': 'F1',
                    'loss_function': 'Logloss',
                    'auto_class_weights': 'Balanced',
                    'od_type': 'Iter',
                    'od_wait': 30,
                    'use_best_model': True,
                    'verbose': False
                }
            },
            {
                'name': 'Conservative_Imbalanced',
                'params': {
                    'iterations': iterations,
                    'depth': 6,
                    'learning_rate': 0.01,
                    'l2_leaf_reg': 10.0,
                    'random_strength': 0.1,
                    'bagging_temperature': 0.2,
                    'border_count': 64,
                    'grow_policy': 'SymmetricTree',
                    'min_data_in_leaf': 100,
                    'random_seed': 44,
                    'eval_metric': 'AUC',
                    'loss_function': 'Logloss',
                    'scale_pos_weight': scale_pos_weight * 1.2,
                    'od_type': 'Iter',
                    'od_wait': 30,
                    'use_best_model': True,
                    'verbose': False
                }
            },
            {
                'name': 'Fast_Imbalanced',
                'params': {
                    'iterations': 800,
                    'depth': 4,
                    'learning_rate': 0.1,
                    'l2_leaf_reg': 1.0,
                    'random_strength': 3.0,
                    'bagging_temperature': 0.8,
                    'border_count': 32,
                    'grow_policy': 'SymmetricTree',
                    'min_data_in_leaf': 20,
                    'random_seed': 45,
                    'eval_metric': 'AUC',
                    'loss_function': 'Logloss',
                    'scale_pos_weight': scale_pos_weight * 0.8,
                    'od_type': 'Iter',
                    'od_wait': 30,
                    'use_best_model': True,
                    'verbose': False
                }
            },
            {
                'name': 'Deep_Imbalanced',
                'params': {
                    'iterations': 2000,
                    'depth': 12,
                    'learning_rate': 0.02,
                    'l2_leaf_reg': 0.1,
                    'random_strength': 0.5,
                    'bagging_temperature': 0.4,
                    'border_count': 254,
                    'grow_policy': 'Depthwise',
                    'min_data_in_leaf': 1,
                    'random_seed': 46,
                    'eval_metric': 'AUC',
                    'loss_function': 'Logloss',
                    'auto_class_weights': 'SqrtBalanced',
                    'od_type': 'Iter',
                    'od_wait': 30,
                    'use_best_model': True,
                    'verbose': False
                }
            }
        ]

        return optimized_configs

    def create_heterogeneous_models(self, X_train, y_train, X_val, y_val, cat_features,
                                  imbalance_method='class_weights'):

        X_train_processed, y_train_processed, imbalance_params = self.handle_class_imbalance(
            X_train, y_train, method=imbalance_method
        )

        imbalance_ratio = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

        model_configs = self.get_imbalanced_optimized_hyperparameters(imbalance_ratio)
        models = []
        weights = []
        val_predictions = []

        for config in model_configs:
            print(f"\nTraining {config['name']}...")

            params = config['params'].copy()

            if torch.cuda.is_available():
                params.update({
                    'task_type': 'GPU',
                    'devices': '0'
                })

            params.update(imbalance_params)

            if 'auto_class_weights' in params and 'scale_pos_weight' in params:
                if config['name'] in ['Balanced_Weighted', 'Deep_Imbalanced']:
                    params.pop('scale_pos_weight', None)
                else:
                    params.pop('auto_class_weights', None)

            try:
                train_pool = Pool(X_train_processed, y_train_processed, cat_features=cat_features)
                eval_pool = Pool(X_val, y_val, cat_features=cat_features)

                model = CatBoostClassifier(**params)
                model.fit(
                    train_pool,
                    eval_set=eval_pool,
                    verbose=100
                )

                y_pred_proba = model.predict_proba(X_val)[:, 1]
                val_predictions.append(y_pred_proba)

                auc_score = roc_auc_score(y_val, y_pred_proba)
                pr_auc = average_precision_score(y_val, y_pred_proba)

                score = 0.7 * auc_score + 0.3 * pr_auc

                models.append((config['name'], model))
                weights.append(score)

                print(f"{config['name']} - Validation AUC: {auc_score:.4f}, PR-AUC: {pr_auc:.4f}, Weight: {score:.4f}")

            except Exception as e:
                print(f"Error training {config['name']}: {str(e)}")
                try:
                    simple_params = {
                        'iterations': 500,
                        'learning_rate': 0.05,
                        'depth': 6,
                        'loss_function': 'Logloss',
                        'eval_metric': 'AUC',
                        'random_seed': 42 + len(models),
                        'verbose': False,
                        'scale_pos_weight': imbalance_ratio,
                        'od_type': 'Iter',
                        'od_wait': 30,
                        'use_best_model': True
                    }
                    if torch.cuda.is_available():
                        simple_params.update({'task_type': 'GPU', 'devices': '0'})
                    
                    train_pool = Pool(X_train_processed, y_train_processed, cat_features=cat_features)
                    eval_pool = Pool(X_val, y_val, cat_features=cat_features)
                    
                    model = CatBoostClassifier(**simple_params)
                    model.fit(train_pool, eval_set=eval_pool, verbose=100)
                    
                    y_pred_proba = model.predict_proba(X_val)[:, 1]
                    val_predictions.append(y_pred_proba)
                    auc_score = roc_auc_score(y_val, y_pred_proba)
                    pr_auc = average_precision_score(y_val, y_pred_proba)
                    score = 0.7 * auc_score + 0.3 * pr_auc
                    
                    models.append((f"{config['name']}_simple", model))
                    weights.append(score)
                    print(f"{config['name']}_simple - Validation AUC: {auc_score:.4f}, PR-AUC: {pr_auc:.4f}")
                except Exception as e2:
                    print(f"Also failed with simple params: {str(e2)}")
                    continue

        return models, weights, val_predictions

    def train_on_folds(self, target_col: str = 'y', imbalance_method: str = 'class_weights'):
        print("\n" + "="*60)
        print(f"TRAINING ON PREDEFINED 5-FOLD DATA (DATA LEAKAGE FIXED)")
        print(f"Imbalance handling method: {imbalance_method}")
        print("="*60)

        fold_results = []

        for fold_num in range(1, 6):
            print(f"\n{'='*40}")
            print(f"Fold {fold_num}/5")
            print(f"{'='*40}")

            fold_data = self.load_fold_data(fold_num, target_col)

            X_train, X_val, X_test, y_train, y_val, y_test, feature_info = \
                self.preprocess_fold_data_safe(
                    fold_data['X_train'], fold_data['X_val'], fold_data['X_test'],
                    fold_data['y_train'], fold_data['y_val'], fold_data['y_test']
                )

            cat_features = feature_info.get('cat_indices', [])

            print("\nCreating Heterogeneous Ensemble with imbalance handling...")
            self.models, self.ensemble_weights, val_predictions = self.create_heterogeneous_models(
                X_train, y_train, X_val, y_val, cat_features, imbalance_method
            )

            y_val_proba = None
            if val_predictions:
                total_weight = sum(self.ensemble_weights) if self.ensemble_weights else 1.0
                weighted_val_predictions = []
                for pred, weight in zip(val_predictions, self.ensemble_weights):
                    weighted_pred = pred * (weight / total_weight) if total_weight > 0 else pred
                    weighted_val_predictions.append(weighted_pred)
                y_val_proba = np.sum(weighted_val_predictions, axis=0)

            if not self.models:
                print("Warning: No models trained successfully. Trying default parameters...")
                train_pool = Pool(X_train, y_train, cat_features=cat_features)
                eval_pool = Pool(X_val, y_val, cat_features=cat_features)
                
                default_model = CatBoostClassifier(
                    iterations=500,
                    learning_rate=0.05,
                    depth=6,
                    loss_function='Logloss',
                    eval_metric='AUC',
                    random_seed=42,
                    verbose=False,
                    scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]),
                    od_type='Iter',
                    od_wait=30,
                    use_best_model=True
                )
                default_model.fit(train_pool, eval_set=eval_pool, verbose=100)
                self.models = [('Default', default_model)]
                self.ensemble_weights = [1.0]
                y_val_proba = default_model.predict_proba(X_val)[:, 1]

            self.save_fold_models(fold_num)

            y_test_pred, y_test_proba = self.weighted_ensemble_predict(X_test)

            metrics = self.calculate_imbalanced_metrics(
                y_test, y_test_pred, y_test_proba, 
                y_val_true=y_val, y_val_proba=y_val_proba
            )

            feature_importance = self._aggregate_feature_importance(X_train.columns)

            test_predictions_df = self._create_test_predictions_df(
                fold_data['test_data'], y_test, y_test_pred, y_test_proba, fold_num
            )

            self._save_test_predictions(test_predictions_df, fold_num)

            self._save_val_predictions(y_val, y_val_proba, fold_num)

            fold_result = FoldResults(
                fold=fold_num,
                train_data=fold_data['train_data'],
                val_data=fold_data['val_data'],
                test_data=fold_data['test_data'],
                model=self.models.copy(),
                metrics=metrics,
                predictions=y_test_pred,
                probabilities=y_test_proba,
                feature_importances=feature_importance,
                training_time=0,
                y_test_true=y_test,
                y_val_true=y_val,
                y_val_proba=y_val_proba,
                test_predictions_df=test_predictions_df
            )

            fold_results.append(fold_result)

            self._save_fold_metrics(metrics, fold_num)

            print(f"\nFold {fold_num} Results:")
            print(f"  AUC: {metrics.get('auroc', 0):.4f}")
            print(f"  PR-AUC: {metrics.get('pr_auc', 0):.4f}")
            print(f"  F1-Score: {metrics.get('f1_score', 0):.4f}")
            print(f"  Precision: {metrics.get('precision', 0):.4f}")
            print(f"  Recall: {metrics.get('recall', 0):.4f}")
            print(f"  Sensitivity: {metrics.get('sensitivity', 0):.4f}")
            print(f"  Specificity: {metrics.get('specificity', 0):.4f}")

        return fold_results

    def _create_test_predictions_df(self, test_data_raw, y_test, y_test_pred, y_test_proba, fold_num):
        df = test_data_raw.copy()
        df['fold'] = fold_num
        df['y_true'] = y_test
        df['y_pred'] = y_test_pred
        df['y_proba'] = y_test_proba
        df['y_proba_0'] = 1 - y_test_proba
        
        return df

    def _save_test_predictions(self, test_predictions_df, fold_num):
        predictions_path = self.predictions_dir / f"fold_{fold_num}_test_predictions.csv"
        test_predictions_df.to_csv(predictions_path, index=False)
        print(f"Saved test predictions to: {predictions_path}")
        
        simple_df = test_predictions_df[['fold', 'y_true', 'y_pred', 'y_proba']].copy()
        simple_path = self.predictions_dir / f"fold_{fold_num}_test_predictions_simple.csv"
        simple_df.to_csv(simple_path, index=False)
        
        return predictions_path

    def _save_val_predictions(self, y_val, y_val_proba, fold_num):
        if y_val_proba is not None:
            val_df = pd.DataFrame({
                'fold': fold_num,
                'y_true': y_val,
                'y_proba': y_val_proba
            })
            val_path = self.predictions_dir / f"fold_{fold_num}_val_predictions.csv"
            val_df.to_csv(val_path, index=False)
            print(f"Saved validation predictions to: {val_path}")

    def _save_fold_metrics(self, metrics, fold_num):
        metrics_path = self.metrics_dir / f"fold_{fold_num}_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        metrics_df = pd.DataFrame([metrics])
        metrics_df['fold'] = fold_num
        csv_path = self.metrics_dir / f"fold_{fold_num}_metrics.csv"
        metrics_df.to_csv(csv_path, index=False)
        print(f"Saved metrics to: {metrics_path}")

    def calculate_imbalanced_metrics(self, y_true, y_pred, y_proba, y_val_true=None, y_val_proba=None):
        from sklearn.metrics import (
            roc_auc_score, average_precision_score, f1_score,
            precision_score, recall_score, confusion_matrix,
            balanced_accuracy_score, roc_curve
        )

        fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba)
        roc_data = {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': roc_thresholds.tolist()
        }

        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_proba)
        pr_data = {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'thresholds': pr_thresholds.tolist() if pr_thresholds is not None else []
        }

        if y_val_true is not None and y_val_proba is not None and len(np.unique(y_val_proba)) > 1:
            val_precision, val_recall, val_thresholds = precision_recall_curve(y_val_true, y_val_proba)
            if len(val_thresholds) > 0:
                val_f1_scores = 2 * val_precision[:-1] * val_recall[:-1] / (val_precision[:-1] + val_recall[:-1] + 1e-10)
                best_threshold = val_thresholds[np.argmax(val_f1_scores)]
                y_pred_optimized = (y_proba >= best_threshold).astype(int)
                threshold_source = 'validation'
            else:
                y_pred_optimized = y_pred
                best_threshold = 0.5
                threshold_source = 'default'
        else:
            if len(np.unique(y_proba)) > 1 and len(pr_thresholds) > 0:
                f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-10)
                best_threshold = pr_thresholds[np.argmax(f1_scores)]
                y_pred_optimized = (y_proba >= best_threshold).astype(int)
                threshold_source = 'test'
            else:
                y_pred_optimized = y_pred
                best_threshold = 0.5
                threshold_source = 'default'

        metrics = {
            'auroc': float(roc_auc_score(y_true, y_proba)),
            'pr_auc': float(average_precision_score(y_true, y_proba)),
            'f1_score': float(f1_score(y_true, y_pred_optimized)),
            'f1_macro': float(f1_score(y_true, y_pred_optimized, average='macro')),
            'precision': float(precision_score(y_true, y_pred_optimized, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred_optimized, zero_division=0)),
            'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred_optimized)),
            'best_threshold': float(best_threshold),
            'threshold_source': threshold_source,
            'roc_data': roc_data,
            'pr_data': pr_data
        }

        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred_optimized).ravel()
            metrics.update({
                'sensitivity': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
                'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
                'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}
            })
        except ValueError:
            metrics.update({
                'sensitivity': 0.0,
                'specificity': 0.0,
                'confusion_matrix': {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0}
            })

        return metrics

    def weighted_ensemble_predict(self, X):
        all_proba = []
        total_weight = sum(self.ensemble_weights) if self.ensemble_weights else 1.0

        for (name, model), weight in zip(self.models, self.ensemble_weights):
            try:
                proba = model.predict_proba(X)[:, 1]
                weighted_proba = proba * (weight / total_weight) if total_weight > 0 else proba
                all_proba.append(weighted_proba)
            except Exception as e:
                print(f"Error in model {name}: {str(e)}")
                continue

        if not all_proba:
            try:
                model_name, model = self.models[0]
                proba = model.predict_proba(X)[:, 1]
                all_proba = [proba]
                print(f"Using single model {model_name} for prediction")
            except Exception as e:
                raise ValueError(f"No valid models for prediction: {str(e)}")

        ensemble_proba = np.sum(all_proba, axis=0)
        ensemble_pred = (ensemble_proba > 0.5).astype(int)

        return ensemble_pred, ensemble_proba

    def save_fold_models(self, fold_num):
        if not self.models:
            print("Warning: No models to save")
            return

        fold_model_dir = self.model_dir / f"fold_{fold_num}"
        fold_model_dir.mkdir(parents=True, exist_ok=True)

        for i, (model_name, model) in enumerate(self.models):
            model_path = fold_model_dir / f"{model_name}_{i}.cbm"
            try:
                model.save_model(str(model_path))
                print(f"Saved model: {model_path}")
            except Exception as e:
                print(f"Error saving model {model_name}: {str(e)}")

        weights_dict = {
            'model_names': [name for name, _ in self.models],
            'weights': self.ensemble_weights,
            'fold': fold_num,
            'timestamp': self.timestamp
        }

        weights_path = fold_model_dir / "ensemble_weights.pkl"
        with open(weights_path, 'wb') as f:
            pickle.dump(weights_dict, f)

        print(f"Saved ensemble weights to: {weights_path}")
        return fold_model_dir

    def _aggregate_feature_importance(self, feature_names):
        importance_dict = {name: [] for name in feature_names}

        for model_name, model in self.models:
            try:
                importance = model.get_feature_importance()
                for name, imp in zip(feature_names, importance):
                    importance_dict[name].append(imp)
            except Exception as e:
                print(f"Error getting feature importance from {model_name}: {str(e)}")
                continue

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'mean_importance': [np.mean(importance_dict[name]) if importance_dict[name] else 0.0 for name in feature_names],
            'std_importance': [np.std(importance_dict[name]) if importance_dict[name] else 0.0 for name in feature_names]
        }).sort_values('mean_importance', ascending=False)

        return importance_df

    def calculate_overall_metrics(self, fold_results):
        all_metrics = {}

        metric_names = ['auroc', 'pr_auc', 'f1_score', 'precision', 'recall',
                       'balanced_accuracy', 'sensitivity', 'specificity']

        for metric in metric_names:
            values = []
            for result in fold_results:
                if metric in result.metrics:
                    value = result.metrics[metric]
                    if isinstance(value, (int, float)):
                        values.append(float(value))

            if values:
                from scipy import stats
                mean_val = np.mean(values)
                std_val = np.std(values)
                n = len(values)

                if n > 1:
                    t_value = stats.t.ppf(0.975, n - 1)
                    ci_low = mean_val - t_value * (std_val / np.sqrt(n))
                    ci_high = mean_val + t_value * (std_val / np.sqrt(n))
                else:
                    ci_low = mean_val
                    ci_high = mean_val

                all_metrics[metric] = {
                    'mean': mean_val,
                    'std': std_val,
                    'ci_low': ci_low,
                    'ci_high': ci_high,
                    'values': values
                }

        overall_metrics_path = self.metrics_dir / "overall_metrics.json"
        with open(overall_metrics_path, 'w') as f:
            json.dump(all_metrics, f, indent=2, default=str)
        print(f"\nSaved overall metrics to: {overall_metrics_path}")

        overall_df = pd.DataFrame([
            {
                'metric': metric,
                'mean': stats['mean'],
                'std': stats['std'],
                'ci_low': stats['ci_low'],
                'ci_high': stats['ci_high']
            }
            for metric, stats in all_metrics.items()
        ])
        overall_csv_path = self.metrics_dir / "overall_metrics.csv"
        overall_df.to_csv(overall_csv_path, index=False)

        self._merge_all_fold_predictions(fold_results)

        return all_metrics

    def _merge_all_fold_predictions(self, fold_results):
        all_test_predictions = []
        all_val_predictions = []
        
        for fold_result in fold_results:
            if fold_result.test_predictions_df is not None:
                all_test_predictions.append(fold_result.test_predictions_df)
            
            val_path = self.predictions_dir / f"fold_{fold_result.fold}_val_predictions.csv"
            if val_path.exists():
                val_df = pd.read_csv(val_path)
                all_val_predictions.append(val_df)
        
        if all_test_predictions:
            combined_test = pd.concat(all_test_predictions, ignore_index=True)
            combined_test_path = self.predictions_dir / "all_folds_test_predictions.csv"
            combined_test.to_csv(combined_test_path, index=False)
            print(f"Combined all test predictions: {combined_test_path}")
            
            simple_test = combined_test[['fold', 'y_true', 'y_pred', 'y_proba']].copy()
            simple_test_path = self.predictions_dir / "all_folds_test_predictions_simple.csv"
            simple_test.to_csv(simple_test_path, index=False)
        
        if all_val_predictions:
            combined_val = pd.concat(all_val_predictions, ignore_index=True)
            combined_val_path = self.predictions_dir / "all_folds_val_predictions.csv"
            combined_val.to_csv(combined_val_path, index=False)
            print(f"Combined all validation predictions: {combined_val_path}")

    def save_visualization_data(self, fold_results):
        print("\n" + "="*60)
        print("SAVING VISUALIZATION DATA")
        print("="*60)
        
        roc_data = []
        for fold_result in fold_results:
            if 'roc_data' in fold_result.metrics:
                roc_data.append({
                    'fold': fold_result.fold,
                    'fpr': fold_result.metrics['roc_data']['fpr'],
                    'tpr': fold_result.metrics['roc_data']['tpr']
                })
        
        if roc_data:
            roc_path = self.visualization_dir / "roc_curve_data.json"
            with open(roc_path, 'w') as f:
                json.dump(roc_data, f, indent=2)
            print(f"Saved ROC curve data to: {roc_path}")
        
        pr_data = []
        for fold_result in fold_results:
            if 'pr_data' in fold_result.metrics:
                pr_data.append({
                    'fold': fold_result.fold,
                    'precision': fold_result.metrics['pr_data']['precision'],
                    'recall': fold_result.metrics['pr_data']['recall']
                })
        
        if pr_data:
            pr_path = self.visualization_dir / "pr_curve_data.json"
            with open(pr_path, 'w') as f:
                json.dump(pr_data, f, indent=2)
            print(f"Saved PR curve data to: {pr_path}")
        
        confusion_data = []
        for fold_result in fold_results:
            if 'confusion_matrix' in fold_result.metrics:
                confusion_data.append({
                    'fold': fold_result.fold,
                    'tn': fold_result.metrics['confusion_matrix']['tn'],
                    'fp': fold_result.metrics['confusion_matrix']['fp'],
                    'fn': fold_result.metrics['confusion_matrix']['fn'],
                    'tp': fold_result.metrics['confusion_matrix']['tp']
                })
        
        if confusion_data:
            confusion_path = self.visualization_dir / "confusion_matrix_data.json"
            with open(confusion_path, 'w') as f:
                json.dump(confusion_data, f, indent=2)
            print(f"Saved confusion matrix data to: {confusion_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Train on predefined 5-fold data (imbalanced data focused) - data leakage fixed version')
    parser.add_argument('--folds_dir', type=str, default='folds_data',
                       help='5-fold data directory (default: folds_data)')
    parser.add_argument('--config_path', type=str, default='configs/model_config.yaml',
                       help='Model config file path (default: configs/model_config.yaml)')
    parser.add_argument('--target_col', type=str, default='y',
                       help='Target column name (default: y)')
    parser.add_argument('--imbalance_method', type=str, default='class_weights',
                       choices=['class_weights', 'smote', 'undersample', 'smote_tomek'],
                       help='Imbalanced data handling method (default: class_weights)')

    args = parser.parse_args()

    print(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")

    trainer = ImbalancedDataCatBoostTrainer(
        folds_dir=args.folds_dir,
        config_path=args.config_path
    )

    print("\n" + "="*60)
    print("IMBALANCED DATA HANDLING CONFIGURATION (DATA LEAKAGE FIXED)")
    print("="*60)
    print(f"Imbalance method: {args.imbalance_method}")
    print("="*60)

    fold_results = trainer.train_on_folds(
        target_col=args.target_col,
        imbalance_method=args.imbalance_method
    )

    overall_metrics = trainer.calculate_overall_metrics(fold_results)

    trainer.save_visualization_data(fold_results)

    print("\n" + "="*60)
    print("OVERALL PERFORMANCE (95% CI) - DATA LEAKAGE FREE")
    print("="*60)

    key_metrics = ['auroc', 'pr_auc', 'f1_score', 'precision', 'recall', 'sensitivity', 'specificity']
    for metric in key_metrics:
        if metric in overall_metrics:
            values = overall_metrics[metric]
            if isinstance(values, dict) and 'mean' in values:
                mean_val = values['mean']
                ci_low = values.get('ci_low', mean_val)
                ci_high = values.get('ci_high', mean_val)
                print(f"{metric.upper():20s}: {mean_val:.4f} ({ci_low:.4f}-{ci_high:.4f})")

    print("\n" + "="*60)
    print("EXPERIMENT COMPLETED - DATA LEAKAGE FREE")
    print("="*60)
    print(f"All models and results saved in: {trainer.model_dir}")
    print(f"All predictions saved in: {trainer.predictions_dir}")
    print(f"All metrics saved in: {trainer.metrics_dir}")
    print(f"Visualization data saved in: {trainer.visualization_dir}")
    
    print("\n" + "="*60)
    print("HOW TO USE THE SAVED DATA FOR VISUALIZATION")
    print("="*60)
    print("1. For ROC curves, use files in: predictions/")
    print("   - all_folds_test_predictions_simple.csv (for ROC curves)")
    print("   - visualization/roc_curve_data.json (pre-computed ROC points)")
    print("2. For metrics analysis, use files in: metrics/")
    print("3. For confusion matrices, use: visualization/confusion_matrix_data.json")


if __name__ == "__main__":
    main()