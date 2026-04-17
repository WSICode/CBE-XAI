import numpy as np
import pandas as pd
import pickle
import warnings
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
import argparse
import sys
import gc

sys.path.append('utils')
try:
    from robust_data_preprocessor import RobustBankDataPreprocessor
except ImportError:
    print("Warning: Could not import RobustBankDataPreprocessor")
    RobustBankDataPreprocessor = None

warnings.filterwarnings('ignore')
np.random.seed(42)


class PreDeploymentAdaptiveFineTuning:
    """
    Pre-deployment Adaptive Fine-Tuning Strategy
    
    Corresponds to Section 3.3 of the paper:
    - Target institution uses local historical labeled data for lightweight fine-tuning of pre-trained model
    - Uses composite loss function to balance global knowledge retention and local distribution adaptation
    - This strategy is optional: can use original model directly if no local data is available
    """
    
    def __init__(self, experiment_dir: str, folds_dir: str = 'folds_data'):
        """
        Initialize fine-tuner
        
        Parameters:
        -----------
        experiment_dir : str
            Pre-trained model directory (contains fold_1/fold_2/... subdirectories)
        folds_dir : str
            Original data directory (for obtaining preprocessing statistics)
        """
        self.experiment_dir = Path(experiment_dir)
        self.folds_dir = Path(folds_dir)
        
        if not self.experiment_dir.exists():
            raise ValueError(f"Experiment directory not found: {experiment_dir}")
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"pre_deployment_aft/experiment_{self.timestamp}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "fine_tuned_models").mkdir(exist_ok=True)
        (self.output_dir / "results").mkdir(exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)
        
        print(f"Output directory: {self.output_dir}")
        
        self.model_names_map = {
            'Focal_Loss': 'FocalLoss',
            'Balanced_Weighted': 'BalancedWeighted',
            'Conservative_Imbalanced': 'Conservative',
            'Fast_Imbalanced': 'Fast',
            'Deep_Imbalanced': 'Deep'
        }
    
    def load_source_preprocessor_and_stats(self, fold_num: int = 1):
        """
        Load source domain (Portuguese data) preprocessor and imputation statistics
        
        These statistics will be used for type-sensitive imputation of target domain data
        """
        print(f"\nLoading source domain preprocessing statistics (Fold {fold_num})...")
        
        fold_data_dir = self.folds_dir / f"fold_{fold_num}"
        if not fold_data_dir.exists():
            raise ValueError(f"Fold data directory not found: {fold_data_dir}")
        
        train_data = pd.read_csv(fold_data_dir / "train_data.csv")
        
        if RobustBankDataPreprocessor is not None:
            preprocessor = RobustBankDataPreprocessor()
            train_combined = train_data.copy()
            X_train_processed, y_train_processed, feature_info = preprocessor.preprocess(
                train_combined, is_training=True
            )
            
            numerical_cols = feature_info.get('numerical_features', [])
            categorical_cols = feature_info.get('categorical_features', [])
            
            mean_vals = {}
            mode_vals = {}
            
            for col in numerical_cols:
                if col in X_train_processed.columns:
                    mean_vals[col] = X_train_processed[col].mean()
            
            for col in categorical_cols:
                if col in X_train_processed.columns:
                    mode_vals[col] = X_train_processed[col].mode().iloc[0] if len(X_train_processed[col].mode()) > 0 else 0
            
            print(f"  Numerical features: {len(mean_vals)}")
            print(f"  Categorical features: {len(mode_vals)}")
            
            return {
                'preprocessor': preprocessor,
                'mean_vals': mean_vals,
                'mode_vals': mode_vals,
                'feature_names': X_train_processed.columns.tolist(),
                'numerical_cols': numerical_cols,
                'categorical_cols': categorical_cols
            }
        else:
            print("  Using simplified preprocessing")
            return {
                'preprocessor': None,
                'mean_vals': {},
                'mode_vals': {},
                'feature_names': train_data.drop(columns=['y']).columns.tolist(),
                'numerical_cols': train_data.select_dtypes(include=['int64', 'float64']).columns.tolist(),
                'categorical_cols': train_data.select_dtypes(include=['object']).columns.tolist()
            }
    
    def load_pretrained_ensemble(self, fold_num: int = 1):
        """
        Load pre-trained ensemble model (trained on Portuguese data)
        """
        print(f"\nLoading pre-trained ensemble model (Fold {fold_num})...")
        
        model_dir = self.experiment_dir / f"fold_{fold_num}"
        if not model_dir.exists():
            raise ValueError(f"Model directory not found: {model_dir}")
        
        weights_path = model_dir / "ensemble_weights.pkl"
        with open(weights_path, 'rb') as f:
            weights_dict = pickle.load(f)
        
        model_names = weights_dict['model_names']
        weights = weights_dict['weights']
        
        models = []
        for i, name in enumerate(model_names):
            model_path = model_dir / f"{name}_{i}.cbm"
            if model_path.exists():
                model = CatBoostClassifier()
                model.load_model(str(model_path))
                models.append({
                    'name': self.model_names_map.get(name, name),
                    'model': model,
                    'weight': weights[i]
                })
                print(f"  Loaded: {name} (weight: {weights[i]:.4f})")
        
        if not models:
            raise ValueError("No models loaded")
        
        print(f"  Total {len(models)} base learners loaded")
        return models
    
    def impute_local_data(self, X_raw: pd.DataFrame, source_stats: dict) -> pd.DataFrame:
        """
        Type-sensitive imputation: use source domain statistics to impute local data
        
        Corresponds to Section 3.3.1 of the paper
        """
        X_imputed = X_raw.copy()
        
        for col, mean_val in source_stats.get('mean_vals', {}).items():
            if col in X_imputed.columns and X_imputed[col].isnull().any():
                X_imputed[col] = X_imputed[col].fillna(mean_val)
        
        for col, mode_val in source_stats.get('mode_vals', {}).items():
            if col in X_imputed.columns and X_imputed[col].isnull().any():
                X_imputed[col] = X_imputed[col].fillna(mode_val)
                if X_imputed[col].dtype == 'float64':
                    X_imputed[col] = X_imputed[col].astype(int)
        
        return X_imputed
    
    def prepare_local_data(self, local_data_path: str, source_stats: dict, 
                          target_col: str = 'y', val_ratio: float = 0.3):
        """
        Prepare local data: imputation + split into fine-tuning set and test set
        
        Parameters:
        -----------
        local_data_path : str
            Local historical data CSV path
        source_stats : dict
            Source domain statistics (for imputation)
        target_col : str
            Target variable column name
        val_ratio : float
            Test set ratio (default 0.3, 7:3 split)
        
        Returns:
        --------
        X_ft, y_ft, X_test, y_test : fine-tuning set and test set
        """
        print(f"\nLoading local data: {local_data_path}")
        
        df = pd.read_csv(local_data_path)
        print(f"  Original data shape: {df.shape}")
        
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in local data")
        
        X_raw = df.drop(columns=[target_col])
        y = df[target_col].map({'yes': 1, 'no': 0} if df[target_col].dtype == 'object' else {})
        
        print("  Performing type-sensitive imputation...")
        X_imputed = self.impute_local_data(X_raw, source_stats)
        
        source_features = source_stats.get('feature_names', [])
        common_features = [f for f in X_imputed.columns if f in source_features]
        if len(common_features) < len(source_features):
            print(f"  Note: local data missing {len(source_features) - len(common_features)} source domain features")
        
        X_imputed = X_imputed[common_features]
        
        from sklearn.model_selection import train_test_split
        X_ft, X_test, y_ft, y_test = train_test_split(
            X_imputed, y, test_size=val_ratio, random_state=42, stratify=y
        )
        
        print(f"  Fine-tuning set: {X_ft.shape}, Test set: {X_test.shape}")
        print(f"  Fine-tuning set positive ratio: {y_ft.mean():.2%}")
        print(f"  Test set positive ratio: {y_test.mean():.2%}")
        
        return X_ft, y_ft, X_test, y_test
    
    def fine_tune_single_model(self, base_model_info: dict, 
                               X_ft: pd.DataFrame, y_ft: pd.Series,
                               X_source: pd.DataFrame, y_source: pd.Series,
                               beta: float = 0.7, learning_rate: float = 1e-4,
                               n_iterations: int = 200) -> CatBoostClassifier:
        """
        Fine-tune a single base learner
        
        Corresponds to Section 3.3.2 composite loss function optimization
        
        Parameters:
        -----------
        base_model_info : dict
            Base learner info containing 'name', 'model', 'weight'
        X_ft, y_ft : local fine-tuning set
        X_source, y_source : source domain training set (for global knowledge retention)
        beta : balance coefficient (β∈[0.6,0.8])
        learning_rate : learning rate (fixed small value, e.g., 1e-4)
        n_iterations : fine-tuning iterations
        
        Returns:
        --------
        fine_tuned_model : fine-tuned CatBoost model
        """
        print(f"    Fine-tuning {base_model_info['name']} (β={beta}, lr={learning_rate})...")
        
        if len(X_source) > 1000:
            sample_idx = np.random.choice(len(X_source), 1000, replace=False)
            X_source_sample = X_source.iloc[sample_idx]
            y_source_sample = y_source.iloc[sample_idx]
        else:
            X_source_sample = X_source
            y_source_sample = y_source
        
        pretrained_model = base_model_info['model']
        
        cat_features_indices = []
        for i, col in enumerate(X_ft.columns):
            if col in X_ft.select_dtypes(include=['int64']).columns:
                cat_features_indices.append(i)
        
        params = {
            'iterations': n_iterations,
            'learning_rate': learning_rate,
            'depth': pretrained_model.get_params().get('depth', 6),
            'loss_function': 'Logloss',
            'verbose': False,
            'random_seed': 42,
            'cat_features': cat_features_indices if cat_features_indices else None,
            'od_type': 'Iter',
            'od_wait': 20,
            'use_best_model': True
        }
        
        n_source = len(X_source_sample)
        n_local = len(X_ft)
        
        n_source_sample = min(n_source, n_local * 2)
        source_idx = np.random.choice(n_source, n_source_sample, replace=False)
        
        X_combined = pd.concat([
            X_ft,
            X_source_sample.iloc[source_idx]
        ], axis=0, ignore_index=True)
        
        y_combined = pd.concat([
            y_ft,
            y_source_sample.iloc[source_idx]
        ], axis=0, ignore_index=True)
        
        sample_weight = np.ones(len(X_combined))
        local_weight_ratio = (1 - beta) / beta if beta > 0 else 10.0
        
        sample_weight[:len(y_ft)] = local_weight_ratio
        
        fine_tuned_model = CatBoostClassifier(**params)
        fine_tuned_model.fit(
            X_combined, y_combined,
            sample_weight=sample_weight,
            verbose=False
        )
        
        return fine_tuned_model
    
    def fine_tune_ensemble(self, pretrained_models: list,
                          X_ft: pd.DataFrame, y_ft: pd.Series,
                          X_source: pd.DataFrame, y_source: pd.Series,
                          beta: float = 0.7, learning_rate: float = 1e-4,
                          n_iterations: int = 200) -> list:
        """
        Fine-tune all base learners of the ensemble model
        
        Corresponds to Section 3.3.2: Fine-tune each base learner independently, keep ensemble weights unchanged
        """
        print(f"\nStarting pre-deployment adaptive fine-tuning...")
        print(f"  β={beta}, learning_rate={learning_rate}, iterations={n_iterations}")
        
        fine_tuned_models = []
        
        for model_info in pretrained_models:
            ft_model = self.fine_tune_single_model(
                model_info, X_ft, y_ft, X_source, y_source,
                beta=beta, learning_rate=learning_rate, n_iterations=n_iterations
            )
            
            fine_tuned_models.append({
                'name': model_info['name'],
                'model': ft_model,
                'weight': model_info['weight']
            })
        
        print(f"  Fine-tuning complete, total {len(fine_tuned_models)} base learners")
        return fine_tuned_models
    
    def ensemble_predict(self, models: list, X: pd.DataFrame) -> np.ndarray:
        """
        Ensemble prediction: weighted average
        """
        all_probas = []
        total_weight = 0
        
        for model_info in models:
            try:
                proba = model_info['model'].predict_proba(X)[:, 1]
                weight = model_info['weight']
                all_probas.append(proba * weight)
                total_weight += weight
            except Exception as e:
                print(f"  Prediction failed {model_info['name']}: {str(e)[:50]}")
                continue
        
        if not all_probas:
            raise ValueError("No models could make predictions")
        
        ensemble_proba = np.sum(all_probas, axis=0) / total_weight
        return ensemble_proba
    
    def evaluate(self, y_true: pd.Series, y_proba: np.ndarray, threshold: float = 0.5) -> dict:
        """
        Evaluate model performance
        """
        y_pred = (y_proba >= threshold).astype(int)
        
        return {
            'auc': roc_auc_score(y_true, y_proba),
            'accuracy': accuracy_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'specificity': recall_score(y_true, y_pred, pos_label=0)
        }
    
    def run_pre_deployment_experiment(self, local_data_path: str, fold_num: int = 1,
                                      beta: float = 0.7, val_ratio: float = 0.3,
                                      fine_tune_iterations: int = 200,
                                      save_models: bool = True) -> dict:
        """
        Run complete pre-deployment fine-tuning experiment
        
        Process:
        1. Load source domain preprocessor statistics
        2. Load pre-trained ensemble model
        3. Prepare local data (imputation + split)
        4. Fine-tune model
        5. Evaluate original model vs fine-tuned model
        
        Parameters:
        -----------
        local_data_path : str
            Local historical labeled data CSV path
        fold_num : int
            Which fold's pre-trained model to use
        beta : float
            Composite loss balance coefficient (0.6-0.8)
        val_ratio : float
            Test set ratio (default 0.3)
        fine_tune_iterations : int
            Fine-tuning iterations
        save_models : bool
            Whether to save fine-tuned models
        
        Returns:
        --------
        results : dict
            Contains evaluation results and models
        """
        print("\n" + "="*70)
        print("Pre-deployment Adaptive Fine-Tuning Strategy Validation Experiment")
        print("="*70)
        
        source_stats = self.load_source_preprocessor_and_stats(fold_num)
        
        pretrained_models = self.load_pretrained_ensemble(fold_num)
        
        X_ft, y_ft, X_test, y_test = self.prepare_local_data(
            local_data_path, source_stats, val_ratio=val_ratio
        )
        
        fold_data_dir = self.folds_dir / f"fold_{fold_num}"
        train_data = pd.read_csv(fold_data_dir / "train_data.csv")
        X_source_raw = train_data.drop(columns=['y'])
        y_source = train_data['y'].map({'yes': 1, 'no': 0})
        
        if source_stats['preprocessor'] is not None:
            train_combined = pd.concat([X_source_raw, y_source], axis=1)
            X_source, _, _ = source_stats['preprocessor'].preprocess(train_combined, is_training=False)
        else:
            X_source = X_source_raw
        
        print("\n[Strategy A] Original pre-trained model direct prediction...")
        y_proba_original = self.ensemble_predict(pretrained_models, X_test)
        metrics_original = self.evaluate(y_test, y_proba_original)
        print(f"  AUROC: {metrics_original['auc']:.4f}")
        print(f"  F1-Score: {metrics_original['f1']:.4f}")
        
        print("\n[Strategy C] Pre-deployment adaptive fine-tuning...")
        fine_tuned_models = self.fine_tune_ensemble(
            pretrained_models, X_ft, y_ft, X_source, y_source,
            beta=beta, learning_rate=1e-4, n_iterations=fine_tune_iterations
        )
        
        y_proba_finetuned = self.ensemble_predict(fine_tuned_models, X_test)
        metrics_finetuned = self.evaluate(y_test, y_proba_finetuned)
        print(f"  AUROC: {metrics_finetuned['auc']:.4f}")
        print(f"  F1-Score: {metrics_finetuned['f1']:.4f}")
        
        improvement_auc = (metrics_finetuned['auc'] - metrics_original['auc']) / metrics_original['auc'] * 100
        improvement_f1 = (metrics_finetuned['f1'] - metrics_original['f1']) / metrics_original['f1'] * 100
        
        print(f"\nPerformance improvement:")
        print(f"  AUROC: {metrics_original['auc']:.4f} → {metrics_finetuned['auc']:.4f} (+{improvement_auc:.2f}%)")
        print(f"  F1: {metrics_original['f1']:.4f} → {metrics_finetuned['f1']:.4f} (+{improvement_f1:.2f}%)")
        
        results = {
            'beta': beta,
            'fine_tune_iterations': fine_tune_iterations,
            'original_metrics': metrics_original,
            'finetuned_metrics': metrics_finetuned,
            'improvement_auc_pct': improvement_auc,
            'improvement_f1_pct': improvement_f1,
            'pretrained_models': pretrained_models,
            'fine_tuned_models': fine_tuned_models
        }
        
        results_df = pd.DataFrame([
            {'Strategy': 'Original (No Fine-tuning)', **metrics_original},
            {'Strategy': 'Pre-deployment AFT', **metrics_finetuned}
        ])
        results_df.to_csv(self.output_dir / "results" / "experiment_results.csv", index=False)
        
        if save_models:
            for i, model_info in enumerate(fine_tuned_models):
                model_path = self.output_dir / "fine_tuned_models" / f"{model_info['name']}_finetuned.cbm"
                model_info['model'].save_model(str(model_path))
            print(f"\nFine-tuned models saved to: {self.output_dir / 'fine_tuned_models'}")
        
        self.plot_comparison(metrics_original, metrics_finetuned)
        
        print("\n" + "="*70)
        print("Experiment complete!")
        print(f"Results saved to: {self.output_dir}")
        print("="*70)
        
        return results
    
    def plot_comparison(self, metrics_original: dict, metrics_finetuned: dict):
        """Plot performance comparison between original and fine-tuned models"""
        metrics = ['auc', 'accuracy', 'f1', 'precision', 'recall', 'specificity']
        labels = ['AUROC', 'Accuracy', 'F1', 'Precision', 'Recall', 'Specificity']
        
        original_values = [metrics_original[m] for m in metrics]
        finetuned_values = [metrics_finetned[m] for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar(x - width/2, original_values, width, label='Original (No Fine-tuning)', color='#FF6B6B')
        bars2 = ax.bar(x + width/2, finetuned_values, width, label='Pre-deployment AFT', color='#4ECDC4')
        
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Performance Comparison: Original vs. Pre-deployment Fine-tuning', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend(loc='lower right')
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3, axis='y')
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Comparison chart saved: {self.output_dir / 'plots' / 'performance_comparison.png'}")


def main():
    parser = argparse.ArgumentParser(description='Pre-deployment adaptive fine-tuning strategy validation')
    parser.add_argument('--experiment_dir', type=str, required=True,
                       help='Pre-trained model directory')
    parser.add_argument('--local_data', type=str, required=True,
                       help='Local historical labeled data CSV path')
    parser.add_argument('--folds_dir', type=str, default='folds_data',
                       help='5-fold data directory')
    parser.add_argument('--fold_num', type=int, default=1,
                       help='Which fold of pre-trained model to use (default: 1)')
    parser.add_argument('--beta', type=float, default=0.7,
                       help='Composite loss balance coefficient (default: 0.7, recommended range 0.6-0.8)')
    parser.add_argument('--val_ratio', type=float, default=0.3,
                       help='Test set ratio (default: 0.3, 7:3 split)')
    parser.add_argument('--iterations', type=int, default=200,
                       help='Fine-tuning iterations (default: 200)')
    
    args = parser.parse_args()
    
    aft = PreDeploymentAdaptiveFineTuning(args.experiment_dir, args.folds_dir)
    
    results = aft.run_pre_deployment_experiment(
        local_data_path=args.local_data,
        fold_num=args.fold_num,
        beta=args.beta,
        val_ratio=args.val_ratio,
        fine_tune_iterations=args.iterations
    )
    
    print("\nFinal results:")
    print(f"  Original model AUROC: {results['original_metrics']['auc']:.4f}")
    print(f"  Fine-tuned model AUROC: {results['finetuned_metrics']['auc']:.4f}")
    print(f"  Improvement: +{results['improvement_auc_pct']:.2f}%")


if __name__ == "__main__":
    main()