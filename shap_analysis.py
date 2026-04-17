import numpy as np
import pandas as pd
import pickle
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from pathlib import Path
from catboost import CatBoostClassifier
from datetime import datetime
import argparse
import sys
import gc

sys.path.append('utils')
try:
    from robust_data_preprocessor import RobustBankDataPreprocessor
except ImportError:
    print("Error: Unable to import RobustBankDataPreprocessor, ensure utils/robust_data_preprocessor.py exists.")

warnings.filterwarnings('ignore')
np.random.seed(42)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class HierarchicalSHAPAnalyzer:
    def __init__(self, experiment_dir, folds_dir='folds_data'):
        self.experiment_dir = Path(experiment_dir)
        self.folds_dir = Path(folds_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        self.output_dir = Path(f"comprehensive_shap_{self.timestamp}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        (self.output_dir / "charts").mkdir(exist_ok=True)
        (self.output_dir / "plot_data").mkdir(exist_ok=True)
        
        self.original_bank_features = [
            'age', 'job', 'marital', 'education', 'default', 'housing', 'loan',
            'contact', 'month', 'day_of_week', 'duration', 'campaign', 'pdays',
            'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx', 
            'cons.conf.idx', 'euribor3m', 'nr.employed'
        ]

    def _load_resources(self, fold_num):
        model_dir = self.experiment_dir / f"fold_{fold_num}"
        fold_data_dir = self.folds_dir / f"fold_{fold_num}"
        
        train_df = pd.read_csv(fold_data_dir / "train_data.csv")
        test_df = pd.read_csv(fold_data_dir / "test_data.csv")
        
        raw_columns = test_df.columns.tolist()
        if 'y' in raw_columns: raw_columns.remove('y')
        
        preprocessor = RobustBankDataPreprocessor()
        X_train_p, _, _ = preprocessor.preprocess(train_df, is_training=True)
        X_test_p, _, _ = preprocessor.preprocess(test_df, is_training=False)
        
        with open(model_dir / "ensemble_weights.pkl", 'rb') as f:
            w_dict = pickle.load(f)
            
        models = []
        for i, name in enumerate(w_dict['model_names']):
            m_path = model_dir / f"{name}_{i}.cbm"
            m = CatBoostClassifier().load_model(str(m_path))
            models.append((name, m))
            
        return models, w_dict['weights'], X_test_p, X_train_p.columns.tolist(), raw_columns

    def run_analysis(self, fold_num, max_feats=15, sample_size=1000):
        print(f"Starting comprehensive hierarchical SHAP analysis [Fold {fold_num}]...")
        models, weights, X_test, proc_feat_names, raw_cols = self._load_resources(fold_num)
        
        if len(X_test) > sample_size:
            X_test_sub = X_test.sample(n=sample_size, random_state=42)
        else:
            X_test_sub = X_test

        individual_importances = {}
        ensemble_shap_matrix = np.zeros(X_test_sub.shape)
        total_w = 0

        for i, (name, m) in enumerate(models):
            print(f"Calculating SHAP: {name}...")
            explainer = shap.TreeExplainer(m)
            s_vals = explainer.shap_values(X_test_sub)
            
            if isinstance(s_vals, list): s_vals = s_vals[1]
            elif len(s_vals.shape) == 3: s_vals = s_vals[:, :, 1]
            
            w = weights[i]
            individual_importances[name] = np.abs(s_vals).mean(axis=0)
            ensemble_shap_matrix += s_vals * w
            total_w += w
            
            del s_vals, explainer
            gc.collect()

        ensemble_shap_matrix /= total_w

        available_orig = [c for c in proc_feat_names if c in self.original_bank_features or c in raw_cols]
        orig_indices = [proc_feat_names.index(c) for c in available_orig]
        
        df_full_imp = pd.DataFrame(individual_importances, index=proc_feat_names)
        df_full_imp['Ensemble'] = np.abs(ensemble_shap_matrix).mean(axis=0)
        
        df_orig_imp = df_full_imp.loc[available_orig].copy()
        ensemble_shap_orig = ensemble_shap_matrix[:, orig_indices]
        X_test_orig = X_test_sub[available_orig]

        top_feats = df_orig_imp.sort_values('Ensemble', ascending=False).head(max_feats).index.tolist()
        df_plot = df_orig_imp.loc[top_feats]

        self._save_all_data(df_orig_imp, df_plot, ensemble_shap_orig, available_orig, fold_num)

        self._plot_global_bar(df_plot, fold_num)
        self._plot_summary(ensemble_shap_orig, X_test_orig, available_orig, fold_num, max_feats)
        self._plot_heatmap(df_plot, fold_num)
        self._plot_grouped_bar(df_plot.head(10), fold_num)
        
        print(f"All analysis complete! Results path: {self.output_dir}")

    def _save_all_data(self, df_orig_imp, df_plot, ensemble_shap_matrix, feat_names, fold_num):
        data_dir = self.output_dir / "plot_data"
        df_orig_imp.to_csv(data_dir / f"fold{fold_num}_global_importance.csv")
        df_plot.to_csv(data_dir / f"fold{fold_num}_top_hierarchical_data.csv")
        pd.DataFrame(ensemble_shap_matrix, columns=feat_names).to_csv(data_dir / f"fold{fold_num}_ensemble_shap_matrix.csv", index=False)

    def _plot_global_bar(self, df, fold_num):
        plt.figure(figsize=(12, 8))
        data = df.sort_values('Ensemble', ascending=True)
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(data)))
        plt.barh(data.index, data['Ensemble'], color=colors)
        plt.title(f"Chart A: Global Feature Importance (Ensemble - Fold {fold_num})", fontsize=15, pad=20)
        plt.xlabel("Mean(|SHAP Value|)", fontsize=12)
        plt.grid(axis='x', linestyle='--', alpha=0.6)
        plt.savefig(self.output_dir / "charts" / "Chart_A_Global_Importance.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_summary(self, shap_values, X, feature_names, fold_num, max_display):
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X, feature_names=feature_names, max_display=max_display, show=False)
        plt.title(f"Chart B: SHAP Summary Plot (Ensemble - Fold {fold_num})", fontsize=15, pad=20)
        plt.savefig(self.output_dir / "charts" / "Chart_B_Summary_Plot.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_heatmap(self, df, fold_num):
        plt.figure(figsize=(14, 8))
        models_only = df.drop(columns=['Ensemble'])
        df_norm = models_only.div(models_only.max(axis=0), axis=1)
        sns.heatmap(df_norm.T, annot=models_only.T, fmt=".3f", cmap="YlGnBu", cbar_kws={'label': 'Norm. Sensitivity'})
        plt.title(f"Chart C: Base Learner Sensitivity Heatmap (Fold {fold_num})", fontsize=15, pad=20)
        plt.xlabel("Original Features", fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.savefig(self.output_dir / "charts" / "Chart_C_Heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_grouped_bar(self, df, fold_num):
        df_melt = df.reset_index().melt(id_vars='index', var_name='Model', value_name='Importance')
        df_melt.columns = ['Feature', 'Model', 'Mean(|SHAP|)']
        plt.figure(figsize=(15, 9))
        sns.barplot(data=df_melt, x='Mean(|SHAP|)', y='Feature', hue='Model', palette='muted')
        plt.title(f"Chart D: Hierarchical SHAP Comparison (Fold {fold_num})", fontsize=15, pad=20)
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.grid(axis='x', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(self.output_dir / "charts" / "Chart_D_GroupedBar.png", dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_dir', type=str, required=True)
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--sample_size', type=int, default=1000)
    parser.add_argument('--max_features', type=int, default=15)
    args = parser.parse_args()

    analyzer = HierarchicalSHAPAnalyzer(args.experiment_dir)
    analyzer.run_analysis(args.fold, sample_size=args.sample_size, max_feats=args.max_features)