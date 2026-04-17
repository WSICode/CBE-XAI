"""
Advanced Metrics Calculator with 95% Confidence Intervals
For SCI Paper Reporting Standards
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                             recall_score, precision_score, confusion_matrix,
                             average_precision_score, roc_curve, precision_recall_curve)
from typing import Dict, Tuple, List, Optional
import warnings

warnings.filterwarnings('ignore')


class MetricsCalculator:
    """高级指标计算器（带95%置信区间）"""

    def __init__(self, alpha: float = 0.05, n_bootstrap: int = 1000):
        self.alpha = alpha  # 显著性水平
        self.n_bootstrap = n_bootstrap  # 自助法重采样次数

    def calculate_all_metrics(self, y_true: np.ndarray,
                              y_pred: np.ndarray,
                              y_proba: np.ndarray) -> Dict:
        """计算所有指标（带95%置信区间）"""

        metrics = {}

        # 1. AUROC (Area Under ROC Curve)
        auroc_result = self.calculate_auroc(y_true, y_proba)
        metrics['auroc'] = auroc_result

        # 2. Accuracy
        accuracy_result = self.calculate_accuracy(y_true, y_pred)
        metrics['accuracy'] = accuracy_result

        # 3. F1-Score
        f1_result = self.calculate_f1_score(y_true, y_pred)
        metrics['f1_score'] = f1_result

        # 4. Sensitivity (Recall)
        sensitivity_result = self.calculate_sensitivity(y_true, y_pred)
        metrics['sensitivity'] = sensitivity_result

        # 5. Specificity
        specificity_result = self.calculate_specificity(y_true, y_pred)
        metrics['specificity'] = specificity_result

        # 6. Precision
        precision_result = self.calculate_precision(y_true, y_pred)
        metrics['precision'] = precision_result

        # 7. Average Precision
        avg_precision_result = self.calculate_average_precision(y_true, y_proba)
        metrics['average_precision'] = avg_precision_result

        # 8. Matthews Correlation Coefficient
        mcc_result = self.calculate_mcc(y_true, y_pred)
        metrics['mcc'] = mcc_result

        # 9. Balanced Accuracy
        balanced_acc_result = self.calculate_balanced_accuracy(y_true, y_pred)
        metrics['balanced_accuracy'] = balanced_acc_result

        # 10. G-Mean
        gmean_result = self.calculate_gmean(y_true, y_pred)
        metrics['gmean'] = gmean_result

        return metrics

    def _ensure_numpy_1d(self, data):
        """确保输入数据是一维 numpy 数组"""
        if isinstance(data, pd.Series):
            return data.values
        elif isinstance(data, pd.DataFrame):
            return data.values.ravel()
        elif isinstance(data, list):
            return np.array(data)
        elif isinstance(data, np.ndarray) and len(data.shape) > 1:
            return data.ravel()
        else:
            return np.array(data)

    def calculate_auroc(self, y_true, y_proba) -> Dict:
        """计算AUROC（带置信区间）"""

        # 转换为 numpy 数组
        y_true_np = self._ensure_numpy_1d(y_true)
        y_proba_np = self._ensure_numpy_1d(y_proba)

        # 检查长度一致
        if len(y_true_np) != len(y_proba_np):
            raise ValueError(f"Length mismatch: y_true={len(y_true_np)}, y_proba={len(y_proba_np)}")

        # 点估计
        try:
            auroc_point = roc_auc_score(y_true_np, y_proba_np)
        except Exception as e:
            print(f"Warning in AUROC calculation: {str(e)}")
            auroc_point = 0.5

        # 自助法
        auroc_bootstrap = []
        n = len(y_true_np)

        for _ in range(self.n_bootstrap):
            indices = np.random.choice(n, n, replace=True)
            y_true_boot = y_true_np[indices]
            y_proba_boot = y_proba_np[indices]

            if len(np.unique(y_true_boot)) < 2:
                continue

            try:
                auroc_boot = roc_auc_score(y_true_boot, y_proba_boot)
                auroc_bootstrap.append(auroc_boot)
            except:
                continue

        if auroc_bootstrap:
            ci_lower = np.percentile(auroc_bootstrap, self.alpha / 2 * 100)
            ci_upper = np.percentile(auroc_bootstrap, (1 - self.alpha / 2) * 100)
            std_err = np.std(auroc_bootstrap)
        else:
            ci_lower = ci_upper = auroc_point
            std_err = 0

        return {
            'mean': auroc_point,
            'ci_low': ci_lower,
            'ci_high': ci_upper,
            'std': std_err,
            'n_samples': n
        }

    def calculate_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """计算准确率（带置信区间）"""

        # 点估计
        acc_point = accuracy_score(y_true, y_pred)

        # 使用二项分布置信区间
        n = len(y_true)
        k = np.sum(y_true == y_pred)

        # Wilson分数区间
        z = stats.norm.ppf(1 - self.alpha / 2)
        denominator = 1 + z ** 2 / n

        centre_adjusted_probability = k / n + z ** 2 / (2 * n)
        adjusted_standard_deviation = np.sqrt(
            (k / n) * (1 - k / n) / n + z ** 2 / (4 * n ** 2)
        )

        ci_lower = (centre_adjusted_probability - z * adjusted_standard_deviation) / denominator
        ci_upper = (centre_adjusted_probability + z * adjusted_standard_deviation) / denominator

        return {
            'mean': acc_point,
            'ci_low': ci_lower,
            'ci_high': ci_upper,
            'std': np.sqrt(acc_point * (1 - acc_point) / n),
            'n_samples': n
        }

    def calculate_f1_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """计算F1分数（带置信区间）"""

        # 点估计
        f1_point = f1_score(y_true, y_pred)

        # 自助法计算置信区间
        f1_bootstrap = []
        n = len(y_true)

        for _ in range(self.n_bootstrap):
            indices = np.random.choice(n, n, replace=True)
            if len(np.unique(y_true[indices])) < 2:
                continue
            try:
                f1_boot = f1_score(y_true[indices], y_pred[indices])
                f1_bootstrap.append(f1_boot)
            except:
                continue

        if f1_bootstrap:
            ci_lower = np.percentile(f1_bootstrap, self.alpha / 2 * 100)
            ci_upper = np.percentile(f1_bootstrap, (1 - self.alpha / 2) * 100)
            std_err = np.std(f1_bootstrap)
        else:
            ci_lower = ci_upper = std_err = np.nan

        return {
            'mean': f1_point,
            'ci_low': ci_lower,
            'ci_high': ci_upper,
            'std': std_err
        }

    def calculate_sensitivity(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """计算敏感度（召回率）"""
        return self._calculate_binary_metric(
            y_true, y_pred, metric_func=recall_score, pos_label=1
        )

    def calculate_specificity(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """计算特异度"""
        return self._calculate_binary_metric(
            y_true, y_pred, metric_func=recall_score, pos_label=0
        )

    def calculate_precision(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """计算精确率"""
        return self._calculate_binary_metric(
            y_true, y_pred, metric_func=precision_score
        )

    def _calculate_binary_metric(self, y_true: np.ndarray, y_pred: np.ndarray,
                                 metric_func, **kwargs) -> Dict:
        """计算二分类指标（带置信区间）"""

        # 点估计
        metric_point = metric_func(y_true, y_pred, **kwargs)

        # 自助法计算置信区间
        metric_bootstrap = []
        n = len(y_true)

        for _ in range(self.n_bootstrap):
            indices = np.random.choice(n, n, replace=True)
            if len(np.unique(y_true[indices])) < 2:
                continue
            try:
                metric_boot = metric_func(y_true[indices], y_pred[indices], **kwargs)
                metric_bootstrap.append(metric_boot)
            except:
                continue

        if metric_bootstrap:
            ci_lower = np.percentile(metric_bootstrap, self.alpha / 2 * 100)
            ci_upper = np.percentile(metric_bootstrap, (1 - self.alpha / 2) * 100)
            std_err = np.std(metric_bootstrap)
        else:
            ci_lower = ci_upper = std_err = np.nan

        return {
            'mean': metric_point,
            'ci_low': ci_lower,
            'ci_high': ci_upper,
            'std': std_err
        }

    def calculate_average_precision(self, y_true: np.ndarray, y_proba: np.ndarray) -> Dict:
        """计算平均精确率"""
        ap_point = average_precision_score(y_true, y_proba)

        # 自助法
        ap_bootstrap = []
        n = len(y_true)

        for _ in range(self.n_bootstrap):
            indices = np.random.choice(n, n, replace=True)
            if len(np.unique(y_true[indices])) < 2:
                continue
            try:
                ap_boot = average_precision_score(y_true[indices], y_proba[indices])
                ap_bootstrap.append(ap_boot)
            except:
                continue

        if ap_bootstrap:
            ci_lower = np.percentile(ap_bootstrap, self.alpha / 2 * 100)
            ci_upper = np.percentile(ap_bootstrap, (1 - self.alpha / 2) * 100)
            std_err = np.std(ap_bootstrap)
        else:
            ci_lower = ci_upper = std_err = np.nan

        return {
            'mean': ap_point,
            'ci_low': ci_lower,
            'ci_high': ci_upper,
            'std': std_err
        }

    def calculate_mcc(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """计算马修斯相关系数"""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        mcc_point = (tp * tn - fp * fn) / np.sqrt(
            (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + 1e-10
        )

        return {
            'mean': mcc_point,
            'ci_low': mcc_point - 1.96 * np.sqrt((1 - mcc_point ** 2) / len(y_true)),
            'ci_high': mcc_point + 1.96 * np.sqrt((1 - mcc_point ** 2) / len(y_true))
        }

    def calculate_balanced_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """计算平衡准确率"""
        sensitivity = recall_score(y_true, y_pred, pos_label=1)
        specificity = recall_score(y_true, y_pred, pos_label=0)
        balanced_acc = (sensitivity + specificity) / 2

        return {
            'mean': balanced_acc,
            'ci_low': balanced_acc - 1.96 * np.sqrt(balanced_acc * (1 - balanced_acc) / len(y_true)),
            'ci_high': balanced_acc + 1.96 * np.sqrt(balanced_acc * (1 - balanced_acc) / len(y_true))
        }

    def calculate_gmean(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """计算G-Mean"""
        sensitivity = recall_score(y_true, y_pred, pos_label=1)
        specificity = recall_score(y_true, y_pred, pos_label=0)
        gmean = np.sqrt(sensitivity * specificity)

        return {
            'mean': gmean,
            'ci_low': gmean - 1.96 * np.sqrt(gmean * (1 - gmean) / len(y_true)),
            'ci_high': gmean + 1.96 * np.sqrt(gmean * (1 - gmean) / len(y_true))
        }

    def generate_latex_table(self, metrics_dict: Dict) -> str:
        """生成LaTeX表格（用于论文）"""

        latex_lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{Model Performance Metrics with 95\\% Confidence Intervals}",
            "\\label{tab:performance_metrics}",
            "\\begin{tabular}{lccc}",
            "\\toprule",
            "Metric & Mean & 95\\% CI Lower & 95\\% CI Upper \\\\",
            "\\midrule"
        ]

        metric_names = {
            'auroc': 'AUROC',
            'accuracy': 'Accuracy',
            'f1_score': 'F1-Score',
            'sensitivity': 'Sensitivity',
            'specificity': 'Specificity',
            'precision': 'Precision',
            'average_precision': 'Average Precision',
            'mcc': 'MCC',
            'balanced_accuracy': 'Balanced Accuracy',
            'gmean': 'G-Mean'
        }

        for key, display_name in metric_names.items():
            if key in metrics_dict:
                metric = metrics_dict[key]
                if isinstance(metric, dict):
                    line = f"{display_name} & {metric['mean']:.4f} & {metric['ci_low']:.4f} & {metric['ci_high']:.4f} \\\\"
                    latex_lines.append(line)

        latex_lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}"
        ])

        return "\n".join(latex_lines)