"""
健壮的银行数据预处理器，修复NaN转换问题
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import warnings

warnings.filterwarnings('ignore')


class RobustBankDataPreprocessor:
    """健壮的银行数据预处理器"""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.selected_features = None

        # 预定义特征类型
        self.categorical_features = [
            'job', 'marital', 'education', 'default', 'housing', 'loan',
            'contact', 'month', 'day_of_week', 'poutcome'
        ]

        self.numerical_features = [
            'age', 'duration', 'campaign', 'pdays', 'previous',
            'emp.var.rate', 'cons.price.idx', 'cons.conf.idx',
            'euribor3m', 'nr.employed'
        ]

    def preprocess(self, data: pd.DataFrame, target_col: str = 'y') -> tuple:
        """完整的预处理流程"""
        print("Starting robust preprocessing for banking data...")
        print(f"Initial data shape: {data.shape}")

        # 复制数据
        df = data.copy()

        # 1. 处理目标变量
        if target_col in df.columns:
            y = self._encode_target(df[target_col])
            df = df.drop(columns=[target_col])
            print(f"Target variable encoded. Positive ratio: {y.mean():.3f}")
        else:
            raise ValueError(f"Target column {target_col} not found")

        # 2. 基本清洗
        df = self._basic_cleaning(df)

        # 3. 识别特征类型
        self._identify_feature_types(df)

        # 4. 处理分类变量
        df, cat_indices = self._encode_categorical(df)

        # 5. 处理数值特征
        df = self._process_numerical(df)

        # 6. 高级特征工程（健壮版）
        df = self._robust_feature_engineering(df)

        # 7. 处理缺失值
        df = self._handle_missing_values(df)

        # 8. 特征选择（可选）
        df, selected_features = self._feature_selection(df, y)

        # 9. 特征缩放
        df = self._scale_features(df)

        # 准备特征信息
        feature_info = {
            'cat_indices': cat_indices,
            'categorical_features': [col for col in self.categorical_features if col in df.columns],
            'numerical_features': [col for col in self.numerical_features if col in df.columns],
            'selected_features': selected_features,
            'original_columns': data.columns.tolist(),
            'processed_columns': df.columns.tolist()
        }

        print(f"Preprocessing complete. Final shape: {df.shape}")
        return df, y, feature_info

    def _robust_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """健壮的特征工程"""
        print("Performing robust feature engineering...")

        # 1. 年龄分组 - 处理边界和NaN
        if 'age' in df.columns:
            try:
                # 确保age没有NaN
                age_clean = df['age'].fillna(df['age'].median())
                # 使用更安全的边界
                bins = [0, 25, 35, 45, 55, 65, 100]
                labels = list(range(len(bins)-1))
                age_group = pd.cut(age_clean, bins=bins, labels=labels, include_lowest=True)
                # 将NaN填充为-1
                df['age_group'] = age_group.cat.add_categories([-1]).fillna(-1).astype('int64')
            except Exception as e:
                print(f"Warning: Age grouping skipped: {e}")
                df['age_group'] = -1

        # 2. 通话时长相关特征
        if 'duration' in df.columns:
            df['duration_minutes'] = df['duration'] / 60
            # 使用安全的log转换
            df['log_duration'] = np.log1p(df['duration'].clip(lower=0))

        # 3. 营销活动相关特征
        if 'campaign' in df.columns and 'previous' in df.columns:
            df['total_contacts'] = df['campaign'] + df['previous']
            df['contact_intensity'] = df['campaign'] / (df['previous'].clip(lower=0) + 1)

        # 4. 经济指标交互特征
        economic_features = ['emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m']
        if all(feat in df.columns for feat in economic_features):
            df['economic_stress'] = (df['emp.var.rate'] + df['cons.conf.idx']) / 2

        # 5. 时间特征编码 - 使用字典get方法避免KeyError
        if 'month' in df.columns:
            month_mapping = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                           'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
            df['month'] = df['month'].apply(lambda x: month_mapping.get(str(x).lower(), 0))
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        if 'day_of_week' in df.columns:
            day_mapping = {'mon': 1, 'tue': 2, 'wed': 3, 'thu': 4, 'fri': 5}
            df['day_of_week'] = df['day_of_week'].apply(lambda x: day_mapping.get(str(x).lower(), 0))
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 5)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 5)

        # 6. 客户价值相关特征
        if 'age' in df.columns and 'duration' in df.columns:
            df['age_duration_ratio'] = df['age'] / (df['duration'].clip(lower=1) + 1)

        print(f"Created {len([col for col in df.columns if col not in self.categorical_features + self.numerical_features])} new features")
        return df

    # 其他方法保持不变...
    def _encode_target(self, target: pd.Series) -> pd.Series:
        """编码目标变量"""
        if target.dtype == 'object':
            target_encoded = target.map({'yes': 1, 'no': 0, '1': 1, '0': 0})
            if target_encoded.isna().any():
                print("Warning: Unknown values in target variable")
                target_encoded = target_encoded.fillna(0)
            return target_encoded
        return target

    def _basic_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """基本数据清洗"""
        print("Performing basic data cleaning...")

        # 移除常数特征
        constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
        if constant_cols:
            print(f"Removing constant columns: {constant_cols}")
            df = df.drop(columns=constant_cols)

        # 移除高缺失率特征
        missing_rates = df.isna().mean()
        high_missing_cols = missing_rates[missing_rates > 0.5].index.tolist()
        if high_missing_cols:
            print(f"Removing columns with >50% missing: {high_missing_cols}")
            df = df.drop(columns=high_missing_cols)

        # 处理特殊值
        if 'pdays' in df.columns:
            df['pdays'] = df['pdays'].replace(999, -1)

        return df

    def _identify_feature_types(self, df: pd.DataFrame):
        """识别特征类型"""
        self.categorical_features = [col for col in self.categorical_features if col in df.columns]
        self.numerical_features = [col for col in self.numerical_features if col in df.columns]

        remaining_cols = [col for col in df.columns if col not in self.categorical_features + self.numerical_features]

        for col in remaining_cols:
            if df[col].dtype == 'object' or df[col].nunique() < 20:
                self.categorical_features.append(col)
            else:
                self.numerical_features.append(col)

        print(f"Identified {len(self.categorical_features)} categorical features")
        print(f"Identified {len(self.numerical_features)} numerical features")

    def _encode_categorical(self, df: pd.DataFrame) -> tuple:
        """编码分类变量"""
        print("Encoding categorical variables...")

        cat_indices = []
        encoded_count = 0

        for i, col in enumerate(df.columns):
            if col in self.categorical_features:
                # 确保分类特征是字符串类型
                df[col] = df[col].astype(str)

                # 处理未知值
                df[col] = df[col].replace('unknown', 'missing')
                df[col] = df[col].replace('nonexistent', 'missing')
                df[col] = df[col].fillna('missing')

                # 使用LabelEncoder
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                self.label_encoders[col] = le

                # 记录分类特征索引
                cat_indices.append(i)
                encoded_count += 1

        print(f"Encoded {encoded_count} categorical variables")
        return df, cat_indices

    def _process_numerical(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理数值特征"""
        print("Processing numerical features...")

        num_cols = [col for col in self.numerical_features if col in df.columns]

        if not num_cols:
            return df

        # 处理异常值
        for col in num_cols:
            if col in df.columns and col != 'pdays':
                q1 = df[col].quantile(0.01)
                q3 = df[col].quantile(0.99)
                df[col] = df[col].clip(lower=q1, upper=q3)

        # 创建统计特征
        if len(num_cols) > 1:
            available_num_cols = [col for col in num_cols if col in df.columns]
            if available_num_cols:
                df['numeric_mean'] = df[available_num_cols].mean(axis=1)
                df['numeric_std'] = df[available_num_cols].std(axis=1)

        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理缺失值"""
        print("Handling missing values...")

        # 对于分类特征，用众数填充
        for col in self.categorical_features:
            if col in df.columns and df[col].isna().any():
                mode_val = df[col].mode()[0] if not df[col].mode().empty else 0
                df[col] = df[col].fillna(mode_val)

        # 对于数值特征，用中位数填充
        for col in self.numerical_features:
            if col in df.columns and df[col].isna().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)

        # 处理新创建特征的缺失值
        for col in df.columns:
            if col not in self.categorical_features + self.numerical_features:
                if df[col].isna().any():
                    if df[col].dtype in ['int64', 'int32', 'float64']:
                        df[col] = df[col].fillna(df[col].median())
                    else:
                        df[col] = df[col].fillna(0)

        return df

    def _feature_selection(self, df: pd.DataFrame, y: pd.Series) -> tuple:
        """特征选择"""
        print("Performing feature selection...")

        if len(df.columns) <= 50:
            print("Number of features is small, skipping feature selection")
            return df, df.columns.tolist()

        try:
            mi_scores = mutual_info_classif(df, y, random_state=self.random_state)
            mi_series = pd.Series(mi_scores, index=df.columns)

            n_features = min(50, len(df.columns))
            selected_features = mi_series.nlargest(n_features).index.tolist()

            df_selected = df[selected_features]
            print(f"Selected {len(selected_features)} features using mutual information")

        except Exception as e:
            print(f"Feature selection failed: {e}. Using all features.")
            df_selected = df
            selected_features = df.columns.tolist()

        return df_selected, selected_features

    def _scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """特征缩放"""
        print("Scaling numerical features...")

        num_cols_to_scale = [col for col in df.columns
                           if col in self.numerical_features or
                           col not in self.categorical_features]

        if num_cols_to_scale:
            df[num_cols_to_scale] = self.scaler.fit_transform(df[num_cols_to_scale])

        return df