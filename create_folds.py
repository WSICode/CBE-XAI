import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import warnings
import argparse

warnings.filterwarnings('ignore')


class FoldDataCreator:

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def create_5fold_splits(self, data_path: str, output_dir: str = "folds_data",
                            target_col: str = 'y', val_frac: float = 0.1, test_frac: float = 0.2):

        print(f"Loading data from: {data_path}")
        data = pd.read_csv(data_path)
        print(f"Original data shape: {data.shape}")

        if target_col not in data.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")

        original_nan_count = data[target_col].isna().sum()
        print(f"Original target variable NaN count: {original_nan_count}")

        if original_nan_count > 0:
            print("Warning: Original data contains NaN in target variable")
            print("Filling NaN values with 0")
            data[target_col] = data[target_col].fillna(0)

        if data[target_col].dtype == 'object':
            print("Encoding target variable...")
            data[target_col] = data[target_col].astype(str).str.lower()

            mapping_dict = {
                'yes': 1, 'no': 0, '1': 1, '0': 0,
                'true': 1, 'false': 0, 't': 1, 'f': 0,
                '是': 1, '否': 0, '成功': 1, '失败': 0,
                'positive': 1, 'negative': 0, 'pos': 1, 'neg': 0
            }

            data[target_col] = data[target_col].map(mapping_dict)

            unmapped_mask = data[target_col].isna()
            if unmapped_mask.any():
                unmapped_values = data.loc[unmapped_mask, target_col].unique()
                print(f"Warning: Found unmapped values: {unmapped_values}")
                print(f"Number of unmapped samples: {unmapped_mask.sum()}")
                data.loc[unmapped_mask, target_col] = 0

            data[target_col] = pd.to_numeric(data[target_col], errors='coerce').fillna(0).astype(int)

        final_nan_count = data[target_col].isna().sum()
        if final_nan_count > 0:
            raise ValueError(f"Target variable still contains {final_nan_count} NaN values after processing")

        print(f"Final class distribution:\n{data[target_col].value_counts(normalize=True)}")
        print(f"Target variable unique values: {data[target_col].unique()}")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        original_data_path = output_path / "original_data.csv"
        data.to_csv(original_data_path, index=False)
        print(f"Original data saved to: {original_data_path}")

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)

        X = data.drop(columns=[target_col])
        y = data[target_col]

        fold_info = []

        for fold, (train_val_idx, test_idx) in enumerate(skf.split(X, y)):
            print(f"\n{'=' * 50}")
            print(f"Creating Fold {fold + 1}/5")
            print(f"{'=' * 50}")

            X_train_val, X_test = X.iloc[train_val_idx].copy(), X.iloc[test_idx].copy()
            y_train_val, y_test = y.iloc[train_val_idx].copy(), y.iloc[test_idx].copy()

            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val,
                test_size=val_frac,
                stratify=y_train_val,
                random_state=self.random_state
            )

            self._validate_fold_data(X_train, y_train, "train", fold + 1)
            self._validate_fold_data(X_val, y_val, "val", fold + 1)
            self._validate_fold_data(X_test, y_test, "test", fold + 1)

            fold_dir = output_path / f"fold_{fold + 1}"
            fold_dir.mkdir(exist_ok=True)

            datasets = {
                'train': (X_train, y_train),
                'val': (X_val, y_val),
                'test': (X_test, y_test),
                'train_val': (X_train_val, y_train_val)
            }

            for name, (X_data, y_data) in datasets.items():
                if y_data.isna().any():
                    raise ValueError(f"Target variable in {name} set contains NaN values")

                X_data = X_data.reset_index(drop=True)
                y_data = y_data.reset_index(drop=True)

                data_to_save = pd.concat([X_data, y_data], axis=1)
                file_path = fold_dir / f"{name}_data.csv"
                data_to_save.to_csv(file_path, index=False)
                print(f"  {name}: {X_data.shape} -> saved to {file_path}")

            indices = {
                'train_idx': [int(idx) for idx in X_train.index.tolist()],
                'val_idx': [int(idx) for idx in X_val.index.tolist()],
                'test_idx': [int(idx) for idx in X_test.index.tolist()],
                'train_val_idx': [int(idx) for idx in X_train_val.index.tolist()]
            }

            indices_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in indices.items()]))
            indices_path = fold_dir / "indices.csv"
            indices_df.to_csv(indices_path, index=False)

            stats = {
                'Dataset': ['train', 'val', 'test', 'train_val'],
                'Samples': [int(len(X_train)), int(len(X_val)), int(len(X_test)), int(len(X_train_val))],
                'Positive_Ratio': [
                    float(y_train.mean()), float(y_val.mean()), float(y_test.mean()), float(y_train_val.mean())
                ],
                'Positive_Count': [
                    int(y_train.sum()), int(y_val.sum()), int(y_test.sum()), int(y_train_val.sum())
                ],
                'NaN_Count': [
                    int(y_train.isna().sum()), int(y_val.isna().sum()),
                    int(y_test.isna().sum()), int(y_train_val.isna().sum())
                ]
            }

            stats_df = pd.DataFrame(stats)
            stats_path = fold_dir / "dataset_stats.csv"
            stats_df.to_csv(stats_path, index=False)

            fold_info.append({
                'fold': int(fold + 1),
                'train_samples': int(len(X_train)),
                'val_samples': int(len(X_val)),
                'test_samples': int(len(X_test)),
                'train_pos_ratio': float(y_train.mean()),
                'val_pos_ratio': float(y_val.mean()),
                'test_pos_ratio': float(y_test.mean()),
                'train_nan_count': int(y_train.isna().sum()),
                'val_nan_count': int(y_val.isna().sum()),
                'test_nan_count': int(y_test.isna().sum()),
                'train_data_path': str(fold_dir / "train_data.csv"),
                'val_data_path': str(fold_dir / "val_data.csv"),
                'test_data_path': str(fold_dir / "test_data.csv"),
                'indices_path': str(fold_dir / "indices.csv")
            })

        fold_info_df = pd.DataFrame(fold_info)
        fold_info_path = output_path / "fold_info_summary.csv"
        fold_info_df.to_csv(fold_info_path, index=False)

        config = {
            'data_path': str(data_path),
            'output_dir': str(output_path),
            'target_col': str(target_col),
            'val_frac': float(val_frac),
            'test_frac': float(test_frac),
            'random_state': int(self.random_state),
            'total_samples': int(len(data)),
            'positive_ratio': float(y.mean()),
            'target_cleaning_log': {
                'original_nan_count': int(original_nan_count),
                'final_nan_count': int(final_nan_count),
                'target_values': [int(val) for val in y.unique().tolist()]
            }
        }

        import json
        config_path = output_path / "split_config.json"

        class NumpySafeEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                else:
                    return super().default(obj)

        try:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2, cls=NumpySafeEncoder)
            print(f"Configuration saved to: {config_path}")
        except Exception as e:
            print(f"Warning: Could not save config as JSON: {e}")
            config_text_path = output_path / "split_config.txt"
            with open(config_text_path, 'w') as f:
                for key, value in config.items():
                    f.write(f"{key}: {value}\n")
            print(f"Configuration saved as text to: {config_text_path}")

        print(f"\n{'=' * 50}")
        print("5-FOLD DATA SPLITTING COMPLETED")
        print(f"{'=' * 50}")
        print(f"Output directory: {output_path}")
        print(f"Target variable validation: ✓ No NaN values")

        self._validate_all_folds(output_path, target_col)

        return fold_info_df

    def _validate_fold_data(self, X, y, dataset_name, fold_num):
        y_nan = int(y.isna().sum())
        if y_nan > 0:
            raise ValueError(f"Fold {fold_num} {dataset_name} set: target contains {y_nan} NaN values")

        X_nan = int(X.isna().sum().sum())
        if X_nan > 0:
            print(f"Warning: Fold {fold_num} {dataset_name} set: features contain {X_nan} NaN values")

        print(f"  {dataset_name} set validation: {int(len(y))} samples, {y_nan} target NaN, {X_nan} feature NaN")

    def _validate_all_folds(self, output_path, target_col):
        print(f"\nValidating all folds...")

        all_valid = True
        for fold_num in range(1, 6):
            fold_dir = output_path / f"fold_{fold_num}"
            if not fold_dir.exists():
                print(f"Fold {fold_num} directory not found: {fold_dir}")
                all_valid = False
                continue

            try:
                train_data = pd.read_csv(fold_dir / "train_data.csv")
                val_data = pd.read_csv(fold_dir / "val_data.csv")
                test_data = pd.read_csv(fold_dir / "test_data.csv")

                train_nan = int(train_data[target_col].isna().sum())
                val_nan = int(val_data[target_col].isna().sum())
                test_nan = int(test_data[target_col].isna().sum())

                if train_nan + val_nan + test_nan == 0:
                    print(f"✓ Fold {fold_num}: All target variables are clean")
                else:
                    print(f"✗ Fold {fold_num}: Target NaN - Train: {train_nan}, Val: {val_nan}, Test: {test_nan}")
                    all_valid = False

            except Exception as e:
                print(f"Error validating fold {fold_num}: {e}")
                all_valid = False

        if all_valid:
            print("✓ All folds validation passed successfully!")
        else:
            print("✗ Some folds have validation issues")

        return all_valid


def main():
    parser = argparse.ArgumentParser(description='Create 5-fold cross-validation dataset splits')
    parser.add_argument('--data_path', type=str, default='bank_subscription.csv',
                        help='Path to the original data file (default: bank_subscription.csv)')
    parser.add_argument('--output_dir', type=str, default='folds_data',
                        help='Output directory (default: folds_data)')
    parser.add_argument('--target_col', type=str, default='y',
                        help='Target column name (default: y)')
    parser.add_argument('--val_frac', type=float, default=0.1,
                        help='Validation set fraction (default: 0.1)')
    parser.add_argument('--test_frac', type=float, default=0.2,
                        help='Test set fraction (default: 0.2)')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed (default: 42)')

    args = parser.parse_args()

    print("=" * 60)
    print("5-FOLD CROSS-VALIDATION DATA SPLITTER (FIXED VERSION)")
    print("=" * 60)
    print(f"Input data: {args.data_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Target column: {args.target_col}")
    print(f"Validation fraction: {args.val_frac}")
    print(f"Test fraction: {args.test_frac}")
    print(f"Random state: {args.random_state}")
    print("=" * 60)

    creator = FoldDataCreator(random_state=args.random_state)

    try:
        fold_info = creator.create_5fold_splits(
            data_path=args.data_path,
            output_dir=args.output_dir,
            target_col=args.target_col,
            val_frac=args.val_frac,
            test_frac=args.test_frac
        )

        print(f"\nFold summary:")
        print(fold_info[['fold', 'train_samples', 'val_samples', 'test_samples',
                         'train_pos_ratio', 'val_pos_ratio', 'test_pos_ratio',
                         'train_nan_count', 'val_nan_count', 'test_nan_count']].to_string(index=False))

    except Exception as e:
        print(f"Error creating folds: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()