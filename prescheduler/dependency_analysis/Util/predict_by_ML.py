import logging
import pickle
import argparse
import time
import sys
import numpy as np
import pandas as pd
import os
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
sys.path.append("../../dependency_analysis")
sys.path.append("../../dependency_analysis/DATA/")

from glob_arg import *
from multiprocessing import freeze_support
import gc

DATA_LIMIT = 10000
os.makedirs('logs', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
    handlers=[
        logging.FileHandler("logs/{}_{}.log".format(DATA_LIMIT, time.strftime("%Y%m%d-%H%M%S"))),
        logging.StreamHandler()
    ])
logger = logging.getLogger(__name__)

def load_pkl(path):
    with open(path, 'rb') as f:
        pkl = pickle.load(f)
    return pkl

def create_features(combined_left, combined_right):
    """
    从combined_left和combined_right创建特征
    """
    # 基本特征：left和right的值
    features = np.concatenate([combined_left, combined_right], axis=1)
    
    # 差值特征
    diff_features = combined_right - combined_left
    
    # 比值特征（避免除零和无穷值）
    ratio_features = np.zeros_like(combined_right, dtype=float)
    valid_mask = (combined_left != 0) & np.isfinite(combined_left) & np.isfinite(combined_right)
    ratio_features[valid_mask] = combined_right[valid_mask] / combined_left[valid_mask]
    
    # 处理无穷值和NaN
    ratio_features = np.nan_to_num(ratio_features, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # 组合所有特征
    all_features = np.concatenate([features, diff_features, ratio_features], axis=1)
    
    # 最终清理：确保没有无穷值或NaN
    all_features = np.nan_to_num(all_features, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return all_features

def train_store_logistic_regression(hdf_path, model_path, result_path):
    """
    训练逻辑回归模型并返回模型数据
    """
    # if os.path.exists(model_path):
    #     print(f"Model path {model_path} already exists, loading existing model")
    #     return load_pkl(model_path)
        
    # 加载数据
    results = load_pkl(result_path)
    df = pd.read_hdf(hdf_path, key="dataframe")
    
    # 检查数据类型并处理非数值列
    numeric_cols = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
    
    num_cols = [i for i, col in enumerate(df.columns) if col in numeric_cols]
    
    combined_left = results['combined_conditions_left'][:, num_cols]
    combined_right = results['combined_conditions_right'][:, num_cols]
    dependency = results['dependency']
    
    # 创建特征
    X = create_features(combined_left, combined_right)
    y = dependency
    
    # 数据验证和清理
    logger.info(f"Original feature shape: {X.shape}")
    logger.info(f"Features contain inf: {np.isinf(X).any()}")
    logger.info(f"Features contain nan: {np.isnan(X).any()}")
    
    # 移除包含无效值的行
    valid_rows = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X_clean = X[valid_rows]
    y_clean = y[valid_rows]
    
    logger.info(f"After cleaning - Feature shape: {X_clean.shape}")
    logger.info(f"Removed {len(X) - len(X_clean)} rows with invalid values")
    
    if len(X_clean) == 0:
        raise ValueError("No valid data remaining after cleaning")
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean, test_size=0.2, random_state=42, stratify=y_clean
    )
    
    logger.info(f"Training set size: {X_train.shape[0]}")
    logger.info(f"Test set size: {X_test.shape[0]}")
    
    # 数据预处理
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # 训练逻辑回归模型
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    # 准备模型数据
    model_data = {
        'model': model,
        'scaler': scaler,
        'num_cols': num_cols,
        'df_len': len(df),
        'valid_rows': valid_rows,
        'X_test': X_test,
        'y_test': y_test,
        'combined_left': combined_left[valid_rows],
        'combined_right': combined_right[valid_rows]
    }
    
    # 清理内存
    gc.collect()
    
    return model_data

def train_store_random_forest(hdf_path, model_path, result_path):
    """
    训练随机森林模型并返回模型数据
    """
    if os.path.exists(model_path):
        print(f"Model path {model_path} already exists, loading existing model")
        return load_pkl(model_path)
        
    # 加载数据
    results = load_pkl(result_path)
    df = pd.read_hdf(hdf_path, key="dataframe")
    
    # 检查数据类型并处理非数值列
    numeric_cols = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
    
    num_cols = [i for i, col in enumerate(df.columns) if col in numeric_cols]
    
    combined_left = results['combined_conditions_left'][:, num_cols]
    combined_right = results['combined_conditions_right'][:, num_cols]
    dependency = results['dependency']
    
    # 创建特征
    X = create_features(combined_left, combined_right)
    y = dependency
    
    # 数据验证和清理
    logger.info(f"Original feature shape: {X.shape}")
    logger.info(f"Features contain inf: {np.isinf(X).any()}")
    logger.info(f"Features contain nan: {np.isnan(X).any()}")
    
    # 移除包含无效值的行
    valid_rows = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X_clean = X[valid_rows]
    y_clean = y[valid_rows]
    
    logger.info(f"After cleaning - Feature shape: {X_clean.shape}")
    logger.info(f"Removed {len(X) - len(X_clean)} rows with invalid values")
    
    if len(X_clean) == 0:
        raise ValueError("No valid data remaining after cleaning")
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean, test_size=0.2, random_state=42, stratify=y_clean
    )
    
    logger.info(f"Training set size: {X_train.shape[0]}")
    logger.info(f"Test set size: {X_test.shape[0]}")
    
    # 训练随机森林模型（不需要标准化）
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # 准备模型数据
    model_data = {
        'model': model,
        'scaler': None,  # 随机森林不需要标准化
        'num_cols': num_cols,
        'df_len': len(df),
        'valid_rows': valid_rows,
        'X_test': X_test,
        'y_test': y_test,
        'combined_left': combined_left[valid_rows],
        'combined_right': combined_right[valid_rows]
    }
    
    # 清理内存
    gc.collect()
    
    return model_data

def dependency_prediction(model_data, result_path):
    """
    使用逻辑回归模型进行依赖预测（在测试集上评估）
    """
    model = model_data['model']
    scaler = model_data['scaler']
    X_test = model_data['X_test']
    y_test = model_data['y_test']
    
    # 预测测试集
    if scaler is not None:
        X_test_scaled = scaler.transform(X_test)
        dependency_prediction = model.predict_proba(X_test_scaled)[:, 1]
    else:
        dependency_prediction = model.predict_proba(X_test)[:, 1]
    
    logger.info(f"Test set predictions: {len(dependency_prediction)} samples")
    
    # 评估分类性能（只在测试集上）
    metrics = evaluate_classification(y_test, dependency_prediction)
    
    # 计算统计信息 - 使用保存的combined_left和combined_right数据
    dependency_nonzero = np.where(y_test == 0)
    if len(dependency_nonzero[0]) > 0:
        # 使用保存的数据而不是重新加载
        combined_left_clean = model_data['combined_left']
        combined_right_clean = model_data['combined_right']
        
        # 获取测试集对应的索引（使用train_test_split的相同参数）
        train_size = int(0.8 * len(combined_left_clean))
        combined_left_test = combined_left_clean[train_size:]
        combined_right_test = combined_right_clean[train_size:]
        
        combined_left_nonzero = combined_left_test[dependency_nonzero]
        combined_right_nonzero = combined_right_test[dependency_nonzero]
        
        right_greater = combined_right_nonzero >= combined_left_nonzero
        true_count = np.sum(np.all(right_greater, axis=1))
        left_greater_ratio = (len(y_test) - true_count) / len(y_test)
    else:
        left_greater_ratio = 0.0
    
    logger.info(f"Left greater ratio: {left_greater_ratio}")
    
    metrics["left_greater_ratio"] = left_greater_ratio
    
    # 清理内存
    gc.collect()
    
    return metrics

def dependency_prediction_rf(model_data, result_path):
    """
    使用随机森林模型进行依赖预测（在测试集上评估）
    """
    model = model_data['model']
    X_test = model_data['X_test']
    y_test = model_data['y_test']
    
    # 预测测试集
    dependency_prediction = model.predict_proba(X_test)[:, 1]
    
    logger.info(f"Test set predictions: {len(dependency_prediction)} samples")
    
    # 评估分类性能（只在测试集上）
    metrics = evaluate_classification(y_test, dependency_prediction)
    
    # 计算统计信息 - 使用保存的combined_left和combined_right数据
    dependency_nonzero = np.where(y_test == 0)
    if len(dependency_nonzero[0]) > 0:
        # 使用保存的数据而不是重新加载
        combined_left_clean = model_data['combined_left']
        combined_right_clean = model_data['combined_right']
        
        # 获取测试集对应的索引（使用train_test_split的相同参数）
        train_size = int(0.8 * len(combined_left_clean))
        combined_left_test = combined_left_clean[train_size:]
        combined_right_test = combined_right_clean[train_size:]
        
        combined_left_nonzero = combined_left_test[dependency_nonzero]
        combined_right_nonzero = combined_right_test[dependency_nonzero]
        
        right_greater = combined_right_nonzero >= combined_left_nonzero
        true_count = np.sum(np.all(right_greater, axis=1))
        left_greater_ratio = (len(y_test) - true_count) / len(y_test)
    else:
        left_greater_ratio = 0.0
    
    logger.info(f"Left greater ratio: {left_greater_ratio}")
    
    metrics["left_greater_ratio"] = left_greater_ratio
    
    # 清理内存
    gc.collect()
    
    return metrics

def compute_roc_curve(db_name, y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('{} ROC Curve'.format(db_name))
    plt.legend(loc="lower right")
    os.makedirs(f'figures/{DATA_LIMIT}', exist_ok=True)
    plt.savefig(f'figures/{DATA_LIMIT}/roc_curve_{db_name}_{time.strftime("%Y%m%d-%H%M%S")}.png')
    plt.close()

def evaluate_classification(y_true, y_pred):
    
    roc_y_pred = np.where(y_pred > 1, 1, y_pred)

    y_pred0 = np.where(y_pred > 0 , 1, 0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred0).ravel()
    
    # 计算各项指标
    accuracy0 = (tp + tn) / (tp + tn + fp + fn)
    y_pred1 = np.where(y_pred > 0.5, 1, 0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred1).ravel()
    accuracy1 = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    acc_num = tp + tn
    length = len(y_true)

    y_pred2 = np.where(y_pred > 1, 1, 0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred2).ravel()
    accuracy2 = (tp + tn) / (tp + tn + fp + fn)

    # 返回所有指标
    return {
        "accuracy": accuracy0,
        "accuracy1": accuracy1,
        "accuracy2": accuracy2,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "specificity": specificity,
        "acc_num": acc_num,
        "length": length,
        "roc_curve": (y_true, roc_y_pred)
    }

def main(database_name, table_name):
    logger.info(f"===================================================================")
    logger.info(f"Start processing database: {database_name}, table: {table_name}")
    start_time_total = time.time()
    
    hdf_file = f"data/{database_name}/{table_name}.hdf"
    model_file = f"model/{DATA_LIMIT}/{database_name}/{table_name}_lr.pkl"
    os.makedirs(os.path.dirname(model_file), exist_ok=True)
    result_file = f"data/{database_name}/{table_name}_combined_results.pkl"
    
    # Phase 1: Train Logistic Regression model
    logger.info(f"Phase 1: Start training Logistic Regression model - {table_name}")
    start_time_train = time.time()
    model_data = train_store_logistic_regression(hdf_file, model_file, result_file)
    train_time = time.time() - start_time_train
    logger.info(f"=============================")
    logger.info(f"Phase 1: Logistic Regression model training completed - Time spent: {train_time:.2f} seconds")
    
    # Phase 2: Dependency prediction
    logger.info(f"=============================")
    logger.info(f"Phase 2: Start dependency prediction - {table_name}")
    start_time_predict = time.time()
    metrics = dependency_prediction(model_data, result_file)
    predict_time = time.time() - start_time_predict
    logger.info(f"Phase 2: Dependency prediction completed - Time spent: {predict_time:.4f} seconds")
    
    # Total time
    total_time = time.time() - start_time_total
    logger.info(f"Processing completed - Total time: {total_time:.2f} seconds, "
                f"Train time: {train_time:.2f} seconds, Predict time: {predict_time:.4f} seconds")
    logger.info(f"accuracy: {metrics['accuracy']:.4f}, "
                f"precision: {metrics['precision']:.4f}, "
                f"recall: {metrics['recall']:.4f}, "
                f"f1_score: {metrics['f1_score']:.4f}, "
                f"specificity: {metrics['specificity']:.4f}, "
                f"acc_num: {metrics['acc_num']}, "
                f"length: {metrics['length']}, ")

    return train_time, predict_time, total_time, metrics

def main_rf(database_name, table_name):
    """
    使用随机森林模型的主函数
    """
    logger.info(f"===================================================================")
    logger.info(f"Start processing database: {database_name}, table: {table_name} with Random Forest")
    start_time_total = time.time()
    
    hdf_file = f"data/{database_name}/{table_name}.hdf"
    model_file = f"model/{DATA_LIMIT}/{database_name}/{table_name}_rf.pkl"
    os.makedirs(os.path.dirname(model_file), exist_ok=True)
    result_file = f"data/{database_name}/{table_name}_combined_results.pkl"
    
    # Phase 1: Train Random Forest model
    logger.info(f"Phase 1: Start training Random Forest model - {table_name}")
    start_time_train = time.time()
    model_data = train_store_random_forest(hdf_file, model_file, result_file)
    train_time = time.time() - start_time_train
    logger.info(f"=============================")
    logger.info(f"Phase 1: Random Forest model training completed - Time spent: {train_time:.2f} seconds")
    
    # Phase 2: Dependency prediction
    logger.info(f"=============================")
    logger.info(f"Phase 2: Start dependency prediction - {table_name}")
    start_time_predict = time.time()
    metrics = dependency_prediction_rf(model_data, result_file)
    predict_time = time.time() - start_time_predict
    logger.info(f"Phase 2: Dependency prediction completed - Time spent: {predict_time:.4f} seconds")
    
    # Total time
    total_time = time.time() - start_time_total
    logger.info(f"Processing completed - Total time: {total_time:.2f} seconds, "
                f"Train time: {train_time:.2f} seconds, Predict time: {predict_time:.4f} seconds")
    logger.info(f"accuracy: {metrics['accuracy']:.4f}, "
                f"precision: {metrics['precision']:.4f}, "
                f"recall: {metrics['recall']:.4f}, "
                f"f1_score: {metrics['f1_score']:.4f}, "
                f"specificity: {metrics['specificity']:.4f}, "
                f"acc_num: {metrics['acc_num']}, "
                f"length: {metrics['length']}, ")

    return train_time, predict_time, total_time, metrics

if __name__ == '__main__':
    logger.info("Starting batch processing of all databases and tables")

    # 可以选择使用逻辑回归或随机森林
    use_random_forest = False  # 设置为True使用随机森林，False使用逻辑回归

    results = {}
    for database_name, table_names in DB_TABLE_DICT.items():
        logger.info(f"Start processing database: {database_name}")
        database_start_time = time.time()

        database_name = database_name.replace("_joined", "")

        y_true = []
        y_pred = []
        
        for table_name in table_names:
            if use_random_forest:
                train_time, predict_time, total_time, metrics = main_rf(database_name, table_name)
                model_type = "RF"
            else:
                train_time, predict_time, total_time, metrics = main(database_name, table_name)
                model_type = "LR"
                
            results[(database_name, table_name, model_type)] = {
                "train_time": train_time,
                "predict_time": predict_time,
                "total_time": total_time,
                "accuracy": metrics['accuracy'],
                "accuracy1": metrics['accuracy1'],
                "accuracy2": metrics['accuracy2'],
                "precision": metrics['precision'],
                "recall": metrics['recall'],
                "f1_score": metrics['f1_score'],
                "specificity": metrics['specificity'],
                "acc_num": metrics['acc_num'],
                "length": metrics['length'],
                "roc_curve": metrics['roc_curve'],
                "left_greater_ratio": metrics['left_greater_ratio']
            }
            y_true.extend(metrics['roc_curve'][0])
            y_pred.extend(metrics['roc_curve'][1])
        
        # Compute ROC curve for the entire database
        roc_curve_dict = compute_roc_curve(f"{database_name}_{model_type}", y_true, y_pred)

        database_time = time.time() - database_start_time
        logger.info(f"Database {database_name} processing completed - Time spent: {database_time:.2f} seconds\n\n")

    # Print result summary
    logger.info(f"\n==== Results Jointmary ({model_type}) ====")
    for (db, table, mt), metrics in results.items():
        logger.info(f"============================================")
        logger.info(f"============== {db} - {table} ({mt}) ==============")
        logger.info(f"Train Time: {metrics['train_time']:.2f} s, "
                    f"Predict Time: {metrics['predict_time']:.4f} s, "
                    f"Total Time: {metrics['total_time']:.2f} s, ")
        logger.info(f"RightGreater Ratio: {metrics['left_greater_ratio']:.4f}, "
                    f"Acc>0: {metrics['accuracy']:.4f}, "
                    f"Acc>0.5: {metrics['accuracy1']:.4f}, "
                    f"Acc>1: {metrics['accuracy2']:.4f}, ")
        logger.info(f"Precision: {metrics['precision']:.4f}, "
                    f"Recall: {metrics['recall']:.4f}, "
                    f"F1 Score: {metrics['f1_score']:.4f}, "
                    f"Specificity: {metrics['specificity']:.4f}, ")
        logger.info(f"Acc Num: {metrics['acc_num']}, "
                    f"Length: {metrics['length']}, ")
