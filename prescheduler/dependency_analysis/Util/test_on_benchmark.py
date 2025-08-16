import logging
import pickle
import argparse
import time
import sys
import numpy as np
import pandas as pd
import os
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
sys.path.append("../../dependency_analysis")
sys.path.append("../../dependency_analysis/DATA")

from glob_arg import *
from multiprocessing import freeze_support
from Structure.nodes import Context
from Structure.leaves.parametric.Parametric import Categorical
from Structure.model import ASPN
from Learning.learningWrapper import learn_ASPN
from Structure.nodes import Context
import gc

freeze_support()

DATA_LIMIT = 20000
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

def train_store_aspn(hdf_path, aspn_path):
    if os.path.exists(aspn_path):
        print(f"aspn_path {aspn_path} already exists, skip training")
        return    
    df = pd.read_hdf(hdf_path, key="dataframe")
    # 检查数据类型并处理非数值列
    num_cols = []
    numeric_cols = []
    categorical_cols = []
    
    for i, col in enumerate(df.columns):
        # 检查列是否为数值型
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
            num_cols.append(i)
        else:
            categorical_cols.append(col)
            
    if len(df)>DATA_LIMIT:
        df = df.sample(n=DATA_LIMIT, random_state=1)
    
    # 先过滤掉 NaN 值并进行安全的类型转换
    sample_data = df[numeric_cols].to_numpy().astype(int)
    
    parametric_types = [Categorical for i in range(len(numeric_cols))]
    
    ds_context = Context(parametric_types=parametric_types).add_domains(sample_data)
    aspn = learn_ASPN(sample_data, ds_context, rdc_sample_size=100000, rdc_strong_connection_threshold=0.7, multivariate_leaf=True, threshold=0.3)
    model = ASPN()
    model.model = aspn
    model.store_factorize_as_dict()
    model.num_list = num_cols
    model.df_len = len(df)

    if not os.path.exists(aspn_path):
        os.makedirs(os.path.dirname(aspn_path), exist_ok=True)

    pickle.dump(model, open(aspn_path, 'wb'), pickle.HIGHEST_PROTOCOL)

    # 清理内存
    gc.collect()

def dependency_prediction(aspn_path, result_path):
    model = load_pkl(aspn_path)
    results = load_pkl(result_path)
    combined_left=results['combined_conditions_left'][:,model.num_list]
    combined_right=results['combined_conditions_right'][:,model.num_list]
    dependency=results['dependency']    
    dependency_prediction = model.probability((combined_left, combined_right), calculated=dict()) * DATA_LIMIT

    dependency_nonzero = np.where(dependency == 0)
    # 计算在dependency不为0的位置上，combined_left和combined_right的值
    combined_left_nonzero = combined_left[dependency_nonzero]
    combined_right_nonzero = combined_right[dependency_nonzero]

    # 右边的值是否完全大于左边的值
    right_greater = combined_right_nonzero >= combined_left_nonzero

    # 一整行都为true的行数
    true_count = np.sum(np.all(right_greater, axis=1))

    # 计算左边大于右边的比例
    left_greater_ratio = (len(dependency)-true_count) / len(dependency)  
    logger.info(f"Left greater ratio: {left_greater_ratio}")

    # dependency_prediction if greater than 1, considered as 1, otherwise 保持原值
    metrics = evaluate_classification(dependency, dependency_prediction)

    metrics["left_greater_ratio"] = left_greater_ratio

    # logger.info(f"Prediction results: {metrics}")

    # 清理内存1
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
    
    hdf_file = f"data/{database_name}/{table_name}.hdf" # Assume HDF files are in the data subdirectory
    aspn_file = f"model/{DATA_LIMIT}/{database_name}/{table_name}.pkl"
    os.makedirs(os.path.dirname(aspn_file), exist_ok=True)
    result_file = f"data/{database_name}/{table_name}_combined_results.pkl"
    
    # Phase 1: Train and store ASPN model
    logger.info(f"Phase 1: Start training ASPN model - {table_name}")
    start_time_train = time.time()
    train_store_aspn(hdf_file, aspn_file)
    train_time = time.time() - start_time_train
    logger.info(f"=============================")
    logger.info(f"Phase 1: ASPN model training completed - Time spent: {train_time:.2f} seconds")
    
    # Phase 2: Dependency prediction
    logger.info(f"=============================")
    logger.info(f"Phase 2: Start dependency prediction - {table_name}")
    start_time_predict = time.time()
    metrics = dependency_prediction(aspn_file, result_file)
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

    return train_time,predict_time,total_time, metrics


if __name__ == '__main__':
    logger.info("Starting batch processing of all databases and tables")

    # If you need to process all tables, uncomment the code below
    results = {}
    for database_name, table_names in DB_TABLE_DICT.items():
        logger.info(f"Start processing database: {database_name}")
        database_start_time = time.time()

        database_name = database_name.replace("_joined", "")  # Adjust the database name if needed

        y_true = []
        y_pred = []
        
        for table_name in table_names:
            train_time, predict_time, total_time, metrics = main(database_name, table_name)
            results[(database_name, table_name)] = {
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
        roc_curve_dict = compute_roc_curve(database_name, y_true, y_pred)

        database_time = time.time() - database_start_time
        logger.info(f"Database {database_name} processing completed - Time spent: {database_time:.2f} seconds\n\n")

    pickle.dump(results, open(f"model/{DATA_LIMIT}/results.pkl", 'wb'), pickle.HIGHEST_PROTOCOL)

    # Print result summary
    logger.info("\n==== Results Jointmary ====")
    for (db, table), metrics in results.items():
        logger.info(f"============================================")
        logger.info(f"============== {db} - {table} ==============")
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
