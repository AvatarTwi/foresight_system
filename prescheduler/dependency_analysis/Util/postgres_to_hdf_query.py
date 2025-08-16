from glob_arg import *
from sqlalchemy import create_engine
import pandas as pd

import sys
sys.path.append('../../dependency_analysis/DATA')

# 从查询中提取to hdf
def store_query_content_to_hdf(DB_TABLE_JOIN_DICT, root_hdf_file_path, hdf_key='dataframe'):
    """
    执行给定的 SQL 查询，并将结果存储到 HDF5 文件中。

    参数:
    querys (str): 要执行的 SQL 查询。
    db_name (str): 数据库的名称。
    hdf_file_path (str): HDF5 文件的保存路径。
    hdf_key (str): HDF5 文件中存储 DataFrame 的键。默认为 'dataframe'。
    """
    try:
        user="postgres"
        password="postgres"
        host="192.168.75.130"
        port="5432"

        for db_name, table_dict in DB_TABLE_JOIN_DICT.items():
            db_connection_string = f"postgresql://{user}:{password}@{host}:{port}/{db_name}"
            engine = create_engine(db_connection_string)
            for table_name, queries in table_dict.items():
               for query in queries:
                    df = pd.read_sql_query(query, engine)

                    # 检查并转换日期类型列，避免HDF5存储问题
                    date_columns = [col for col in df.columns if col.endswith('date') or col.endswith('time')]
                    df.drop(columns=date_columns, inplace=True) # 删除日期列

                    # 只保留数值类型为int的列
                    int_columns = df.select_dtypes(include=['int']).columns
                    df = df[int_columns]

                    if df.empty:
                        print(f"警告: 查询 '{query}' 的结果为空。")
                        return
                    hdf_file_path = f"{root_hdf_file_path}/{db_name}/{table_name}.hdf"  # 确保每个表的结果存储在不同的 HDF5 文件中

                    df.to_hdf(hdf_file_path, key=hdf_key, mode='w', format='table', data_columns=True)

        print(f"成功将查询结果存储到 HDF5 文件: {hdf_file_path}")

    except Exception as e:
        print(f"处理查询时发生错误: {e}")

store_query_content_to_hdf(DB_TABLE_JOIN_DICT, "data/", hdf_key='dataframe')