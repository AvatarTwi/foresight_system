import pandas as pd
from sqlalchemy import create_engine
import sys
import os
sys.path.append(r'E:\\itc_transaction_schedule\\prescheduler\\dependency_analysis\\Evaluation')

def store_table_to_hdf(engine, table_name, hdf_file_path, hdf_key='dataframe'):
    """
    使用现有的数据库引擎读取指定的表，并将其存储到 HDF5 文件中。

    参数:
    engine (sqlalchemy.engine.Engine): SQLAlchemy 数据库引擎实例。
    table_name (str): 要从数据库中读取的表名。
    hdf_file_path (str): HDF5 文件的保存路径。
    hdf_key (str): HDF5 文件中存储 DataFrame 的键。默认为 'dataframe'。
    """
    try:
        print(f"正在从表 '{table_name}' 读取数据...")
        query = f"SELECT * FROM {table_name}" # Consider schema if not always public
        df = pd.read_sql_query(query, engine)

        print(f"成功从表 '{table_name}' 读取数据。")
        
        if df.empty:
            print(f"警告: 从表 '{table_name}' 读取的数据为空。")
        
        # 检查并转换日期类型列，避免HDF5存储问题
        date_columns = [col for col in df.columns if col.endswith('date') or col.endswith('time')]

        df.drop(columns=date_columns, inplace=True) # 删除日期列

        # 只保留数值类型为int的列
        int_columns = df.select_dtypes(include=['int']).columns
        df = df[int_columns]
            
        # 确保输出目录存在 (由调用方 store_all_tables_to_hdf 创建基础目录 db_name/)
        # 此处确保文件自身的目录存在，对于 db_name/table.hdf，os.path.dirname 是 db_name
        output_dir = os.path.dirname(hdf_file_path)
        if output_dir and not os.path.exists(output_dir):
            # This case should ideally be handled by the caller creating the base db_name directory
            print(f"警告: 目录 '{output_dir}' 在 store_table_to_hdf 中创建。")
            os.makedirs(output_dir)
            
        print(f"正在将数据存储到 HDF5 文件: '{hdf_file_path}' (key: '{hdf_key}')...")
        df.attrs['table_name'] = table_name # 将表名存储为 HDF 文件的属性
        df.to_hdf(hdf_file_path, key=hdf_key, mode='w', format='table', data_columns=True)
        
        print("数据成功存储到 HDF5 文件。")
        print(f"  文件路径: {hdf_file_path}")
        print(f"  HDF 键: {hdf_key}")
        print(f"  存储的表名 (属性): {df.attrs.get('table_name')}")
        print(f"  数据行数: {len(df)}")
        print(f"  数据列数: {len(df.columns)}")

    except ImportError:
        print("错误: 缺少必要的库。请确保已安装 pandas, sqlalchemy 和 psycopg2 (或适用于您的 PostgreSQL 的驱动程序)。")
        print("  pip install pandas sqlalchemy psycopg2-binary")
    except Exception as e:
        print(f"处理表 '{table_name}' 时发生错误: {e}")
        print("请检查以下几点:")
        # print(f"  1. 数据库连接字符串是否正确。") # Connection string not directly used here
        print("  1. 数据库服务器是否正在运行且可访问。")
        print(f"  2. 用户是否有权限访问表 '{table_name}'。")
        print(f"  3. 指定的表 '{table_name}' 是否存在于数据库中。")
        print("  4. HDF5 文件路径 '{hdf_file_path}' 是否有效且具有写入权限。")


def store_all_tables_to_hdf(db_connection_string, db_name, hdf_key='dataframe', target_tables: list[str] | None = None):
    """
    连接到 PostgreSQL 数据库，读取指定的表或 public schema 中的所有表，
    并将每个表分别存储到以数据库名和表名命名的 HDF5 文件中。

    参数:
    db_connection_string (str): SQLAlchemy 数据库连接字符串。
    db_name (str): 数据库的名称，也将用作输出子目录的名称。
    hdf_key (str): HDF5 文件中存储 DataFrame 的键。
    target_tables (list[str] | None, optional): 要读取的特定表名的列表。
                                              如果为 None 或空列表，则读取 public schema 中的所有表。
                                              默认为 None。
    """
    try:
        print(f"--- 开始处理数据库: {db_name} ---")
        engine = create_engine(db_connection_string)
        
        table_names_to_process = []

        if target_tables and len(target_tables) > 0:
            print(f"目标表已指定: {target_tables}")
            # 验证目标表是否存在 (可选但推荐)
            # 为了简单起见，这里我们直接使用提供的列表
            # 您可能希望添加一个检查，以确保这些表确实存在于数据库中
            table_names_to_process = target_tables
        else:
            print("未指定目标表，将尝试读取 public schema 中的所有表。")
            # 获取 public schema 中的所有表名
            query_tables = "SELECT tablename FROM pg_tables WHERE schemaname = 'public';"
            print(f"执行查询以获取表名: {query_tables}")
            df_tables = pd.read_sql_query(query_tables, engine)
            
            if df_tables.empty:
                print(f"在数据库 '{db_name}' 的 public schema 中未找到任何表。")
                return
            table_names_to_process = df_tables['tablename'].tolist()
            print(f"在 public schema 中找到的表: {table_names_to_process}")

        if not table_names_to_process:
            print("没有要处理的表。")
            return

        # HDF 文件的基础输出目录 (以数据库名命名)
        db_output_dir = "data/"+db_name 
        if not os.path.exists(db_output_dir):
            print(f"基础输出目录 '{db_output_dir}' (用于数据库 '{db_name}') 不存在，正在创建...")
            os.makedirs(db_output_dir)
        else:
            print(f"将使用现有的基础输出目录 '{db_output_dir}' (用于数据库 '{db_name}')。")

        for table_name in table_names_to_process:
            print(f"\n--- 开始处理表: {table_name} ---")
            hdf_file_path = os.path.join(db_output_dir, f"{table_name}.hdf")
            store_table_to_hdf(engine, table_name, hdf_file_path, hdf_key)
            
        print(f"\n--- 完成处理数据库 '{db_name}' 中的所有表 ---")

    except ImportError:
        print("错误: 缺少必要的库。请确保已安装 pandas, sqlalchemy 和 psycopg2 (或适用于您的 PostgreSQL 的驱动程序)。")
        print("  pip install pandas sqlalchemy psycopg2-binary")
    except Exception as e:
        print(f"处理数据库 '{db_name}' 时发生严重错误: {e}")
        print("请检查数据库连接字符串、服务器状态和权限。")

DB_TABLE_DICT = {
    "benchmarksql": ["bmsql_customer","bmsql_history","bmsql_order_line",'bmsql_district','bmsql_item','bmsql_new_order','bmsql_oorder','bmsql_stock'],
    "imdbload": ["movie_companies", "movie_info", "title"],
    "tpch_1g": ["lineitem", "part", "partsupp", "supplier", "nation", "orders", "customer"],
    "gas_data": ["gas_discrete_numeric"]
}

if __name__ == '__main__':
    # --- 配置参数 ---
    # 示例连接字符串 (请根据您的 PostgreSQL 配置进行修改)
    # 格式: "postgresql://<USER>:<PASSWORD>@<HOST_ADDRESS>:<PORT>/<DATABASE_NAME>"
    user="postgres"
    password="postgres"
    host="192.168.75.130"
    port="5432"

    for database_name, target_tables in DB_TABLE_DICT.items():

        # 连接字符串
        DB_CONNECTION_STRING = f"postgresql://{user}:{password}@{host}:{port}/{database_name}"
        
        # HDF5 存储键
        HDF_STORAGE_KEY = "dataframe" 

        print(f"\n--- PostgreSQL 数据库 '{database_name}' 所有表到 HDF5 转换脚本 ---")
        
        # 调用函数处理指定数据库中的所有表
        store_all_tables_to_hdf(DB_CONNECTION_STRING, database_name, HDF_STORAGE_KEY, target_tables=target_tables)