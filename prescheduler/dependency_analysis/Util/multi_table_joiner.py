import pandas as pd
import os
# shutil is no longer needed for the revised example
# import shutil 

def join_multiple_hdf_tables(
    hdf_file_paths: list[str],
    join_on_columns: list[str] = None,
    hdf_key: str = 'dataframe',
    join_type: str = 'inner',
    output_hdf_path: str = None,
    output_hdf_format: str = 'table',
) -> pd.DataFrame | None:
    """
    从多个 HDF5 文件中读取 DataFrame，并根据指定的列逐步连接它们。
    可选择将连接结果保存到新的 HDF5 文件中。

    参数:
        hdf_file_paths (list[str]): HDF5 文件路径的列表。
        join_on_columns (list[str], optional): 用于连接的列名列表，当左右表的连接列名相同时使用。
                                     如果提供此参数，则忽略 left_on_columns 和 right_on_columns。
        hdf_key (str, optional): 从 HDF5 文件中读取 DataFrame 的键。
                                 默认为 'dataframe'。
        join_type (str, optional): 要执行的合并类型。
                                   可以是 'left', 'right', 'outer', 'inner', 'cross'。
                                   默认为 'inner'。
        output_hdf_path (str, optional): 如果提供，连接后的 DataFrame 将被保存到此 HDF5 文件路径。
                                         如果为 None，则不保存结果。
        output_hdf_format (str, optional): HDF5 存储格式，可以是 'table' 或 'fixed'。
                                         默认为 'table'。
    返回:
        pd.DataFrame | None: 连接后的 DataFrame，如果发生错误或无法连接，则返回 None。
    """    
    if not hdf_file_paths:
        print("错误: 未提供 HDF 文件路径。")
        return None    # 如果 output_hdf_path 未提供，则根据输入文件自动生成
    if output_hdf_path is None and hdf_file_paths:
        try:
            first_file_dir = os.path.dirname(hdf_file_paths[0])
            if not first_file_dir: # 如果第一个路径是相对路径且在当前目录
                first_file_dir = "."
            
            base_names = []
            for p in hdf_file_paths:
                base_name_with_ext = os.path.basename(p)
                base_name_without_ext, _ = os.path.splitext(base_name_with_ext)
                base_names.append(base_name_without_ext)
            
            generated_filename = "_".join(base_names) + "_joined.hdf"
            output_hdf_path = os.path.join(first_file_dir, generated_filename)
            print(f"信息: output_hdf_path 未提供，自动生成为: {output_hdf_path}")
        except Exception as e:
            print(f"警告: 自动生成 output_hdf_path 时出错: {e}。将不保存结果。")
            output_hdf_path = None # 出错则不保存
    
    # 读取第一个文件
    try:
        print(f"正在读取第一个 HDF 文件: {hdf_file_paths[0]}")
        merged_df = pd.read_hdf(hdf_file_paths[0], key=hdf_key)
        print(f"成功读取第一个 HDF 文件，形状: {merged_df.shape}")
    except KeyError as e:
        print(f"错误: 在文件 '{hdf_file_paths[0]}' 中未找到键 '{hdf_key}'。请检查 HDF 文件的键名或使用不同的 hdf_key 参数值。错误详情: {e}")
        return None
    except Exception as e:
        print(f"错误: 读取文件 '{hdf_file_paths[0]}' 时出错: {e}")
        return None
    
    # 读取第二个文件
    try:
        print(f"正在读取第二个 HDF 文件: {hdf_file_paths[1]}")
        current_df = pd.read_hdf(hdf_file_paths[1], key=hdf_key)
        print(f"成功读取第二个 HDF 文件，形状: {current_df.shape}")
    except KeyError as e:
        print(f"错误: 在文件 '{hdf_file_paths[1]}' 中未找到键 '{hdf_key}'。请检查 HDF 文件的键名或使用不同的 hdf_key 参数值。错误详情: {e}")
        return None
    except Exception as e:
        print(f"错误: 读取文件 '{hdf_file_paths[1]}' 时出错: {e}")
        return None
            
    try:
        print(f"正在合并两个 DataFrame，使用左表列 '{join_on_columns[0]}' 和右表列 '{join_on_columns[1]}'")
        merged_df = pd.merge(
            merged_df,
            current_df,
            left_on=join_on_columns[0],
            right_on=join_on_columns[1],
            how=join_type
        )
        print(f"成功合并 DataFrame，结果形状: {merged_df.shape}")
    except KeyError as e:
        print(f"错误: 合并时找不到连接列。错误详情: {e}")
        print(f"左表可用列: {merged_df.columns.tolist()}")
        print(f"右表可用列: {current_df.columns.tolist()}")
        return None
    except Exception as e:
        print(f"错误: 合并 DataFrame 时出错: {e}")
        return None

    # 如果提供了输出路径，则将结果保存到HDF文件
    if merged_df is not None and output_hdf_path is not None:
        try:
            # 确保输出目录存在
            output_dir = os.path.dirname(output_hdf_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                
            # 保存DataFrame到HDF文件
            merged_df.to_hdf(output_hdf_path, key=hdf_key, mode='w', format=output_hdf_format)
            print(f"连接后的 DataFrame 已保存到: {output_hdf_path}")
        except Exception as e:
            print(f"保存连接结果到 HDF 文件 '{output_hdf_path}' 时出错: {e}")
            # 保存失败不应影响函数的返回值，仍然返回合并后的DataFrame

    return merged_df if merged_df is not None else pd.DataFrame()


def inspect_hdf_file(file_path: str, verbose: bool = True) -> dict:
    """
    检查HDF文件的结构，并返回包含文件信息的字典。
    
    参数:
        file_path (str): HDF文件的路径
        verbose (bool): 是否打印详细信息
        
    返回:
        dict: 包含文件信息的字典，如果文件不存在或出错则返回空字典
    """
    if not os.path.exists(file_path):
        print(f"错误: 文件 '{file_path}' 不存在")
        return {}
    
    try:
        import tables
        info = {}
        with tables.open_file(file_path, 'r') as h5file:
            if verbose:
                print(f"\n文件 '{file_path}' 的HDF结构:")
                print(f"文件标题: {h5file.title}")
                print("根组结构:")
            
            # 收集根组中的所有组和叶节点
            groups = []
            leaves = []
            
            for node in h5file.root:
                if isinstance(node, tables.Group):
                    groups.append(node._v_pathname)
                    if verbose:
                        print(f"  组: {node._v_pathname}")
                else:
                    leaves.append(node._v_pathname)
                    if verbose:
                        print(f"  叶节点: {node._v_pathname}")
            
            info['title'] = h5file.title
            info['groups'] = groups
            info['leaves'] = leaves
            
            # 尝试使用pandas读取，看看能识别哪些键
            try:
                with pd.HDFStore(file_path, 'r') as store:
                    keys = store.keys()
                    info['pandas_keys'] = keys
                    if verbose:
                        print(f"Pandas可识别的键: {keys}")
            except Exception as e:
                if verbose:
                    print(f"Pandas无法读取此文件的键: {e}")
                info['pandas_keys'] = []
                
        return info
    except Exception as e:
        print(f"检查HDF文件时出错: {e}")
        return {}


if __name__ == '__main__':
    print("--- 多表连接器演示 (从预先存在的 HDF 文件读取) ---")

    # 定义此脚本所在的目录
    base_script_dir = os.path.dirname(os.path.abspath(__file__))
    # 假设 'data' 目录 (由 postgres_to_hdf.py 创建) 与此脚本位于同一目录级别下
    # 例如，如果此脚本在 Evaluation/ 中，则 data 目录应为 Evaluation/data/
    data_root_dir = os.path.join(base_script_dir, "data")
    hdf_key = 'dataframe'

    print(f"期望 HDF 数据位于以下子目录中: {data_root_dir}")
    print("请确保您已运行 'postgres_to_hdf.py' 来生成这些 HDF 文件。")
    print("此示例假设 HDF 文件中的连接键（如果它们在原始数据库中不同）已被调整为具有相同的名称。")

    # --- 示例 1: 连接 TPC-H Customer 和 Orders ---
    print("\n--- 示例 1: 连接 TPC-H Customer 和 Orders (例如，在 'c_custkey' 上) ---")
    # 如果您的 postgres_to_hdf.py 为 TPC-H 使用了不同的子文件夹名称，请调整 'tpch_db_name'
    tpch_db_name_ex1 = "tpch_1g" 
    customer_hdf_path_ex1 = os.path.join(data_root_dir, tpch_db_name_ex1, "customer.hdf")
    orders_hdf_path_ex1 = os.path.join(data_root_dir, tpch_db_name_ex1, "orders.hdf")

    # 重要假设:
    # 为使此示例与当前的 join_multiple_hdf_tables 函数一起使用，
    # 'customer.hdf' 和 'orders.hdf' 必须共享一个用于客户键的共同列名。
    # 假设 'c_custkey' 是两个 HDF 文件中共有的键名。
    # 原始 TPC-H 模式: CUSTOMER.C_CUSTKEY, ORDERS.O_CUSTKEY
    # 如果您的 HDF 文件具有不同的名称，您需要预处理它们或修改
    # join_multiple_hdf_tables 函数以支持 left_on/right_on。
    join_cols_ex1 = ['c_custkey','o_custkey'] # 如果您的共同键名不同，请调整

    if os.path.exists(customer_hdf_path_ex1) and os.path.exists(orders_hdf_path_ex1):
        files_ex1 = [customer_hdf_path_ex1, orders_hdf_path_ex1]

        # 在连接之前先检查HDF文件
        for file_path in files_ex1:
            if os.path.exists(file_path):
                print(f"\n检查HDF文件: {file_path}")
                file_info = inspect_hdf_file(file_path)
                if not file_info.get('pandas_keys'):
                    print(f"警告: 文件 '{file_path}' 中未找到有效的pandas键。请确保文件是由pandas创建的有效HDF文件。")

        joined_df_ex1 = join_multiple_hdf_tables(files_ex1, join_cols_ex1, hdf_key=hdf_key, join_type='inner')

        if joined_df_ex1 is not None and not joined_df_ex1.empty:
            print(f"连接后的 TPC-H DataFrame (Customer, Orders) 基于 {join_cols_ex1}:")
            print(f"形状: {joined_df_ex1.shape}")
            print(joined_df_ex1.head())
        elif joined_df_ex1 is not None: # 结果是空的 DataFrame
             print(f"TPC-H Customer 和 Orders 连接成功，但结果为空 DataFrame (可能连接条件未匹配任何行)。")
        else: # joined_df_ex1 is None
            print(f"从 {tpch_db_name_ex1} 数据库连接 TPC-H Customer 和 Orders 失败。")
    else:
        print(f"跳过 TPC-H 示例 1: 缺少所需的 HDF 文件。")
        if not os.path.exists(customer_hdf_path_ex1): print(f"  缺少: {customer_hdf_path_ex1}")
        if not os.path.exists(orders_hdf_path_ex1): print(f"  缺少: {orders_hdf_path_ex1}")

    # --- 示例 2: 连接 IMDB title 和 movie_companies ---
    print("\n--- 示例 2: 连接 JOB (IMDB) title 和 name (通过 role_type 为中间表) ---")
    imdb_db_name_ex2 = "imdbload" 
    title_hdf_path_ex2 = os.path.join(data_root_dir, imdb_db_name_ex2, "title.hdf")
    movie_company_hdf_path_ex2 = os.path.join(data_root_dir, imdb_db_name_ex2, "movie_companies.hdf")
    company_type_hdf_path_ex2 = os.path.join(data_root_dir, imdb_db_name_ex2, "company_type.hdf")
    
    # 重要假设:
    # 假设 title.id -> 连接到 -> role_type.movie_id
    # 假设 name.id -> 连接到 -> role_type.person_id
    # 原始 IMDB 模式中的关系:
    # title.id <-> role_type.movie_id
    # role_type.person_id <-> name.id (通过 role_type.person_id)
    
    # 检查所有需要的文件是否存在
    missing_files_ex2 = []
    if not os.path.exists(title_hdf_path_ex2): missing_files_ex2.append(title_hdf_path_ex2)
    if not os.path.exists(movie_company_hdf_path_ex2): missing_files_ex2.append(movie_company_hdf_path_ex2)
    if not os.path.exists(company_type_hdf_path_ex2): missing_files_ex2.append(company_type_hdf_path_ex2)

    print("  步骤 1: 连接 title 和 role_type 表")
    title_role_join_cols = ['id','movie_id']  # 假设这是连接列
    title_role_files = [title_hdf_path_ex2, movie_company_hdf_path_ex2]
    title_role_df = join_multiple_hdf_tables(title_role_files, title_role_join_cols, hdf_key=hdf_key, join_type='inner')

    # 与 name 表连接
    name_join_cols = ['company_type','id']  # 假设这是 name 表中的连接列
    role_name_files = [movie_company_hdf_path_ex2, company_type_hdf_path_ex2]
    final_df = join_multiple_hdf_tables(role_name_files, name_join_cols, hdf_key=hdf_key, join_type='inner')

    # --- 示例 3: TPC-C item 和 stock ---
    # print("\n--- 示例 3: 连接 TPC-H item 和 stock (例如，在 'i_id' 上) ---")
    # tpcc_db_name_ex3 = "benchmarksql" # 或 "tpcc"，根据您的 postgres_to_hdf.py 输出进行调整
    # item_hdf_path_ex3 = os.path.join(data_root_dir, tpcc_db_name_ex3, "bmsql_item.hdf")
    # stock_hdf_path_ex3 = os.path.join(data_root_dir, tpcc_db_name_ex3, "bmsql_stock.hdf")

    # 重要假设:
    # 假设 'item.hdf' 和 'stock.hdf' 共享一个用于商品 ID 的共同列名，例如 'i_id'。
    # 原始 TPC-C 模式: ITEM.I_ID, STOCK.S_I_ID
    # join_cols_ex3 = ['i_id','s_i_id'] # 如果您的共同键名不同，请调整

    # if os.path.exists(item_hdf_path_ex3) and os.path.exists(stock_hdf_path_ex3):
    #     files_ex3 = [item_hdf_path_ex3, stock_hdf_path_ex3]
    #     joined_df_ex3 = join_multiple_hdf_tables(files_ex3, join_cols_ex3, hdf_key=hdf_key, join_type='inner')

    #     if joined_df_ex3 is not None and not joined_df_ex3.empty:
    #         print(f"连接后的 TPC-C DataFrame (item, stock) 基于 {join_cols_ex3}:")
    #         print(f"形状: {joined_df_ex3.shape}")
    #         print(joined_df_ex3.head())
    #     elif joined_df_ex3 is not None:
    #         print(f"TPC-C item 和 stock 连接成功，但结果为空 DataFrame。")
    #     else:
    #         print(f"从 {tpcc_db_name_ex3} 数据库连接 TPC-C item 和 stock 失败。")
    # else:
    #     print(f"跳过 TPC-C 示例 3: 缺少所需的 HDF 文件。")
    #     if not os.path.exists(item_hdf_path_ex3): print(f"  缺少: {item_hdf_path_ex3}")
    #     if not os.path.exists(stock_hdf_path_ex3): print(f"  缺少: {stock_hdf_path_ex3}")

    print("\n--- 演示完成 ---")
