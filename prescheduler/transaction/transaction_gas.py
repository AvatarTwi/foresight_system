import random
import re # Added for SQL parsing

# df = pd.read_hdf("gas_data/gas_discrete.hdf", key="dataframe")
# print(len(df), len(df.columns))
# df.head(3)
# Time	Humidity	Temperature	Flow_rate	Heater_voltage	R1	R5	R7
# 0	0	36	37	61	13	30	10	8
# 1	0	36	37	61	12	38	14	13
# 2	0	36	37	61	11	45	24	20

# SELECT COUNT(*) FROM climate WHERE R1 >= 0 AND R1 <= 5 AND R5 >= 1 AND R5 <= 6 AND R7 >= 0 AND R7 <= 3||222835
# SELECT COUNT(*) FROM climate WHERE R1 >= 0 AND R1 <= 3 AND R5 >= 1 AND R5 <= 6 AND R7 >= 1 AND R7 <= 6||113648
# SELECT COUNT(*) FROM climate WHERE Flow_rate >= 3 AND Flow_rate <= 34 AND R1 >= 3 AND R1 <= 6 AND R5 >= 3 AND R5 <= 6 AND R7 >= 4 AND R7 <= 7||6773

# Utility function to parse SQL queries
def parse_sql_query(sql_string):
    """
    Parses a 'SELECT COUNT(*) FROM table WHERE conditions' SQL query.
    Returns table name and a list of filter conditions.
    """
    # Extract table name
    table_match = re.search(r"FROM\s+(\w+)", sql_string, re.IGNORECASE)
    if not table_match:
        raise ValueError(f"Could not parse table name from query: {sql_string}")
    table_name = table_match.group(1)

    # Extract WHERE clause
    where_clause_match = re.search(r"WHERE\s+(.+);?", sql_string, re.IGNORECASE)
    if not where_clause_match:
        return table_name, [] # No WHERE clause, no filters
    
    conditions_str = where_clause_match.group(1)
    
    # Split conditions by 'AND'
    # Regex to split by 'AND' while handling potential spaces around it
    # and also to capture each condition part: column, operator, value
    condition_parts = re.findall(r"(\w+)\s*([><=!]+)\s*(\w+)", conditions_str)
    
    filters = []
    for part in condition_parts:
        column, operator, value = part
        # Attempt to convert value to int, otherwise keep as string
        try:
            value = int(value)
        except ValueError:
            pass # Keep as string if not an integer
        filters.append({"column": column, "operator": operator, "value": value})
        
    return table_name, filters

class GasQuery:
    def __init__(self, sql_query_string):
        self.query_string = sql_query_string
        self.table_name, self.filters = parse_sql_query(sql_query_string)
        # is_update_operation is not relevant for SELECT COUNT(*) queries
        print(f"GasQuery initialized for table '{self.table_name}' with filters: {self.filters}")

class Storage:
    def __init__(self):
        # Simplified storage, may not be actively used for schema if queries are self-descriptive
        # Could hold table schemas if needed in the future
        self.schema = {} # Example: {"climate": ["R1", "R2", "R3", "R4", ...]}
        print(f"Gas Storage initialized (simplified).")

class GasTransaction:
    def __init__(self, coordinator_id, partition_id, db, context, random_gen, storage, transaction_id, sql_query):
        self.coordinator_id = coordinator_id
        self.partition_id = partition_id # May need re-evaluation based on new model
        self.db = db
        self.context = context
        self.random_gen = random_gen # May not be needed if queries are predefined
        self.storage = storage
        self.query = GasQuery(sql_query) # Initialize GasQuery with the SQL string
        self.read_set = []
        self.write_set = [] # Will remain empty for SELECT COUNT(*) queries
        self.id = transaction_id
        self.execution_phase = False # Kept for consistency, but role might change
        self.epoch = 0
        self.waw = False
        self.war = False
        self.raw = False
        self.sqls = [sql_query] # Store the original SQL query
        print(f"GasTransaction {self.id} initialized for query: {sql_query.strip()}")

    # get_partition_id_for_key is removed as it's key-based and we now use table/column info.
    # Partitioning logic would need to be based on table name or other query properties if used.

    # make_gas_query is removed as the query is now passed directly to __init__

    def execute(self, worker_id):
        """
        Simulates the execution of a SQL query based transaction.
        Populates read_set based on the table and columns in filters.
        """
        print(f"GasTransaction {self.id} (query: '{self.query.query_string.strip()}') executing by worker {worker_id}.")

        # For SELECT COUNT(*) FROM table WHERE conditions...
        # The "read" operation involves the table and all columns mentioned in the WHERE clause.
        # We don't have individual keys here in the YCSB sense.
        # The read_set will reflect the table and columns being accessed.
        
        accessed_columns = set()
        for filt in self.query.filters:
            accessed_columns.add(filt["column"])

        if not self.execution_phase: # Log reads
            self.read_set.append({
                "table": self.query.table_name,
                "columns_in_filter": sorted(list(accessed_columns)), # Log columns used in WHERE
                "filter_conditions": self.query.filters, # Log the actual filter conditions
                # "partition_id": self.partition_id # Partitioning needs to be re-evaluated
            })
        
        # Since these are SELECT COUNT(*) queries, there are no writes.
        # self.write_set will remain empty.
        # The old logic for search_for_update, update, and iterating ops_per_transaction is removed.
        
        print(f"GasTransaction {self.id} finished execution simulation. Read set: {self.read_set}")
        return "READY_TO_COMMIT" # Or other status

    # search_for_read, search_for_update, update methods are removed as their logic
    # is now incorporated into execute() or is not applicable for SELECT COUNT(*) queries.

# Example of how to use (for testing or integration)
if __name__ == '__main__':
    # Mock context for testing
    mock_context = {
        # "partition_num": 1, # Partitioning strategy might change
        "execution_phase_for_writes": False # No writes in these queries
    }
    mock_random_gen = random.Random(0) # May not be used extensively now
    
    # Create simplified storage instance
    shared_storage = Storage() 

    # Sample SQL queries (similar to those in query_sc.sql)
    sample_sql_queries = [
        "SELECT COUNT(*) FROM climate WHERE R1 >= 0 AND R1 <= 1 AND R2 >= 0 AND R2 <= 1 AND R3 >= 0 AND R3 <= 1 AND R4 >= 0 AND R4 <= 1;",
        "SELECT COUNT(*) FROM climate WHERE R1 >= 10 AND R1 <= 11 AND R2 >= 10 AND R2 <= 11 AND R3 >= 10 AND R3 <= 11 AND R4 >= 10 AND R4 <= 11;",
        # Add more queries from query_sc.sql or read from the file
    ]

    # Path to the SQL query file (replace with actual path if needed for dynamic loading)
    # query_file_path = 'e:\\itc_transaction_schedule\\prescheduler\\dependency_analysis\\Evaluation\\gas_data\\query_sc.sql'
    # loaded_sql_queries = []
    # try:
    #     with open(query_file_path, 'r') as f:
    #         loaded_sql_queries = [line.strip() for line in f if line.strip()]
    # except FileNotFoundError:
    #     print(f"Warning: SQL query file not found at {query_file_path}. Using sample queries.")
    #     loaded_sql_queries = sample_sql_queries
    # if not loaded_sql_queries: # Fallback if file is empty
    #     loaded_sql_queries = sample_sql_queries
    
    # Using the provided sample queries for now
    sql_queries_to_run = sample_sql_queries


    transaction_id_counter = 100
    for sql_query in sql_queries_to_run:
        if not sql_query: continue # Skip empty lines

        transaction_id_counter += 1
        print(f"\n--- Creating Transaction {transaction_id_counter} for SQL: {sql_query.strip()} ---")
        tx = GasTransaction(
            coordinator_id=1,
            partition_id=0, # Home partition, may need adjustment
            db=None, # No actual DB connection for simulation
            context=mock_context,
            random_gen=mock_random_gen,
            storage=shared_storage,
            transaction_id=transaction_id_counter,
            sql_query=sql_query
        )

        # Execute the transaction
        status = tx.execute(worker_id=1)
        print(f"Transaction {tx.id} executed with status: {status}")
        print(f"  Read set: {tx.read_set}")
        print(f"  Write set: {tx.write_set}") # Should be empty

    print("\nExample usage finished.")
