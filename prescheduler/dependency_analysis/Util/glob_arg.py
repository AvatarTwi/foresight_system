DB_TABLE_DICT = {
    "gas_data": ["gas_discrete_numeric"],
    # "benchmarksql": ["bmsql_history","bmsql_order_line","bmsql_item"],
    # "imdbload": ["movie_companies", "movie_info"],
    # "tpch_1g": ["nation","supplier","partsupp"],
    # "benchmarksql_joined": ["district_warehouse_joined", "customer_district_joined", "orders_customer_joined", "order_line_orders_joined", "stock_item_joined", "new_order_orders_joined", "history_customer_joined"],
    # "imdbload_joined": ["movie_companies_title_joined", "movie_info_info_type_joined"],
    # "tpch_1g_joined": ["lineitem_part_joined", "partsupp_supplier_joined", "nation_region_joined", "orders_customer_joined"],
}

TABLE_MAPPING = {
    'district_warehouse_joined': 'bmsql_district,bmsql_warehouse',
    'customer_district_joined': 'bmsql_customer,bmsql_district',
    'orders_customer_joined': 'bmsql_oorder,bmsql_customer',
    'order_line_orders_joined': 'bmsql_order_line,bmsql_oorder',
    'stock_item_joined': 'bmsql_stock,bmsql_item',
    'new_order_orders_joined': 'bmsql_new_order,bmsql_oorder',
    'history_customer_joined': 'bmsql_history,bmsql_customer',

    'district,warehouse': 'bmsql_district,bmsql_warehouse',
    'customer,district': 'bmsql_customer,bmsql_district',
    'orders,customer': 'bmsql_oorder,bmsql_customer',
    'order_line,orders': 'bmsql_order_line,bmsql_oorder',
    'stock,item': 'bmsql_stock,bmsql_item',
    'new_order,orders': 'bmsql_new_order,bmsql_oorder',
    'history,customer': 'bmsql_history,bmsql_customer',
    # 可能的 bmsql_ 前缀版本
    'bmsql_district_warehouse_joined': 'bmsql_district,bmsql_warehouse',
    'bmsql_customer_district_joined': 'bmsql_customer,bmsql_district',
    'bmsql_orders_customer_joined': 'bmsql_oorder,bmsql_customer',
    'bmsql_order_line_orders_joined': 'bmsql_order_line,bmsql_oorder',
    'bmsql_stock_item_joined': 'bmsql_stock,bmsql_item',
    'bmsql_new_order_orders_joined': 'bmsql_new_order,bmsql_oorder',
    'bmsql_history_customer_joined': 'bmsql_history,bmsql_customer',
    # 其他可能的连接表
    'title_kind_type_joined': 'title,kind_type',
    'movie_companies_title_joined': 'movie_companies,title',
    'movie_info_info_type_joined': 'movie_info,info_type',
    'lineitem_part_joined': 'lineitem,part',
    'partsupp_supplier_joined': 'partsupp,supplier',
    'nation_region_joined': 'nation,region',
    'orders_customer_joined': 'orders,customer',
}


DB_TABLE_JOIN_DICT={
    'benchmarksql': {
        'district_warehouse_joined': [
            "SELECT * FROM bmsql_district JOIN bmsql_warehouse ON bmsql_district.d_w_id = bmsql_warehouse.w_id"
        ],
        'customer_district_joined': [
            "SELECT * FROM bmsql_customer JOIN bmsql_district ON bmsql_customer.c_d_id = bmsql_district.d_id AND bmsql_customer.c_w_id = bmsql_district.d_w_id"
        ],
        'orders_customer_joined': [
            "SELECT * FROM bmsql_oorder JOIN bmsql_customer ON bmsql_oorder.o_c_id = bmsql_customer.c_id AND bmsql_oorder.o_d_id = bmsql_customer.c_d_id AND bmsql_oorder.o_w_id = bmsql_customer.c_w_id"
        ],
        'order_line_orders_joined': [
            "SELECT * FROM bmsql_order_line JOIN bmsql_oorder ON bmsql_order_line.ol_o_id = bmsql_oorder.o_id AND bmsql_order_line.ol_d_id = bmsql_oorder.o_d_id AND bmsql_order_line.ol_w_id = bmsql_oorder.o_w_id"
        ],
        'stock_item_joined': [
            "SELECT * FROM bmsql_stock JOIN bmsql_item ON bmsql_stock.s_i_id = bmsql_item.i_id"
        ],
        'new_order_orders_joined': [
            "SELECT * FROM bmsql_new_order JOIN bmsql_oorder ON bmsql_new_order.no_o_id = bmsql_oorder.o_id AND bmsql_new_order.no_d_id = bmsql_oorder.o_d_id AND bmsql_new_order.no_w_id = bmsql_oorder.o_w_id"
        ],
        'history_customer_joined': [
            "SELECT * FROM bmsql_history JOIN bmsql_customer ON bmsql_history.h_c_id = bmsql_customer.c_id AND bmsql_history.h_c_d_id = bmsql_customer.c_d_id AND bmsql_history.h_c_w_id = bmsql_customer.c_w_id"
        ]
    },
    'tpch_1g':{
        'lineitem_part_joined': [
            "SELECT * FROM lineitem JOIN part ON lineitem.l_partkey = part.p_partkey"
        ],
        'partsupp_supplier_joined': [
            "SELECT * FROM partsupp JOIN supplier ON partsupp.ps_suppkey = supplier.s_suppkey"
        ],
        'nation_region_joined': [
            "SELECT * FROM nation JOIN region ON nation.n_regionkey = region.r_regionkey"
        ],
        'orders_customer_joined': [
            "SELECT * FROM orders JOIN customer ON orders.o_custkey = customer.c_custkey"
        ]
    },
    'imdbload':{
        'aka_name_name_joined': [
            "SELECT * FROM aka_name JOIN name ON aka_name.person_id = name.id"
        ],
        'title_kind_type_joined': [
            "SELECT * FROM title JOIN kind_type ON title.kind_id = kind_type.id"
        ],
        'movie_companies_title_joined': [
            "SELECT * FROM movie_companies JOIN title ON movie_companies.movie_id = title.id"
        ],
        'movie_info_info_type_joined': [
            "SELECT * FROM movie_info JOIN info_type ON movie_info.info_type_id = info_type.id"
        ],
    }
}
