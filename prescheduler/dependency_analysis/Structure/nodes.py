import numpy as np
import collections
from collections import deque, OrderedDict
from Learning.utils import convert_to_scope_domain
import logging

logger = logging.getLogger(__name__)


class Node(object):
    """
    Base class for all nodes in SPN
    Defines basic properties and operations for nodes
    """
    def __init__(self):
        self.id = 0                # Node unique identifier
        self.scope = []            # Node scope (set of variables it handles)
        #self.scope_idx = []
        self.condition = []        # Node conditions (set of conditional variables)
        #self.condition_idx = []
        self.range = dict()        # Value range dictionary

    @property
    def name(self):
        """Returns the node name in format: ClassName_ID"""
        return f"{self.__class__.__name__}Node_{self.id}"

    @property
    def parameters(self):
        """Returns node parameters, subclasses must implement this method"""
        raise Exception("Not Implemented")

    def __repr__(self):
        """Returns string representation of the node"""
        return self.name

    def __rmul__(self, weight):
        """
        Defines right multiplication operation, allowing weight * node syntax to create weighted nodes
        Used to create Joint nodes
        """
        assert type(weight) == int or type(weight) == float
        self._tmp_weight = weight
        return self

    def __mul__(self, node):
        """
        Defines multiplication operation, allowing node1 * node2 syntax to create Independent nodes
        """
        assert isinstance(node, Node)
        assert len(node.scope) > 0, "right node has no scope"
        assert len(self.scope) > 0, "left node has no scope"
        assert len(set(node.scope).intersection(set(self.scope))) == 0, "children's scope is not disjoint"
        assert set(node.condition) == set(self.condition), "condition not matched, should use factorized nodes"
        result = Independent()
        result.children.append(self)
        result.children.append(node)
        result.scope.extend(self.scope)
        result.scope.extend(node.scope)
        result.condition.extend(self.condition)
        #_, result.scope_idx, result.condition_idx = convert_to_scope_domain(result.scope, result.condition)
        assign_ids(result)
        return result

    def __add__(self, node):
        """
        Defines addition operation, allowing node1 + node2 syntax to create Joint nodes
        """
        assert isinstance(node, Node)
        assert hasattr(node, "_tmp_weight"), "right node has no weight"
        assert hasattr(self, "_tmp_weight"), "left node has no weight"
        assert len(node.scope) > 0, "right node has no scope"
        assert len(self.scope) > 0, "left node has no scope"
        assert set(node.scope) == (set(self.scope)), "children's scope are not the same"
        assert set(node.condition) == (set(self.condition)), "children's condition are not the same"

        from numpy import isclose

        assert isclose(
            1.0, self._tmp_weight + node._tmp_weight
        ), "unnormalized weights, maybe trying to add many nodes at the same time?"

        result = Joint()
        result.children.append(self)
        result.weights.append(self._tmp_weight)
        result.children.append(node)
        result.weights.append(node._tmp_weight)
        result.scope.extend(self.scope)
        result.condition.extend(self.condition)
        result._tmp_weight = self._tmp_weight + node._tmp_weight
        #_, result.scope_idx, result.condition_idx = convert_to_scope_domain(result.scope, result.condition)
        assign_ids(result)
        return result

    def factor_mul(self, node):
        """
        Defines factorized multiplication, creating Decomposition nodes
        Used to represent conditional probability P(Y|X), where X is conditional variable and Y is target variable
        """
        assert isinstance(node, Node)
        assert len(node.scope) > 0, "right node has no scope"
        assert len(self.scope) > 0, "left node has no scope"
        assert len(set(node.scope).intersection(set(self.scope))) == 0, "children's scope is not disjoint"
        assert set(node.condition) == set(self.scope+self.condition), "scope does not match with others' condition"

        result = Decomposition()
        result.children.append(self)
        result.left_child = self
        result.children.append(node)
        result.right_child = self
        result.scope.extend(self.scope)
        result.scope.extend(node.scope)
        result.condition.extend(self.condition)
        #_, result.scope_idx, result.condition_idx = convert_to_scope_domain(result.scope, result.condition)
        assign_ids(result)
        return result


class Joint(Node):
    """
    Sum node class, representing mixture models
    Weighted sum of child nodes, weights represent probability of selecting each sub-distribution
    """
    def __init__(self, weights=None, children=None, cluster_centers=None, cardinality=None):
        Node.__init__(self)
        if weights is None:
            weights = []
        self.weights = weights         # Child node weights list

        if children is None:
            children = []
        self.children = children       # Child nodes list

        if cluster_centers is None:
            cluster_centers = []
        self.cluster_centers = cluster_centers    # Cluster centers (used to save cluster centers during row splitting)

        if cardinality is None:
            cardinality = 0
        self.cardinality = cardinality     # Cardinality (number of samples)
        #_, self.scope_idx, self.condition_idx = convert_to_scope_domain(self.scope, self.condition)

    @property
    def parameters(self):
        """Returns node parameters: tuples of child node IDs and corresponding weights"""
        sorted_children = sorted(self.children, key=lambda c: c.id)
        params = [(n.id, self.weights[i]) for i, n in enumerate(sorted_children)]
        return tuple(params)


class Independent(Node):
    """
    Product node class, representing independence between variables
    Multiplication of probability density functions of child nodes
    """
    def __init__(self, children=None):
        Node.__init__(self)
        if children is None:
            children = []
        self.children = children       # Child nodes list

    @property
    def parameters(self):
        """Returns node parameters: tuple of child node IDs sorted by ID"""
        return tuple(map(lambda n: n.id, sorted(self.children, key=lambda c: c.id)))


class Decomposition(Node):
    """
    Decomposition node class, used to represent conditional probability
    Decomposes joint distribution into conditional probability form P(X,Y) = P(X)P(Y|X)
    """
    def __init__(self, children=None):
        Node.__init__(self)
        if children is None:
            children = []
        self.children = children       # Child nodes list

    @property
    def parameters(self):
        """Returns node parameters: tuple of left and right child node IDs"""
        return (self.left_child.id, self.right_child.id)


class Leaf(Node):
    """
    Leaf node class, representing probability distributions
    Leaf nodes in SPN, defining univariate or multivariate probability distributions
    """
    def __init__(self, scope=None, condition=None, cardinality=0):
        Node.__init__(self)
        if scope is not None:
            if type(scope) == int:
                self.scope.append(scope)
            elif type(scope) == list:
                self.scope.extend(scope)
            else:
                raise Exception("invalid scope type %s " % (type(scope)))

        if condition is not None:
            if type(condition) == int:
                self.condition.append(condition)
            elif type(condition) == list:
                self.condition.extend(condition)
            else:
                raise Exception("invalid condition type %s " % (type(condition)))
        self.cardinality = cardinality     # Cardinality (number of samples)

        # Convert scope and conditions to internal index format
        _, self.scope_idx, self.condition_idx = convert_to_scope_domain(self.scope, self.condition)

    def query(self, query, attr):
        """
        Compute probability of the query, subclasses must implement this method
        
        Args:
            query: Query range
            attr: Attribute list
            
        Returns:
            Query probability
        """
        raise NotImplemented

    def likelihood(self, data, attr, log):
        """
        Compute likelihood of the data, subclasses must implement this method
        
        Args:
            data: Input data
            attr: Attribute list
            log: Whether to return log likelihood
            
        Returns:
            Likelihood value
        """
        pass


class Context:
    """
    Context class, stores meta-information related to data
    Including variable types, domains, parameter types, etc.
    """
    def __init__(self, meta_types=None, domains=None, parametric_types=None, feature_names=None):
        self.meta_types = meta_types           # Meta types list (type of each variable)
        self.domains = domains                 # Domains list (value range of each variable)
        self.parametric_types = parametric_types   # Parameter types list
        self.feature_names = feature_names     # Feature names list
        self.fanout_attr = []                  # Fanout attributes list
        self.fanout_attr_positive = []         # Positive fanout attributes list
        self.fanout_attr_inverse = []          # Inverse fanout attributes list

        # If parameter types are provided but no meta types, extract meta types from parameter types
        if meta_types is None and parametric_types is not None:
            self.meta_types = []
            for p in parametric_types:
                self.meta_types.append(p.type.meta_type)

    def get_meta_types_by_scope(self, scopes):
        """Returns meta types list for specified scopes"""
        return [self.meta_types[s] for s in scopes]

    def get_domains_by_scope(self, scopes):
        """Returns domains list for specified scopes"""
        return [self.domains[s] for s in scopes]

    def get_parametric_types_by_scope(self, scopes):
        """Returns parameter types list for specified scopes"""
        return [self.parametric_types[s] for s in scopes]

    def add_domains(self, data):
        """
        Add domain information based on data
        
        Args:
            data: Input data
            
        Returns:
            Updated context object
        """
        assert len(data.shape) == 2, "data is not 2D?"
        assert data.shape[1] == len(self.meta_types), "Data columns and metatype size doesn't match"

        from Structure.StatisticalTypes import MetaType

        domain = []

        # Process each column, generate corresponding domain based on meta type
        for col in range(data.shape[1]):
            feature_meta_type = self.meta_types[col]
            min_val = np.nanmin(data[:, col])
            max_val = np.nanmax(data[:, col])
            domain_values = [min_val, max_val]

            if feature_meta_type == MetaType.REAL:
                domain.append(domain_values)                     # Continuous variables, domain is min and max values
            elif feature_meta_type == MetaType.BINARY:
                domain.append([0, 1])                            # Binary variables, domain is 0 and 1
            elif feature_meta_type == MetaType.DISCRETE:
                domain.append(np.sort(np.unique(data[:, col])))  # Discrete variables, domain is all unique values
            else:
                raise Exception("Unkown MetaType " + str(feature_meta_type))

        # self.domains = np.asanyarray(domain) replaced with following code
        self.domains = np.array(domain, dtype=object)

        return self

# The following are various utility functions for SPN operations and traversal

def get_number_of_edges(node):
    """Calculate the number of edges in SPN"""
    return sum([len(c.children) for c in get_nodes_by_type(node, (Joint, Independent))])


def get_number_of_nodes(spn, node_type=Node):
    """Calculate the number of nodes of specified type"""
    return len(get_nodes_by_type(spn, node_type))


def get_parents(node, includ_pos=True):
    """
    Get parent nodes of all nodes
    
    Args:
        node: Root node
        includ_pos: Whether to include position of node in parent's children list
        
    Returns:
        Dictionary mapping nodes to their parent node lists
    """
    parents = OrderedDict({node: []})
    for n in get_nodes_by_type(node):
        if not isinstance(n, Leaf):
            for i, c in enumerate(n.children):
                parent_list = parents.get(c, None)
                if parent_list is None:
                    parents[c] = parent_list = []
                if includ_pos:
                    parent_list.append((n, i))
                else:
                    parent_list.append(n)
    return parents


def get_depth(node):
    """Calculate depth of SPN (longest path)"""
    node_depth = {}

    def count_layers(node):
        ndepth = node_depth.setdefault(node, 1)

        if hasattr(node, "children"):
            for c in node.children:
                node_depth.setdefault(c, ndepth + 1)

    bfs(node, count_layers)

    return max(node_depth.values())


def rebuild_scopes_bottom_up(node):
    """
    Rebuild node scopes bottom-up
    Note: This function will modify node scopes
    
    Args:
        node: Root node
        
    Returns:
        Root node with updated scopes
    """
    for n in get_topological_order(node):
        if isinstance(n, Leaf):
            continue

        new_scope = set()
        for c in n.children:
            new_scope.update(c.scope)
        n.scope = list(new_scope)

    return node


def bfs(root, func):
    """
    Breadth-first search traversal of SPN
    
    Args:
        root: Root node
        func: Function to execute on each node
    """
    seen, queue = set([root]), collections.deque([root])
    while queue:
        node = queue.popleft()
        func(node)
        if not isinstance(node, Leaf):
            for c in node.children:
                if c not in seen:
                    seen.add(c)
                    queue.append(c)


def get_topological_order(node):
    """
    Get topological ordering of SPN (bottom-up order)
    
    Args:
        node: Root node
        
    Returns:
        List of nodes in topological order
    """
    nodes = get_nodes_by_type(node)

    # Calculate in-degree and parent nodes for each node
    parents = OrderedDict({node: []})
    in_degree = OrderedDict()
    for n in nodes:
        in_degree[n] = in_degree.get(n, 0)
        if not isinstance(n, Leaf):
            for c in n.children:
                parent_list = parents.get(c, None)
                if parent_list is None:
                    parents[c] = parent_list = []
                parent_list.append(n)
                in_degree[n] += 1

    # Initialize queue with nodes having no incoming edges (leaf nodes)
    S = deque()  # Set of all nodes with no incoming edge
    for u in in_degree:
        if in_degree[u] == 0:
            S.appendleft(u)

    L = []  # Result list

    # Kahn's algorithm to compute topological sort
    while S:
        n = S.pop()           # Remove a node with no incoming edges
        L.append(n)           # Add to result list

        for m in parents[n]:  # For each parent node m of n
            in_degree_m = in_degree[m] - 1  # Remove edge from n to m
            in_degree[m] = in_degree_m
            if in_degree_m == 0:  # If m has no other incoming edges
                S.appendleft(m)    # Add m to queue

    # Ensure graph is DAG (no cycles)
    assert len(L) == len(nodes), "Graph is not DAG, it has at least one cycle"
    return L


def get_topological_order_layers(node):
    """
    Get topological layers of SPN (nodes grouped by layers)
    
    Args:
        node: Root node
        
    Returns:
        List of lists, each containing nodes at that layer
    """
    nodes = get_nodes_by_type(node)

    # Calculate in-degree and parent nodes for each node
    parents = OrderedDict({node: []})
    in_degree = OrderedDict()
    for n in nodes:
        in_degree[n] = in_degree.get(n, 0)
        if not isinstance(n, Leaf):
            for c in n.children:
                parent_list = parents.get(c, None)
                if parent_list is None:
                    parents[c] = parent_list = []
                parent_list.append(n)
                in_degree[n] += 1

    # Initialize first layer (nodes with no incoming edges, usually leaf nodes)
    layer = []  # Set of all nodes with no incoming edge
    for u in in_degree:
        if in_degree[u] == 0:
            layer.append(u)

    L = [layer]  # Add first layer

    # Build topological layers level by level
    added_nodes = len(layer)
    while True:
        layer = []

        # For each node in previous layer
        for n in L[-1]:
            for m in parents[n]:  # For each parent node m of n
                in_degree_m = in_degree[m] - 1  # Remove edge from n to m
                in_degree[m] = in_degree_m
                if in_degree_m == 0:  # If m has no other incoming edges
                    layer.append(m)    # Add m to current layer

        if len(layer) == 0:
            break  # No more layers

        added_nodes += len(layer)
        L.append(layer)

    # Ensure graph is DAG (no cycles)
    assert added_nodes == len(nodes), "Graph is not DAG, it has at least one cycle"
    return L


def get_nodes_by_type(node, ntype=Node):
    """
    Get all nodes of specified type
    
    Args:
        node: Root node
        ntype: Node type or tuple of types
        
    Returns:
        List of nodes of specified type
    """
    assert node is not None

    result = []

    def add_node(node):
        if isinstance(node, ntype):
            result.append(node)

    bfs(node, add_node)

    return result


def get_node_types(node, ntype=Node):
    """
    Get node types that exist in SPN
    
    Args:
        node: Root node
        ntype: Base node type
        
    Returns:
        Set of node types
    """
    assert node is not None

    result = set()

    def add_node(node):
        if isinstance(node, ntype):
            result.add(type(node))

    bfs(node, add_node)

    return result


def assign_ids(node, ids=None):
    """
    Assign unique IDs to all nodes in SPN
    
    Args:
        node: Root node
        ids: Existing ID dictionary (node to ID mapping)
        
    Returns:
        Root node with assigned IDs
    """
    if ids is None:
        ids = {}

    def assign_id(node):
        if node not in ids:
            ids[node] = len(ids)

        node.id = ids[node]

    bfs(node, assign_id)
    return node


def eval_spn_bottom_up(node, eval_functions, all_results=None, debug=False, **args):
    """
    Evaluate SPN bottom-up
    
    Args:
        node: SPN root node
        eval_functions: Dictionary mapping node types to evaluation functions
        all_results: Dictionary to store intermediate results
        debug: Whether to show progress information
        args: Arguments passed to evaluation functions
        
    Returns:
        Evaluation result of root node
    """
    # Get all nodes in topological order
    nodes = get_topological_order(node)

    if debug:
        from tqdm import tqdm
        nodes = tqdm(list(nodes))

    if all_results is None:
        all_results = {}
    else:
        all_results.clear()

    # Set evaluation functions for each node type
    for node_type, func in eval_functions.items():
        if "_eval_func" not in node_type.__dict__:
            node_type._eval_func = []
        node_type._eval_func.append(func)
        node_type._is_leaf = issubclass(node_type, Leaf)
    leaf_func = eval_functions.get(Leaf, None)

    # Temporary list to cache child node results
    tmp_children_list = []
    len_tmp_children_list = 0
    
    # Evaluate each node in topological order
    for n in nodes:
        try:
            func = n.__class__._eval_func[-1]
            n_is_leaf = n.__class__._is_leaf
        except:
            if isinstance(n, Leaf) and leaf_func is not None:
                func = leaf_func
                n_is_leaf = True
            else:
                raise AssertionError("No lambda function associated with type: %s" % (n.__class__.__name__))

        # Leaf nodes call evaluation function directly
        if n_is_leaf:
            result = func(n, **args)
        else:
            # Non-leaf nodes first collect results from all child nodes
            len_children = len(n.children)
            if len_tmp_children_list < len_children:
                tmp_children_list.extend([None] * len_children)
                len_tmp_children_list = len(tmp_children_list)
            for i in range(len_children):
                ci = n.children[i]
                tmp_children_list[i] = all_results[ci]
            result = func(n, tmp_children_list[0:len_children], **args)

        all_results[n] = result

    # Clean up temporary attributes
    for node_type, func in eval_functions.items():
        del node_type._eval_func[-1]
        if len(node_type._eval_func) == 0:
            delattr(node_type, "_eval_func")

    return all_results[node]


def eval_spn_top_down(root, eval_functions, all_results=None, parent_result=None, **args):
    """
    Evaluate SPN top-down
    
    Args:
        root: SPN root node
        eval_functions: Dictionary mapping node types to evaluation functions
        all_results: Dictionary to store intermediate results
        parent_result: Initial value passed to root node
        args: Arguments passed to evaluation functions
        
    Returns:
        Evaluation result of root node
    """
    if all_results is None:
        all_results = {}
    else:
        all_results.clear()

    # Set evaluation functions for each node type
    for node_type, func in eval_functions.items():
        if "_eval_func" not in node_type.__dict__:
            node_type._eval_func = []
        node_type._eval_func.append(func)

    # Set initial result for root node
    all_results[root] = [parent_result]

    # Evaluate layer by layer from top to bottom
    for layer in reversed(get_topological_order_layers(root)):
        for n in layer:
            func = n.__class__._eval_func[-1]

            # Get parameters passed down from parent nodes
            param = all_results[n]
            result = func(n, param, **args)

            # If there are results and it's not a leaf node, continue passing down
            if result is not None and not isinstance(n, Leaf):
                assert isinstance(result, dict)

                # Pass results to child nodes
                for child, param in result.items():
                    if child not in all_results:
                        all_results[child] = []
                    all_results[child].append(param)

    # Clean up temporary attributes
    for node_type, func in eval_functions.items():
        del node_type._eval_func[-1]
        if len(node_type._eval_func) == 0:
            delattr(node_type, "_eval_func")

    return all_results[root]
