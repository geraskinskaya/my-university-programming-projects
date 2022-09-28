from typing import Union, List
from itertools import product, combinations
import pandas as pd
import numpy as np
from copy import deepcopy
from BayesNet import BayesNet


class BNReasoner:
    def __init__(self, net: Union[str, BayesNet]):
        """
        :param net: either file path of the bayesian network in BIFXML format or BayesNet object
        """
        if type(net) == str:
            # constructs a BN object
            self.bn = BayesNet()
            # Loads the BN from an BIFXML file
            self.bn.load_from_bifxml(net)
        else:
            self.bn = net

    # TODO: This is where your methods should go

    def dseparation(self, x, z, y):

        """
        Based on graph pruning.
        Inputs should be lists of variables/nodes (see examples).
        e.g. ['Y'], ['X','Y'], ['O'].
        method 'get_parents' has been added to the BayenNet (see BayesNet.py).
        
        """
        query_variables = [x, y, z]
        query_variables = [item for sublist in query_variables for item in sublist]
        all_variables = self.bn.get_all_variables()
        non_instantiated_variables = list(set(all_variables) - set(query_variables))

        # iteratively removes leaf nodes that are not in the set of {x,y,z}
        counter = 0
        while counter <= len(non_instantiated_variables):

            if non_instantiated_variables == []:
                break

            for var in non_instantiated_variables[:]:
                if self.bn.get_children(var) == []:
                    non_instantiated_variables.remove(var)
                    self.bn.del_var(var)
                    counter -= 1
                else:
                    counter += 1

        # remove outgoing edges from all nodes in Z
        for var in z:
            for child in self.bn.get_children(var):
                self.bn.del_edge((var, child))

        # X and Y are d-separated by Z in if X and Y are disconnected in the pruned graph w.r.t. Z
        # For each var in X, check if that variable reaches any of the variables in Y
        # -if this is the case, then they are not d-separated --> return False
        # -if this is not the case, then they are d-separated --> return True
        num_of_connections = 0
        for node in x:
            nodes = [node]
            visited = []

            while len(nodes) > 0:
                for var in nodes[:]:
                    visited.append(var)
                    for c in self.bn.get_children(var):
                        if c not in visited:
                            nodes.append(c)
                    for p in self.bn.get_parents(var):
                        if p not in visited:
                            nodes.append(p)
                    nodes.pop(0)

                if any(item in y for item in visited):
                    num_of_connections += 1
                    break

        # print('Num_of_connections: ' + str(num_of_connections))

        if num_of_connections == 0:
            return True
        else:
            return False

    def network_pruning(self, q, e):

        """
        :param q: set of query vars --> e.g. ['A','B','C']
        :param e: evidence set --> e.g. {'A': True, 'B': False}
        """

        evidence = list(e.keys())
        query_plus_evidence_variables = [q, evidence]
        query_plus_evidence_variables = [item for sublist in query_plus_evidence_variables for item in sublist]
        all_variables = self.bn.get_all_variables()
        non_instantiated_variables = list(set(all_variables) - set(query_plus_evidence_variables))

        # iteratively removes leaf nodes that are not in the set of {q,e}
        counter = 0
        while counter <= len(non_instantiated_variables):

            if non_instantiated_variables == []:
                break

            for var in non_instantiated_variables[:]:
                if self.bn.get_children(var) == []:
                    non_instantiated_variables.remove(var)
                    self.bn.del_var(var)
                    counter -= 1
                else:
                    counter += 1

        # remove outgoing edges from all nodes in Z, and update CPTs
        for var in evidence:
            for child in self.bn.get_children(var):
                self.bn.del_edge((var, child))

                # new = self.bn.get_compatible_instantiations_table(pd.Series(e), self.bn.get_cpt(child))
                new = self.bn.get_compatible_instantiations_table(pd.Series({var: e.get(var)}), self.bn.get_cpt(child))
                new = new.drop([var], axis=1)
                self.bn.update_cpt(child, new)

    def sum_out_factors(self, factor: Union[str, pd.DataFrame], subset: Union[str, list]) -> pd.DataFrame:
        """
        Sum out variable(s) in subset from a factor.
        :param factor:  factor over variables X
        :param subset:  a subset of variables X
        :return:        a factor corresponding to the factor with the subset summed out
        """
        if isinstance(factor, str):
            factor = self.bn.get_cpt(factor)
        if isinstance(subset, str):
            subset = [subset]

        variables = [v for v in factor.keys() if v not in subset + ['p']]  # Set subtraction: Factor - Subset - p
        new_factor = self.init_factor(variables, 0)
        subset_factor = self.init_factor(subset, 0)

        for i, y in new_factor.iterrows():
            for _, z in subset_factor.iterrows():
                new_factor.loc[i, 'p'] = new_factor.loc[i, 'p'] + self.bn.get_compatible_instantiations_table(
                    y[:-1].append(z[:-1]), factor)['p'].sum()
                # sum() instead of float() here, since the compatible table can be empty at times, this works around it

        return new_factor

    def multiply_factors(self, factors: List[Union[str, pd.DataFrame]]) -> pd.DataFrame:
        """
        Multiply multiple factors with each other.
        :param factors: a list of factors
        :return:        a factor corresponding to the product of all given factors
        """
        # If there are strings in the input-list of factors, replace them with the corresponding cpt
        for x, y in enumerate(factors):
            if isinstance(y, str):
                factors[x] = self.bn.get_cpt(y)

        variables = list(set().union(*factors))  # All variables occurring in all given factors (screws up ordering tho)
        variables.remove('p')  # Remove 'p' col to add it again in the next step, ensuring it ends up as the last col
        new_factor = self.init_factor(variables, 1)

        for i, z in new_factor.iterrows():
            for _, f in enumerate(factors):
                new_factor.loc[i, 'p'] = new_factor.loc[i, 'p'] * self.bn.get_compatible_instantiations_table(
                    z[:-1], f)['p'].sum()
                # sum() instead of float() here, since the compatible table can be empty at times, this works around it

        return new_factor

    def compute_marginal(self, query: List[str], evidence: pd.Series = None, order: List[str] = None) -> pd.DataFrame:
        """
        Compute the prior marginal Pr(query) or joint/posterior marginal Pr(query, evidence).
        :param query:       variables in network N
        :param evidence:    optional; instantiation of some variables in network N
        :param order:       optional; ordering of variables in network N
        :return:            a factor describing the marginal
        """
        if order is None:
            order = self.min_degree_order()

        S = self.bn.get_all_cpts()

        if evidence is not None:  # If there's evidence, reduce all CPTs using the evidence
            for var in self.bn.get_all_variables():
                var_cpt = self.bn.get_cpt(var)
                if any(evidence.keys().intersection(var_cpt.columns)):  # If the evidence occurs in the cpt
                    new_cpt = self.bn.get_compatible_instantiations_table(evidence, var_cpt)
                    S[var] = new_cpt

        pi = [nv for nv in order if nv not in query]
        for var_pi in pi:
            # Pop all functions from S, which mention var_pi...
            func_k = [S.pop(key) for key, cpt in deepcopy(S).items() if var_pi in cpt]

            new_factor = self.multiply_factors(func_k)
            new_factor = self.sum_out_factors(new_factor, var_pi)
            # And replace them with the new factor
            S[var_pi] = new_factor

        res_factor = self.multiply_factors(list(S.values())) if len(S) > 1 else S.popitem()[1]

        if evidence is not None:  # Normalizing over pr_evidence
            cpt_e = self.compute_marginal(list(evidence.keys()), order=order)
            pr_evidence = float(self.bn.get_compatible_instantiations_table(evidence, cpt_e)['p'])
            res_factor['p'] = res_factor['p'] / pr_evidence

        return res_factor

    def min_degree_order(self) -> List[str]:
        """
        Orders variables in self.bn by choosing nodes wth the smallest degrees.
        :return: an ordering of all variables in self.bn
        """
        G = self.bn.get_interaction_graph()
        order = []
        for i in range(len(self.bn.get_all_variables())):
            min_degree_var = ""
            min_n_neighbors = np.inf
            min_neighbors = []

            # For each node, count and remember the neighbors; append the node with the smallest count to order
            for node in G.nodes:
                neighbors = []
                n = 0
                for neighbor in G.neighbors(node):
                    neighbors.append(neighbor)
                    n += 1
                if n < min_n_neighbors:
                    min_degree_var = node
                    min_n_neighbors = n
                    min_neighbors = neighbors
            order.append(min_degree_var)

            if min_n_neighbors > 1:
                for pair in combinations(min_neighbors, 2):
                    G.add_edge(pair[0], pair[1])
            G.remove_node(min_degree_var)

        return order

    @staticmethod
    def init_factor(variables: list, value=0) -> pd.DataFrame:
        truth_table = product([True, False], repeat=len(variables))
        factor = pd.DataFrame(truth_table, columns=variables)
        factor['p'] = value
        return factor


def main():
    # bnr = BNReasoner('testing/lecture_example.BIFXML')
    # a = bnr.sum_out_factors('Wet Grass?', 'Wet Grass?')
    # print(a)
    #
    # bnr2 = BNReasoner('testing/multiply_example.BIFXML')
    # b = bnr2.multiply_factors(['D', 'E'])
    # print(b)

    bnr3 = BNReasoner('testing/marginals_example.BIFXML')
    c = bnr3.compute_marginal(['C'], pd.Series({'A': True}), bnr3.min_degree_order())
    print(c)
    cc = bnr3.compute_marginal(['C'], order=bnr3.min_degree_order())
    print(cc)

    bnr4 = BNReasoner('testing/lecture_example.BIFXML')
    d = bnr4.compute_marginal(['Wet Grass?', 'Slippery Road?'], order=bnr4.min_degree_order())
    print(d)
    dd = bnr4.compute_marginal(['Wet Grass?', 'Slippery Road?'], pd.Series({'Winter?': True, 'Sprinkler?': False}),
                               order=bnr4.min_degree_order())
    print(dd)


if __name__ == '__main__':
    main()
