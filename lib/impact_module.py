import csv
import pandas as pd

def filter_dic(m_dic, nodes):
    m_dic_f = {}
    for key in nodes:
        if key in m_dic:
            m_dic_f[key] = m_dic[key]
        else:
            m_dic_f[key] = 0
    
    return m_dic_f

class OutWImpact:
    def __init__(self, filename = None, df_input = None):
        self.impact_dic = {}
        df = df_input
        if df is None:
            df = pd.read_csv(filename, sep="\t")

        for index, row in df.iterrows():
            self.impact_dic[row['src']]= row['weight']

    def get(self, non_reachable, dep_graph=None, child_graph=None):
        s_dic = filter_dic(self.impact_dic, non_reachable)
        return s_dic

class StdImpact:
    def __init__(self, ge_std_filepath):
        self.ge_std_dic = {}
        with open(ge_std_filepath, "r") as f_in:
            reader = csv.reader(f_in, delimiter = ",")
            next(reader) # skip header

            for row in reader:
                gname = row[0]
                gstd = float(row[1])
                self.ge_std_dic[gname] = gstd

    def __get_nodes(self, dep_dic):
        nodes = set()
        for key, value in dep_dic.items():
            nodes.add(key)
            nodes.update(value)

        return nodes

    def __get_node_in_stds(self, node_stds, nodes, dep_dic):
        in_stds = {}
        sub_dep_dic = {k: dep_dic[k] for k in dep_dic.keys() & nodes} # extract dep_dic related to given nodes list

        for nodeid, parents in sub_dep_dic.items():
            in_stds[nodeid] = 0
            for parent in parents:
                in_stds[nodeid] += node_stds[parent]
        
        return in_stds

    def __get_impacts(self, nodes, node_stds, node_stds_in, child_graph):
        impact_dic = {}
        eplison = 0.0001 # to avoid division by zero
        for node in nodes:
            progeny_std_sum = child_graph.get_progeny_std_sum(node, node_stds)
            impact_dic[node] = progeny_std_sum/(node_stds_in[node]+eplison)
        
        return impact_dic

    def get(self, non_reachable, dep_graph, child_graph):
        node_stds = filter_dic(self.ge_std_dic, dep_graph.get_nodes())
        node_stds_in = self.__get_node_in_stds(node_stds, non_reachable, dep_graph.dep_dic)
        non_reachable_impacts = self.__get_impacts(non_reachable, node_stds, node_stds_in, child_graph)

        return non_reachable_impacts