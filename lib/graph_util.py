import os,sys
import csv
from toposort import toposort, toposort_flatten
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import pprint
pp = pprint.PrettyPrinter(indent=4)

class ChildGraph:
    def __init__(self, edges):
        self.child_dic = {}
        for edge in edges:
            src_nodeid = edge[0]
            dst_nodeid = edge[1]
            if src_nodeid == dst_nodeid: # skips self loops
                continue

            self.__add_node(src_nodeid)
            self.__add_node(dst_nodeid)
            self.__add_child(src_nodeid, dst_nodeid)
    
    def remove_edge(self, edge):
        src_nodeid = edge[0]
        dst_nodeid = edge[1]
        self.child_dic[src_nodeid].remove(dst_nodeid)

    def __add_node(self, nodeid):
        if nodeid not in self.child_dic:
            self.child_dic[nodeid] = set()

    def __add_child(self, src_nodeid, dst_nodeid):
        self.child_dic[src_nodeid].add(dst_nodeid)

    def __visit_children(self, head, visited_nodes):
        for node in self.child_dic[head]:
            if node not in visited_nodes:
                visited_nodes.append(node)
                self.__visit_children(node, visited_nodes)

    def get_non_reachable(self, heads):
        reachable_nodes = self.get_reachable(heads)

        non_reachable_nodes = []
        for nodeid in self.child_dic.keys():
            if nodeid not in reachable_nodes:
                non_reachable_nodes.append(nodeid)
        
        return non_reachable_nodes

    def get_reachable(self, heads):
        visited_nodes = []

        for head in heads:
            visited_nodes.append(head)
            self.__visit_children(head, visited_nodes)
        return visited_nodes

    def get_progeny_std_sum(self, head, node_stds):
        visited_nodes = []
        self.__visit_children(head, visited_nodes)
        sum_std = node_stds[head] #initialize with head's std
        for node in visited_nodes:
            sum_std += node_stds[node]
        
        return sum_std

class DepGraph:
    def __init__(self, edges):
        self.dep_dic = {}
        for edge in edges:
            src_nodeid = edge[0]
            dst_nodeid = edge[1]
            if src_nodeid == dst_nodeid: # skips self loops
                continue

            self.__add_node(src_nodeid)
            self.__add_node(dst_nodeid)
            self.__add_dep(src_nodeid, dst_nodeid)
    
    def remove_edge(self, edge):
        src_nodeid = edge[0]
        dst_nodeid = edge[1]
        self.dep_dic[dst_nodeid].remove(src_nodeid)

    def __add_node(self, nodeid):
        if nodeid not in self.dep_dic:
            self.dep_dic[nodeid] = set()

    def __add_dep(self, src_nodeid, dst_nodeid):
        self.dep_dic[dst_nodeid].add(src_nodeid)

    def get_roots(self):
        roots = []
        for key, value in self.dep_dic.items():
            if len(value) == 0:
                roots.append(key)
        
        return roots

    def get_nodes(self):
        return self.dep_dic.keys()

    def get_node_in_stds(self, node_stds, nodes):
        in_stds = {}
        sub_dep_dic = {k: self.dep_dic[k] for k in self.dep_dic.keys() & nodes} # extract dep_dic related to given nodes list

        for nodeid, parents in sub_dep_dic.items():
            in_stds[nodeid] = 0
            for parent in parents:
                in_stds[nodeid] += node_stds[parent]
        
        return in_stds

    def sort_save(self, filepath):
        nodes_ordered = toposort_flatten(self.dep_dic)
        dep_graph_ordered_list = self.__dep_dic2list(nodes_ordered)

        # save:
        with open(filepath, "w") as bf:
            for node in dep_graph_ordered_list:
                bf.write('{!s}:'.format(node[0]))
                bf.write(",".join(node[1]))
                bf.write("\n")
        print("saved: {!s}".format(filepath))
    
    def __dep_dic2list(self, nodes_ordered):
        """ Save dependencies in dic in a list instead with given order
        Args:
            graph_dic: node dependencies in dictionary
            nodes_ordered: order of nodes

        Return: node dependencies with provided order
        """

        graph_list = []
        for nodeid in nodes_ordered:
            curr_node_deps = {}
            if nodeid in self.dep_dic:
                curr_node_deps = self.dep_dic[nodeid]
            graph_list.append([nodeid, curr_node_deps])
        
        return graph_list

class DirGraphReal:
    def __init__(self, graph_tsv_filepath, impact_m, has_header = False):
        if not graph_tsv_filepath:
            raise Exception("need .tsv file")
        
        self.load_from_tsv(graph_tsv_filepath, has_header)

        # 1) get list of genes with no dep
        self.dep_graph = DepGraph(self.graph_edges)
        self.mr = self.dep_graph.get_roots()

        # 2) non_reachable = BFS/DFS to mark all nodes strarting from self.mr
        self.child_graph = ChildGraph(self.graph_edges)
        non_reachable = self.child_graph.get_non_reachable(self.mr)
        non_reachable_impacts = impact_m.get(non_reachable, self.dep_graph, self.child_graph)

        # 3) ToDo: calculate amount of std in progenies divided by it's in_std
        while len(non_reachable) > 0:
            non_reachable_impacts = self.__get_sub_dic(non_reachable_impacts, non_reachable)
            node_max_std = max(non_reachable_impacts, key=non_reachable_impacts.get)

            #self.mr.append(node_max_std)
            
            self.__update_mrs_with_new_mr(node_max_std)
            non_reachable = self.child_graph.get_non_reachable(self.mr)
            print(".", end="")
    
    def __update_mrs_with_new_mr(self, new_node):
        new_node_reachable_list = self.child_graph.get_reachable([new_node])

        mr_copy = self.mr.copy()
        for node in mr_copy:
            if node in new_node_reachable_list:
                self.mr.remove(node)
        
        self.mr.append(new_node)

    def __get_sub_dic(self, dic_in, nodes):
        return {k: dic_in[k] for k in dic_in.keys() & nodes}

    def load_from_tsv(self, filepath, has_header = False):
        """ Load graph from ".tsv" file
        """

        with open(filepath, "r") as f:
            tsv_reader = csv.reader(f, delimiter = "\t", skipinitialspace=True)
            if has_header:
                next(tsv_reader, None)
            self.graph_edges = list(tsv_reader)
        
        for edge in self.graph_edges:
            if len(edge) == 3:
                edge.remove(edge[2])

    def get_mr(self):
        return self.mr

    def get_nmr(self):
        all_nodes_set = set(self.dep_graph.get_nodes())
        mr_set = set(self.mr)
        
        return list(all_nodes_set.difference(mr_set))

    def save_as_linear_dep_graph(self, roots, filepath):
        back_edges = self.__get_back_edges(roots)
        for edge in back_edges:
            self.graph_edges.remove(edge)
            self.dep_graph.remove_edge(edge)
            self.child_graph.remove_edge(edge)
        
        self.dep_graph.sort_save(filepath)
        

    def __get_back_edges(self, roots):
        backEdges = []
        visited_dic = {}
        for node in roots:
            backEdges = backEdges + self.__get_backEdges_for_root(node, [], visited_dic)

        return backEdges
    
    def __get_backEdges_for_root(self, node, ancestor_nodes, visited_dic):
        backEdges = []
        if visited_dic.get(node) or len(self.child_graph.child_dic[node]) < 1:
            return backEdges
        
        visited_dic[node] = True
        for child in self.child_graph.child_dic.get(node):
            ancestor_nodes.append(node)

            if child in ancestor_nodes:
                backEdges.append([node, child])
            else:
                projeny_backEdges = self.__get_backEdges_for_root(child, ancestor_nodes, visited_dic)
                backEdges = backEdges + projeny_backEdges

            ancestor_nodes.remove(node)
        return backEdges
    



class DirGraph:
    def __init__(self, file_path):
        if file_path:
            self.load_from_tsv(file_path)

    def load_from_tsv(self, file_path):
        """ Load graph from ".tsv" file
        """

        with open(file_path, "r") as f:
            tsv_reader = csv.reader(f, delimiter = "\t", skipinitialspace=True)
            self.graph_edges = list(tsv_reader)

    def save_as_linear_dep_graph(self, file_path):
        # remove back edge
        self.remove_backEdges()
        
        # convert:
        dep_graph_dic = self.__get_dep_graph_as_dic()
        nodes_ordered = toposort_flatten(dep_graph_dic)
        dep_graph_ordered_list = self.__dep_dic2list(dep_graph_dic, nodes_ordered)

        # save:
        with open(file_path, "w") as bf:
            for node in dep_graph_ordered_list:
                bf.write('{!s}:'.format(node[0]))
                bf.write(",".join(node[1]))
                bf.write("\n")

    def remove_backEdges(self):
        backEdges = self.__get_backEdges()
        for edge in self.graph_edges:
            cur_edge = [edge[0], edge[1]]
            if cur_edge in backEdges:
                self.graph_edges.remove(edge)
                print("removed back edge: {!s}".format(str(cur_edge)))

    def __get_backEdges(self):
        root_nodes = []
        self.children_graph_dic, self.root_nodes = self.__get_children_graph_as_dic()

        backEdges = []
        visited_dic = {}
        for node in self.root_nodes:
            backEdges = backEdges + self.__get_backEdges_for_root(node, [], visited_dic)
        
        return backEdges 

    def __get_backEdges_for_root(self, node, ancestor_nodes, visited_dic):
        backEdges = []
        if visited_dic.get(node) or not self.children_graph_dic.__contains__(node):
            return backEdges
        
        visited_dic[node] = True
        for child in self.children_graph_dic.get(node):
            ancestor_nodes.append(node)

            if child in ancestor_nodes:
                backEdges.append([node, child])
            else:
                projeny_backEdges = self.__get_backEdges_for_root(child, ancestor_nodes, visited_dic)
                backEdges = backEdges + projeny_backEdges

            ancestor_nodes.remove(node)
        
        return backEdges

    def __get_children_graph_as_dic(self):
        graph_dic = {}
        graph_dic_non_root = {}
        for edge in self.graph_edges:
            src_nodeid = edge[0]
            dst_nodeid = edge[1]
            if src_nodeid == dst_nodeid: # skips self loops
                continue

            if src_nodeid not in graph_dic:
                graph_dic[src_nodeid] = set([dst_nodeid])
            else:
                graph_dic[src_nodeid].add(dst_nodeid)

            # to fill graph_dic_non_root
            if dst_nodeid not in graph_dic_non_root:
                graph_dic_non_root[dst_nodeid] = True

        # find roots
        root_nodes = []
        for key, value in graph_dic.items():
            if not graph_dic_non_root.__contains__(key):
                root_nodes.append(key)
        
        return graph_dic, root_nodes

    def __get_dep_graph_as_dic(self, skip_self_loops = True):
        graph_dic = {}
        for edge in self.graph_edges:
            src_nodeid = edge[0]
            dst_nodeid = edge[1]
            if src_nodeid == dst_nodeid: # skips self loops
                continue

            if dst_nodeid not in graph_dic:
                graph_dic[dst_nodeid] = set([src_nodeid])
            else:
                graph_dic[dst_nodeid].add(src_nodeid)
        
        return graph_dic
    
    def __dep_dic2list(self, graph_dic, nodes_ordered):
        """ Save dependencies in dic in a list instead with given order
        Args:
            graph_dic: node dependencies in dictionary
            nodes_ordered: order of nodes

        Return: node dependencies with provided order
        """

        graph_list = []
        for nodeid in nodes_ordered:
            curr_node_deps = {}
            if nodeid in graph_dic:
                curr_node_deps = graph_dic[nodeid]
            graph_list.append([nodeid, curr_node_deps])
        
        return graph_list