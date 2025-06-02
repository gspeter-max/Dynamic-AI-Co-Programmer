import torch 
import torch.nn as nn 
from tree_sitter import Parser, Language
import tree_sitter_python

code = """ 
def add(a, b):
    return a + b
    """ 
lan = Language(tree_sitter_python.language()) 
parser = Parser(lan) 

code_graph = parser.parse(bytes(code, "utf8")) 
class edge_type(nn.Module): 
    
    def __init__(self, code_graph): 
        self.code_graph = code_graph 
        self.node_id_encoding = {} 
        self.value = -1   # to start encoding from 0
        self.node_type_str_to_id = {}  
        self.type_id = -1  # to start encoding from 0 
        self.node_features = [] 
        self.edge_type = []
        
        self.Edge_Type_Parent_Child = 0  # Parent-Child relationship 
        
    def get_node_id(self, node):
        
        if node not in self.node_id_encoding:
            self.value += 1 # increment value for each new node 
            self.node_id_encoding[node] = self.value
        
        if node.type not in self.node_type_str_to_id:# check if node type is already str_to_id 
            self.type_id += 1 
            self.node_type_str_to_id[node.type] = self.type_id
        
        return self.node_id_encoding[node] , self.node_type_str_to_id[node.type] 
    
    def traverse(self,node,source_nodes, destination_nodes):
            node_id , node_str_to_id  = self.get_node_id(node)
            
            for child in node.children: 
                
                    child_id , child_str_to_id = self.get_node_id(child) 
                    
                    # Add the parent-child relationship to the edge lists
                    
                    source_nodes.append(node_id) 
                    destination_nodes.append(child_id)  
                    if (node_id,node_str_to_id) not in self.node_features:
                        self.node_features.append((node_id,node_str_to_id))
                    
                    if (child_id,child_str_to_id) not in self.node_features:
                        self.node_features.append((child_id,child_str_to_id))
                        
                    self.edge_type.append(self.Edge_Type_Parent_Child) 
                    
                    if child.children: 
                        self.traverse(child, source_nodes, destination_nodes) 
            
            return source_nodes, destination_nodes 
    
    def parent_child(self): 
        source_nodes = [] 
        destination_nodes= [] 
        
        
        source_nodes, destination_nodes = self.traverse(self.code_graph.root_node, source_nodes, destination_nodes)                

        edge_index = [] 
        edge_index.append(source_nodes) 
        edge_index.append(destination_nodes) 
        edge_index = torch.tensor(edge_index, dtype=torch.int32).contiguous() 
        
        edge_attr = torch.zeros(len(self.edge_type),3, dtype=torch.int32 ) 
        for i, edge in enumerate(self.edge_type):
            edge_attr[i, edge] = 1
        
        x = torch.zeros(len(self.node_features),len(self.node_type_str_to_id), dtype=torch.int32) 
        for i, feature in self.node_features:
            x[i, feature] = 1
        
        return x, edge_index, edge_attr 

edge = edge_type(code_graph) 
x, edge_index, edge_attr = edge.parent_child() 
print("Node Features:\n", x) 
print("Edge Index:\n", edge_index) 
print("Edge Attributes:\n", edge_attr) 
print("Node Type String to ID Mapping:\n", edge.node_type_str_to_id) 
print("Node ID Encoding:\n", edge.node_id_encoding) 
print(f'node features shape: {edge.node_features}')
print("Edge Types:\n", edge.edge_type) 
