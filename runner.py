# ! pip install tree_sitter tree_sitter_python
from  tree_sitter import Language , Parser
import  tree_sitter_python
import torch
import torch.nn as nn
import os

# source_file_path = '/content/drive/MyDrive/downloaded/scikit-learn/sklearn/ensemble'
# with open('data.text' , 'w') as write_file:
#     for data_file in os.listdir(source_file_path):
#         data_file = source_file_path + '/' + data_file
#         if '.' in data_file[-7:]:  
#             with open(data_file, 'r') as data_files:
#                 write_file.write(data_files.read())

with open('data.text', 'r') as f:
    data = f.read()

lan = Language(tree_sitter_python.language())
parser_obj = Parser(lan)
code_graph = parser_obj.parse(
    bytes(data, 'utf8')
)
terminals = []
non_terminals_or_type_nodes = []

def extract_all(node):
    not_important_structure_type = {
            'for_satement',
            'if_statement',
            'while_statement',
            'function_defination'
        }                                                                                      # that is because some type like for statement is not need to include be
                                                                                                # because there childrens is give the identity of that like human
                                                                                                # model is tring to understand that is for_function using
                                                                                                # ( for, i, in, range() ,block)
    # Terminal nodes (leaves) always captured
    if not node.children:
        # print(f'-------->-------->{node.type}--------> {node.text}')
        terminals.append(node.text.decode('utf8'))

    # Non-terminal nodes (structure) EXCLUDING pure keywords
    elif node.children:
        if node.type not in  not_important_structure_type:
            # print(f'--> {node}')
            non_terminals_or_type_nodes.append(node.type)

    # Recurse for all children
    for i,child in enumerate(node.children):
        extract_all(child)

extract_all(code_graph.root_node)
non_terminals_sorted = sorted(list(set(non_terminals_or_type_nodes)))
type_to_index = {types : index for index, types in enumerate(non_terminals_sorted)}
print(len(non_terminals_sorted)); print(len(type_to_index))

