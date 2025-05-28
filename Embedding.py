from transformers.models.gpt2 import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

class make_embedding:

    def __init__(self,type_to_index = type_to_index ,max_sub_tokens = 10,tokenizer = tokenizer):

        self.tokenizer = tokenizer
        vocab_size = len(tokenizer.get_vocab())
        self.type_to_index = type_to_index
        self.type_embedding = nn.Embedding(len(type_to_index),128)
        self.text_embedding = nn.Embedding(vocab_size,128)
        self.max_sub_tokens = max_sub_tokens
        self.overall_maximium_tokens_recored = 0
        self.overall_maximium_tokens_recored_text = None

        # spacial types child is too smart and completely show our father identity ( we are push model to learn about spcial node from ndoe childrens )
        # its banificial like vocab size and etc

        self.ignore_types = not_important_structure_type = {
                                    'for_satement',
                                    'if_statement',
                                    'while_statement',
                                    'function_defination'
                        }

        self.compressing_linear_layer = nn.Linear(self.max_sub_tokens * 128, 128)

    def get_embedding_for_this_node(self,node,temp = None):

        convered_2d_to_1d = None
        if temp is None :
            temp = []

        for child in node.children:

            if not child.children:
                child_text  = child.text.decode('utf8')
                child_node_text = self.tokenizer(child_text)

                try:
                    text_index = torch.tensor(child_node_text['input_ids'])

                    max_tokens_and_actual_token_diff = self.max_sub_tokens - len(text_index)

                    if len(text_index) > self.overall_maximium_tokens_recored :
                        self.overall_maximium_tokens_recored = len(text_index)
                        self.overall_maximium_tokens_recored_text = child_text

                    if max_tokens_and_actual_token_diff  ==  0:

                        input_for_compressing_linear_layer = self.text_embedding(text_index)
                        convered_2d_to_1d = input_for_compressing_linear_layer.view(-1)

                    elif max_tokens_and_actual_token_diff < 0:
                        print(' warrning sub_tokens len large then "max_sub_tokens" limit \n we are continue with max_sub_tokens ')

                        after_reducing_sub_tokens_len_to_max_sub_tokens = self.text_embedding(text_index)[:self.max_sub_tokens]
                        convered_2d_to_1d = after_reducing_sub_tokens_len_to_max_sub_tokens.view(-1)

                    else:
                        ''' we are need to add more embedding becuase max_sub_tokens is match the size of sub_tokens'''

                        generated_left_embeddings_for_compressing_linear_layer = torch.zeros(max_tokens_and_actual_token_diff,128)        # zeros dot not update or any change
                                                                                                                                        # in weight and bias but have drawback
                        try:
                            text_embedding = self.text_embedding(text_index)
                            combined_generated_existed_embedding = torch.cat((self.text_embedding(text_index),generated_left_embeddings_for_compressing_linear_layer),dim = 0)
                        except Exception as e :
                            raise RuntimeWarning(f' error : {e} \n  error for this children node  : {text_index} --> {type(text_index)}')
                        convered_2d_to_1d = combined_generated_existed_embedding.view(-1)


                except Exception as e :
                    raise RuntimeWarning(f'error is founded for this text ;-- {child_node_text} \n this node_type : {child.type} \n text index - {torch.tensor(child_node_text["input_ids"])}')


                transformed_by_linear_layer_output = self.compressing_linear_layer(convered_2d_to_1d)# make sure we are append 1d
                transformed_by_linear_layer_output = transformed_by_linear_layer_output.unsqueeze(0)
                temp.append(transformed_by_linear_layer_output)

            else :
                try:
                    if child.type not in self.ignore_types:
                        index = torch.tensor(self.type_to_index.get(child.type))
                        parent_node_embedding = self.type_embedding(index).unsqueeze(0)
                        # print(f'___________embedding_for_parent_node________{parent_node_embedding.shape}')

                    else:
                        parent_node_embedding = torch.zeros(1,128)
                    # make sure parent_node_embedding  is 2d because logic says ( we are need to combine child and parent node embedding so dims is important)


                    # remember torch.stack() that is ok for computaional graph for torch torch know how to comptue that gradient of that
                    # and converting that List[torch.tensors()] --> when you compute torch.tensor() that is brack computation graph
                    # so the soltuion is torch.stack() but remember that , it is add or combine things with new demision unlike( torch.cat)

                    child_embedding_uisng_recursion = self.get_embedding_for_this_node(child).unsqueeze(0)  # after stack that become List() --> 2d

                    # that time you see this error that means only parent is exists not child
                    # or might be you see 0D --> means no child

                    # another case ! i know you are too smart

                    a = torch.cat((parent_node_embedding,child_embedding_uisng_recursion), dim = 0)
                    a = torch.mean(a, dim = 0,keepdim = True)
                    # here you say do  not append (2,128)  that mean we are combine that infromation by using mean i think but that is fine
                    # tell me about that
                    temp.append(a)

                except Exception as e:
                    raise RuntimeError(f' {e} \n error for this node  : {node.type} ... {temp}')

        return torch.mean(torch.cat(temp, dim = 0),dim = 0) # mean remove dim  like --> [1,128] after mean [128] keep in mind

    def __call__(self,node):        # for do that we are need to walk throw code_Graph
                                    # because how you give node ( type suing that type how you compute that children )

        index = torch.tensor(self.type_to_index.get(node.type))
        node_type_embedding = self.type_embedding(index)
        node_child_emebedding = self.get_embedding_for_this_node(node)
        # that fuction user is give node and get the combine embeding ( parent_type + text_embedding + child_embedding)

        return node_type_embedding + node_child_emebedding # becuase we are call the function is handle child only not include type of that particuler node os i am right over here
