import json
import collections
   

class TrieNode:
    """A node in the trie structure"""

    def __init__(self, subword_id):
        # the subword stored in this node
        self.subword_id = subword_id

        # whether this can be the end of a word
        self.is_word = False

        # whether this can be the end of an entity
        self.is_entity = False

        # a dictionary of child nodes: {token: nodes}
        self.children = {}
   
   
class Trie(object):
    """The trie object"""

    def __init__(self, tokenizer):
        """
        The trie has at least the root node.
        The root node does not store any character
        """
        self.root = TrieNode('')

        self.tokenizer = tokenizer
        
        self.tmp_node = dict()

    def insert(self, entity):
        """Insert an entity into the trie

        Args:
            - entity: list of entity tokens
        """
        node = self.root

        # For each token in the entity, create a new child for the current node if there is no child containing the token.
        for token in entity:
            subwords = self.tokenizer.tokenize(token)
            subword_ids = self.tokenizer.convert_tokens_to_ids(subwords)
            
            for subword_id in subword_ids:
                if subword_id in node.children:
                    node = node.children[subword_id]
                else:
                    # If a token is not found, create a new node in the trie
                    new_node = TrieNode(subword_id)
                    node.children[subword_id] = new_node
                    node = new_node
                    
            # Mark the end of a word
            node.is_word = True

        # Mark the end of an entity
        node.is_entity = True
    
    def search(self, node, x):
        """Traversal of the trie

        Args:
            - node: the node to start with
            - prefix: the current prefix, for tracing a word while traversing the trie
        """
        # Search the trie to get all candidates
        while len(node.children) != 0:
            # Check if the prefix is in the trie
            if x in node.children:
                node = node.children[x]
                return list(node.children), node.is_word, node.is_entity
            else:
                node = node.children

        # cannot found the prefix, return empty list
        return [], False, False
   
    def query(self, batch_id, x):
        """
        Given an input (a prefix), retrieve all words stored in
        the trie with that prefix
        Return:
            - candidate: a list of candidate subwords
            - is_word:   whether the subword is the end of a word
            - is_entity: whether the subword is the end of an entity
        """
        if x in self.tmp_node[batch_id].children:
            self.tmp_node[batch_id] = self.tmp_node[batch_id].children[x]
            candidates, is_word, is_entity = list(self.tmp_node[batch_id].children), self.tmp_node[batch_id].is_word, self.tmp_node[batch_id].is_entity
            
            if len(candidates) == 0:
                self.tmp_node[batch_id] = self.root
        else:
            candidates, is_word, is_entity = [], False, False

        return candidates, is_word, is_entity
   
    def traverse(self, root):
        if len(root.children) != 0:
            tree_dict = {}
            for child_key in root.children:
                child = root.children[child_key]
                child_subword = self.tokenizer.decode(child.subword_id)
                
                if len(child.children) != 0:
                    tree_dict[child_subword] = self.traverse(child)
                else:
                    return {child_subword: '<EOS>'}
                
            return tree_dict
   
    def print_trie(self, root):
        self.dump_dict = collections.defaultdict(dict)

        if root.subword_id != '':
            entity_type = self.tokenizer.convert_ids_to_tokens([root.subword_id])[0]
            self.dump_dict['<BOS>'] = {entity_type: self.traverse(root)}
        else:
            self.dump_dict['<BOS>'] = self.traverse(root)

        print(json.dumps(self.dump_dict, sort_keys=True, indent=4))
