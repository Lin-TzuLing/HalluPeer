import json
import pickle
import uuid


def remove_node(root_node, target_uuid: str):
    """Remove a node by its UUID from the tree"""
    # If this node is the target, cannot remove root from itself
    if root_node.uuid == target_uuid:
        raise ValueError("Cannot remove root node from itself. Remove from parent instead.")
    
    # Check if target is in direct children
    for i, child in enumerate(root_node.children):
        if child.uuid == target_uuid:
            root_node.children.pop(i)
            child.parent = None  # Clear parent reference
            return True
    
    # Recursively search in children's subtrees
    for child in root_node.children:
        if remove_node(child, target_uuid):
            return True
    
    return False


class HallucinationTaxonomyTreeNode:
    def __init__(
        self,
        concept: str,
        concept_desc: str,
        depth: int,
        parent=None,
        children: list = None,
    ):
        
        # basic attributes
        self.uuid = str(uuid.uuid4())
        self.concept = concept
        self.concept_desc = concept_desc
        self.depth = depth
        self.parent = parent
        self.children = children if children is not None else []
        
        # processing status
        self._processed = False
        self._processed_time = 0
        self._is_leaf = False
        


    def add_child(self, child_node):
        if isinstance(child_node, list):
            for child in child_node:
                child.parent = self
            self.children.extend(child_node)
        else:
            child_node.parent = self
            self.children.append(child_node)


    def get_siblings(self):
        sibs = (
            [child for child in self.parent.children if child.concept != self.concept]
            if self.parent
            else []
        )
        sibs_list = [
            {
                "concept": sib.concept,
                "description": sib.concept_desc,
            }
            for sib in sibs
        ]
        return sibs_list


    def get_leaf_nodes(self):
        if not self.children:
            self._is_leaf = True
            return [self]
        leaves = []
        for child in self.children:
            leaves.extend(child.get_leaf_nodes())
        return leaves
    
    
    
    def to_dict(self):
        result = {
            "concept": self.concept,
            "concept_desc": self.concept_desc,
            "depth": self.depth,
            "children": [child.to_dict() for child in self.children],
            "processed": self._processed,
            "processed_time": self._processed_time,
            "is_leaf": self._is_leaf,
        }
        return result


    def to_json_dict(self):
        """Plain nested dict (concept, concept_desc, depth, children) used for the released JSON taxonomy."""
        return {
            "concept": self.concept,
            "concept_desc": self.concept_desc,
            "depth": self.depth,
            "children": [child.to_json_dict() for child in self.children],
        }


    @classmethod
    def from_json_dict(cls, data, parent=None):
        node = cls(
            concept=data["concept"],
            concept_desc=data["concept_desc"],
            depth=data["depth"],
            parent=parent,
        )
        node.children = [cls.from_json_dict(c, parent=node) for c in data.get("children", [])]
        return node


    def print_tree(self, level=0):
        indent = "  " * level
        print(
            f"{indent}- {self.concept} (Depth: {self.depth}) (Processed: {self._processed}, Processed Time: {self._processed_time}, "
            f"Is Leaf: {self._is_leaf}"
        )
        for child in self.children:
            child.print_tree(level + 1)

    
    def mark_processed(self):
        self._processed = True
        self._processed_time += 1
    
    
    def __repr__(self):
        return f"Node(concept={self.concept}, depth={self.depth}, children={self.children})"
    
    def get_size(self):
        """Get the total number of nodes in the subtree rooted at this node"""
        size = 1  # Count this node
        for child in self.children:
            size += child.get_size()
        return size
    
    def get_node_by_uuid(self, target_uuid):
        """Retrieve a node by its UUID"""
        if self.uuid == target_uuid:
            return self
        for child in self.children:
            result = child.get_node_by_uuid(target_uuid)
            if result:
                return result
        return None
    
    def get_path_to_root(self):
        """Get the path from this node to the root as a list of nodes"""
        path = []
        current = self
        while current:
            path.append(current.concept)
            current = current.parent
        return path[::-1]  # Reverse to get path from root to this node
    
    
    # constraint the taxonomy tree to a certain depth
    def clone_subtree(self, max_depth):
        if self.depth > max_depth:
            return None
        new_node = HallucinationTaxonomyTreeNode(
            concept=self.concept,
            concept_desc=self.concept_desc,
            depth=self.depth,
        )
        for child in self.children:
            cloned_child = child.clone_subtree(max_depth)
            if cloned_child:
                new_node.add_child(cloned_child)
        return new_node
    
    def get_children_by_depth(self, node, depth):
        if node.depth == depth:
            return [node]
        result = []
        for child in node.children:
            result.extend(self.get_children_by_depth(child, depth))
        return result


def load_taxonomy(path):
    """Load a taxonomy tree from the released JSON file or from a pickle produced by the pipeline."""
    path = str(path)
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            return HallucinationTaxonomyTreeNode.from_json_dict(json.load(f))
    with open(path, "rb") as f:
        return pickle.load(f)
