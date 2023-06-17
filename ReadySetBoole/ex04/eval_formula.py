from collections import deque

BINARY_OP = "&|^>="
UNARY_OP = "!"


class Node:
    """
    Create a node for our syntax tree
    """
    def __init__(self, value=None, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right


class SyntaxTree:
    """
    Simple functions to read and evaluate a syntax tree
    """

    def inorder(self, root: Node) -> None:
        """
        in order traversal of the tree
        """
        if not root:
            return
        self.inorder(root.left)
        print(root.value, end=" ")
        self.inorder(root.right)

    def preorder(self, root: Node) -> None:
        """
        pre order traversal of the tree
        """
        if not root:
            return
        print(root.value, end=" ")
        self.preorder(root.left)
        self.preorder(root.right)

    def postorder(self, root: Node) -> None:
        """
        post order traversal of the tree (aka reverse polish)
        """
        if not root:
            return
        self.postorder(root.left)
        self.postorder(root.right)
        print(root.value, end=" ")

    def evaluate(self, root: Node) -> int:
        """
        a function that takes as input the root of a syntax tree,
        evaluates this tree, then returns the result.
        """
        if root is None:  # empty tree
            return
        if root.left is None and root.right is None:  # leaf
            return int(root.value)
        leftResult = self.evaluate(root.left)  # evaluate left tree
        rightResult = self.evaluate(root.right)  # evaluate right tree

        if root.value == "&":
            return leftResult & rightResult
        elif root.value == "|":
            return leftResult | rightResult
        elif root.value == "^":
            return leftResult ^ rightResult
        elif root.value == "=":
            return leftResult == rightResult
        elif root.value == ">":  # -> equivalent !a | b
            return (not leftResult) | rightResult
        elif root.value == "!":
            return not leftResult


def errorMessage():
    """ guess what it does"""
    print("syntax error, missing elements")
    return None


def postfixToSyntaxTree(formula: str) -> Node:
    """
    a function that takes as input a string that contains
    a propositional formula in reverse polish notation,
    then returns the corresponding syntax tree
    """
    stack = deque()
    for letter in formula:
        if letter in BINARY_OP:
            if len(stack) < 2:
                return errorMessage()
            x = stack.pop()
            y = stack.pop()
            z = Node(letter, y, x)
            stack.append(z)
        elif letter in UNARY_OP:
            if len(stack) < 1:
                return errorMessage()
            x = stack.pop()
            z = Node(letter, x)
            stack.append(z)
        else:
            stack.append(Node(letter))
    if len(stack) != 1:
        return errorMessage()
    return stack.pop()


def eval_formula(formula: str) -> bool:
    """
    a function that takes as input a string that contains
    a propositional formula in reverse polish notation,
    evaluates this formula, then returns the result as a boolean.
    """
    tree = SyntaxTree()
    root = postfixToSyntaxTree(formula)
    if not root:
        return
    return bool(tree.evaluate(root))
