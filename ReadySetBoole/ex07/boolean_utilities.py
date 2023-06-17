from collections import deque
import itertools

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
    Simple functions to read, transform and evaluate a syntax tree
    """

    def transformImplicationNode(self, root: Node) -> None:
        """
        Transform implication AB> in A!B|
        """
        left = root.left
        z = Node("!", left)
        root.left = z
        root.value = "|"

    def transformEquivalenceNode(self, root: Node) -> None:
        """
        Transform equivalence AB= in A!B|AB!|&
        """
        left = root.left
        right = root.right
        z = Node("!", left)
        root.left = Node("|", z, right)
        y = Node("!", right)
        root.right = Node("|", left, y)
        root.value = "&"

    def transformXorNode(self, root: Node) -> None:
        """
        Transform XOR AB^ in AB|AB&!&
        """
        left = root.left
        right = root.right
        root.value = "&"
        z = Node("|", left, right)
        root.left = z
        x = Node("&", left, right)
        y = Node("!", x)
        root.right = y

    def transformInitialNegationNode(self, root: Node) -> Node:
        """
        If the tree begins with a negation
        """
        if not root:
            return None
        if root.left.value == "!":
            return root.left.left
        elif root.left.value == "0":
            return Node("1")
        elif root.left.value == "1":
            return Node("0")
        elif root.left.value == "&" or root.left.value == "|":
            a = root.left.left
            b = root.left.right
            x = Node("!", a)
            y = Node("!", b)
            if root.left.value == "&":
                return Node("|", x, y)
            elif root.left.value == "|":
                return Node("&", x, y)

    def eliminateImplicationAndEquivalence(self, root: Node) -> None:
        """
        Recursive elimination of implications, equivalence and xor
        """
        if not root:
            return
        if root.value == ">":
            self.transformImplicationNode(root)
        elif root.value == "=":
            self.transformEquivalenceNode(root)
        elif root.value == "^":
            self.transformXorNode(root)

        self.eliminateImplicationAndEquivalence(root.left)
        self.eliminateImplicationAndEquivalence(root.right)

    def pushDisjunctionDown(self, root: Node) -> None:
        """
        Push the disjunctions down the tree, and conjunctions up
        """
        if not root:
            return
        if root.value == "|":
            if root.left.value == "|":
                self.pushDisjunctionDown(root.left)
            if root.right.value == "|":
                self.pushDisjunctionDown(root.right)
            if root.left.value == "&":
                a = root.left.left
                b = root.left.right
                c = root.right
                root.value = "&"
                root.left = Node("|", c, a)
                root.right = Node("|", c, b)
            elif root.right.value == "&":
                a = root.right.left
                b = root.right.right
                c = root.left
                root.value = "&"
                root.left = Node("|", c, a)
                root.right = Node("|", c, b)
        self.pushDisjunctionDown(root.left)
        self.pushDisjunctionDown(root.right)

    def rearrangeDisjunction(self, root: Node) -> None:
        """
        Rearrange disjunctions at the end of each clause
        """
        if not root:
            return
        if root.value == "|":
            while root.left.value == "|" and (root.right.value not in ("&")):
                a = root.left.left
                b = root.left.right
                c = root.right
                root.left = a
                root.right = Node("|", b, c)
        self.rearrangeDisjunction(root.left)
        self.rearrangeDisjunction(root.right)

    def rearrangeConjunction(self, root: Node) -> None:
        """
        Rearrange conjunctions at the end of the expression
        """
        if not root:
            return
        if root.value == "&":
            while root.left.value == "&" and (root.right.value not in ("|")):
                a = root.left.left
                b = root.left.right
                c = root.right
                root.left = a
                root.right = Node("&", b, c)
        self.rearrangeConjunction(root.left)
        self.rearrangeConjunction(root.right)

    def manageNegations(self, root: Node) -> None:
        """
        Manage double negations and negation distributivity
        in a recursive way
        """
        if not root:
            return
        if root.left:
            if root.left.value == "!" and not root.left.left.value.isupper():
                root.left = self.transformInitialNegationNode(root.left)
        if root.right:
            if root.right.value == "!" and not root.right.left.value.isupper():
                root.right = self.transformInitialNegationNode(root.right)
        self.manageNegations(root.left)
        self.manageNegations(root.right)

    def inorder(self, root: Node, output: list) -> None:
        """
        in order traversal of the tree
        """
        if not root:
            return
        self.inorder(root.left)
        # print(root.value, end=" ")
        output.append(root.value)
        self.inorder(root.right)

    def preorder(self, root: Node, output: list) -> None:
        """
        pre order traversal of the tree
        """
        if not root:
            return
        # print(root.value, end=" ")
        output.append(root.value)
        self.preorder(root.left)
        self.preorder(root.right)

    def postorder(self, root: Node, output: list) -> None:
        """
        post order traversal of the tree (aka reverse polish)
        """
        if not root:
            return
        self.postorder(root.left, output)
        self.postorder(root.right, output)
        # print(root.value, end=" ")
        output.append(root.value)

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


def find_number_variable(formula: str) -> list:
    """
    Take a reverse polish formula and extract
    and sort the variables as a list
    """
    variables = []
    for c in formula:
        if c.isupper():
            variables.append(c)
    return sorted(list(set(variables)))


def printFirstLines(variables: list, n: int) -> None:
    """
    Print the first lines of the truth table
    """
    if not variables or not n:
        errorMessage()
    first = "| "
    for variable in variables:
        first = first + variable + " | "
    first = first + "= |"
    print(first)
    print("|---" * (n + 1) + "|")


def printTruth(formula: str, variables: list, table: list) -> None:
    """
    Complete the truth table
    """
    for rank in table:
        n = len(rank)
        line = ""
        changedFormula = formula
        for i in range(n):
            line = line + "| " + str(rank[i]) + " "
            changedFormula = changedFormula.replace(variables[i], str(rank[i]))
        eval = eval_formula(changedFormula)
        if eval is None:
            return None
        line = line + "| " + str(int(eval)) + " |"
        print(line)


def print_truth_table(formula: str) -> None:
    """
    a function that takes as input a string
    that contains a propositional formula in reverse polish notation,
    and writes its truth table on the standard output
    """
    variables = find_number_variable(formula)
    n = len(variables)
    printFirstLines(variables, n)
    table = list(itertools.product([0, 1], repeat=n))
    printTruth(formula, variables, table)


def adder(a: int, b: int) -> int:
    """
    a function that takes as parameters two natural numbers
    a and b and returns one natural number that equals a + b,
    bitwise
    """
    while b != 0:
        carry = a & b  # la retenue est mise qd 1 + 1
        a = a ^ b  # l'addition sans retenue
        b = carry << 1  # decalage vers la gauche de la retenue
    return a


def multiplier(a: int, b: int) -> int:
    """
    a function that takes as parameters two natural numbers
    a and b and returns one natural number that equals a * b,
    bitwise
    """
    if not (isinstance(a, int)) or not (isinstance(b, int)) or a < 0 or b < 0:
        print("Only natural numbers ")
        return None

    if b > a:
        a, b = b, a  # afin que b soit le plus petit
    res = 0
    while b > 0:
        if b & 1:  # si b est impair
            res = adder(res, a)
        a = a << 1
        b = b >> 1
    return res


def gray_code(n: int) -> int:
    """
    A function that takes an integer n and returns its equivalent in Gray code
    """
    return n ^ (n >> 1)


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
            if letter.isupper() or letter in ["0", "1"]:
                stack.append(Node(letter))
            else:
                return errorMessage()
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


def negation_normal_form(formula: str) -> str:
    """
    A function that takes as input a string that contains
    a propositional formula in reverse polish notation,
    and returns an equivalent formula in Negation Normal Form (NNF)
    """
    tree = SyntaxTree()
    root = postfixToSyntaxTree(formula)

    tree.eliminateImplicationAndEquivalence(root)
    if not root:
        return None
    while root.value == "!" and not root.left.value.isupper():
        root = tree.transformInitialNegationNode(root)
    tree.manageNegations(root)

    output = []
    tree.postorder(root, output)
    return ''.join(output)


def nnfForCnf(formula: str) -> Node:
    """
    Same function than negation_normal_form, but returns
    an equivalent formula in Negation Normal Form (NNF)
    as a tree
    """
    tree = SyntaxTree()
    root = postfixToSyntaxTree(formula)

    tree.eliminateImplicationAndEquivalence(root)
    if not root:
        return None
    while root.value == "!" and not root.left.value.isupper():
        root = tree.transformInitialNegationNode(root)
    tree.manageNegations(root)
    return root


def conjunctive_normal_form(formula: str) -> str:
    """
    a function that takes as input a string that contains
    a propositional formula in reverse polish notation,
    and returns an equivalent formula in Conjunctive Normal Form (CNF)
    """
    tree = SyntaxTree()
    root = nnfForCnf(formula)
    tree.pushDisjunctionDown(root)
    tree.rearrangeDisjunction(root)
    tree.rearrangeConjunction(root)

    output = []
    tree.postorder(root, output)
    return ''.join(output)


def cnfForSAT(formula: str) -> Node:
    """
    Same function as conjunctive_normal_form,
    but returns the root of the syntax tree
    """
    tree = SyntaxTree()
    root = nnfForCnf(formula)
    tree.pushDisjunctionDown(root)
    tree.rearrangeDisjunction(root)
    tree.rearrangeConjunction(root)
    return root
