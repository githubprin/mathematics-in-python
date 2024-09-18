import sys 
import math
import re

from numbers import Number 
from typing import Dict, Callable, Iterable, Union
from typing_extensions import TypeAlias, Self 

from ADT.stack import Stack
from ADT.tree import Tree

number: TypeAlias = Union[Number, int, float]

class Token(list):
    """
    Represents a single token in the parsing process of an equation or expression.

    A Token is an instance that holds the type and value of a token extracted from an equation string.
    It is used during the parsing process to build the abstract syntax tree.

    Parameters:
        - tok_type (str): The type of the token. Possible values include 'op', 'unary', 'para', 'num', 'var', 'comma', 'func'.
        - tok_val (Any): The value of the token. For example, '+', '-', '*', '/', '(', ')', a number, a variable name, etc.

    Attributes:
        - tok_type (str): The type of the token.
        - tok_val (Any): The value of the token.

    Methods:
        - is_operator(): Returns True if the token is a binary operator.
        - is_unary_operator(): Returns True if the token is a unary operator.

    Examples:
        >>> token = Token('op', '+')
        >>> token.tok_type
        'op'
        >>> token.tok_val
        '+'
        >>> token.is_operator()
        True

        >>> token = Token('num', 3.14)
        >>> token.tok_type
        'num'
        >>> token.tok_val
        3.14
        >>> token.is_operator()
        False

    Detailed Explanation:
        The Token class is designed to encapsulate the information about a single token during the parsing process.
        It inherits from the list class to store the tuple (tok_type, tok_val) and provides additional attributes for ease of access.
        The class also contains class variables that define operator priorities, function dictionaries, and token tables used in parsing.

    Step-by-Step Implementation Guide:
        1. Initialize a Token instance with the token type and value.
        2. The __init__ method stores the tok_type and tok_val as attributes.
        3. The is_operator() method checks if the token type is 'op'.
        4. The is_unary_operator() method checks if the token type is 'unary'.

    """
    OPERATOR_PRIORITY = {\
        '+' : 1, 
        '-' : 1, 
        '*' : 2, 
        '/' : 2, 
        '^' : 3, 
        'unary -' : 4, 
        '=' : 0}

    PARA = ['(', ')', '[', ']', '{', '}',]

    FUNCTION_DICT = {\
        'cos' : (math.cos, math.sin, 
                        '-1*sin(placeholder)'),
        'sin' : (math.sin, lambda x:math.cos(x), 
                        'cos(placeholder)',),
        'tan' : (math.tan, lambda x:1/(math.cos(x)**2), 
                        '1/cos(placeholder)^2'),
        'ln' : (math.log, lambda x:1/x, 
                        '1/placeholder'),
        # custom functions 
        }

    TOKEN_TABLE = {\
        'op' : ['^', '+', '-', '*',  '/', '='],
        'unary' :  ['-'],
        'para' : ['(', ')'],
        'num' : [r"[1-9][0-9]*\.?[0-9]*|0",],
        'var' : [r"[a-zA-Z]+_?[0-9]*",], 
        'comma' : [',']}

    OPERATION_WITH_FUNCTIONS = {\
        '+' : lambda x, y : x + y, 
        '-' : lambda x, y : x - y,
        '*' : lambda x, y : x * y,
        '/' : lambda x, y : x / y,
        '^' : lambda x, y : x ** y,
    }

    def __init__(self, tok_type: str, tok_val: Union[str, number]):
        """
        Initializes a Token instance.

        Parameters:
            - tok_type (str): The type of the token.
            - tok_val (Any): The value of the token.

        Returns:
            - None
        """
        super().__init__((tok_type, tok_val))
        self.tok_type = tok_type 
        self.tok_val = tok_val 

    def is_operator(self) -> bool:
        """
        Checks if the token is a operator.

        Returns:
            - bool: True if the token type is 'op', False otherwise.
        """
        return self.tok_type == 'op'

    def is_unary_operator(self) -> bool:
        """
        Checks if the token is a unary operator.

        Returns:
            - bool: True if the token type is 'op', False otherwise.
        """
        return self.tok_type == 'unary'

    def is_variable(self) -> bool:
        """
        Checks if the token is a variable.

        Returns:
            - bool: True if the token type is 'op', False otherwise.
        """
        return self.tok_type == 'var'

    def is_number(self) -> bool:
        return self.tok_type == 'num'

    def __eq__(self, other: Token) -> bool :
        if isinstance(other, Token):
            return self.tok_type == other.tok_type and self.tok_val == other.tok_val
        return False 

    def __hash__(self):
        return hash((self.tok_type, self.tok_val))

class FormulaTree(Tree):
    """
    Represents a abstract syntax tree (AST) of a mathematical formula.

    FormulaTree is a subclass of the Tree class, specialized for representing parsed mathematical expressions.
    Each node contains a Token and potentially a list of child FormulaTree nodes.

    Parameters:
        - root (Token): The root token of the tree or subtree. Must be an instance of Token.
        - children (list of FormulaTree, optional): A list of child subtrees. Defaults to an empty list.

    Attributes:
        - datum (Token): The token associated with the node.
        - tok_type (str): The type of the token (e.g., 'op', 'num', 'var', etc.).
        - tok_val (Any): The value of the token.

    Methods:
        - iter_subtree(): Generator that yields all subtrees starting from the current node.
        - leaves(): Generator that yields all leaf tokens in the subtree.

    Examples:
        >>> root_token = Token('op', '+')
        >>> child1 = FormulaTree(Token('num', 3))
        >>> child2 = FormulaTree(Token('num', 4))
        >>> tree = FormulaTree(root_token, [child1, child2])
        >>> [leaf.tok_val for leaf in tree.leaves()]
        [3, 4]

    Detailed Explanation:
        FormulaTree is used to construct the AST of a mathematical expression parsed from a string.
        Each node in the tree represents an operator, operand, or function, with child nodes representing sub-expressions.
        The iter_subtree() method allows traversal of the tree in a depth-first manner.
        The leaves() method collects all leaf nodes, which are tokens without any children (i.e., numbers or variables).

    Step-by-Step Implementation Guide:
        1. Initialize a FormulaTree with a root Token and optional children.
        2. The __init__ method asserts that the root is a Token and initializes attributes.
        3. The iter_subtree() method recursively yields the current node and all its descendants.
        4. The leaves() method uses iter_subtree() to find and yield all leaf tokens.

    """
    def __init__(self, root: Token, children: list[FormulaTree] = []):
        assert isinstance(root, Token)
        super().__init__(root, children) 
        self.datum = root 
        self.tok_type = root.tok_type 
        self.tok_val = root.tok_val 

    def iter_subtree(self) -> Iterator[FormulaTree]:
        """
        Generator to iterate over all subtrees starting from the current node.

        Yields:
            - FormulaTree: Subtrees in a depth-first manner.

        Example:
            >>> root_token = Token('op', '*')
            >>> child1 = FormulaTree(Token('num', 5))
            >>> child2 = FormulaTree(Token('num', 6))
            >>> tree = FormulaTree(root_token, [child1, child2])
            >>> for subtree in tree.iter_subtree():
            ...     print(subtree.tok_val)
            *
            5
            6

        Detailed Explanation:
            The iter_subtree() method uses recursion to yield the current node and then iterates through each child,
            recursively yielding their subtrees. This provides a full traversal of the tree in depth-first order.

        Step-by-Step Implementation Guide:
            1. Yield the current node (self).
            2. For each child in self.children:
                a. Recursively call iter_subtree() on the child.
                b. Yield all subtrees from the child.
        """
        yield self 

    def iter_subtree_with_address(self) -> Iterator[tuple[list[int], FormulaTree]]:
        """
        Generator to iterate over all subtrees with their addresses in the tree.

        Yields:
            - tuple: A pair (address, subtree), where address is a list of indices indicating the position.

        Example:
            >>> root_token = Token('op', '+')
            >>> child1 = FormulaTree(Token('num', 1))
            >>> child2 = FormulaTree(Token('op', '*'), [FormulaTree(Token('num', 2)), FormulaTree(Token('num', 3))])
            >>> tree = FormulaTree(root_token, [child1, child2])
            >>> for addr, subtree in tree.iter_subtree_with_address():
            ...     print(addr, subtree.tok_val)
            [] +
            [0] 1
            [1] *
            [1, 0] 2
            [1, 1] 3

        Detailed Explanation:
            The iter_subtree_with_address() method traverses the tree and keeps track of the address of each node.
            The address is a list of indices that represent the path from the root to the node.

        Step-by-Step Implementation Guide:
            1. Yield the current node with an empty address ([]).
            2. For each child in self.children, along with its index:
                a. Recursively call iter_subtree_with_address() on the child.
                b. For each address and subtree from the child, prepend the current index to the address.
                c. Yield the updated address and subtree.
        """
        yield [], self 

    def leaves(self) -> Iterator[Token]:
        """
        Generator to yield all leaf tokens in the subtree.

        Yields:
            - Token: Leaf tokens without any children.

        Example:
            >>> root_token = Token('op', '+')
            >>> child1 = FormulaTree(Token('num', 4))
            >>> child2 = FormulaTree(Token('num', 5))
            >>> tree = FormulaTree(root_token, [child1, child2])
            >>> for leaf in tree.leaves():
            ...     print(leaf.tok_val)
            4
            5

        Detailed Explanation:
            The leaves() method traverses the tree using iter_subtree() and yields tokens from nodes that have no children.
            These are considered leaf nodes, typically representing variables or numbers.

        Step-by-Step Implementation Guide:
            1. Iterate over all subtrees using iter_subtree().
            2. If a subtree has no children (i.e., it's a leaf node):
                a. Yield the datum (token) of the subtree.
        """
        yield self 

    def replace_subtree(self, src: FormulaTree, dest: FormulaTree):
        """
        Replaces a subtree matching 'src' with 'dest' in the tree.

        Parameters:
            - src (FormulaTree): The subtree to be replaced.
            - dest (FormulaTree): The subtree to replace with.

        Returns:
            - None

        Example:
            >>> root_token = Token('op', '+')
            >>> child1 = FormulaTree(Token('num', 2))
            >>> child2 = FormulaTree(Token('num', 3))
            >>> tree = FormulaTree(root_token, [child1, child2])
            >>> new_subtree = FormulaTree(Token('num', 5))
            >>> tree.replace_subtree(child2, new_subtree)
            >>> [leaf.tok_val for leaf in tree.leaves()]
            [2, 5]

        Detailed Explanation:
            The replace_subtree() method searches through the tree for a subtree that matches 'src'.
            When found, it deletes that subtree and inserts 'dest' at the same position.

        Step-by-Step Implementation Guide:
            1. Use iter_subtree_with_address() to traverse the tree with addresses.
            2. For each subtree, check if it is equal to 'src'.
            3. If a match is found:
                a. Delete the subtree at the found address using self.delete(addr).
                b. Insert 'dest' at the same address using self.insert(addr, dest).
        """
        return

    def __eq__(self, other: FormulaTree) -> bool:
        """
        Checks equality between two FormulaTree instances.

        Parameters:
            - other (FormulaTree): Another FormulaTree instance to compare with.

        Returns:
            - bool: True if the trees are equal, False otherwise.

        Example:
            >>> tree1 = FormulaTree(Token('num', 3))
            >>> tree2 = FormulaTree(Token('num', 3))
            >>> tree1 == tree2
            True

            >>> tree3 = FormulaTree(Token('num', 4))
            >>> tree1 == tree3
            False

        Detailed Explanation:
            The __eq__() method compares the root tokens and the structure of the child nodes.
            It returns True only if both the tokens and the structure of the trees are identical.

        Step-by-Step Implementation Guide:
            1. Check if 'other' is an instance of FormulaTree.
            2. Compare the root tokens (self.datum and other.datum).
            3. Check if the number of children is the same.
            4. Recursively compare each pair of child nodes.
            5. Return True if all checks pass, else return False.
        """
        return True      

    def flatten(self) -> str:
        """
        Flattens the FormulaTree into a string representation of the mathematical expression.

        This method recursively traverses the FormulaTree and reconstructs the original mathematical expression as a string.
        It handles variables, numbers, binary operators, and unary operators appropriately.

        Returns:
            - str: The string representation of the mathematical expression.

        Examples:
            >>> # Constructing a FormulaTree for the expression: 2 * x + 3
            >>> token_plus = Token('op', '+')
            >>> token_mul = Token('op', '*')
            >>> token_num2 = Token('num', '2')
            >>> token_varx = Token('var', 'x')
            >>> token_num3 = Token('num', '3')
            >>> tree_mul = FormulaTree(token_mul, [FormulaTree(token_num2), FormulaTree(token_varx)])
            >>> tree = FormulaTree(token_plus, [tree_mul, FormulaTree(token_num3)])
            >>> expression = tree.flatten()
            >>> print(expression)
            '2*x+3'

            >>> # Flattening a unary operation: -x
            >>> token_unary_minus = Token('unary', '-')
            >>> token_varx = Token('var', 'x')
            >>> tree_unary = FormulaTree(token_unary_minus, [FormulaTree(token_varx)])
            >>> expression = tree_unary.flatten()
            >>> print(expression)
            '-x'

        Detailed Explanation:
            The flatten() method reconstructs the mathematical expression represented by the FormulaTree into a string format.
            It operates recursively, handling different types of nodes:

            - If the node represents a variable or a number, it returns the string representation of its value.
            - If the node represents a binary operator, it recursively flattens each of its child nodes, collects the results in a list,
            and joins them using the operator's value.
            - If the node represents a unary operator, it recursively flattens its single child and concatenates the operator's value with the result.

        Step-by-Step Implementation Guide:
            1. Check if the current node's datum represents a variable or a number:
                - Use self.datum.is_variable() or self.datum.is_number() to determine the type.
                - If true, return the string representation of self.datum.tok_val.
            2. Check if the current node's datum represents a binary operator:
                - Initialize an empty list 'res' to store the flattened strings of child nodes.
                - Iterate over each child in self.children:
                    a. Recursively call child.flatten() to flatten the child subtree.
                    b. Append the result to the 'res' list.
                - Join the elements of 'res' using self.datum.tok_val (the operator) as the separator.
                - Return the resulting string.
            3. Check if the current node's datum represents a unary operator:
                - Assert that there is exactly one child node (since it's unary).
                - Recursively call flatten() on the single child.
                - Concatenate self.datum.tok_val (the unary operator) with the result of the recursive call.
                - Return the resulting string.
        """
        return ''

class Formula:
    """
    Represents a mathematical formula parsed from a string into an abstract syntax tree.

    The Formula class encapsulates a mathematical expression, allowing for operations such as evaluation,
    substitution, simplification, and algebraic manipulations. It parses the input string into a FormulaTree
    and provides methods to work with the formula.

    Parameters:
        - eq (str): The equation or expression in string form to be parsed.
        - constant_variable (list, optional): A list of variables to be treated as constants. Defaults to an empty list.
        - debug (bool, optional): Flag to enable debug mode. Defaults to False.

    Attributes:
        - eq (str): The original equation string.
        - tree (FormulaTree): The abstract syntax tree representation of the equation.
        - variables (set): A set of variable tokens present in the formula.

    Methods:
        - __add__(other): Defines addition between Formulas.
        - __sub__(other): Defines subtraction between Formulas.
        - __mul__(other): Defines multiplication between Formulas.
        - __div__(other): Defines division between Formulas.
        - __str__(): Returns the equation string.
        - terms(): Returns the terms of the formula if the root operator is '+'.
        - substitute(var, val): Substitutes a variable with another Formula or value.
        - simplify(): Simplifies the formula (implementation pending).
        - solve(variables): Solves the equation for the given variables (implementation pending).
        - __call__(*value_list, **value_dict): Evaluates the formula with given variable values.

    Examples:
        >>> f = Formula('2*x + 3')
        >>> print(f)
        2*x + 3
        >>> f.variables
        {'x'}
        >>> f(5)  # Assuming _calculate is properly implemented
        13

        >>> g = Formula('x^2')
        >>> h = f + g
        >>> print(h)
        (2*x + 3)+(x^2)

    Detailed Explanation:
        Upon initialization, the Formula class parses the input equation string into a FormulaTree using the parse() function.
        It then extracts the variables from the leaves of the tree. The class supports arithmetic operations by overloading
        operators such as __add__, __sub__, etc., which create new Formula instances combining the expressions.

        The __call__ method allows the Formula instance to be called as a function to evaluate the expression with given variable values.
        The _calculate static method is used internally to compute the numerical value by traversing the FormulaTree.

    Step-by-Step Implementation Guide:
        1. Initialize the Formula with an equation string.
        2. Parse the equation into a FormulaTree.
        3. Extract variables from the tree leaves.
        4. Define arithmetic operations to return new Formulas.
        5. Implement the __call__ method to evaluate the formula with variable values.
        6. Implement additional methods for substitution, simplification, and solving equations.

    """
    def __init__(self, eq: str, constant_variables: list[str] = [], debug: bool = False):
        """
        Initializes a Formula instance.

        Parameters:
            - eq (str): The equation or expression to parse.
            - constant_variable (list, optional): Variables to treat as constants. Defaults to [].
            - debug (bool, optional): Enable debug mode. Defaults to False.

        Returns:
            - None
        """
        self.eq = eq
        tree = parse(eq)
        self.tree = tree
        var_list = [x.tok_val for x in tree.leaves() if x.is_variable()]
        var_list = list(set(var_list))
        self.variables = set(var_list)
        self.constant_variables = constant_variables

    def __add__(self, other: Formula) -> Formula: 
        """
        Defines addition for Formulas. Similar for all other mathematical operational magic methods. 

        Parameters:
            - other (Formula): Another Formula instance.

        Returns:
            - Formula: A new Formula representing the sum.

        Example:
            >>> f = Formula('2*x')
            >>> g = Formula('3')
            >>> h = f + g
            >>> print(h)
            (2*x)+(3)
        """
        return Formula('(' + self.eq + ')+(' + other.eq + ')',)

    def __sub__(self, other: Formula) -> Formula:
        return ''

    def __mul__(self, other: Formula) -> Formula:
        return ''

    def __div__(self, other: Formula) -> Formula:
        return ''

    def __str__(self) -> str:
        return self.eq

    def substitute(self, var: str, val: number) -> Formula:
        # find var token and replace to val Tree or Formula
        return Formula('1')

    def __call__(self, *value_list, **value_dict) -> number:
        """
        Evaluates the formula with given variable values.

        Parameters:
            - *value_list: Positional arguments for variable values.
            - **value_dict: Keyword arguments for variable values.

        Returns:
            - float: The numerical result of the evaluation.

        Raises:
            - TypeError: If input values are invalid.

        Example:
            >>> f = Formula('2*x + 3')
            >>> f(5)
            13  # Assuming _calculate is properly implemented
        """
        return 0

    @staticmethod    
    def _calculate(tree, value_dict: Dict[str, number]) -> number:
        """
        Recursively calculates the numerical value of the FormulaTree.

        Parameters:
            - tree (FormulaTree): The tree to evaluate.
            - value_dict (dict): Dictionary of variable values.

        Returns:
            - float: The numerical result.

        Raises:
            - AssertionError: If an invalid token type is encountered.
        """
        return 0 

def tokenizer(equation: str) -> list[Token]:
    """
    Tokenizes a mathematical equation string into a sequence of Token instances.

    This function processes the input equation string and yields Token objects representing numbers, variables, operators, parentheses, and other symbols according to the defined TOKEN_TABLE in the Token class.

    Parameters:
        - equation (str): The mathematical equation or expression to tokenize.

    Returns:
        - list[Token]: List of Token instances representing a part of the equation.

    Examples:
        >>> tokens = list(tokenizer('2*x + 3'))
        >>> for token in tokens:
        ...     print(token.tok_type, token.tok_val)
        num 2.0
        op *
        var x
        op +
        num 3.0

    Detailed Explanation:
        The tokenizer function removes all whitespace from the input equation and iteratively processes the string.
        It uses regular expressions to match numbers and variables, and checks for operators and other symbols defined in the TOKEN_TABLE.
        The function keeps track of a num_flag to handle cases like unary operators correctly.

    Step-by-Step Implementation Guide:
        1. Remove all whitespace from the input equation.
        2. Initialize the list of token strings from the TOKEN_TABLE.
        3. Define a helper function find_key_from_elem to find the token type for a given token value.
        4. Iterate over the remaining string 'left' until it is empty.
        5. For each possible token, attempt to match it at the beginning of 'left'.
            a. If it matches a number pattern, yield a 'num' Token and update 'left'.
            b. If it matches a variable pattern, yield a 'var' Token and update 'left'.
            c. If it matches an operator or other symbol, yield the appropriate Token and update 'left'.
        6. Handle unary operators by checking the num_flag.

    Raises:
        - AssertionError: If an unexpected token is encountered.

    """
    res = []
    left = equation.replace(' ', '')
    
    return res 

def compress_tok(tokenizer: list[Token]) -> list[Token]:
    """
    Processes tokens to identify function names and reclassify them as 'func' tokens.

    This function takes an iterator of Token objects and yields new Token instances.
    If a 'var' token matches a function name in FUNCTION_DICT, it is reclassified as a 'func' token.

    Parameters:
        - tokenizer (iterator of Token): An iterator that yields Token instances.

    Returns:
        - Token: A Token instance, potentially reclassified as a 'func'.

    Examples:
        >>> tokens = tokenizer('sin(x)')
        >>> compressed_tokens = list(compress_tok(tokens))
        >>> for token in compressed_tokens:
        ...     print(token.tok_type, token.tok_val)
        func sin
        para (
        var x
        para )

    Detailed Explanation:
        The compress_tok function scans through the tokens produced by the tokenizer.
        It checks if any 'var' token corresponds to a function name defined in FUNCTION_DICT.
        If it does, it yields a new Token with type 'func' instead of 'var'.
        All other tokens are unchanged. 

    Step-by-Step Implementation Guide:
        1. Iterate over the tokens produced by the tokenizer.
        2. For each token, check if it is a 'var' token and if its value is a function name.
        3. If it is a function name, yield a new Token with type 'func'.
        4. Otherwise, yield the token unchanged.

    Raises:
        - AssertionError: If the input is not an instance of Token.

    """
    return []

def recursive_descent(tokens: list[Token]) -> FormulaTree:
    """
    Parses a sequence of tokens into an abstract syntax tree using recursive descent parsing.

    This function implements the recursive descent parsing algorithm to convert a list of Token instances
    into a FormulaTree representing the parsed expression.

    Parameters:
        - tokens (list of Token): A list of Token instances to parse.

    Returns:
        - FormulaTree: The root of the parsed abstract syntax tree.

    Examples:
        >>> tokens = list(compress_tok(tokenizer('2*x + 3')))
        >>> tree = recursive_descent(tokens)
        >>> tree.tok_type
        'op'
        >>> tree.tok_val
        '+'

    Detailed Explanation:
        The recursive_descent function initializes operator and operand stacks and starts the parsing process by calling expr().
        It ensures all tokens are instances of Token and uses the expr() function to parse the expression starting from index 0.
        After parsing, it pops the final result from the operand stack.

    Step-by-Step Implementation Guide:
        1. Verify that all elements in tokens are instances of Token.
        2. Initialize empty operator and operand stacks.
        3. Call the expr() function to parse the tokens starting from index 0.
        4. After parsing, pop the resulting FormulaTree from the operand stack.
        5. Return the FormulaTree as the parsed abstract syntax tree.

    Raises:
        - AssertionError: If any element in tokens is not an instance of Token.
    """
    for t in tokens: 
        assert isinstance(t, Token), t
    operator = Stack()
    operand = Stack() 
    idx = expr(operator, operand, tokens, 0)
    res = operand.pop()

    return res

def expr(operator: Stack, operand: Stack, tokens: list[Token], idx: int) -> int:
    """
    Parses an expression from the token list starting at the given index.

    The expr function implements the grammar rule: expr := part (binary_operator part)*

    Parameters:
        - operator (Stack): A stack to hold operators during parsing.
        - operand (Stack): A stack to hold operands (FormulaTree instances) during parsing.
        - tokens (list of Token): The list of tokens to parse.
        - idx (int): The current index in the token list.

    Returns:
        - int: The updated index after parsing the expression.

    Detailed Explanation:
        The expr function first parses a 'part' of the expression.
        It then enters a loop to handle any binary operators followed by another 'part'.
        For each operator found, it calls push_operator to handle operator precedence.
        After processing all operators, it ensures that any remaining operators are popped from the stack.

    Step-by-Step Implementation Guide:
        1. Call the part() function to parse the first part of the expression.
        2. Increment idx to move to the next token.
        3. While there are tokens left and the next token is an operator:
            a. Push the operator onto the operator stack using push_operator().
            b. Increment idx and parse the next part.
        4. After processing all parts, pop any remaining operators from the operator stack.
        5. Return the updated index.

    Raises:
        - IndexError: If the token list is exhausted unexpectedly.
    """
    # expr := part (binary part)*
    return 0 

def find_match(tokens: list[Token], t_idx: int) -> int:
    """
    Finds the index of the matching closing parenthesis for an opening parenthesis.

    Parameters:
        - tokens (list of Token): The list of tokens containing the parentheses.
        - t_idx (int): The index of the opening parenthesis '(' in the token list.

    Returns:
        - int: The index of the matching closing parenthesis ')'.

    Raises:
        - AssertionError: If the token at t_idx is not an opening parenthesis.
        - AssertionError: If parentheses are mismatched.

    Examples:
        >>> tokens = list(tokenizer('(2 + (3 * x)) + 5'))
        >>> idx = find_match(tokens, 0)
        >>> idx
        7  # Assuming the closing parenthesis is at index 7

    Detailed Explanation:
        The find_match function starts from the given index of the opening parenthesis and scans forward.
        It uses a counter 'cnt' to keep track of nested parentheses:
            - Increment 'cnt' when encountering an opening parenthesis.
            - Decrement 'cnt' when encountering a closing parenthesis.
        The function returns the index when 'cnt' returns to zero, indicating the matching parenthesis.

    Step-by-Step Implementation Guide:
        1. Verify that the token at t_idx is an opening parenthesis '('.
        2. Initialize a counter 'cnt' to zero.
        3. Iterate over the tokens starting from t_idx:
            a. If an opening parenthesis is found, increment 'cnt'.
            b. If a closing parenthesis is found, decrement 'cnt'.
            c. If 'cnt' becomes negative, raise an assertion error (unmatched closing parenthesis).
            d. If 'cnt' returns to zero, return the current index.
        4. If the end of tokens is reached without finding a match, return len(tokens) + 1.
    """
    tok_type, tok = tokens[t_idx]
    
    assert tokens[t_idx].is_para() and tok in Token.OPENING_PARANTHESES, f'Invalid token, should be one of {Token.OPENING_PARANTHESES}, now {tok}' 
    
    return 0 

def part(operator: Stack, operand: Stack, tokens: list[Token], idx: int) -> int:
    """
    Parses a part of an expression starting from the given index.

    The grammar rules for 'part' are:
        - part := num | var
        - part := "(" expr ")"
        - part := func "(" (expr ,)* expr ")"
        - part := unary_operator part

    Parameters:
        - operator (Stack): A stack to hold operators.
        - operand (Stack): A stack to hold operands (FormulaTree instances).
        - tokens (list of Token): The list of tokens to parse.
        - idx (int): The current index in the token list.

    Returns:
        - int: The updated index after parsing the part.

    Detailed Explanation:
        The part function handles different types of parts in an expression:
        1. If the next token is a number or variable, it creates a FormulaTree and pushes it onto the operand stack.
        2. If the next token is an opening parenthesis '(', it recursively parses the expression inside the parentheses.
        3. If the next token is a unary operator, it pushes the operator onto the operator stack and parses the following part.
        4. If the next token is a function, it parses the function arguments and creates a FormulaTree with the function and its arguments.

    Step-by-Step Implementation Guide:
        1. Get the next token at index idx.
        2. Check the token type and handle accordingly:
            a. For 'num' or 'var', create a FormulaTree and push onto operand stack.
            b. For '(', find the matching ')' and recursively parse the expression inside.
            c. For 'unary', push the operator and parse the next part.
            d. For 'func', parse the function arguments and create a FormulaTree.
        3. Return the updated index after parsing.

    Raises:
        - AssertionError: If an unexpected token is encountered.

    """
    
    # part := num | var 
    #      := "(" expr ")"
    #      := func "(" (expr ,)* expr ")"
    #      := unary part 
    return 0 

def pop_operator(operator: Stack, operand: Stack) -> None:
    """
    Pops an operator from the operator stack and applies it to operands from the operand stack.

    This function handles both binary and unary operators, constructing a FormulaTree for the operation.

    Parameters:
        - operator (Stack): The operator stack.
        - operand (Stack): The operand stack containing FormulaTree instances.

    Raises:
        - AssertionError: If the popped operator is not a valid operator.

    Detailed Explanation:
        The pop_operator function performs the following:
        1. Pops the top operator from the operator stack.
        2. If it's a binary operator:
            a. Pops two operands from the operand stack.
            b. Creates a new FormulaTree with the operator as root and the two operands as children.
            c. Pushes the new FormulaTree onto the operand stack.
        3. If it's a unary operator:
            a. Pops one operand from the operand stack.
            b. If the operand is a number, negates it directly.
            c. Otherwise, creates a multiplication FormulaTree with -1 and the operand.
            d. Pushes the new FormulaTree onto the operand stack.

    Step-by-Step Implementation Guide:
        1. Pop the top operator from the operator stack.
        2. Check if the operator is binary or unary.
        3. For binary operators:
            a. Pop two operands.
            b. Create a FormulaTree and push onto operand stack.
        4. For unary operators:
            a. Pop one operand.
            b. Handle negation as per the operand type.
            c. Push the result onto the operand stack.
        5. If the operator is invalid, raise an assertion error.

    """
    return 

def push_operator(operator: Stack, operand: Stack, op: Token) -> None:
    """
    Pushes an operator onto the operator stack, handling operator precedence.

    This function ensures that operators with higher precedence are applied before lower precedence operators.

    Parameters:
        - operator (Stack): The operator stack.
        - operand (Stack): The operand stack.
        - op (Token): The operator Token to be pushed.

    Detailed Explanation:
        The push_operator function compares the precedence of the incoming operator 'op' with the operator at the top of the operator stack.
        While the top operator has higher precedence, it pops operators from the operator stack and applies them using pop_operator().
        After handling higher precedence operators, it pushes the incoming operator onto the operator stack.

    Step-by-Step Implementation Guide:
        1. If the operator stack is not empty:
            a. Get the operator at the top of the stack.
            b. While the top operator has higher precedence than 'op':
                i. Pop operators and apply them using pop_operator().
                ii. Update the top operator.
        2. Push the incoming operator 'op' onto the operator stack.

    """
    return 
    
def parse(eq: str) -> FormulaTree:
    """
    Parses a mathematical equation string into an abstract syntax tree.

    This function combines tokenization, token compression, and recursive descent parsing to produce a FormulaTree.

    Parameters:
        - eq (str): The equation or expression string to parse.

    Returns:
        - FormulaTree: The root of the parsed abstract syntax tree.

    Examples:
        >>> tree = parse('2*x + 3')
        >>> tree.tok_type
        'op'
        >>> tree.tok_val
        '+'

    Detailed Explanation:
        The parse function performs the following steps:
        1. Tokenizes the input equation string using tokenizer().
        2. Compresses the tokens to identify functions using compress_tok().
        3. Parses the tokens into a FormulaTree using recursive_descent().

    Step-by-Step Implementation Guide:
        1. Call tokenizer(eq) to get an iterator of tokens.
        2. Pass the tokens to compress_tok() to adjust token types.
        3. Convert the compressed tokens into a list.
        4. Call recursive_descent() with the list of tokens to parse.
        5. Return the resulting FormulaTree.
    """
    return recursive_descent(compress_tok(tokenizer(eq))))

if __name__ == '__main__':

    # tests, tests, more tests! 

    # simple numbers
    eq1 = '(1)'
    eq2 = '3'
    eq3 = '-1'

    # +,- 
    eq4 = '1+1'
    eq5 = '1-1-2' # check
    eq6 = '-1-2-3-4-5'

    # +,-,*,/ 
    eq7 = '1+2/3+2'
    eq8 = '3*4+2'
    eq9 = '4/2'
    eq10 = '3+4*2'
    eq11 = '3+4/2'
    eq12 = '3/4/2'
    eq13 = '(3/4)/2' # check
    eq14 = '3/(4/2)'
    eq15 = '1+2/3'

    # +,-,*,/,^ with (,)
    eq16 = '(1+2)/3'
    eq17 = '(1*2)/3'
    eq18 = '(1+2)*3'
    eq19 = '3*(1+2)'
    eq20 = '3*(2-1)'
    eq21 = '3*(1-2)'
    eq22 = '3*(-2+1)'
    eq23 = '-3-2^3'
    eq24 = '-3-2^(3+2)'
    eq25 = '-2^3'
    eq26 = '-2^-3'

    # +,-,*,/ with nested (,)
    eq27 = '-1+(-1-2)'
    eq28 = '-(2+2)'
    eq29 = '3+(2^(-(2+2)))'
    eq30 = '3*(2*2+1)'
    eq31 = '2-3*(2*2+1)'
    eq32 = '2-3*(2*(2+1))'
    eq33 = '((3+2)*4-(2*4+2^(2-5)))*(2+(3+2)*5^2)'
    eq34 = '2+(3+2)*5^2'
    eq35 = '1+2^2*1'

    eq36 = 'x'
    eq37 = '-x_0*z+y'
    eq40 = '1+3^3*c'
    eq45 = 'a+b+C+d+e+f+g+h'
    eq46 = '1'
    eq47 = '0'

    eq48 = 'sin(x+y, z+1, cos(x+5))'
    eq49 = 'y = a*x + b'

    # for tok in compress_tok(tokenizer(eq48)):
    #     print(type(tok))

    # print(parse(eq48))

    # for i in range(100):
    for i in [37]:
        try:
            eq = eval('eq%d'%i)
        except NameError:
            continue
        print('=============')
        print(eq)
        for t, tok in compress_tok(tokenizer(eq)):
            print(t, tok)
        print(parse(eq))
        print(Formula(eq)(x_0 = 1, z = 5, y = 3))
        print('=============')
    # eq = Formula(eq37)
    # print(eq(1,2,3))
    # print(eq(x_0 = 1, y = 2, z = 3)) 
