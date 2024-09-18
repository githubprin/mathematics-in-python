from formula import tokenizer, compress_tok, recursive_descent, Token, FormulaTree

# Phase 1: Without unary operators and function calls
test_cases_phase1 = [
    {
        'expression': "2 + 3",
        'expected_tokens': [
            Token('num', 2.0),
            Token('op', '+'),
            Token('num', 3.0)
        ],
        'description': "Simple addition"
    },
    {
        'expression': "2 + 3 * 4",
        'expected_tokens': [
            Token('num', 2.0),
            Token('op', '+'),
            Token('num', 3.0),
            Token('op', '*'),
            Token('num', 4.0)
        ],
        'description': "Mixed operators with precedence"
    },
    {
        'expression': "(2 + 3) * 4",
        'expected_tokens': [
            Token('para', '('),
            Token('num', 2.0),
            Token('op', '+'),
            Token('num', 3.0),
            Token('para', ')'),
            Token('op', '*'),
            Token('num', 4.0)
        ],
        'description': "Expressions with parentheses"
    },
    {
        'expression': "x * y + 5",
        'expected_tokens': [
            Token('var', 'x'),
            Token('op', '*'),
            Token('var', 'y'),
            Token('op', '+'),
            Token('num', 5.0)
        ],
        'description': "Variables in expressions"
    },
    {
        'expression': "10 - 2 + 3",
        'expected_tokens': [
            Token('num', 10.0),
            Token('op', '-'),
            Token('num', 2.0),
            Token('op', '+'),
            Token('num', 3.0)
        ],
        'description': "Multiple operators with same precedence"
    },
    {
        'expression': "(1 + (2 * (3 + 4)))",
        'expected_tokens': [
            Token('para', '('),
            Token('num', 1.0),
            Token('op', '+'),
            Token('para', '('),
            Token('num', 2.0),
            Token('op', '*'),
            Token('para', '('),
            Token('num', 3.0),
            Token('op', '+'),
            Token('num', 4.0),
            Token('para', ')'),
            Token('para', ')'),
            Token('para', ')')
        ],
        'description': "Nested parentheses"
    },
    {
        'expression': "8 / 4 * 2",
        'expected_tokens': [
            Token('num', 8.0),
            Token('op', '/'),
            Token('num', 4.0),
            Token('op', '*'),
            Token('num', 2.0)
        ],
        'description': "Division and multiplication"
    },
    {
        'expression': "2 ^ 3 ^ 2",
        'expected_tokens': [
            Token('num', 2.0),
            Token('op', '^'),
            Token('num', 3.0),
            Token('op', '^'),
            Token('num', 2.0)
        ],
        'description': "Exponentiation"
    }
]

# Phase 2: With all features (including unary operators and function calls)
test_cases_phase2 = [
    {
        'expression': "-5 + 3",
        'expected_tokens': [
            Token('unary', 'unary -'),
            Token('num', 5.0),
            Token('op', '+'),
            Token('num', 3.0)
        ],
        'description': "Unary negation"
    },
    {
        'expression': "sin(0)",
        'expected_tokens': [
            Token('func', 'sin'),
            Token('para', '('),
            Token('num', 0.0),
            Token('para', ')')
        ],
        'description': "Function call with single argument"
    },
    {
        'expression': "-x * y",
        'expected_tokens': [
            Token('unary', 'unary -'),
            Token('var', 'x'),
            Token('op', '*'),
            Token('var', 'y')
        ],
        'description': "Unary negation with variables"
    },
    {
        'expression': "cos(2 * x)",
        'expected_tokens': [
            Token('func', 'cos'),
            Token('para', '('),
            Token('num', 2.0),
            Token('op', '*'),
            Token('var', 'x'),
            Token('para', ')')
        ],
        'description': "Function call with expression argument"
    },
    {
        'expression': "max(a, b, c)",
        'expected_tokens': [
            Token('func', 'max'),
            Token('para', '('),
            Token('var', 'a'),
            Token('comma', ','),
            Token('var', 'b'),
            Token('comma', ','),
            Token('var', 'c'),
            Token('para', ')')
        ],
        'description': "Function call with multiple arguments"
    },
    {
        'expression': "sin(-x) + cos(y)",
        'expected_tokens': [
            Token('func', 'sin'),
            Token('para', '('),
            Token('unary', 'unary -'),
            Token('var', 'x'),
            Token('para', ')'),
            Token('op', '+'),
            Token('func', 'cos'),
            Token('para', '('),
            Token('var', 'y'),
            Token('para', ')')
        ],
        'description': "Nested functions and unary operators"
    },
    {
        'expression': "-3 + 4 * sin(theta) / cos(phi) - log(1 + x)",
        'expected_tokens': [
            Token('unary', 'unary -'),
            Token('num', 3.0),
            Token('op', '+'),
            Token('num', 4.0),
            Token('op', '*'),
            Token('func', 'sin'),
            Token('para', '('),
            Token('var', 'theta'),
            Token('para', ')'),
            Token('op', '/'),
            Token('func', 'cos'),
            Token('para', '('),
            Token('var', 'phi'),
            Token('para', ')'),
            Token('op', '-'),
            Token('func', 'ln'),  # Assuming 'log' maps to 'ln'
            Token('para', '('),
            Token('num', 1.0),
            Token('op', '+'),
            Token('var', 'x'),
            Token('para', ')')
        ],
        'description': "Complex expression with all features"
    },
    {
        'expression': "2 ^ -3",
        'expected_tokens': [
            Token('num', 2.0),
            Token('op', '^'),
            Token('unary', 'unary -'),
            Token('num', 3.0)
        ],
        'description': "Exponentiation with unary operator"
    },
    {
        'expression': "pow(2, x + y)",
        'expected_tokens': [
            Token('func', 'pow'),
            Token('para', '('),
            Token('num', 2.0),
            Token('comma', ','),
            Token('var', 'x'),
            Token('op', '+'),
            Token('var', 'y'),
            Token('para', ')')
        ],
        'description': "Function call with multiple expression arguments"
    },
    {
        'expression': "- (a + b) * c",
        'expected_tokens': [
            Token('unary', 'unary -'),
            Token('para', '('),
            Token('var', 'a'),
            Token('op', '+'),
            Token('var', 'b'),
            Token('para', ')'),
            Token('op', '*'),
            Token('var', 'c')
        ],
        'description': "Unary operator preceding parentheses"
    }
]

# Function to test the tokenizer
def test_tokenizer(test_cases):
    for idx, test_case in enumerate(test_cases, 1):
        expression = test_case['expression']
        expected_tokens = test_case['expected_tokens']
        description = test_case.get('description', '')
        print(f"Test Case {idx}: {description}")
        print(f"Expression: {expression}")
        # Tokenize the expression
        tokens = list(compress_tok(tokenizer(expression)))
        # Check if the tokens match the expected tokens
        if tokens == expected_tokens:
            print("Tokenizer Test Passed.")
        else:
            print("Tokenizer Test Failed.")
            print("Expected Tokens:")
            for token in expected_tokens:
                print(f"  {token}")
            print("Actual Tokens:")
            for token in tokens:
                print(f"  {token}")
        print("-" * 40)

# Function to test the parser
def test_parser(test_cases):
    for idx, test_case in enumerate(test_cases, 1):
        expression = test_case['expression']
        description = test_case.get('description', '')
        print(f"Test Case {idx}: {description}")
        print(f"Expression: {expression}")
        # Parse the expression
        try:
            tree = parse(expression)
            print("Parser Test Passed.")
            # Optionally, you can print the parse tree
            print("Parse Tree:")
            print(tree)
        except Exception as e:
            print("Parser Test Failed.")
            print(f"Error: {e}")
        print("-" * 40)

# Run the tests for Phase 1
print("Testing Phase 1: Without Unary Operators and Function Calls")
test_tokenizer(test_cases_phase1)
test_parser(test_cases_phase1)

# Run the tests for Phase 2
print("\nTesting Phase 2: With All Features")
test_tokenizer(test_cases_phase2)
test_parser(test_cases_phase2)
