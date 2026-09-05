'''
input: formula strings
output: the answer of the mathematical formula
'''
from sympy.parsing.latex import parse_latex
from sympy import sympify 

# import wolframalpha
query = '1+2*3'


def WolframAlphaCalculator(query: str):
    allowed_chars = set('0123456789+-*/()maxmin., ')
    if any(char not in allowed_chars for char in query.lower()):
        return f"Error: Invalid characters in query, cann't calculate."
    try:
        return sympify(query).evalf()
    except Exception as e:
        return f"Error: {e}"
    # return parse_latex(query)
    # operators = {
    #     '+': add,
    #     '-': sub,
    #     '*': mul,
    #     '/': truediv,
    # }
    # query = re.sub(r'\s+', '', query)
    # if query.isdigit():
    #     return float(query)
    # for c in operators.keys():
    #     left, operator, right = query.partition(c)
    #     if operator in operators:
    #         return round(operators[operator](calculator(left), calculator(right)),2)

# def WolframAlphaCalculator(input_query: str):
#     wolfram_alpha_appid = "YOUR_WOLFRAMALPHA_APPID"
#     wolfram_client = wolframalpha.Client(wolfram_alpha_appid)
#     res = wolfram_client.query(input_query)
#     assumption = next(res.pods).text
#     answer = next(res.results).text
#     # return f"Assumption: {assumption} \nAnswer: {answer}"
#     return answer

if __name__ == "__main__":
    print(calculator("Max(1, 2+2, 3)"))