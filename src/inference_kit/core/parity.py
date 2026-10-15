"""Small preregistered float tolerance; exact integers/booleans/status/identities."""
import math
from .io import require


def equal(actual,expected,path='$',abs_tol=1e-10,rel_tol=1e-9):
    if type(expected) is float:
        require(type(actual) is float and math.isfinite(expected) and math.isfinite(actual)
                and math.isclose(actual,expected,abs_tol=abs_tol,rel_tol=rel_tol),'Float parity mismatch '+path)
    elif isinstance(expected,dict):
        require(isinstance(actual,dict) and actual.keys()==expected.keys(),'Object identity mismatch '+path)
        for key in expected:equal(actual[key],expected[key],path+'.'+str(key),abs_tol,rel_tol)
    elif isinstance(expected,list):
        require(isinstance(actual,list) and len(actual)==len(expected),'Length mismatch '+path)
        for i,(a,b) in enumerate(zip(actual,expected)):equal(a,b,path+f'[{i}]',abs_tol,rel_tol)
    else:require(type(actual) is type(expected) and actual==expected,'Exact value/status mismatch '+path)
