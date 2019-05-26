import numpy as np

def get_projections(P: np.ndarray, N: np.ndarray, points_to_project: np.ndarray) -> np.ndarray:
    """
    Returns projections of given "points_to_project" to plane defined by point "P" and normal vector "N"
    
    >>> get_projections(np.array([0,0,0]), np.array([0,0,1]), np.array([np.array([1,1,1])]))
    array([[1., 1., 0.]])
    """
    # Obtain A,B,C,D params for plane equation.
    # See http://www.academiaxxi.ru/Collections/La_Ag/Electr_book/Ag/03/02/t.htm
    A,B,C = N
    D = -A*P[0] - B*P[1] - C*P[2]
    
    # Calculate projection, see vector formula at https://bit.ly/30GZbRe
    get_projection = lambda p: p - (p.dot(N) + D)/(N.dot(N))*N
    
    return np.array([get_projection(p) for p in points_to_project])