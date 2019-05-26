import numpy as np

def get_projections(P: np.ndarray, N: np.ndarray, points_to_project: np.ndarray) -> np.ndarray:
    """
    Returns projections of given "points_to_project" to plane defined by point "P" and normal vector "N"
    
    See http://www.academiaxxi.ru/Collections/La_Ag/Electr_book/Ag/03/02/t.htm
        and http://cyclowiki.org/wiki/%D0%9F%D1%80%D0%BE%D0%B5%D0%BA%D1%86%D0%B8%D1%8F_%D1%82%D0%BE%D1%87%D0%BA%D0%B8_%D0%BD%D0%B0_%D0%BF%D0%BB%D0%BE%D1%81%D0%BA%D0%BE%D1%81%D1%82%D1%8C
        
    >>> get_projections(np.array([0,0,0]), np.array([0,0,1]), np.array([np.array([1,1,1])]))
    array([[1., 1., 0.]])
    """
    A,B,C = N
    D = -A*P[0] - B*P[1] - C*P[2]
    get_projection = lambda p: p - (p.dot(N) + D)/(N.dot(N))*N
    return np.array([get_projection(p) for p in points_to_project])