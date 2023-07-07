from typing import List

from sklearn import metrics


def calculate_trapezoidal_rule_area(x: List[float], y: List[float]) -> float:
    """Calculates the area under the curve using the trapezoidal rule

    Args:
        x (list or array): Array of x values
        y (list or array): Array of y values corresponding to each x value

    Returns:
        float: The estimated area under the curve
    
    Raises:
        ValueError: If the lengths of x and y are different
    """
    if len(x) != len(y):
        raise ValueError("x and y must be the same length")
    
    areas_sum = 0.0
    
    # Area of a trapezoid is 1/2 x 
    # (sum of the lengths of the parallel sides) x 
    # perpendicular distance between parallel sides
    for i in range(1, len(x)):
        y_sum = y[i] + y[i-1]
        width = x[i] - x[i-1]
        areas_sum += 0.5 * y_sum * width
    
    return round(areas_sum, 2)


if __name__ == "__main__":
    x = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5]
    y = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.8, 0.6, 0.4, 0.2]
    
    auc = calculate_trapezoidal_rule_area(x, y)
    print(f"My implementation: {auc}\n")

    sklearn_auc = metrics.auc(x, y)
    print(f"scikit-learn implementation: {sklearn_auc}")
