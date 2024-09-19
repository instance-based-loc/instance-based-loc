import numpy as np

def generate_onehot_array(arr):
    if not isinstance(arr, np.ndarray) or arr.ndim != 1:
        raise ValueError("Input must be a 1-dimensional numpy array of integers.")
    
    # Determine the size of the 2D array (max index + 1)
    max_index = len(arr)
    num_rows = np.sum(arr)
    
    # Initialize the 2D array with zeros
    result = np.zeros((num_rows, max_index), dtype=float)
    
    row_start = 0
    for index, count in enumerate(arr):
        if count > 0:
            # Create count rows of one-hot encoding with 1 in position `index`
            result[row_start:row_start + count, index] = 1
            row_start += count
    
    return result

# Example usage
arr = np.array([2, 5, 3, 2, 2])
result = generate_onehot_array(arr)
print(result)