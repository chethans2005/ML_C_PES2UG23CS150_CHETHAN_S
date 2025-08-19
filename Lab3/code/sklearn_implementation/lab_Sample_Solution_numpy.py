import numpy as np
from collections import Counter

def get_entropy_of_dataset(data: np.ndarray) -> float:
    """
    Calculate the entropy of the entire dataset based on the target variable (last column).
    
    Formula:
        Entropy = -Σ(p_i * log2(p_i)) 
        where p_i is the probability of class i.
    
    Args:
        data (np.ndarray): Dataset where the last column is the target variable.
    
    Returns:
        float: Entropy value of the dataset.
    """
    if data.shape[0] == 0:
        return 0.0

    # Extract the target column (last column)
    target_column = data[:, -1]

    # Get unique classes and their counts
    unique_classes, counts = np.unique(target_column, return_counts=True)

    # Calculate probabilities of each class
    total_samples = data.shape[0]
    probabilities = counts / total_samples

    # Compute entropy
    entropy = -np.sum([prob * np.log2(prob) for prob in probabilities if prob > 0])

    return entropy


def get_avg_info_of_attribute(data: np.ndarray, attribute: int) -> float:
    """
    Calculate the average information (weighted entropy) of a specific attribute.
    
    Formula:
        Avg_Info = Σ((|S_v|/|S|) * Entropy(S_v)) 
        where S_v is the subset of data with attribute value v.
    
    Args:
        data (np.ndarray): Dataset where the last column is the target variable.
        attribute (int): Index of the attribute column to calculate average information for.
    
    Returns:
        float: Weighted average entropy for the attribute.
    """
    if data.shape[0] == 0 or attribute < 0 or attribute >= data.shape[1] - 1:
        return 0.0

    # Extract the attribute column
    attribute_column = data[:, attribute]
    total_samples = data.shape[0]

    # Get unique values of the attribute
    unique_values = np.unique(attribute_column)

    # Calculate weighted entropy for each unique value
    avg_info = 0.0
    for value in unique_values:
        # Create a subset of data where the attribute equals the current value
        subset = data[attribute_column == value]

        # Calculate the weight and entropy of the subset
        weight = subset.shape[0] / total_samples
        subset_entropy = get_entropy_of_dataset(subset)

        # Add weighted entropy to the total
        avg_info += weight * subset_entropy

    return avg_info


def get_information_gain(data: np.ndarray, attribute: int) -> float:
    """
    Calculate the Information Gain for a specific attribute.
    
    Formula:
        Information_Gain = Entropy(S) - Avg_Info(attribute)
    
    Args:
        data (np.ndarray): Dataset where the last column is the target variable.
        attribute (int): Index of the attribute column to calculate information gain for.
    
    Returns:
        float: Information gain for the attribute, rounded to 4 decimal places.
    """
    if data.shape[0] == 0:
        return 0.0

    # Calculate dataset entropy and average information for the attribute
    dataset_entropy = get_entropy_of_dataset(data)
    avg_info = get_avg_info_of_attribute(data, attribute)

    # Compute information gain
    information_gain = dataset_entropy - avg_info

    return round(information_gain, 4)


def get_selected_attribute(data: np.ndarray) -> tuple:
    """
    Select the best attribute based on the highest information gain.
    
    Args:
        data (np.ndarray): Dataset where the last column is the target variable.
    
    Returns:
        tuple: A tuple containing:
            - dict: Dictionary mapping attribute indices to their information gains.
            - int: Index of the attribute with the highest information gain.
    
    Example:
        ({0: 0.123, 1: 0.768, 2: 1.23}, 2)
    """
    if data.shape[0] == 0 or data.shape[1] <= 1:
        return ({}, -1)

    # Calculate information gain for all attributes (except the target variable)
    num_attributes = data.shape[1] - 1
    gain_dictionary = {i: get_information_gain(data, i) for i in range(num_attributes)}

    # Find the attribute with the highest information gain
    selected_attribute_index = max(gain_dictionary, key=gain_dictionary.get)

    return gain_dictionary, selected_attribute_index