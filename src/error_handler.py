def check_list_of_lists(*args):
    """
    Check if all the args provided are valid lists of lists
    """
    for arg in args:
        if not any(isinstance(i, list) for i in arg):
            raise ValueError("List of lists not provided")
        if not all(isinstance(i, list) for i in arg):
            raise ValueError("List of lists contains non-list element")
