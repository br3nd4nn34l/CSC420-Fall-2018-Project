def chunks(lst, n):
    """
    Yield chunks of size n from lst
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def tf_model_input_sizes(tf_model):
    """
    Returns a list of tuples of of integers of a
    TensorFlow model's input sizes
    """

    # Tensorflow has a special "dimension" datatype (why???)
    raw_sizes = (
        inp.shape[1:]
        for inp in tf_model.inputs
    )

    # Convert to integers
    return [
        tuple([int(x) for x in size])
        for size in raw_sizes
    ]