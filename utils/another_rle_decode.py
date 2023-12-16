def decode_rle(rle_codes, image_size):
    """
    Decode RLE encoded list to binary mask.

    :param rle_codes: The RLE codes.
    :param image_size: A tuple of (height, width).
    :return: A 2D array representing the decoded binary mask.
    """
    # Create an empty mask with all zeros
    mask = [[0 for _ in range(image_size[1])] for _ in range(image_size[0])]
    
    # Loop through the RLE codes and fill in the ones
    for code in rle_codes:
        row, col = code
        if row < 0 or row >= image_size[0] or col < 0 or col >= image_size[1]:
            raise ValueError(f"Index ({row}, {col}) out of bounds for image of size {image_size}")
        mask[row][col] = 1
    
    return mask

# Example usage:
rle_codes = [
    [169,228],[168,228],[167,228],[166,228],[165,228],[164,228],[163,228],
    [163,227],[162,227],[161,227], # ...
    # Add the rest of the RLE codes here
]

image_size = (300, 300)  

mask = decode_rle(rle_codes, image_size)

