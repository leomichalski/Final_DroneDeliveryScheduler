import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def item_from_color(color, col_size=32, row_size=32):
    """Generates a matrix of size (col_size, row_size, len(color)) if color is 1D or a matrix of size (col_size, row_size) if color is a digit.
    Parameters
    ----------
    color : int or list
        RGB, gray or binary color i.e [255,255,255], 255 or 1. Maybe even RGBA or whatever.
    col_size : int
        Color size.
    row_size : int
        Size of a row.
    Returns
    -------
    numpy.ndarray
        2D matrix representing an image.
    """
    return np.repeat(
        [np.repeat([color], col_size, axis=0)],
        row_size,
        axis=0
    )


def adj_matrix_to_image(adj_matrix,
                        value_to_color={1: [80, 30, 50],
                                        0: [168, 50, 125],
                                        -1: [0, 0, 255]},
                        num_channels=3,
                        num_cols=7, num_rows=7,
                        col_size=32, row_size=32,
                        img_dtype=np.uint8):
    """Convert a node list to an image. Do that by coloring the image items at the coordinates in the node_list.
    Parameters
    ----------
    adj_matrix : numpy.ndarray
        Adjacent matrix where the meaning of each value is specified in the label_utils.py module.
    value_to_color : dict
        Dict to map each value in the adj_matrix to a differenct color.
    num_cols : int
        Number of columns in each adj_matrix (or height_map) item.
    num_rows : int
        Number of rows in each adj_matrix (or height_map) item.
    col_size : int
        Width of each adj_matrix (or height_map) item.
    row_size : int
        Height of each adj_matrix (or height_map) item.
    img_dtype : int
        Data type of the resulting image.
    Returns
    -------
    numpy.ndarray
        2D matrix representing an image.
    """
    img = np.zeros(
        (num_cols*col_size, num_rows*row_size, num_channels),
        dtype=img_dtype
    )
    value_to_item = {
        1: item_from_color(value_to_color[1]),
        0: item_from_color(value_to_color[0]),
        -1: item_from_color(value_to_color[-1]),
    }
    for i, col in enumerate(adj_matrix):
        x1 = col_size*i
        x2 = col_size*(i+1)
        for j, value in enumerate(col):
            y1 = row_size*j
            y2 = row_size*(j+1)
            img[x1:x2, y1:y2, :] = value_to_item[value].copy()
    return img


def node_list_to_image(node_list,
                       item_color=[255, 0, 0],
                       num_channels=3,
                       num_cols=7, num_rows=7,
                       col_size=32, row_size=32,
                       img_dtype=np.uint8):
    """Convert a node list to an image. Do that by coloring the image items at the coordinates in the node_list.
    Parameters
    ----------
    node_list : list
        List of node coordinates i.e [(0,0), (0,1), (0,2), (1,2)].
    item_color : list
        Color of each node/item.
    num_cols : int
        Number of columns in each adj_matrix (or height_map) item.
    num_rows : int
        Number of rows in each adj_matrix (or height_map) item.
    col_size : int
        Width of each adj_matrix (or height_map) item.
    row_size : int
        Height of each adj_matrix (or height_map) item.
    img_dtype : int
        Data type of the resulting image.
    Returns
    -------
    numpy.ndarray
        2D matrix representing an image.
    """
    img = np.zeros(
        (num_cols*col_size, num_rows*row_size, num_channels),
        dtype=img_dtype
    )
    item = item_from_color(item_color)
    for node in node_list:
        x1 = col_size*node[0]
        x2 = col_size*(node[0]+1)
        y1 = row_size*node[1]
        y2 = row_size*(node[1]+1)
        img[x1:x2, y1:y2, :] = item.copy()
    return img


def plot_frame(frame, width=224, height=224, images_to_overlay=[]):
    """Method to plot the frame.
    Parameters
    ----------
    frame : numpy.ndarray
        2D matrix representing an image.
    width : int
        Frame width.
    height : int
        Frame height.
    images_to_overlay : list
        List of images to overlay on top of the frame.
    """
    # overlay images on the frame
    if images_to_overlay:
        img_to_overlay = images_to_overlay[0]
    for img in images_to_overlay[1:]:
        img_to_overlay += img
    if images_to_overlay:
        frame = cv2.addWeighted(img_to_overlay, 1.0, frame, 1.0, 0.0)
    # resize to desired size with bilinear interpolation
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
    dpi = matplotlib.rcParams['figure.dpi']
    figsize = width / float(dpi), height / float(dpi)
    plt.figure(figsize=figsize)
    # convert from BGR to RGB for plotting purposes
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # plot the resulting frame
    plt.imshow(frame)
    plt.show()

