import numpy as np
import cv2
import os
from delivery import label_utils


class AerialImageData(object):

    def __init__(self, map=None, adj_matrix=None, weight_map=None,
                 coord_list=None, helipad_coord=None):
        self.map = map
        self.adj_matrix = adj_matrix
        self.weight_map = weight_map
        self.coord_list = coord_list
        self.helipad_coord = helipad_coord


# UTILITIES TO GENERATE RANDOM DATA
MOCK_DATA_PATH = os.path.join('mock_data', 'images')
ZONE_OPTIONS_IMAGES = [
    os.path.join(MOCK_DATA_PATH, 'high_speed.jpg'),
    os.path.join(MOCK_DATA_PATH, 'low_speed.jpg'),
    os.path.join(MOCK_DATA_PATH, 'low_risk.jpg'),
    os.path.join(MOCK_DATA_PATH, 'no_fly.jpg'),
]
GOAL_OPTIONS_IMAGES = [
    os.path.join(MOCK_DATA_PATH, 'goal.png'),
]
HELIPAD_OPTIONS_IMAGES = [
    os.path.join(MOCK_DATA_PATH, 'helipad.png'),
]

ZONE_OPTIONS_LABELS = [
    label_utils.HIGH_SPEED,
    label_utils.LOW_SPEED,
    label_utils.LOW_RISK,
    label_utils.NO_FLY,
]
GOAL_OPTIONS_LABELS = [
    label_utils.LOW_SPEED,
]
HELIPAD_OPTIONS_LABELS = [
    label_utils.LOW_SPEED,
]

ZONE_OPTIONS_WEIGHTS = [
    0,
    -1,
    -2,
    -99999,
]
GOAL_OPTIONS_WEIGHTS = [
    -1,
]
HELIPAD_OPTIONS_WEIGHTS = [
    -1,
]


class RandomAerialImageDataGenerator(object):

    def __init__(self, width=224, height=224, channels=3, dtype=np.uint8,
                 col_size=32, row_size=32,
                 zone_options_images=ZONE_OPTIONS_IMAGES,
                 zone_options_labels=ZONE_OPTIONS_LABELS,
                 zone_options_weights=ZONE_OPTIONS_WEIGHTS,
                 goal_options_images=GOAL_OPTIONS_IMAGES,
                 goal_options_labels=GOAL_OPTIONS_LABELS,
                 goal_options_weights=GOAL_OPTIONS_WEIGHTS,
                 helipad_options_images=HELIPAD_OPTIONS_IMAGES,
                 helipad_options_labels=HELIPAD_OPTIONS_LABELS,
                 helipad_options_weights=HELIPAD_OPTIONS_WEIGHTS):
        self.width = width
        self.height = height
        self.channels = channels
        self.dtype = dtype
        self.col_size = col_size
        self.row_size = row_size
        self.zone_options_images = zone_options_images
        self.zone_options_labels = zone_options_labels
        self.zone_options_weights = zone_options_weights
        self.goal_options_images = goal_options_images
        self.goal_options_labels = goal_options_labels
        self.goal_options_weights = goal_options_weights
        self.helipad_options_images = helipad_options_images
        self.helipad_options_labels = helipad_options_labels
        self.helipad_options_weights = helipad_options_weights

        self.num_cols = self.width // self.col_size
        self.num_rows = self.height // self.row_size

        self.zone_options_images = self.load_img_items(
            self.zone_options_images
        )
        self.goal_options_images = self.load_img_items(
            self.goal_options_images
        )
        self.helipad_options_images = self.load_img_items(
            self.helipad_options_images
        )

    def __read_img_item(self, filename, **kwargs):
        """Read an image item from filename.
        Parameters
        ----------
        filename : str
            File path.
        **kwargs : dict
            **kwargs.
        Returns
        -------
        numpy.ndarray
            2D matrix representing an image.
        """
        return cv2.imread(
            filename,
            **kwargs
        )

    def __resize_img_item(self, img_item,
                          resize_method=cv2.INTER_LINEAR, **kwargs):
        """Resize an image item.
        Parameters
        ----------
        img_item : numpy.ndarray
            2D matrix representing an image.
        resize_method : int
            Type of the resize method i.e cv2.INTER_LINEAR.
        **kwargs : dict
            **kwargs
        Returns
        -------
        numpy.ndarray
            Resized image.
        """
        return cv2.resize(
            src=img_item,
            dsize=(self.col_size, self.row_size),
            interpolation=resize_method,
            **kwargs
        )

    def load_img_items(self, img_items):
        """Read and transform a list of image items.
        Parameters
        ----------
        img_items : list
            List of paths, or list of numpy.ndarrays.
        Returns
        -------
        list
            List of the transformed images.
        """
        for i, img in enumerate(img_items):
            # read images
            if type(img) == str:
                img = self.__read_img_item(img)
            # transform images
            img = self.__resize_img_item(img)
            img_items[i] = img
        return img_items

    def place_goals_on_frame(self, frame, adj_matrix, weight_map,
                             goals_quantity):
        assert goals_quantity <= (self.num_cols * self.num_rows)/2
        unique_random_coords = []
        while len(unique_random_coords) < goals_quantity:
            coord = [
                np.random.choice(self.num_cols),
                np.random.choice(self.num_rows)
            ]
            if coord not in unique_random_coords:
                unique_random_coords.append(coord)
        for (i, j) in unique_random_coords:
            x1 = self.col_size*i
            x2 = self.col_size*(i+1)
            y1 = self.row_size*j
            y2 = self.row_size*(j+1)
            chosen_item_idx = np.random.choice(len(self.goal_options_images))
            weight_map[i][j] = self.goal_options_weights[chosen_item_idx]
            adj_matrix[i][j] = self.goal_options_labels[chosen_item_idx]
            frame[x1:x2, y1:y2, :] = self.goal_options_images[
                chosen_item_idx
            ]
        return frame, adj_matrix, weight_map, unique_random_coords

    def place_helipad_on_frame(self, frame, adj_matrix, weight_map):
        i = np.random.choice(self.num_cols)
        x1 = self.col_size*i
        x2 = self.col_size*(i+1)
        j = np.random.choice(self.num_rows)
        y1 = self.row_size*j
        y2 = self.row_size*(j+1)
        chosen_item_idx = np.random.choice(len(self.helipad_options_images))
        weight_map[i][j] = self.helipad_options_weights[chosen_item_idx]
        adj_matrix[i][j] = self.helipad_options_labels[chosen_item_idx]
        frame[x1:x2, y1:y2, :] = self.helipad_options_images[
            chosen_item_idx
        ]
        return frame, adj_matrix, weight_map, [i, j]

    def generate(self, goals_quantity, place_helipad=True):
        data = AerialImageData(
            map=np.empty(
                (
                    self.num_cols*self.col_size,
                    self.num_rows*self.row_size,
                    self.channels,
                ),
                dtype=self.dtype,
            ),
            adj_matrix=np.empty(
                (self.num_cols, self.num_rows),
                dtype=np.float32,
            ),
            weight_map=np.empty(
                (self.num_cols, self.num_rows),
                dtype=np.float32,
            ),

        )
        # populate with the zone options
        for i in range(self.num_cols):
            x1 = self.col_size*i
            x2 = self.col_size*(i+1)
            for j in range(self.num_rows):
                y1 = self.row_size*j
                y2 = self.row_size*(j+1)
                chosen_item_idx = np.random.choice(
                    len(self.zone_options_images)
                )
                data.weight_map[i][j] = self.zone_options_weights[
                    chosen_item_idx
                ]
                data.adj_matrix[i][j] = \
                    self.zone_options_labels[chosen_item_idx]
                data.map[x1:x2, y1:y2, :] = \
                    self.zone_options_images[chosen_item_idx]
        # choose a single zone item randomly and subtitute it with a goal
        data.map, data.adj_matrix, data.weight_map, data.goal_coord_list =\
            self.place_goals_on_frame(
                frame=data.map,
                adj_matrix=data.adj_matrix,
                weight_map=data.weight_map,
                goals_quantity=goals_quantity
            )
        if place_helipad:
            data.map, data.adj_matrix, data.weight_map, data.helipad_coord =\
                self.place_helipad_on_frame(
                    frame=data.map,
                    adj_matrix=data.adj_matrix,
                    weight_map=data.weight_map,
                )
        return data
