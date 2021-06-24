import numpy as np

class BeeConfig(object):

    #name of dataset
    NAME = 'training'

    PATH='dataset/synthetic_data/'
    #Path to individual Bees
    CLASSES_PATH = 'dataset/raw/Bee_Cutout/'

    #Background
    BACKGROUNDS_PATH = 'dataset/raw/flower_bushes_data/'



    # Folder names
    IMAGE_DIR = 'images'

    ANNOT_DIR = 'annots'

    # Configurations for synthetic images
    MIN_NUM_OBJECT_PER_IMAGE = 5

    MAX_NUM_OBJECT_PER_IMAGE = 15

    RANGE_NUM_EQUAL_OBJECTS = (3, 6)

    MUTATE = True

    OBJECT_SIZE = np.arange(0.8, 1.2, 0.6)

    # Label_map
    CREATE_LABEL_MAP = False

    LABEL_MAP_NAME = 'label_map.json'

    LABEL_MAP_PATH = 'dataset/synthetic_data/'

    NUM_CLASSES = None

    def __init__(self, name=None, num_images=None, label_map=None, root=None):
        if name is not None:
            self.NAME = name


        if num_images is not None:
            self.NUM_IMAGES = num_images

        if label_map:
            self.CREATE_LABEL_MAP = True
            if isinstance(label_map, str):
                self.LABEL_MAP_NAME = label_map + '.json'

        self.LABEL_MAP_PATH = self.PATH + self.LABEL_MAP_NAME


        if isinstance(root, str):
            if root[-1] != '/':
                root += '/'
            print('root is ' + str(root))

            self.PATH = root + self.PATH
            self.CLASSES_PATH = root + self.CLASSES_PATH
            self.BACKGROUNDS_PATH = root + self.BACKGROUNDS_PATH
            self.LABEL_MAP_PATH = root + self.LABEL_MAP_PATH

    def set_label_map(self, name):
        self.CREATE_LABEL_MAP = True

        if not name.endswith('.json'):
            name = name + '.json'
        self.LABEL_MAP_NAME = name

        self.LABEL_MAP_PATH = self.PATH + self.LABEL_MAP_NAME

