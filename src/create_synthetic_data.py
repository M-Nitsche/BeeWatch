from src.config_dataset import BeeConfig
import os
import shutil
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from src.pascalVOC import *
from src.create_label_map import *
import random
import itertools



# Class that inherits from the Class Annotated_Object
class Annotated_Unknown_Object(Annotated_Object):

    def __init__(self, name, pose, truncated, difficult, bndBox):
        Annotated_Object.__init__(self, name, pose, truncated, difficult, bndBox)
        self.name = name
        self.pose = pose
        self.truncated = truncated
        self.difficult = difficult
        self.bndBox = bndBox

    def toPascalVOCFormat(self):
        output = "<unknown_object>"
        output += "\n\t<name>" + 'Bee' + "</name>" #replace Bee with str(self.name)
        output += "\n\t<pose>" + str(self.pose) + "</pose>"
        output += "\n\t<truncated>" + str(self.truncated) + "</truncated>"
        output = appendPascalVOC(output, self.bndBox.toPascalVOCFormat())
        output += "\n</unknown_object>"
        return output


# Helper functions
def get_backgrounds(config):
    base_bkgs_path = config.BACKGROUNDS_PATH
    bkg_images = [f for f in os.listdir(base_bkgs_path)
                  if not f.startswith(".") and f.endswith('.jpg')]
    return bkg_images


def get_bees(config=None):
    objs_path = config.CLASSES_PATH

    obj_images = [f for f in os.listdir(objs_path)
                  if not f.startswith(".") and f.endswith('.png')]
    return obj_images




def draw_random_subset_of_bees(list, size):
    return random.sample(list, min(size, len(list)))


def choose_flower_bushes_randomly(list):
    return random.choice(list)


def get_obj_positions(obj, bkg, size, count=1, margin=0):
    obj_w, obj_h = [], []
    x_positions, y_positions = [], []
    bkg_w, bkg_h = bkg.size
    # Rescale our obj to have a couple different sizes
    obj_size = [tuple([int(size * x) for x in obj.size])]
    for w, h in obj_size:
        obj_w.extend([w] * count)
        obj_h.extend([h] * count)
        max_x, max_y = bkg_w - w - margin, bkg_h - h  - margin
        x_positions.extend(list(np.random.randint(margin, max_x, count)))
        y_positions.extend(list(np.random.randint(margin, max_y, count)))
    return obj_h, obj_w, x_positions, y_positions


def get_box(obj_w, obj_h, max_x, max_y):
    x1, y1 = np.random.randint(0, max_x, 1), np.random.randint(0, max_y, 1)
    x2, y2 = x1 + obj_w, y1 + obj_h
    return [x1[0], y1[0], x2[0], y2[0]]


# check if two boxes intersect
def intersects(box, new_box):
    box_x1, box_y1, box_x2, box_y2 = box
    x1, y1, x2, y2 = new_box
    return not (box_x2 < x1 or box_x1 > x2 or box_y1 > y2 or box_y2 < y1)



def resize_image(img, mutate_size, rotate=True):
    # resize image
    # resize_rate = random.choice(sizes)
    img = img.resize([int(img.width * mutate_size), int(img.height * mutate_size)], Image.BILINEAR)


    random_num=random.randint(0, 10)
    if rotate:
        if random_num < 8 :                    #isinstance(rotate, list)
            rotate_angle = random.choice(rotate)
        else:
            rotate_angle = random.randint(0, 360)
    else:
        rotate_angle = 0

    # rotate image for random andle and generate exclusion mask
    img = img.rotate(rotate_angle, expand=True)
    mask = Image.new('L', img.size, 255)
    mask = mask.rotate(rotate_angle, expand=True)

    return img, mask


def mutate_image(img):
    # perform some enhancements on image
    enhancers = [ImageEnhance.Brightness, ImageEnhance.Color, ImageEnhance.Contrast, ImageEnhance.Sharpness]
    enhancers_count = random.randint(0, 3)
    for i in range(0, enhancers_count):
        enhancer = random.choice(enhancers)
        enhancers.remove(enhancer)
        img = enhancer(img).enhance(random.uniform(0.5, 1.5))

    return img

def blur_image(img):
    # perform filtering on image
    radius=random.randint(0, 1)
    filters= [ImageFilter.GaussianBlur(radius),ImageFilter.BoxBlur(radius),ImageFilter.BLUR]
    random_num= random_num=random.randint(0, 10)
    if random_num > 8:
        chosen_filter= random.choice(filters)
        img=img.filter(chosen_filter)

    return img

def filter_image(img):
    # perform filtering on image
    filters= [ImageFilter.MinFilter, ImageFilter.DETAIL, ImageFilter.EDGE_ENHANCE, ImageFilter.EDGE_ENHANCE_MORE,
              ImageFilter.SMOOTH, ImageFilter.SMOOTH_MORE, ImageFilter.SHARPEN]
    random_num= random_num=random.randint(0, 10)
    if random_num > 8:
        chosen_filter= random.choice(filters)
        img=img.filter(chosen_filter)

    return img


def reset_dir(path):
    try:
        shutil.rmtree(path)

    except OSError:
        print ("Deletion of the directory %s failed" % path)
    else:
        print ("Successfully deleted the directory %s" % path)

    try:
        os.mkdir(path)

    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s" % path)


def run_create_synthetic_images(config=None):

    # Prepare data creation pipeline
    ann_path = config.PATH + config.ANNOT_DIR
    icon_sizes = config.OBJECT_SIZE
    output_images = config.PATH + config.IMAGE_DIR
    annotations_list = []

    # Delete Output and Annotations folders and create new ones
    reset_dir(output_images)
    reset_dir(ann_path)

    bkg_images = get_backgrounds(config)



    print('Create images (known classes):')

    if config.CREATE_LABEL_MAP:
        create_label_map(objs_path=config.CLASSES_PATH, label_map_path=config.LABEL_MAP_PATH)


    for n in range(config.NUM_IMAGES):

        random.seed(n)

        flower_bush = choose_flower_bushes_randomly(bkg_images)
        bees = get_bees(config=config)

        number_bees_per_image = random.randint(config.MIN_NUM_OBJECT_PER_IMAGE, config.MAX_NUM_OBJECT_PER_IMAGE)
        bees = draw_random_subset_of_bees(bees, number_bees_per_image)
        # Duplicate each icon
        for i in range(len(bees)):
            for x in range(random.randint(*config.RANGE_NUM_EQUAL_OBJECTS)):
                bees.append(bees[i])




        # Load the background flower bush
        bkg_img = Image.open(config.BACKGROUNDS_PATH + flower_bush)
        bkg_img = bkg_img.convert('RGBA')
        bkg_x, bkg_y = bkg_img.size

        # Load bee objects
        # Initialize lists for Pascal VOC Format
        processed_bnd_boxes = []
        objects = []
        obj_coordinates = []
        bee_size_list = []
        # Copy background
        bkg_w_obj = bkg_img.copy()


        for i, idx in enumerate(bees):

            i_path = config.CLASSES_PATH + idx
            obj_img = Image.open(i_path)
            (width, height) = (obj_img.width // 4, obj_img.height // 4)
            obj_img  = obj_img.resize((width, height))
            #print(obj_img.size)
            obj_img = obj_img.convert('RGBA')
            label_class = idx.split(".png")[0]
            #print(label_class)

            # Generate images with different sizes of symbols
            # Get an array of random obj positions (from top-left corner)
            icon_size = random.choice(icon_sizes)
            bee_size_list.append(icon_size)
            obj_h, obj_w, x_pos, y_pos = get_obj_positions(obj=obj_img, bkg=bkg_img, size=icon_size, count=1)
            obj_h = obj_h[0]
            obj_w = obj_w[0]
            x_pos = x_pos[0]
            y_pos = y_pos[0]

            obj_coordinates.append([x_pos,
                                    y_pos,
                                    x_pos + obj_w,
                                    y_pos + obj_h])



        # print('Length of object coordinates : ', len(obj_coordinates))

        #Check if icons position overlap
        counter=0
        for row1, row2 in itertools.combinations(obj_coordinates, 2):
            if intersects(row1, row2):
                if row1 in obj_coordinates:
                    counter+=1
                    index = obj_coordinates.index(row1)
                    del bees[index]            #remove/comment out for icons to overlap
                    del obj_coordinates[index]  #remove(comment out for icons to overlap

        # print('Länge der icons removed: ', len(icons))
        # print('Länge der object coordinates removed: ', len(obj_coordinates))


        for i, idx in enumerate(bees):

            i_path = config.CLASSES_PATH + idx
            obj_img = Image.open(i_path)
            (width, height) = (obj_img.width // 4, obj_img.height // 4)

            obj_img  = obj_img.resize((width, height))
            obj_img = obj_img.convert('RGBA')
            label_class ='bee'       #idx.split(".png")[0]
            #print(label_class)
            #print(obj_img.size)


            # Create synthetic images based on positions
            obj_img, mask = resize_image(obj_img, mutate_size=bee_size_list[i], rotate=[0, 90, 180, 270])
            obj_img = blur_image(obj_img)
            obj_img= filter_image(obj_img)
            bkg_w_obj.alpha_composite(obj_img, dest=(obj_coordinates[i][0], obj_coordinates[i][1])) #(x_pos, y_pos)
            obj_w, obj_h = obj_img.size

            # for b in boxes:
            processed_bnd_boxes.append([BoundingBox(      # x_pos, y_pos, x_pos + obj_w, y_pos + obj_h),label_class])
                obj_coordinates[i][0],
                obj_coordinates[i][1],
                obj_coordinates[i][0] + obj_w,
                obj_coordinates[i][1] + obj_h
            ), label_class])

        # Mutate image enhancers
        if config.MUTATE:
            bkg_w_obj = mutate_image(bkg_w_obj)
            #bkg_w_obj = blur_image(bkg_w_obj)          #optional
            #bkg_w_obj = filter_image(bkg_w_obj)        #optional


        for b in processed_bnd_boxes:
            objects.append(Annotated_Object(name=b[1], pose="Front",
                                            truncated=0, difficult=0, bndBox=b[0]))

        annotations_list.append(Annotation(folder=output_images.split('/')[-2], filename=str(n) + ".jpg",
                                           path=output_images + '/' + str(n) + ".jpg",
                                           source="Source", size=[bkg_x, bkg_y, 3], segmented=0,
                                           objects=objects))

        # Save the image
        output_fp = output_images + '/' + str(n) + ".jpg"
        bkg_w_obj = bkg_w_obj.convert('RGB')
        bkg_w_obj.save(fp=output_fp, format="JPEG")

        if n+1 % 10 == 0:
            print(n+1, 'images created.')

    print("Saving Annotations in Pascal VOC Format")

    # save each annotation as a file
    for n, ann in enumerate(annotations_list):
        filename = str(n) + ".xml"
        ann_file = open(ann_path + '/' + filename, 'w+')
        ann_file.write(ann.toPascalVOCFormat())
        ann_file.close()

    total_images = len([f for f in os.listdir(output_images) if not f.startswith(".")])
    print(f'Done! Created {total_images} synthetic training images.')


if __name__ == "__main__":
    bee_config = BeeConfig(name='train', num_images=1000, label_map='label_map', root='../')
    run_create_synthetic_images(config=bee_config)
