import os
import json


def create_label_map(objs_path, label_map_path):

    # Read class names in object folder

    classes = ['bee']

    # save label_map as json 
    label_map = {}
    for id, label_class in enumerate(classes, start=1):
        label_map[id] = label_class

    with open(label_map_path, 'w') as f:
        f.write(json.dumps(label_map))
        f.close()

    print('Created Label map: ' + label_map_path)
