import json
import os
from PIL import Image
import numpy as np

parent_dir = os.path.dirname(os.pardir)
genome_pictures_path = os.path.join(parent_dir, "data", "images")
annotations_path = os.path.join(
    parent_dir, "data", "vdqg", "annotations", "annotation.json"
)

def get_img_as_array(img_path):
    img = Image.open(img_path)
    img_arr = np.asarray(img)
    img_arr = Image.fromarray(np.uint8(img_arr))
    img.close()
    
    return img_arr

def load_genome_images():
    image_dict = {}
    for picture_name in os.listdir(genome_pictures_path):
        picture_path = os.path.join(genome_pictures_path, picture_name)
        img = Image.open(picture_path)
        img_arr = np.asarray(img)
        image_dict[picture_name.split(".")[0]] = Image.fromarray(
            np.uint8(img_arr))
        img.close()

    return image_dict

def load_genome_image_paths():
    image_dict = {}
    for picture_name in os.listdir(genome_pictures_path):
        picture_path = os.path.join(genome_pictures_path, picture_name)
        image_dict[picture_name.split(".")[0]] = picture_path

    return image_dict

def add_genome_images_to_annotations() -> dict:
    genome_images = load_genome_image_paths()
    annotations_file = open(annotations_path, "r")
    annotations = json.load(annotations_file)["annotation"]
    result = {}
    for sample_id, annotation in annotations.items():
        object_1 = annotation["object"][0]
        object_2 = annotation["object"][1]

        object_1["image"] = genome_images[object_1["VG_image_id"]]
        object_2["image"] = genome_images[object_2["VG_image_id"]]

        annotation["images"] = annotation["object"]
        annotation["questions_with_scores"] = list(
            sorted(
                filter(
                    lambda x: x[1] > 0,
                    zip(annotation["question"], annotation["question_label"]),
                ),
                key=lambda x: x[1],
                reverse=True
            )
        )
        annotation["org_questions"] = list(zip(
            annotation["question"], annotation["question_label"]))

        del annotation["question_label"]
        del annotation["question"]
        del annotation["object"]
        del annotation["id"]
        result[int(sample_id)] = annotation

    return result