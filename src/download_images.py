import json
import os
import requests
from multiprocessing import Pool

root_dir = os.path.dirname(__file__)
NUM_OF_THREADS = 10

# Create dir if not exists
image_dir = os.path.join(root_dir, "data", "images")
if not os.path.exists(image_dir):
     os.makedirs(image_dir)

def download_img(payload):
    image_url, image_name = payload
    image_path = os.path.join(image_dir, image_name)
    image_data = requests.get(image_url).content
    print(f"Downloading image {image_url}")
    with open(image_path, 'wb') as handler:
        handler.write(image_data)

def download_data():
    genome_data_path = os.path.join(root_dir, "data", "image_data.json")
    vdqg_data_path = os.path.join(root_dir, "data", "vdqg", "annotations", "annotation.json")
    print("genome_data_path: ", genome_data_path)

    genome_file = open(genome_data_path, 'r')
    vdqg_file = open(vdqg_data_path, 'r')
    
    genome_data = json.load(genome_file)
    vdqg_data = json.load(vdqg_file)

    annotations = vdqg_data["annotation"]

    # Convert genome data to a dictionary
    genome_data_dict = {}
    for img in genome_data:
        genome_data_dict[img["id"]] = img
        

    image_urls = []
    for vdqg_id in annotations:
        anno_objects = annotations[vdqg_id]["object"]

        for anno_object in anno_objects:
            vg_image_id = int(anno_object["VG_image_id"])

            
            # check if we have the image in the genome data
            if vg_image_id in genome_data_dict:
                image_url = genome_data_dict[vg_image_id]["url"]
                image_name = f"{vg_image_id}.jpg"
            
                image_urls.append((image_url, image_name))

    with Pool(NUM_OF_THREADS) as p:
        p.map(download_img, image_urls)

    
    print(f"DOWNLOADED {len(image_urls)} images")

if __name__ == "__main__":
    download_data()
