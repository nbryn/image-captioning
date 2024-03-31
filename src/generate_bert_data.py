import json
import random
from src.image_captioning import BlipImageCaptioning, GitBaseImageCaptioner, GitLargeImageCaptioner
from src.preprocess import add_genome_images_to_annotations
from src.util import log, success
from src.config import CAPTION_MODELS
from tqdm import tqdm
from src.util import log, success

def generate_bert_data(args):
    data_size = args.data_size
    annotations = add_genome_images_to_annotations().values()
    annotations = list(annotations)[0:data_size]

    log("GENERATING CAPTIONS")
    img_captioner = None
    match args.caption_model:
        case CAPTION_MODELS.GIT_BASE:
            img_captioner = GitBaseImageCaptioner()
        case CAPTION_MODELS.BLIP:
            img_captioner = BlipImageCaptioning()
        case CAPTION_MODELS.GIT_LARGE:
            img_captioner = GitLargeImageCaptioner()
        case _:
            raise ValueError("Invalid caption model")

    for anno in tqdm(annotations, desc="***Generating image captions...: "):
        img_paths = list(map(lambda x: x["image"], anno["images"]))
        captions = list(
            map(lambda img_path: img_captioner.generate_caption(img_path), img_paths))

        success("Genereated the following captions: " + str(captions))

        anno["context"] = captions

    # Train-test
    test_length = int(len(annotations) * args.test_size)
    random.seed(args.seed)
    random.shuffle(annotations)

    log(f"Splitting data into train and test. Test size: {test_length}")

    train_annotations = annotations[test_length:]
    test_annotations = annotations[0:test_length]

    log(f"Dumping test annotations to file: {args.test_data_path}")
    with open(args.test_data_path, "w") as file:
        json.dump(test_annotations, file)

    log(f"Dumping train annotations to file: {args.train_data_path}")
    with open(args.train_data_path, "w") as file:
        json.dump(train_annotations, file)

    return annotations
