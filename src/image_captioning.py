from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration,   AutoModelForCausalLM, AutoProcessor

class BlipImageCaptioning:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        self.model.to(self.device)

    def generate_caption(self, image: str, text: str = None) -> str:
        """
        Args:
           image (str): Path to image to ask question about
           text (str): Text to condition caption on
        """
        
        image = Image.open(image).convert("RGB")
        if text:
            inputs = self.processor(image, text, return_tensors="pt").to(
                self.device
            )
        else:
            inputs = self.processor(image, return_tensors="pt").to(
                self.device
            )

        out = self.model.generate(**inputs)
        image.close()

        return self.processor.decode(out[0], skip_special_tokens=True)

class GitBaseImageCaptioner:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "microsoft/git-base-coco"
        self.processor = AutoProcessor.from_pretrained(
            self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name)
        self.model.to(self.device)

    def generate_caption(self, image_path: str) -> str:
        """
        Args:
           images (str): Path to image to ask question about
           text (str): Text to condition caption on
        """
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.processor(
            images=[image], return_tensors="pt").to(
                self.device
            ).pixel_values

        generated_ids = self.model.generate(
            pixel_values=pixel_values, max_length=500)
        generated_captions = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True)

        image.close()
        return generated_captions[0]

class GitLargeImageCaptioner:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "microsoft/git-large-coco"
        self.processor = AutoProcessor.from_pretrained(
            self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name)
        self.model.to(self.device)

    def generate_caption(self, image_path: str) -> str:
        """
        Args:
           images (str): Path to image to ask question about
           text (str): Text to condition caption on
        """
        image = Image.open(image_path).convert("RGB")

        pixel_values = self.processor(
            images=[image], return_tensors="pt").to(
                self.device
            ).pixel_values

        generated_ids = self.model.generate(
            pixel_values=pixel_values, max_length=500)
        generated_captions = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True)

        image.close()

        return generated_captions[0]