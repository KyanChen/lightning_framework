from transformers import AutoTokenizer, CLIPModel
from PIL import Image
import requests
from transformers import AutoProcessor, CLIPModel

config_ = "openai/clip-vit-large-patch14-336"
config_ = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(config_)
tokenizer = AutoTokenizer.from_pretrained(config_)
processor = AutoProcessor.from_pretrained(config_)

inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
text_features = model.get_text_features(**inputs)



url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(images=image, return_tensors="pt")

image_features = model.get_image_features(**inputs)
pass
