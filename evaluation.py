from PIL import Image
from clip_interrogator import Config, Interrogator
from lookup_table import lookup_table_stigma


def contains_stigmatizing_words(image_path, lookup_table):
    image = Image.open(image_path).convert("RGB")
    ci = Interrogator(Config(clip_model_name="ViT-L-14/openai"))
    interrogation_result = ci.interrogate(image)
    stigmatizing_words = set()
    for description in lookup_table.values():
        stigmatizing_words.update(description.split(", "))

    return any(word in stigmatizing_words for word in interrogation_result.split())


print(contains_stigmatizing_words("path", lookup_table_stigma))
