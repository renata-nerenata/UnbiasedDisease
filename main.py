from src.pipeline_unbiased_diffusion import UnbiasedDiseasePipeline
import torch
from src.utils import plot_images, get_editing_params
from lookup_table import lookup_table_stigma, params
from torchvision.transforms.functional import to_pil_image


def main():
    pipe = UnbiasedDiseasePipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
    )
    device = "cuda"
    pipe.to(device)
    generator = torch.Generator(device=device)
    prompt = "A person suffering from schizophrenia"

    edited_params = get_editing_params(prompt, lookup_table_stigma, params)

    original_image = pipe(
        prompt=prompt, generator=generator, num_images_per_prompt=1, guidance_scale=7
    )

    edited_image = pipe(
        prompt=prompt,
        generator=generator,
        num_images_per_prompt=1,
        guidance_scale=7,
        **edited_params
    )

    original_image = to_pil_image(original_image.images[0])
    edited_image = to_pil_image(edited_image.images[0])

    image_grid = plot_images([original_image, edited_image], rows=1, cols=2)
    image_grid.show()


if __name__ == "__main__":
    main()
