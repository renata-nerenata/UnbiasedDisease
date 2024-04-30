from PIL import Image


def plot_images(images, num_rows, num_cols, padding=20):
    img_width, img_height = images[0].size
    grid_width = num_cols * img_width + (num_cols - 1) * padding
    grid_height = num_rows * img_height + (num_rows - 1) * padding
    grid = Image.new("RGBA", size=(grid_width, grid_height), color=(255, 255, 255, 0))

    for index, image in enumerate(images):
        x = index % num_cols * (img_width + padding)
        y = index // num_cols * (img_height + padding)
        grid.paste(image, box=(x, y))

    return grid


def get_editing_params(prompt, stigma_in_media, params):
    EDITING_PROMPT = None
    for key in stigma_in_media:
        if key.lower() in prompt.lower():
            EDITING_PROMPT = stigma_in_media[key]
            break

    params["editing_prompt"] = EDITING_PROMPT
    return params
