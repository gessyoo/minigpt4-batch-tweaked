import argparse
import os
import glob
import random
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import ImageOps, Image

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", type=str, default='eval_configs/minigpt4.yaml', help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument("--image-folder", type=str, required=True, help="path to the input image folder")
    parser.add_argument("--target-size", type=int, default=768, help="Target size for the smaller side of the image.")
    parser.add_argument("--beam-search-numbers", type=int, default=8, help="beam search numbers")
    parser.add_argument("--model", type=str, default='llama7b', help="Model to be used for generation. Options: 'llama' (default), 'llama7b'")
    parser.add_argument("--save-in-imgfolder", action="store_true", help="save captions in the input image folder")
    parser.add_argument("--name", type=str, required=False, help="Name for substitution in captions")  # New argument
    parser.add_argument("--name2", type=str, required=False, help="Optional second name for substitution in captions")  # New argument
    options = parser.parse_args()
    return options

def process_images(directory, target_size):
    files = os.listdir(directory)
    sorted_files = sorted(files, key=lambda x: (int(re.sub(r'\D', '', x)), x))

    for file_name in sorted_files:
        image_path = os.path.join(directory, file_name)
        image = Image.open(image_path)

        if image.mode == "RGB":
            image = image.convert("RGB")

        if image.mode == "L":
            print(f"Skipped grayscale image: {file_name}")
            continue

def remove_transparency(im, bg_colour=(255, 255, 255)):
    if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):
        alpha = im.convert('RGBA').split()[-1]
        bg = Image.new("RGBA", im.size, bg_colour + (255,))
        bg.paste(im, mask=alpha)
        return bg
    else:
        return im

def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def resize_image(image, target_size):
    width, height = image.size
    min_dimension = min(width, height)

    if min_dimension >= target_size:
        return image

    scale_factor = target_size / min_dimension
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    resized_image = image.resize((new_width, new_height))
    return resized_image
  

def describe_image(image_path, chat, chat_state, img, num_beams=3, temperature=1.0, repetition_penalty=1.0, min_sentence_length=6, max_sentence_length=25):
    chat_state = CONV_VISION.copy()
    img_list = []

    llm_message = chat.upload_img(resized_image, chat_state, img_list)

    # Modify the prompt to provide more specific instructions
    chat.ask("Describe the people and the scene in the image.", chat_state)
    
   # Generate the caption
    generated_caption = chat.answer(
        conv=chat_state,
        img_list=img_list,
        max_new_tokens=max_sentence_length * 2,  # Estimate a higher value to ensure desired sentence length
        num_beams=num_beams,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        max_length=2500
    )[0]
    
    generated_caption = generated_caption.replace("The woman", args.name).replace("the woman", args.name).replace("a woman", args.name).replace("woman", args.name)  # Substitute the name
    if args.name2:
        generated_caption = generated_caption.replace("The man", args.name2).replace("a man", args.name2).replace("the man", args.name2)

    # Remove unnecessary phrases from the generated caption
    generated_caption = generated_caption.replace("The image shows", "").replace("The image is", "").replace("looking directly at the camera", "").replace("in the image", "").replace("taking a selfie", "").replace("posing for a picture", "").replace("holding a cellphone", "").replace("is wearing a pair of sunglasses", "").replace("pulled back in a ponytail", "").replace("with a large window in the cent", "")

    # Split the caption into sentences
    sentences = generated_caption.split('. ')

    # Check if the last sentence is a fragment and remove it if necessary
    if len(sentences) > 1:
        last_sentence = sentences[-1]
        if len(last_sentence.split()) <= min_sentence_length:
            sentences = sentences[:-1]

    # Keep only the first two sentences and append periods
    sentences = [s.strip() + '.' for s in sentences[:3]]

    generated_caption = ' '.join(sentences)

    generated_caption = remove_duplicates(generated_caption)  # Remove duplicate words

    return generated_caption

def remove_duplicates(string):
    words = string.split(', ')
    unique_words = []

    for word in words:
        if word not in unique_words:
            unique_words.append(word)
        else:
            break

    return ', '.join(unique_words)

if __name__ == '__main__':
    args = parse_args()

    cfg = Config(args)

    model_config = cfg.model_cfg
    if args.model == "llama7b":
        model_config.llama_model = "camenduru/MiniGPT4-7B"

    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:0')

    vis_processor_cfg = cfg.datasets_cfg.cc_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    chat = Chat(model, vis_processor)

    chat_state = CONV_VISION.copy()
    img_list = []
    
    image_folder = args.image_folder
    num_beams = args.beam_search_numbers
    temperature = 1.0  # default temperature
    repetition_penalty = 1.0  # default repetition penalty

    image_extensions = ['jpg', 'jpeg', 'png', 'bmp', "webp"]
    image_paths = []

    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(image_folder, f'*.{ext}')))
        image_paths.extend(glob.glob(os.path.join(image_folder, f'*.{ext.upper()}')))

    if not args.save_in_imgfolder:
        if not os.path.exists("mycaptions"):
            os.makedirs("mycaptions")

    # Process each image
    for i, image_path in enumerate(image_paths):
        # Skip if the file is a text file
        if image_path.endswith('.txt'):
            continue
    
        # Open the image (corrected position)
        gr_img = Image.open(image_path)

        gr_img = ImageOps.autocontrast(gr_img, cutoff=0, ignore=None, mask=None, preserve_tone=False)

        if gr_img.mode in ('RGBA', 'LA') or (gr_img.mode == 'P' and 'transparency' in gr_img.info):
            gr_img = remove_transparency(gr_img)
            gr_img.save(image_path, quality=100)
            print("removed transparecy")

        # Resize the image
        resized_image = resize_image(gr_img, args.target_size)

        # Save the resized image
        resized_image.save(image_path, quality=100)
        
        start_time = time.time()
        caption = describe_image(image_path, chat, chat_state, img_list, num_beams, temperature, repetition_penalty, min_sentence_length=8, max_sentence_length=35)

        if args.save_in_imgfolder:
            output_path = os.path.join(image_folder, "{}.txt".format(os.path.splitext(os.path.basename(image_path))[0]))
        else:
            output_path = "mycaptions/{}.txt".format(os.path.splitext(os.path.basename(image_path))[0])

        with open(output_path, "w") as f:
            f.write(caption)

        end_time = time.time()
        time_taken = end_time - start_time

        print(f"Processing image {i + 1} of {len(image_paths)}")
        print(f"* Caption:   {caption}")
        print(f"Caption for {os.path.basename(image_path)} saved in '{output_path}'")
        print(f"Substituted name: {args.name}")
        if args.name2:
            print(f"Substituted name 2: {args.name2}")
        print(f"Time taken to process caption for {os.path.basename(image_path)} is: {time_taken:.2f} s")
        print("")

    print("Caption generation completed.")
