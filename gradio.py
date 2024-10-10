import gradio as gr
import torch
from lama_cleaner.model_manager import ModelManager
from lama_cleaner.schema import Config, HDStrategy, LDMSampler
from transformers import AutoProcessor, AutoModelForCausalLM
import cv2
import numpy as np
from PIL import Image, ImageDraw
import subprocess

# Install necessary packages
subprocess.run('pip install flash-attn --no-build-isolation', env={'FLASH_ATTENTION_SKIP_CUDA_BUILD': "TRUE"}, shell=True)

# Initialize Llama Cleaner model
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define available models
available_models = [
    'microsoft/Florence-2-base',
    'microsoft/Florence-2-base-ft',
    'microsoft/Florence-2-large',
    'microsoft/Florence-2-large-ft'
]

# Load all models and processors
model_dict = {}
for model_id in available_models:
    florence_model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to("cuda").eval()
    florence_processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model_dict[model_id] = (florence_model, florence_processor)

@spaces.GPU()
def process_image(image, mask, strategy, sampler, fx=1, fy=1):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    if fx != 1 or fy != 1:
        image = cv2.resize(image, None, fx=fx, fy=fy, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, None, fx=fx, fy=fy, interpolation=cv2.INTER_NEAREST)
    
    config = Config(
        ldm_steps=1,
        ldm_sampler=sampler,
        hd_strategy=strategy,
        hd_strategy_crop_margin=32,
        hd_strategy_crop_trigger_size=200,
        hd_strategy_resize_limit=200,
    )

    model = ModelManager(name="lama", device=device)
    result = model(image, mask, config)
    return result

def create_mask(image, prediction):
    mask = Image.new("RGBA", image.size, (0, 0, 0, 255))  # Black background
    draw = ImageDraw.Draw(mask)
    scale = 1
    for polygons in prediction['polygons']:
        for _polygon in polygons:
            _polygon = np.array(_polygon).reshape(-1, 2)
            if len(_polygon) < 3:
                continue
            _polygon = (_polygon * scale).reshape(-1).tolist()
            draw.polygon(_polygon, fill=(255, 255, 255, 255))  # Make selected area white
    return mask
  
def process_images_florence_lama(image, model_choice):
    florence_model, florence_processor = model_dict[model_choice]
    
    # Convert image to OpenCV format
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Run Florence to get mask
    text_input = 'watermark'
    task_prompt = '<REGION_TO_SEGMENTATION>'
    image_pil = Image.fromarray(image_cv)  # Convert array to PIL Image
    inputs = florence_processor(text=task_prompt + text_input, images=image_pil, return_tensors="pt").to("cuda")
    generated_ids = florence_model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )
    generated_text = florence_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = florence_processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image_pil.width, image_pil.height)
    )
    
    # Create mask and process image with Llama Cleaner
    mask_image = create_mask(image_pil, parsed_answer['<REGION_TO_SEGMENTATION>'])
    result_image = process_image(image_cv, np.array(mask_image), HDStrategy.RESIZE, LDMSampler.ddim)
    
    # Convert result back to PIL Image
    result_image_pil = Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    
    return result_image_pil

# Define Gradio interface
demo = gr.Interface(
    fn=process_images_florence_lama,
    inputs=[
        gr.Image(type="pil", label="Input Image"),
        gr.Dropdown(choices=available_models, value='microsoft/Florence-2-large', label="Choose Florence Model")
    ],
    outputs=gr.Image(type="pil", label="Output Image"),
    title="Watermark Remover",
    description="Upload images and remove selected watermarks using Florence and Lama Cleaner.\nhttps://github.com/Damarcreative/rem-wm.git"
)

if __name__ == "__main__":
    demo.launch()
