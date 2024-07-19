import torch
from lama_cleaner.model_manager import ModelManager
from lama_cleaner.schema import Config, HDStrategy, LDMSampler
from transformers import AutoProcessor, AutoModelForCausalLM
import cv2
import numpy as np
from PIL import Image, ImageDraw
import subprocess
import os
from concurrent.futures import ThreadPoolExecutor

# Install necessary packages
subprocess.run('pip install flash-attn --no-build-isolation', env={'FLASH_ATTENTION_SKIP_CUDA_BUILD': "TRUE"}, shell=True)

class WatermarkRemover:
    def __init__(self, model_id='microsoft/Florence-2-large'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize Florence model
        self.florence_model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(self.device).eval()
        self.florence_processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        
        # Initialize Llama Cleaner model
        self.model_manager = ModelManager(name="lama", device=self.device)

    def process_image(self, image, mask, strategy=HDStrategy.RESIZE, sampler=LDMSampler.ddim, fx=1, fy=1):
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

        result = self.model_manager(image, mask, config)
        return result

    def create_mask(self, image, prediction):
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

    def process_images_florence_lama(self, input_image_path, output_image_path):
        # Load input image
        image = Image.open(input_image_path).convert("RGB")
        
        # Convert image to OpenCV format
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Run Florence to get mask
        text_input = 'watermark'  # Teks untuk Florence agar mengenali watermark
        task_prompt = '<REGION_TO_SEGMENTATION>'
        inputs = self.florence_processor(text=task_prompt + text_input, images=image, return_tensors="pt").to(self.device)
        generated_ids = self.florence_model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
        )
        generated_text = self.florence_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.florence_processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(image.width, image.height)
        )
        
        # Create mask and process image with Llama Cleaner
        mask_image = self.create_mask(image, parsed_answer['<REGION_TO_SEGMENTATION>'])
        result_image = self.process_image(image_cv, np.array(mask_image), HDStrategy.RESIZE, LDMSampler.ddim)
        
        # Convert result back to PIL Image
        result_image_pil = Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        
        # Save output image
        result_image_pil.save(output_image_path)

    def process_batch(self, input_dir, output_dir, max_workers=4):
        input_images = [os.path.join(input_dir, img) for img in os.listdir(input_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]
        output_images = [os.path.join(output_dir, os.path.basename(img)) for img in input_images]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(self.process_images_florence_lama, input_images, output_images)
