from remwm import WatermarkRemover

# Initialize WatermarkRemover object with custom Florence model (optional)
remover = WatermarkRemover(model_id='microsoft/Florence-2-large')

# To process a single image
input_image_path = "path/to/input/image.jpg"
output_image_path = "path/to/output/image.jpg"
remover.process_images_florence_lama(input_image_path, output_image_path)

# To batch process images in a folder
input_dir = "path/to/input/folder"
output_dir = "path/to/output/folder"
remover.process_batch(input_dir, output_dir, max_workers=4)