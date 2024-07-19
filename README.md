# Rem-WM: Watermark Remover using Florence and Lama Cleaner

**Rem-WM**, a powerful watermark remover tool that leverages the capabilities of Microsoft Florence and Lama Cleaner models. This tool provides an easy-to-use interface for removing watermarks from images, with support for both individual images and batch processing.

### Test
https://huggingface.co/spaces/DamarJati/Remove-watermark

## Features

- **Watermark Removal**: Automatically detect and remove watermarks from images.
- **Batch Processing**: Efficiently process multiple images using threading.
- **Custom Model Support**: Flexibility to use custom Florence models.
- **Easy Integration**: Simple class-based interface for integration into your projects.

## Installation

First, clone the repository and navigate to the project directory:

```bash
git clone https://github.com/Damarcreative/rem-wm.git
cd rem-wm
```

Install the necessary dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Single Image Processing

To remove a watermark from a single image, use the `WatermarkRemover` class:

```python
from remwm import WatermarkRemover

# Initialize the WatermarkRemover with the default Florence model
remover = WatermarkRemover()

# Define input and output paths
input_image_path = "path/to/input/image.jpg"
output_image_path = "path/to/output/image.jpg"

# Process the image
remover.process_images_florence_lama(input_image_path, output_image_path)
```

### Batch Processing

To process multiple images in a directory, use the `process_batch` method:

```python
from remwm import WatermarkRemover

# Initialize the WatermarkRemover
remover = WatermarkRemover()

# Define input and output directories
input_dir = "path/to/input/folder"
output_dir = "path/to/output/folder"

# Process the batch of images
remover.process_batch(input_dir, output_dir, max_workers=4)
```

### Using a Custom Florence Model

If you want to use a custom Florence model, simply provide the model ID during initialization:

```python
from remwm import WatermarkRemover

# Initialize with a custom Florence model
remover = WatermarkRemover(model_id='facebook/custom-florence-model')

# Process images as usual
input_image_path = "path/to/input/image.jpg"
output_image_path = "path/to/output/image.jpg"
remover.process_images_florence_lama(input_image_path, output_image_path)
```

## Contributing

We welcome contributions to enhance Rem-WM! To contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature-name`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add some feature'`).
5. Push to the branch (`git push origin feature/your-feature-name`).
6. Open a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Microsoft Florence](https://github.com/microsoft/Florence)
- [Lama Cleaner](https://github.com/Sanster/lama-cleaner)

## Contact

For any questions or inquiries, please open an issue or contact us at
dev@damarcreative.my.id .

---

Thank you for using Rem-WM! We hope this tool helps you effectively remove watermarks from your images.
