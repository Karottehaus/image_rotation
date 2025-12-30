from time import time
from PIL import Image
import os
import pathlib


class Generator:
    def __init__(self, config, new_image_size=(224, 224)):
        self.config = config
        self.new_image_size = new_image_size
        self._create_directories()

    def _create_directories(self):
        for dir_path in [
            self.config['output_dir'],
            os.path.join(self.config['output_dir'], 'TRAINING_SET'),
            os.path.join(self.config['output_dir'], 'TEST_SET')
        ]:
            pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)

        for dataset in ['TRAINING_SET', 'TEST_SET']:
            for angle in [0, 90, 180, 270]:
                pathlib.Path(os.path.join(
                    self.config['output_dir'],
                    dataset,
                    str(angle)
                )).mkdir(parents=True, exist_ok=True)

    def get_images_pathnames_list(self, file_pathname):
        with open(file_pathname, mode='r') as paths:
            return [line.strip() for line in paths.readlines()]

    def generate(self, resample=Image.BICUBIC, training_set_flag=True):
        start = time()
        dataset_type = "Training" if training_set_flag else "Test"
        print(f"\nStarting {dataset_type} set generation...")

        file_txt = self.config['train_txt'] if training_set_flag else self.config['test_txt']
        images_pathnames_list = self.get_images_pathnames_list(file_txt)
        total_images = len(images_pathnames_list)

        for idx, image_path in enumerate(images_pathnames_list, 1):
            input_image_path = os.path.join(self.config['input_dir'], image_path.lstrip('/'))
            image_name = os.path.basename(image_path)
            print(f"Processing [{idx}/{total_images}]: {image_name}")

            try:
                original_image = self._open_image(input_image_path)
                resulting_image = self._resize(original_image, resample)

                for angle in [0, 90, 180, 270]:
                    dest_folder = self._create_rotation_folder(angle, training_set_flag)
                    rotated_image = self._rotate(resulting_image, rotation_degree=angle)
                    rgb_image = self._to_RGB(rotated_image)
                    self._save_image(rgb_image, image_name, dest_path=dest_folder)

            except Exception as e:
                print(f"Error processing {image_name}: {str(e)}")
                continue

        end = time()
        self._print_timing(start, end, operation=f'{dataset_type} set generation')

    def _rotate(self, image, rotation_degree):
        if rotation_degree == 0:
            return image
        else:
            rotations = {
                90: Image.ROTATE_90,
                180: Image.ROTATE_180,
                270: Image.ROTATE_270
            }
            return image.transpose(method=rotations.get(rotation_degree))

    def _resize(self, image, resample):
        return image.resize(self.new_image_size, resample)

    def _to_RGB(self, image):
        return image.convert('RGB') if image.mode != 'RGB' else image

    def _open_image(self, image_path):
        return Image.open(image_path)

    def _save_image(self, image, image_name, dest_path):
        output_path = os.path.join(dest_path, image_name)
        image.save(output_path)

    def _create_rotation_folder(self, rotation_angle, training_set_flag):
        dataset_type = 'TRAINING_SET' if training_set_flag else 'TEST_SET'
        return os.path.join(self.config['output_dir'], dataset_type, str(rotation_angle))

    def _print_timing(self, start, end, operation):
        print(f'{operation} completed in {round((end - start), 3)} seconds')


if __name__ == "__main__":
    config = {
        'input_dir': 'indoor/images',
        'output_dir': 'indoor_output',
        'train_txt': 'indoor_TrainImages.txt',
        'test_txt': 'indoor_TestImages.txt'
    }

    generator = Generator(config)
    generator.generate(training_set_flag=True)
    generator.generate(training_set_flag=False)
