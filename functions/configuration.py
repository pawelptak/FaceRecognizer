import yaml


def load_config():
    with open('config.yaml') as file:
        settings = yaml.load(file, Loader=yaml.FullLoader)
        image_source = settings['image_source']
        if image_source:
            print('Image has been loaded:', image_source)
        else:
            print('No image loaded')
        return image_source


def save_image_src(img_source: str):
    data = dict(image_source=img_source)
    with open('config.yaml', 'w') as file:
        yaml.dump(data, file)
        print('Saved:', data)

def clear_image_src():
    data = dict(image_source='')
    with open('config.yaml', 'w') as file:
        yaml.dump(data, file)
        print('Image data cleared')