import yaml


def load_config():
    with open('config.yaml') as file:
        settings = yaml.load(file, Loader=yaml.FullLoader)
        detection_mode = settings['algorithm']
        if detection_mode:
            print('Detection algorithm:', detection_mode)
        else:
            print('Detection algorithm not chosen')
        return detection_mode


def save_settings(value):
    data = dict(algorithm=value)
    with open('config.yaml', 'w') as file:
        yaml.dump(data, file)
        print('Saved:', data)


def clear_image_src():
    data = dict(image_source='')
    with open('config.yaml', 'w') as file:
        yaml.dump(data, file)
        print('Image data cleared')
