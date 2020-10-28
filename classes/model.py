from datetime import datetime
import os
import pickle


class Model:

    def __init__(self, algorithm, encoder, train_set_dir, save_dir):
        self.algorithm = algorithm
        self.encoder = encoder
        now = datetime.now()
        self.creation_date = now.strftime("%d.%m.%Y, %H:%M:%S")
        self.label_dictionary = self.create_label_dictionary(train_set_dir)
        self.name = self.get_algorithm_name() + '_' + now.strftime("%d%m%Y_%H%M%S")
        self.save_path = os.path.join(save_dir, self.name)
        self.save_to_file('model.info')

    def get_algorithm_name(self):
        if self.algorithm == 1:
            return 'LBPH'
        elif self.algorithm == 2:
            return 'Eigenfaces'
        elif self.algorithm == 3:
            return 'Fisherfaces'
        elif self.algorithm == 4:
            return 'DNN'
        else:
            return 'None'

    def create_label_dictionary(self, dir_path):
        dict = {}
        dirs = os.listdir(dir_path)
        label = 1
        for dir_name in dirs:
            dict[label] = dir_name
            label += 1
        return dict

    def save_to_file(self, filename):

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        file_path = os.path.join(self.save_path, filename)
        with open(file_path, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
            print('Model info saved as:', file_path, '| Algorithm:', self.get_algorithm_name())

    def change_dictionary(self, new_dictionary):
        self.label_dictionary = new_dictionary
        self.save_to_file('model.info')
        print('Label dictionary updated.')

    def change_encoder(self, new_encoder):
        self.encoder = new_encoder
        self.save_to_file('model.info')
        print('Encoder updated.')

    def get_labels(self):
        labels = ""
        for value in self.label_dictionary.values():
            labels += value + ', '
        labels = labels[:-2]
        return labels
