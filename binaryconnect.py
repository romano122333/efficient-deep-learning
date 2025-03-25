import torch # type: ignore
import torch.nn as nn # type: ignore
import numpy as np

class BC():
    def __init__(self, model):
        # Compter le nombre de Conv2d et Linear dans le modèle
        count_targets = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                count_targets += 1

        start_range = 0
        end_range = count_targets - 1
        self.bin_range = np.linspace(start_range, end_range, end_range - start_range + 1).astype('int').tolist()

        # Initialisation des paramètres
        self.num_of_params = len(self.bin_range)
        self.saved_params = []  # Liste pour stocker les poids en pleine précision
        self.target_modules = []  # Modules à binariser

        self.model = model  # Le modèle à entraîner et quantifier

        # Création de la copie initiale de tous les paramètres
        index = -1
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                index += 1
                if index in self.bin_range:
                    tmp = m.weight.data.clone()
                    self.saved_params.append(tmp)
                    self.target_modules.append(m.weight)

    def save_params(self):
        """Sauvegarde des poids de pleine précision dans self.saved_params"""
        for index in range(self.num_of_params):
            self.saved_params[index].copy_(self.target_modules[index].data)

    def binarization(self):
        """Binarise tous les poids du modèle"""
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(torch.sign(self.saved_params[index]))
        # print("Poids binarisés (extrait) :", self.target_modules[0].data.flatten()[:10])

    def restore(self):
        """Restaure les poids originaux à partir de self.saved_params"""
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.saved_params[index])

    def clip(self):
        """Clipping des poids dans la plage [-1, 1] en utilisant HardTanh"""
        for param in self.model.parameters():
            param.data = nn.functional.hardtanh(param.data, min_val=-1, max_val=1)  # Clipping avec HardTanh

    def forward(self, x):
        """Fait passer les données dans le modèle"""
        out = self.model(x)
        return out
