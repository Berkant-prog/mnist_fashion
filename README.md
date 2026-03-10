# Fashion-MNIST CNN (PyTorch)

Projet de classification d'images Fashion-MNIST avec un reseau de neurones convolutif (CNN) en PyTorch.

## Apercu

Le notebook principal `mnist_fash.ipynb` couvre :

- le chargement des donnees Fashion-MNIST
- le pretraitement et la normalisation
- la creation des jeux train / validation / test
- l'entrainement d'un CNN
- le suivi des courbes (loss et accuracy)
- l'evaluation finale sur le test set
- une visualisation des predictions

## Structure du projet

```text
mnist_fashion/
|-- mnist_fash.ipynb
|-- .gitignore
`-- data/                  # telecharge automatiquement par torchvision
```

## Prerequis

- Python 3.10+
- pip
- (optionnel) un environnement virtuel

## Installation

```bash
python -m venv .venv
# Windows PowerShell
.\\.venv\\Scripts\\Activate.ps1
pip install --upgrade pip
pip install torch torchvision numpy matplotlib jupyter
```

## Lancer le notebook

```bash
jupyter notebook mnist_fash.ipynb
```

## Pipeline d'entrainement

1. Chargement du dataset via `torchvision.datasets.FashionMNIST`.
2. Transformations : `ToTensor()` puis normalisation `(0.5,), (0.5,)`.
3. Split du train en 90% train / 10% validation.
4. Entrainement sur 10 epochs.
5. Sauvegarde en memoire du meilleur modele selon la validation accuracy.
6. Evaluation du meilleur modele sur le jeu de test.

## Architecture du modele

```text
Conv2d(1, 32, kernel_size=3, padding=1) + ReLU + MaxPool2d(2, 2)
Conv2d(32, 64, kernel_size=3, padding=1) + ReLU + MaxPool2d(2, 2)
Flatten
Linear(64*7*7, 128) + ReLU + Dropout(0.3)
Linear(128, 10)
```

## Hyperparametres utilises

- `batch_size = 64`
- `optimizer = Adam(lr=0.001)`
- `criterion = CrossEntropyLoss`
- `num_epochs = 10`
- seed fixe: `42`

## Resultat

- Precision sur le jeu de test: **92%**
- Le score peut varier legerement selon la machine et l'entrainement.
- 
## Classes Fashion-MNIST

`T-shirt/top`, `Trouser`, `Pullover`, `Dress`, `Coat`, `Sandal`, `Shirt`, `Sneaker`, `Bag`, `Ankle boot`

## Notes GitHub

Le dossier `data/` est ignore via `.gitignore` pour eviter de versionner les fichiers bruts du dataset.
