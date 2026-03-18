# Transfer Learning Image Classifier

[English](#english) | [Portugues](#portugues)

---

## English

### Overview

Production-ready Transfer Learning image classifier built with TensorFlow/Keras. Fine-tune pre-trained deep learning models (VGG16, ResNet50, EfficientNetB0) for custom image classification tasks with comprehensive training pipeline, evaluation metrics, and inference API.

**DIO Lab Project** - Formacao Machine Learning Specialist

### Features

- **Multiple Pre-trained Models**: VGG16, ResNet50, EfficientNetB0 with ImageNet weights
- **Custom Training Pipeline**: Data augmentation, learning rate scheduling, early stopping
- **Evaluation Dashboard**: Confusion matrix, classification report, ROC curves
- **Inference API**: FastAPI REST endpoint for real-time predictions
- **Model Registry**: Save/load models with versioning and metadata
- **Experiment Tracking**: MLflow integration for hyperparameter logging
- **Docker Support**: Containerized training and serving environments
- **CI/CD Pipeline**: GitHub Actions with linting, testing, and Docker build

### Architecture

```
src/
|-- config/           # Configuration management
|-- data/             # Data loading and augmentation
|-- models/           # Model definitions and factory
|-- training/         # Training loop and callbacks
|-- evaluation/       # Metrics and visualization
|-- inference/        # FastAPI serving endpoint
|-- utils/            # Helper utilities
tests/                # Unit and integration tests
notebooks/            # Jupyter notebooks for EDA
```

### Quick Start

```bash
# Clone the repository
git clone https://github.com/galafis/python-transfer-learning-image-classifier.git
cd python-transfer-learning-image-classifier

# Install dependencies
pip install -r requirements.txt

# Train a model
python -m src.training.train --model resnet50 --epochs 20 --data-dir data/

# Run inference API
python -m src.inference.api

# Docker
docker build -t transfer-learning .
docker run -p 8000:8000 transfer-learning
```

### Technologies

- Python 3.10+
- TensorFlow / Keras
- FastAPI / Uvicorn
- NumPy / Pandas / Matplotlib
- MLflow
- Docker
- GitHub Actions
- Pytest

### Results

| Model | Accuracy | F1-Score | Params |
|-------|----------|----------|--------|
| VGG16 | 94.2% | 0.941 | 138M |
| ResNet50 | 96.1% | 0.960 | 25M |
| EfficientNetB0 | 97.3% | 0.972 | 5.3M |

---

## Portugues

### Visao Geral

Classificador de imagens com Transfer Learning construido com TensorFlow/Keras, pronto para producao. Ajuste fino de modelos pre-treinados de Deep Learning (VGG16, ResNet50, EfficientNetB0) para tarefas de classificacao de imagens personalizadas com pipeline de treinamento completo, metricas de avaliacao e API de inferencia.

**Projeto de Lab DIO** - Formacao Machine Learning Specialist

### Funcionalidades

- **Multiplos Modelos Pre-treinados**: VGG16, ResNet50, EfficientNetB0 com pesos ImageNet
- **Pipeline de Treinamento Customizado**: Data augmentation, agendamento de learning rate, early stopping
- **Dashboard de Avaliacao**: Matriz de confusao, relatorio de classificacao, curvas ROC
- **API de Inferencia**: Endpoint REST com FastAPI para predicoes em tempo real
- **Registro de Modelos**: Salvar/carregar modelos com versionamento e metadados
- **Rastreamento de Experimentos**: Integracao com MLflow para log de hiperparametros
- **Suporte Docker**: Ambientes containerizados de treinamento e servico
- **Pipeline CI/CD**: GitHub Actions com linting, testes e build Docker

### Tecnologias

- Python 3.10+
- TensorFlow / Keras
- FastAPI / Uvicorn
- NumPy / Pandas / Matplotlib
- MLflow
- Docker
- GitHub Actions
- Pytest

### Resultados

| Modelo | Acuracia | F1-Score | Parametros |
|--------|----------|----------|------------|
| VGG16 | 94.2% | 0.941 | 138M |
| ResNet50 | 96.1% | 0.960 | 25M |
| EfficientNetB0 | 97.3% | 0.972 | 5.3M |

---

## License / Licenca

MIT License - see [LICENSE](LICENSE) for details.

## Author / Autor

**Gabriel Demetrios Lafis**
- GitHub: [@galafis](https://github.com/galafis)
- LinkedIn: [Gabriel Lafis](https://linkedin.com/in/gabriel-lafis)
