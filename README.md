# Pneumonia detection using Convolutional Neural Networks

Lightweight, reproducible implementation of a convolutional neural network (CNN) to detect pneumonia from chest X-ray images using TensorFlow / Keras.

## Quick links
- App: [app.py](app.py)  
- CLI entry: [main.py](main.py) — calls [`train.run_training`](src/train.py), [`evaluate.run_evaluation`](src/evaluate.py), [`predict.run_prediction`](src/predict.py)  
- Training script: [src/train.py](src/train.py) — contains [`train.run_training`](src/train.py)  
- Evaluation script: [src/evaluate.py](src/evaluate.py) — contains [`evaluate.run_evaluation`](src/evaluate.py)  
- Prediction script: [src/predict.py](src/predict.py) — contains [`predict.run_prediction`](src/predict.py)  
- Grad-CAM util: [`src.utils.generate_gradcam`](src/utils.py) — [src/utils.py](src/utils.py)  
- Dataset root: [Data/chest_xray](Data/chest_xray)  
- Saved model (example): [runs/pneumonia_densenet121.h5](runs/pneumonia_densenet121.h5)  
- Dependencies: [requirements.txt](requirements.txt)  
- Git ignore: [.gitignore](.gitignore)

## Overview
This project trains a DenseNet121-based classifier to distinguish between NORMAL and PNEUMONIA chest X-rays. It includes training, evaluation, single-image prediction, a Streamlit demo app, and a Grad-CAM utility for visualizing model attention.

## Dataset
Expected layout (already present under [Data/chest_xray](Data/chest_xray)):

data/chest_xray/
- train/
  - NORMAL/
  - PNEUMONIA/
- val/
  - NORMAL/
  - PNEUMONIA/
- test/
  - NORMAL/
  - PNEUMONIA/

Ensure patient-level separation between splits to avoid data leakage.

## Requirements
- Python 3.8+
- GPU recommended for training
- Install dependencies:
  - pip install -r [requirements.txt](requirements.txt)

## Install & setup
1. Clone the repository.
2. Create and activate a virtual environment:
   - python -m venv .venv
   - Windows: .venv\Scripts\activate
   - macOS/Linux: source .venv/bin/activate
3. Install dependencies:
   - pip install -r [requirements.txt](requirements.txt)

## Usage

Run any of the main flows via the CLI wrapper:

- Train:
  - python [main.py](main.py) --mode train
  - Internally calls [`train.run_training`](src/train.py). See [src/train.py](src/train.py) for hyperparameters to edit.

- Evaluate:
  - python [main.py](main.py) --mode eval
  - Calls [`evaluate.run_evaluation`](src/evaluate.py) which calculates confusion matrix, classification report, and shows sample predictions.

- Predict (single image):
  - python [main.py](main.py) --mode predict --image path/to/image.jpg
  - Calls [`predict.run_prediction`](src/predict.py).

Streamlit UI
- Interactive demo:
  - streamlit run [app.py](app.py)
  - Upload an X-ray, get a prediction and confidence. The app loads the saved model from [runs/pneumonia_densenet121.h5](runs/pneumonia_densenet121.h5).

Notes on the model & scripts
- The training pipeline uses a pretrained DenseNet121 backbone with a custom head in [src/train.py](src/train.py).
- Saved model path used across scripts: `runs/pneumonia_densenet121.h5` — see [runs/pneumonia_densenet121.h5](runs/pneumonia_densenet121.h5).
- Visual explanations: [`src.utils.generate_gradcam`](src/utils.py) produces a Grad-CAM heatmap for a given image and model.

Example commands
- Full training (from CLI):
  - python [main.py](main.py) --mode train
- Evaluate:
  - python [main.py](main.py) --mode eval
- Predict:
  - python [main.py](main.py) --mode predict --image examples/sample.jpg
- Run Streamlit demo:
  - streamlit run [app.py](app.py)

## Implementation details
- Input resolution: 224×224
- Backbone: DenseNet121 (pretrained on ImageNet), frozen first and then fine-tuned (see [src/train.py](src/train.py))
- Loss: binary cross-entropy
- Metrics: accuracy, ROC-AUC (computed in [src/train.py](src/train.py) / [src/evaluate.py](src/evaluate.py))
- Augmentations: flips, small rotations, shifts and zooms (configured in [src/train.py](src/train.py))

## Outputs & artifacts
- Model checkpoints / final model: saved to `runs/` (example: [runs/pneumonia_densenet121.h5](runs/pneumonia_densenet121.h5))
- Plots: training curves and sample prediction figures are generated inline by scripts (Matplotlib).

## Troubleshooting
- "Model file not found": ensure a model exists at [runs/pneumonia_densenet121.h5](runs/pneumonia_densenet121.h5) or re-run training.
- Dataset not found: confirm [Data/chest_xray](Data/chest_xray) exists and follows the structure above.
- GPU issues: verify TensorFlow sees your GPU (`tf.config.list_physical_devices('GPU')`), and use compatible CUDA/cuDNN versions.

## Contributing
- Report issues or request features via repository issues.
- Preferred workflow: fork → feature branch → tests/docs → PR.
- Keep changes focused; update README and [requirements.txt](requirements.txt) when adding dependencies.

## License
Project is provided under an MIT-style license. Update LICENSE file as needed.

## Useful source references in this repo
- Training flow: [`train.run_training`](src/train.py) — [src/train.py](src/train.py)  
- Evaluation flow: [`evaluate.run_evaluation`](src/evaluate.py) — [src/evaluate.py](src/evaluate.py)  
- Prediction flow: [`predict.run_prediction`](src/predict.py) — [src/predict.py](src/predict.py)  
- Grad-CAM util: [`src.utils.generate_gradcam`](src/utils.py) — [src/utils.py](src/utils.py)  
- Streamlit app: [app.py](app.py)  
- CLI launcher: [main.py](main.py)

```// filepath: c:\Users\karim haddar24\pneumonia-cnn\README.md

# Pneumonia CNN

Lightweight, reproducible implementation of a convolutional neural network (CNN) to detect pneumonia from chest X-ray images using TensorFlow / Keras.

## Quick links
- App: [app.py](app.py)  
- CLI entry: [main.py](main.py) — calls [`train.run_training`](src/train.py), [`evaluate.run_evaluation`](src/evaluate.py), [`predict.run_prediction`](src/predict.py)  
- Training script: [src/train.py](src/train.py) — contains [`train.run_training`](src/train.py)  
- Evaluation script: [src/evaluate.py](src/evaluate.py) — contains [`evaluate.run_evaluation`](src/evaluate.py)  
- Prediction script: [src/predict.py](src/predict.py) — contains [`predict.run_prediction`](src/predict.py)  
- Grad-CAM util: [`src.utils.generate_gradcam`](src/utils.py) — [src/utils.py](src/utils.py)  
- Dataset root: [Data/chest_xray](Data/chest_xray)  
- Saved model (example): [runs/pneumonia_densenet121.h5](runs/pneumonia_densenet121.h5)  
- Dependencies: [requirements.txt](requirements.txt)  
- Git ignore: [.gitignore](.gitignore)

## Overview
This project trains a DenseNet121-based classifier to distinguish between NORMAL and PNEUMONIA chest X-rays. It includes training, evaluation, single-image prediction, a Streamlit demo app, and a Grad-CAM utility for visualizing model attention.

## Dataset
Expected layout (already present under [Data/chest_xray](Data/chest_xray)):

data/chest_xray/
- train/
  - NORMAL/
  - PNEUMONIA/
- val/
  - NORMAL/
  - PNEUMONIA/
- test/
  - NORMAL/
  - PNEUMONIA/

Ensure patient-level separation between splits to avoid data leakage.

## Requirements
- Python 3.8+
- GPU recommended for training
- Install dependencies:
  - pip install -r [requirements.txt](requirements.txt)

## Install & setup
1. Clone the repository.
2. Create and activate a virtual environment:
   - python -m venv .venv
   - Windows: .venv\Scripts\activate
   - macOS/Linux: source .venv/bin/activate
3. Install dependencies:
   - pip install -r [requirements.txt](requirements.txt)

## Usage

Run any of the main flows via the CLI wrapper:

- Train:
  - python [main.py](main.py) --mode train
  - Internally calls [`train.run_training`](src/train.py). See [src/train.py](src/train.py) for hyperparameters to edit.

- Evaluate:
  - python [main.py](main.py) --mode eval
  - Calls [`evaluate.run_evaluation`](src/evaluate.py) which calculates confusion matrix, classification report, and shows sample predictions.

- Predict (single image):
  - python [main.py](main.py) --mode predict --image path/to/image.jpg
  - Calls [`predict.run_prediction`](src/predict.py).

Streamlit UI
- Interactive demo:
  - streamlit run [app.py](app.py)
  - Upload an X-ray, get a prediction and confidence. The app loads the saved model from [runs/pneumonia_densenet121.h5](runs/pneumonia_densenet121.h5).

Notes on the model & scripts
- The training pipeline uses a pretrained DenseNet121 backbone with a custom head in [src/train.py](src/train.py).
- Saved model path used across scripts: `runs/pneumonia_densenet121.h5` — see [runs/pneumonia_densenet121.h5](runs/pneumonia_densenet121.h5).
- Visual explanations: [`src.utils.generate_gradcam`](src/utils.py) produces a Grad-CAM heatmap for a given image and model.

Example commands
- Full training (from CLI):
  - python [main.py](main.py) --mode train
- Evaluate:
  - python [main.py](main.py) --mode eval
- Predict:
  - python [main.py](main.py) --mode predict --image examples/sample.jpg
- Run Streamlit demo:
  - streamlit run [app.py](app.py)

## Implementation details
- Input resolution: 224×224
- Backbone: DenseNet121 (pretrained on ImageNet), frozen first and then fine-tuned (see [src/train.py](src/train.py))
- Loss: binary cross-entropy
- Metrics: accuracy, ROC-AUC (computed in [src/train.py](src/train.py) / [src/evaluate.py](src/evaluate.py))
- Augmentations: flips, small rotations, shifts and zooms (configured in [src/train.py](src/train.py))

## Outputs & artifacts
- Model checkpoints / final model: saved to `runs/` (example: [runs/pneumonia_densenet121.h5](runs/pneumonia_densenet121.h5))
- Plots: training curves and sample prediction figures are generated inline by scripts (Matplotlib).

## Troubleshooting
- "Model file not found": ensure a model exists at [runs/pneumonia_densenet121.h5](runs/pneumonia_densenet121.h5) or re-run training.
- Dataset not found: confirm [Data/chest_xray](Data/chest_xray) exists and follows the structure above.
- GPU issues: verify TensorFlow sees your GPU (`tf.config.list_physical_devices('GPU')`), and use compatible CUDA/cuDNN versions.

## Contributing
- Report issues or request features via repository issues.
- Preferred workflow: fork → feature branch → tests/docs → PR.
- Keep changes focused; update README and [requirements.txt](requirements.txt) when adding dependencies.

## License
Project is provided under an MIT-style license. Update LICENSE file as needed.

## Useful source references in this repo
- Training flow: [`train.run_training`](src/train.py) — [src/train.py](src/train.py)  
- Evaluation flow: [`evaluate.run_evaluation`](src/evaluate.py) — [src/evaluate.py](src/evaluate.py)  
- Prediction flow: [`predict.run_prediction`](src/predict.py) — [src/predict.py](src/predict.py)  
- Grad-CAM util: [`src.utils.generate_gradcam`](src/utils.py) — [src/utils.py](src/utils.py)  
- Streamlit app: [app.py](app.py)  
- CLI launcher: [main.py](main.py)
