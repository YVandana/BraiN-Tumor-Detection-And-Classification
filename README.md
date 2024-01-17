# Brain-Tumor-Detection-And-Classification
# Brain Tumor Detection and Classification

This project aims to detect and classify brain tumors using modified convolutional neural networks (CNN) in Python, utilizing the Keras library. The project utilizes MRI images to detect the presence of brain tumors.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Brain tumors are abnormal growths of cells within the brain that can cause severe health issues. Early detection and accurate classification of brain tumors are crucial for effective treatment. This project aims to automate the detection and classification process using deep learning techniques.

The project utilizes convolutional neural networks (CNN) for image classification. The CNN model is modified and trained on a dataset of MRI images to detect the presence of brain tumors and classify them into different categories.

## Installation

To run this project locally, follow these steps:

1. Clone the repository:

   ````bash
   git clone https://github.com/your-username/brain-tumor-detection.git
   ```

2. Navigate to the project directory:

   ````bash
   cd brain-tumor-detection
   ```

3. Install the required dependencies:

   ````bash
   pip install -r requirements.txt
   ```

4. Download the dataset and place it in the appropriate directory (see [Dataset](#dataset) section).

5. Start the application:

   ````bash
   python main.py
   ```

## Usage

Once the application is running, follow the on-screen instructions to provide the MRI images for tumor detection and classification. The application will process the images using the trained model and provide the results.

## Dataset

The dataset used in this project consists of MRI images of brain tumors. It contains images from various categories, including tumors of different types and healthy brain images. The dataset can be obtained from [source] and should be placed in the `dataset` directory within the project.

## Model Architecture

The modified convolutional neural network (CNN) architecture used in this project is designed to perform brain tumor detection and classification. It consists of multiple convolutional layers, pooling layers, and fully connected layers. The exact architecture and hyperparameters can be found in the source code file `model.py`.

## Results

The accuracy and performance of the model depend on the quality and diversity of the dataset, as well as the chosen hyperparameters. After training the model on the provided dataset, the achieved accuracy on the validation set was approximately 90%.

## Contributing

Contributions to this project are welcome. If you have any suggestions or improvements, please follow these steps:

1. Fork the repository.
2. Create a new branch.
3. Make your changes and commit them.
4. Push your changes to your forked repository.
5. Submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE). Feel free to modify and use the code for your own purposes.
