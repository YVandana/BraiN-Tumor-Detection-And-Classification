# Brain Tumor Detection and Classification

![Pre-Processed MRI Images](https://github.com/YVandana/Brain-Tumor-Detection-And-Classification/assets/80910772/8d7a597a-d842-4fcc-acec-7702b92fb3d0)

This repository contains the code and resources for the project "Brain Tumor Detection and Classification" using Convolutional Neural Networks (CNNs). The project aims to develop an automated system for the early detection and classification of brain tumors from MRI images.

## Introduction

- Malignancy or Brain Tumors are generally created when there is an sharp increase of abnormal cell growth in the  tissue of the brain. There is a need to locate the tumor to be able to set an effective  treatment plan for the patient.

- The image segmentation and classification of an infected tumor from MRIs by the process of enhancement, detection, and extraction are a major concerns as they are time consuming and is complex for even a medical specialist. 

- Using our approach, malignant cell growth will be detected from MRIs of the Brain with the application of CNN. Furthermore, it will also classify these MRIs into one of the following classes, Meningioma, Glioma, Pituitary if a malignant cell growth is found. This model will be accurate and assist technicians to determine the results.


## Project Overview

The project focuses on using deep learning techniques, specifically CNNs, to analyze MRI images and accurately detect and classify brain tumors. It involves the following key steps:
1. Preprocessing: The MRI images are preprocessed to enhance important features and remove noise.
2. Tumor Segmentation: The tumor regions are segmented from the preprocessed MRI images using advanced image processing techniques.
3. CNN Model Training: A CNN model is trained on a labeled dataset of MRI images to learn the patterns and features associated with different types of brain tumors.
4. Tumor Classification: The trained model is used to classify the segmented tumor regions into specific tumor types, such as Pituitary, Glioma, and Meningioma.
5. Performance Evaluation: The accuracy of the model is evaluated using various evaluation metrics to assess its effectiveness in tumor detection and classification.

## Project Architecture

![Architecture](https://github.com/YVandana/Brain-Tumor-Detection-And-Classification/assets/80910772/18a4cbf8-c80c-4757-97d1-0732eea183e5)


- The methodology followed is a 2D CNN with four Activation Functions: ReLU(Rectified Linear Unit) Function, TanH (Hyberbolic Tangent) Function, Exponential  Linear Unit Function and Sigmoid Function. 

- Along with Image Pre Processing techniques such as data cleaning , data integration, image cropping, edge detection and resizing. We have a total of 3246 MRI images to train the model from [Kaggle](https://www.kaggle.com/sartajbhuvaji/brain-tumor-classification-mri). 

- Through the readings the conclusion was reached, that CNN is the best to identify tumors from MRIs. The findings are based on the similarities between the graphs of the functions ReLU, TanH, Sigmoid added with ELU Functions resulting  in mean accuracy of 99.4%. For the advanced classification while the training accuracy  remains to be 99.4%. The validation accuracy is at 89.55%.

## CNN Model Architecture

![General CNN Architecutre](https://github.com/YVandana/Brain-Tumor-Detection-And-Classification/assets/80910772/213457bb-b29b-4845-9cc2-e27495b0fd89)

The CNN model used for brain tumor detection and classification consists of multiple convolutional layers followed by pooling layers and fully connected layers. The architecture is designed to extract informative features from the input MRI images and classify them into different tumor types.

Here is the table representing the CNN Model and it's layers:

![image](https://github.com/YVandana/Brain-Tumor-Detection-And-Classification/assets/80910772/c803cd45-56ac-421d-99f6-48aa6138931a)

| Total Parameters         | 1,032,238  |
| :--------               | :--------: |
| Trainable Parameters     | 1,032,232  |
| Non-Trainable Parameters | 6          |

## Libraries Used

The following libraries were used in the implementation of the project:

- TensorFlow: An open-source deep learning framework for training and deploying neural networks.
- Keras: A high-level neural networks API, running on top of TensorFlow, that simplifies the process of building and training deep learning models.
- NumPy: A fundamental package for scientific computing with Python, used for efficient array operations and mathematical computations.
- OpenCV: A library for computer vision tasks, used for image preprocessing and manipulation.
- Matplotlib: A plotting library for creating visualizations and graphs to analyze the results.For a complete list of dependencies and their versions, please refer to the requirements.txt file in the repository.

## Results
This section will consist of graphical results, images and table.

- The model achieved a training accuracy of 99.44% in 5 epochs while the training loss was 0.431%.


![Training Accuracy of 2 D Model for Tumor Detection](https://github.com/YVandana/Brain-Tumor-Detection-And-Classification/assets/80910772/dca7b032-8ff7-4b41-b7eb-280d7345e00f)

![Training Loss of 2 D Model for Tumor Detection](https://github.com/YVandana/Brain-Tumor-Detection-And-Classification/assets/80910772/7a46652c-4e8b-4e59-bec5-e9e448af4cbc)

![Accuracy: Training vs Valdiation for Tumor Detection](https://github.com/YVandana/Brain-Tumor-Detection-And-Classification/assets/80910772/3ed224b1-d242-4da1-9a21-103e39f5131e)

![Seaborn HeatMap view of the Confusion Matrix for Tumor Detection](https://github.com/YVandana/Brain-Tumor-Detection-And-Classification/assets/80910772/6c8fe648-8476-486e-9069-3381c6af55e0)

Table: Evaluation Metrics for Tumor Detection

| Metric     | Value      |
| ---------- | ---------- |
| Precision  | 0.98       |
| Recall     | 0.99       |
| F-Score    | 0.99       |
| Error      | 0.0298388  |

- For the task of classification the model again achieved an training accuracy of 99.44% with a validation accuracy of 89.55%.While the validation loss was at 4.5%.

![Accuracy: Training vs Valdiation of 2 D Model for Classification](https://github.com/YVandana/Brain-Tumor-Detection-And-Classification/assets/80910772/524d3dad-d30f-42c1-a922-d18a99fcdd7a)


![Loss: Training vs Valdiation of 2 D Model for Classification](https://github.com/YVandana/Brain-Tumor-Detection-And-Classification/assets/80910772/72afe4f5-e3af-461e-9159-c9e5d2a7b018)

Table: Testing Evaluation Metrics for Classification Task 

| Parameters       | Precision | Recall | F1 Score | Support |
| :---             |     :---: |  :---: | :------: | :------: |
| Glioma Tumor     | 0.86      | 0.18   | 0.30     | 100      |
| Meningioma Tumor | 0.61      | 0.93   | 0.74     | 115      |
| No Tumor         | 0.71      | 0.88   | 0.79     | 105      |
| Pituitary Tumor  | 0.84      | 0.77   | 0.80     | 74       |
| Accuracy         | N/A       | N/A    | 0.70     | 394      |
| Macro Avg        | 0.75      | 0.69   | 0.66     | 394      |
| Weighted Avg     | 0.74      | 0.70   | 0.65     | 394      |

## Conclusion

- The result procured from this activation function gives an accuracy of 99.44% for the identification of malignancy. It is better structured and gives coherent results for brain malignancy.

- Furthermore the classification task of detecting whether the given MRI falls under the classes of Meningioma, Glioma or Pituitary Tumor has also been successfully done with an accuracy of 89.55%.


## Repository Structure
- Code: This directory contains the implementation of the brain tumor detection and classification system. It includes scripts for data preprocessing, tumor segmentation, CNN model training, and classification.
- Datasets: This directory is used to store the labeled dataset of MRI images used for training and testing the CNN model.
- LICENSE: This file contains the license information for the project.

## Getting Started
To get started with the project, follow these steps:
1. Clone the repository: git clone https://github.com/YVandana/Brain-Tumor-Detection-And-Classification.git
2. Explore the readme to understand the implementation details.
3. Run the scripts in the appropriate order to preprocess the data, train the CNN model, and perform tumor detection and classification.
4. Refer to the project report in the report/ directory for a comprehensive understanding of the project methodology and results.

## Contribution Guidelines
If you wish to contribute to this project, you can follow these guidelines:
1. Fork the repository.
2. Create a new branch for your contribution: git checkout -b feature/your-feature
3. Make your changes and commit them: git commit -m "Add your message"
4. Push the changes to your forked repository: git push origin feature/your-feature
5. Open a pull request, describing your changes and their purpose.Please ensure that your contributions align with the project's goals and follow the coding conventions and best practices.
## License
This project is licensed under the GNU License.
## Acknowledgments
We would like to express our gratitude to Dr. Y. Vijayalata for her guidance and support throughout the project. We also acknowledge the contributions of Vandana Yalla, M Ananya Varma, Salunke Savitha, and Sudarsi Namrata Ravindra, who were the authors of the project report.
## Contact Information
If you have any questions or suggestions regarding this project, please feel free to contact the project maintainer.
