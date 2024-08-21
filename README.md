<p align="center">
  <img width="10%" src="Prediction of Breast Cancer Incidence- Shivani-23200782/images/bck.jpg">
</p>

<h1 align="center">
  <b>
    Prediction of Breast Cancer Incidence
  </b>
</h1>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-v3.11.4+-red.svg">
  <img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-v2.9.1+-orange.svg">
  <img alt="Contributions welcome" src="https://img.shields.io/badge/contributions-welcome-blue.svg">
</p>

Welcome to the **Prediction of Breast Cancer Incidence** project repository! This project leverages machine learning and deep learning techniques to enhance the accuracy of breast cancer detection. By analyzing two distinct datasets— the **Wisconsin Breast Cancer dataset** and the **BreakHis400x Image Dataset**, this project aims to classify breast tumor images and predict malignancy based on cell nuclei characteristics.

<p align="center">
  <img src="Prediction of Breast Cancer Incidence- Shivani-23200782/images/Normal_VS_Cancer_Defined.jpg">
</p>

**Note**: It is recommended to view this README in light mode for better graph visibility.

## Table of Contents 📑
- [Project Overview](#project-overview)
- [Project Highlights](#project-highlights)
- [Installation](#installation)
- [Usage](#usage)
- [Datasets](#datasets)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)
- [Data Sources](#data-sources)
- [Authors](#authors)
- [Streamlit Demonstration of the Project](#demonstration)
- [Poster](#poster)

## Project Overview 

Breast cancer remains a leading cause of mortality among women globally, making early detection crucial for effective treatment. This project is designed to develop and evaluate machine learning models that accurately classify breast cancer images and predict malignancy using the **Wisconsin Breast Cancer** and **BreaKHis 400X** datasets.

By analyzing the visual characteristics of cell structures and applying deep learning techniques, this project aims to enhance early detection capabilities and support medical diagnosis.

## Project Highlights

- **Multi-Model Approach**: Various machine learning models including Random Forest, Logistic Regression, K-Nearest Neighbors (KNN), and Neural Networks on the Wisconsin dataset have been evaluated and compared to obtain the best performing model.
- **Deep Learning with DenseNet**: The DenseNet CNN architecture has been used for classifying images from the BreakHis400x dataset as benign or malignant.
- **Exploratory Data Analysis**: Detailed Exploratory data analysis on the Wisconsin Dataset has been performed as shown below:
  <p align="center">
    <img src="Prediction of Breast Cancer Incidence- Shivani-23200782/images/correlation_heatmap.png" alt="Correlation Heatmap">
  </p>
  <p align="center">
    <img src="Prediction of Breast Cancer Incidence- Shivani-23200782/images/newplot.png" alt="Scatter Plots of Symmetry, Fractal Dimension, and Concavity">
  </p>
  <p align="center">
    <img src="Prediction of Breast Cancer Incidence- Shivani-23200782/images/newplot (1).png" alt="Symmetry, Fractal Dimension, and Concavity measures by Diagnosis">
  </p>
  <p align="center">
    <img src="Prediction of Breast Cancer Incidence- Shivani-23200782/images/12.png" alt="Perimeter, Smoothness, and Texture measures by Diagnosis">
  </p>
  
- **Scalability**: The project is designed to be scalable for future improvements, including the integration of more diverse datasets and the exploration of ensemble methods.

## Installation

To get started with this project, clone the repository and install the required dependencies.

```bash
git clone https://github.com/Shivani-23200782/breast-cancer-prediction.git
cd breast-cancer-prediction
pip install -r requirements.txt
```

## Usage

After installation, you can run the notebooks for each dataset:

1. **Wisconsin Dataset**:
   - Open the `Breast_Cancer_classification_wisconsin.ipynb` notebook.
   - Execute all the cells to analyze the Wisconsin dataset using multiple machine learning models.

   <p align="center">
     <img src="Prediction of Breast Cancer Incidence- Shivani-23200782/images/classification_report.png" alt="Classification Report">
   </p>

2. **BreaKHis 400X Dataset**:
   - Open the `Breast_cancer_Classification_DenseNet.ipynb` notebook.
   - Execute all the cells to use DenseNet201 to classify breast cancer images.

   <p align="center">
     <img src="Prediction of Breast Cancer Incidence- Shivani-23200782/images/confusion_matrix.png" alt="Confusion Matrix">
   </p>

You can run these notebooks in Jupyter Notebook or any other compatible environment.

## Datasets

- **Wisconsin Breast Cancer Dataset**: Contains numeric features derived from fine needle aspirates of breast masses, used for predicting breast cancer diagnosis.
  - Features include radius, texture, and perimeter of cell nuclei.
  <p align="center">
    <img src="Prediction of Breast Cancer Incidence- Shivani-23200782/images/accuracy_plot.png" alt="Accuracy Plot">
  </p>
- **BreaKHis 400X Dataset**: High-resolution microscopic images of breast tumor tissues categorized into benign and malignant classes.
  - Used to train deep learning models for distinguishing between benign and malignant breast cancer based on visual characteristics.

## Model Architecture

- **DenseNet201**: A deep convolutional neural network used for classifying images in the BreakHis400x dataset. The model achieved a training accuracy of 96% and a test accuracy of 87.45%.
  <p align="center">
    <img src="Prediction of Breast Cancer Incidence- Shivani-23200782/images/accuracy_loss.png" alt="Training and Validation Accuracy">
  </p>
  <p align="center">
    <img src="Prediction of Breast Cancer Incidence- Shivani-23200782/images/ROC_Plot.png" alt="ROC Curve">
  </p>
- **Random Forest, Logistic Regression, KNN, Neural Networks**: Various models evaluated on the Wisconsin dataset, with Random Forest emerging as the best performer with an accuracy of 96.49%.

## Results

- **DenseNet201**: Demonstrated strong classification capabilities on the BreaKHis dataset with a training accuracy of 96%, precision of 0.91, recall of 0.94, and an F1 score of 0.97.
  <p align="center">
    <img src="Prediction of Breast Cancer Incidence- Shivani-23200782/images/styled_table.png" alt="Styled Table of Metrics">
  </p>
- **Random Forest**: Outperformed other models on the Wisconsin dataset with an accuracy of 96.49%, F1 score of 0.952, and balanced accuracy of 0.958.

<p align="center">
  <img src="Prediction of Breast Cancer Incidence- Shivani-23200782/images/styled_classification_report.png" alt="Styled Classification Report">
</p>

## Contributing

Contributions are welcome! If you have ideas for improving this project or want to add more models or datasets, feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

1. Arnold, M., Morgan, E., Rumgay, H., et al. (2022). Current and future burden of breast cancer: Global statistics for 2020 and 2040. *The Breast*, 66, 15–23.
2. Huang, J., Chan, P. S., Lok, V., et al. (2021). Global incidence and mortality of breast cancer: a trend analysis. *Aging (Albany NY)*, 13(4), 5748.
3. Spanhol, F. A., Oliveira, L. S., Petitjean, C., & Heutte, L. (2016). A Dataset for Breast Cancer Histopathological Image Classification. *IEEE Transactions on Biomedical Engineering*, 63(7), 1455-1462.

## Data Sources

- [Wisconsin Breast Cancer Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)
- [BreaKHis 400X Dataset](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/)

## Authors

- **Shivani - 23200782**: MSc Data and Computational Science, UCD, Dublin

## Demonstration

- Please find the link for the Streamlit Application to test the functionality here: [Prediction of Breast Cancer Incidence] (https://breast-cancer-incidence-prediction-shivs-ucd.streamlit.app)

## Poster

Please review the poster presented at the University for the Poster Presentation here:[Poster](https://github.com/ACM40960/project-shivsucd/tree/main/Prediction%20of%20Breast%20Cancer%20Incidence-%20Shivani-23200782/images/Final_Poster_23200782.pdf).
