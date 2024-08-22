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
- [Conclusion and Future Work](#conclusion-and-future-work)
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
    <img src="Prediction of Breast Cancer Incidence- Shivani-23200782/images/12.png" alt="Perimeter, Smoothness, and Texture measures by Diagnosis">
  </p>
  <p align="center">
    <img src="Prediction of Breast Cancer Incidence- Shivani-23200782/images/x.png" alt="Distribution Plot of the Target Variable">
  </p>

- **Exploratory Data Analysis**: Detailed Exploratory data analysis on the BreaHis400x Dataset has been performed as shown below:
  #### Non-Cancerous Cell Image
  <p align="center">
    <img src="Prediction of Breast Cancer Incidence- Shivani-23200782/images/Finalized_Benign.png" alt="Non-Cancerous Cell Image">
  </p>
  
  #### Cancerous Cell Image
  <p align="center">
    <img src="Prediction of Breast Cancer Incidence- Shivani-23200782/images/Finalized_Malignant.png" alt="Cancerous Cell image">
  </p>

  #### Distribution plot for the Images 
  <p align="center">
    <img src="Prediction of Breast Cancer Incidence- Shivani-23200782/images/results_before_augmentation.png" alt="Distribution of Images Before data Augmentation">
  </p>

- **Data Augmentation** : As observed in the above picture, the benign class is heavily imbalanced, which could create a bias in training of the images towards malignancy. In an effort to avoid this, data augmentation is peformed on the benign images by flipping and rotating the benign images existing, and increase the number of images to ensure that the augmented images can aid in better classification. The dataset after the balancing of the dataset is as follows:

#### Distribution plot for the Images after Data Augmentation
  <p align="center">
    <img src="Prediction of Breast Cancer Incidence- Shivani-23200782/images/results_after_augmentation.png" alt="Distribution of Images After data Augmentation">
  </p>
  
- **Scalability**: The project is designed to be scalable for future improvements, including the integration of more diverse datasets and the exploration of ensemble methods.

## Installation

To get started with this project, clone the repository with the following code:

```bash
git clone https://github.com/ACM40960/project-shivsucd.git
cd Prediction of Breast Cancer Incidence- Shivani-23200782
```
Create a Virtual Environment with Python verison 3.12.

To create a virtual environment on Windows OS in Anaconda Command Prompt, you can use the following commands: 
```bash
conda create -n breast_cancer_pred python==3.12
conda activate breast_cancer_pred
```
Install the dependencies with the following command: 
```bash
pip install -r requirements.txt
```

## Usage

After installation, you can run the python notebooks for each dataset:

1. **Wisconsin Dataset**:
   - Open the `Breast_Cancer_classification_wisconsin.ipynb` notebook.
   - Execute all the cells to analyze the Wisconsin dataset using multiple machine learning models.
  
2. **BreaKHis 400X Dataset**:
   - Open the `Breast_cancer_Classification_DenseNet.ipynb` notebook.
   - Execute all the cells to use DenseNet201 to classify breast cancer images.

You can run these notebooks in Jupyter Notebook or any other compatible environment.

## Datasets

- **Wisconsin Breast Cancer Dataset**: Contains numeric features derived from fine needle aspirates of breast masses, used for predicting breast cancer diagnosis.
  - Features include radius, texture, and perimeter of cell nuclei.
- **BreaKHis 400X Dataset**: High-resolution microscopic images of breast tumor tissues categorized into benign and malignant classes.
  - Used to train deep learning models for distinguishing between benign and malignant breast cancer based on visual characteristics.

## Model Architecture

- **DenseNet201**: A deep convolutional neural network used for classifying images in the BreakHis400x dataset. The model achieved a training accuracy of 96% and a test accuracy of 87.45%.
  <p align="center">
    <img src="Prediction of Breast Cancer Incidence- Shivani-23200782/images/densenet.png" alt="Training and Validation Accuracy">
  </p>
- **Random Forest, Logistic Regression, KNN, Neural Networks**: Various models evaluated on the Wisconsin dataset, with Random Forest emerging as the best performer with an accuracy of 96.49%.

## Results

- **DenseNet201**: Demonstrated strong classification capabilities on the BreaKHis dataset with a training accuracy of 96%, precision of 0.91, recall of 0.94, and an F1 score of 0.97.

  #### Confusion Matrix:
   <p align="center">
     <img src="Prediction of Breast Cancer Incidence- Shivani-23200782/images/confusion_matrix.png" alt="Confusion Matrix">
   </p>
#### Evaluation figures:
   <p align="center">
     <img src="Prediction of Breast Cancer Incidence- Shivani-23200782/images/accuracy_plot.png" alt="Accuracy Plot">
   </p>
   <p align="center">
     <img src="Prediction of Breast Cancer Incidence- Shivani-23200782/images/accuracy_loss.png" alt="Loss Accuracy">
   </p>
   <p align="center">
     <img src="Prediction of Breast Cancer Incidence- Shivani-23200782/images/roc_curve.png" alt="ROC-AUC Plot">
   </p>
   <p align="center">
     <img src="Prediction of Breast Cancer Incidence- Shivani-23200782/images/classification_report.png" alt="Classification Report">
   </p>
   
- **Random Forest**: Outperformed other models on the Wisconsin dataset with an accuracy of 96.49%, F1 score of 0.952, Precision of 97.56%, Recall of 93.02%, and a balanced accuracy of 95.8%.

#### Comparative Model Metrics:
  <p align="center">
     <img src="Prediction of Breast Cancer Incidence- Shivani-23200782/images/radar_chart.png" alt="Radar Chart of all the metrics">
   </p>
   
 #### Evaluation Metrics Table:
   <p align="center">
     <img src="Prediction of Breast Cancer Incidence- Shivani-23200782/images/styled_table.png" alt="Styled Table of all metrics">
   </p>

## Conclusion and Future Work
This project demonstrates the potential of machine learning and deep learning models in accurately predicting breast cancer. The DenseNet-based CNN achieved a commendable training accuracy of 96.4% on the BreaKHis dataset, and the Random Forest model emerged as the best-performing model on the Wisconsin dataset with an accuracy of 96.49%. These results highlight the efficacy of advanced algorithms in distinguishing between benign and malignant cases, providing valuable support in medical diagnosis. 

However, the variability in model performance across different datasets emphasizes the importance of dataset-specific model selection and further model refinement for clinical applications.

Future work could focus on integrating more diverse datasets and exploring ensemble methods to enhance model robustness and generalization.

## Contributing

Contributions are welcome! If you have ideas for improving this project or want to add more models or datasets, feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/ACM40960/project-shivsucd/blob/main/LICENSE) file for details.

## References
1. Arnold, M., Morgan, E., Rumgay, H., Mafra, A., Singh, D., Laversanne, M., Vignat, J., Gralow, J. R., Cardoso, F., Siesling, S. et al. (2022). Current and future burden of breast cancer: Global statistics for 2020 and 2040, The Breast 66: 15–23.

2. Huang, J., Chan, P. S., Lok, V., Chen, X., Ding, H., Jin, Y., Yuan, J., Lao, X.-q., Zheng, Z.-J. and Wong, M. C. (2021). Global incidence and mortality of breast cancer: a trend analysis, Aging (Albany NY) 13(4): 5748. 

3. Hussain, S., Ali, M., Naseem, U., Nezhadmoghadam, F., Jatoi, M. A., Gulliver, T. A. and Tamez-Pe˜na, J. G. (2024). Breast cancer risk prediction using machine learning: a systematic review, Frontiers in Oncology 14: 1343627. 

4. Muller, F. M., Li, E. J., Daube-Witherspoon, M. E., Vanhove, C., Vandenberghe, S., Pantel, A. R. and Karp, J. S. (2024). Deep learning denoising for low-dose dual-tracer protocol with 18f-fgln and 18f-fdg in breast cancer imaging, Annual Meeting of the Society of Nuclear Medicine and Molecular Imaging.

5. FA Spanhol, LS Oliveira, C. Petitjean and L. Heutte, "A Dataset for Breast Cancer Histopathological Image Classification," in IEEE Transactions on Biomedical Engineering, vol. 63, no. 7, pp. 1455-1462, July 2016, doi: 10.1109 / TBME.2015.2496264.

6. Machine Learning Algorithms For Breast Cancer Prediction And Diagnosis: Mohammed Amine Naji , Sanaa El Filali , Kawtar Aarika , EL Habib Benlahmar , Rachida Ait Abdelouhahid , Olivier Debauche.

## Data Sources

- [Wisconsin Breast Cancer Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)
- [BreaKHis 400X Dataset](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/)

## Authors

- **Shivani - 23200782**: MSc Data and Computational Science, UCD, Dublin

## Demonstration

- A Streamlit application is deployed that allows you to perform Image Analysis and Cell Nuclei Analysis.
    - To perform Image Analysis, **double click** on **Image Analysis Button** and it will take you to the Image upload section.
    - Download test images from : [Images](https://github.com/ACM40960/project-shivsucd/tree/main/Prediction%20of%20Breast%20Cancer%20Incidence-%20Shivani-23200782/Dataset/BreaKHis%20400X/test_for_upload).
    - Upload the benign or malignant image from the Test folder above, and you will see the prediction at the bottom of the page.
    - Scroll back up to the page, and click on the **X** button just below the upload button to erase the uploaded image, and upload more images, or double click on the home button to return home.
    - To Perform Cell Nuclei Measurements predictions on the Wisconsin dataset, **double click** on **Cell Nuclei Analysis**. Use the slider to slide and adjust the value of the measurements shown, and observe the change in the prediction on the page. The model being used in the Random Forest Model, thus selected as the best model, upon analysis.
 
- Please find the link for the Streamlit Application to test the functionality here: [Prediction of Breast Cancer Incidence](https://breast-cancer-incidence-prediction-shivs-ucd.streamlit.app)

## Poster

Please review the poster presented at the University for the Poster Presentation here:[Poster](https://github.com/ACM40960/project-shivsucd/tree/main/Prediction%20of%20Breast%20Cancer%20Incidence-%20Shivani-23200782/images/Final_Poster_23200782.pdf).


