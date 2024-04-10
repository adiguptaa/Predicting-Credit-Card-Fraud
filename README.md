# Predicting-Credit-Card-Fraud

## Objectives

This project aims to develop a machine learning model capable of identifying fraudulent transactions. Given the challenges posed by imbalanced datasets typical in fraud detection, we employ various data preprocessing techniques, including resampling and feature scaling, and explore different models to improve prediction accuracy and recall.

## Preprocessing Steps

### Data Cleaning

The initial step in preparing our dataset for modeling involved a comprehensive data cleaning process, with a focus on handling both categorical and numerical data effectively. Here's a breakdown of our approach:

1. **Separate the Dataset**: We started by dividing our dataset into two distinct parts:
    - **Categorical Data**: Columns containing non-numeric values, representing various categories.
    - **Numerical Data**: Columns with numeric values, representing measurable quantities.

2. **Encode Categorical Data**: To make our categorical data more meaningful and easier for our models to understand, we applied frequency encoding. This process involved:
    - Creating a `frequency_map` for each categorical column, which maps each category to its frequency of occurrence within the dataset.
    - Using this map, we transformed each category into its corresponding frequency value, effectively encoding our categorical data in a way that highlights the prevalence of each category.

3. **Recombine the Data**: After encoding the categorical data and retaining our numerical data as-is, we concatenated both parts back into a single DataFrame. This recombined dataset, now with frequency-encoded categorical data and untouched numerical data, was used for all subsequent analyses and model training.

This data cleaning and preprocessing approach allowed us to maintain the richness of our dataset's categorical information in a numerical format that's more suitable for machine learning models, while also preserving the integrity of our numerical data.

### Feature Engineering

In addition to our initial data cleaning and preprocessing steps, we employed Principal Component Analysis (PCA) as a key feature engineering technique to enhance Logistic Regression model's performance further. Here's how PCA contributed to our project:

- **Dimensionality Reduction**: PCA was applied to reduce the high dimensionality of our dataset, transforming the original features into a smaller set of principal components that capture the most significant variance and patterns in the data.

- **Improving Model Efficiency and Performance**: By focusing on principal components, we were able to streamline our models, making them faster and often more accurate, as they could now learn from the most informative aspects of the data without being distracted by noise or irrelevant features.

- **Visualization**: PCA also facilitated a better understanding and visualization of the data distribution, enabling us to observe clustering patterns among the transformed features that were not apparent in the original high-dimensional space.

This step was crucial in our preprocessing pipeline, allowing us to build more robust models that are better suited to predicting fraudulent transactions while efficiently managing computational resources.

### Resampling with SMOTEENN

Given the inherent class imbalance typical in fraud detection scenarios, we employed SMOTEENN (a combination of Synthetic Minority Over-sampling Technique and Edited Nearest Neighbors) for resampling our dataset. This approach not only addresses the imbalance by oversampling the minority class and undersampling the majority class but also cleans the synthetic samples to remove any that are likely to be misclassified. This step was crucial in creating a more balanced dataset that allows our models to better learn and generalize from both classes.

### Feature Scaling

Post-resampling, it was essential to standardize the feature scales to ensure that our machine learning models could interpret and learn from the data effectively. We utilized `StandardScaler` from scikit-learn, which standardizes features by removing the mean and scaling to unit variance. This preprocessing step is particularly important for models that are sensitive to the magnitude of input features, such as Logistic Regression and Support Vector Machines, ensuring that all features contribute equally to the model's prediction.

```python
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTEENN

# Resampling with SMOTEENN
smote_enn = SMOTEENN(random_state=42)
X_resampled, y_resampled = smote_enn.fit_resample(X, y)

# Scaling the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_resampled)
```

## Model Development

Throughout this project, we explored various machine learning models to tackle the challenge of fraud detection, leveraging the powerful libraries provided by scikit-learn for implementation and evaluation. Below is an overview of the models we experimented with, along with the rationale and methodology behind each choice.

### Logistic Regression

Initially, we applied **Logistic Regression**, a widely used linear model for binary classification tasks, to establish a baseline performance. This step allowed us to gauge the model's capability to distinguish between fraudulent and non-fraudulent transactions based on the original feature set without any dimensionality reduction.

#### Libraries Used:
- scikit-learn for model training (`LogisticRegression` class) and evaluation metrics.

### Logistic Regression with PCA

To investigate the impact of dimensionality reduction on model performance, we subsequently applied **PCA (Principal Component Analysis)** to our dataset before training the Logistic Regression model. This approach aimed to reduce the complexity of the data, potentially enhancing model training efficiency and effectiveness by focusing on the most informative aspects of the data.

#### Libraries Used:
- scikit-learn for PCA implementation (`PCA` class) and Logistic Regression.

### Random Forest Model

Seeking to leverage the benefits of ensemble learning, we then explored the **Random Forest** model. Known for its robustness and ability to handle imbalanced datasets, the Random Forest model offered a more sophisticated approach compared to the simplicity of Logistic Regression, potentially providing higher accuracy and better generalization.

#### Libraries Used:
- scikit-learn for model training (`RandomForestClassifier` class) and evaluation metrics.

### Hyperparameter Tuning with Grid Search

To further optimize our model's performance, specifically the Random Forest classifier, we employed **Grid Search CV** for hyperparameter tuning. This exhaustive search over specified parameter values aimed to identify the most effective combination of parameters that results in the best model performance.

#### Libraries Used:
- scikit-learn for Grid Search CV (`GridSearchCV` class) and model evaluation.

### Evaluation Framework

Each model was rigorously evaluated using a consistent set of metrics, including accuracy, precision, recall, and F1-score, allowing us to compare their performance directly. This systematic approach to model development and evaluation ensured that our findings were reliable and actionable.

By experimenting with different models and applying both preprocessing techniques like PCA and advanced methods like hyperparameter tuning, we aimed to comprehensively explore the solution space for fraud detection. The utilization of scikit-learn throughout provided a robust and flexible framework for model training, evaluation, and optimization.

## Model Evaluation

A thorough and methodical evaluation process is crucial in assessing the effectiveness of machine learning models, especially in applications as critical as fraud detection. Our evaluation strategy employed a combination of cross-validation techniques and a comprehensive set of performance metrics to ensure the reliability and robustness of our models.

### Evaluation Methodology

**Performance Metrics**: Understanding the nuances of model performance in the context of fraud detection required us to look beyond mere accuracy. We employed several key metrics to gain a comprehensive view of each model's effectiveness:
   - **Accuracy**: The proportion of total predictions that were correct.
   - **Precision (Positive Predictive Value)**: Of all transactions predicted as fraudulent, the percentage that were correctly identified.
   - **Recall (Sensitivity)**: Of all actual fraudulent transactions, the percentage that were correctly identified by the model.
   - **F1 Score**: A harmonic mean of precision and recall, providing a single metric to assess the balance between them.
   - **ROC-AUC**: The Area Under the Receiver Operating Characteristic Curve, measuring the model's ability to distinguish between classes.

### Model Performance

- **Logistic Regression without PCA**: Served as our baseline model. While offering decent accuracy, it struggled with recall, indicating a potential issue with identifying the minority class (fraudulent transactions).
  
- **Logistic Regression with PCA**: Application of PCA slightly improved model efficiency by reducing features, but the impact on overall recall and precision was minimal, suggesting that dimensionality reduction did not significantly enhance our ability to detect fraud within this model framework.

- **Random Forest Model**: Showed a substantial improvement in recall and precision compared to the Logistic Regression models. The inherent ability of Random Forest to manage imbalances and capture complex patterns made it more adept at identifying fraudulent transactions.

- **Random Forest with Tuned Hyperparameters**: The application of GridSearchCV to fine-tune the Random Forest model's hyperparameters further improved performance. This model demonstrated the highest F1 score, recall, and precision, underlining the value of hyperparameter tuning in optimizing model outcomes.

### Discussion

The evaluation process underscored the importance of considering a range of metrics to truly understand model performance, especially in imbalanced datasets like those common in fraud detection. While accuracy remained high across all models, the nuanced improvements in recall, precision, and F1 scores were pivotal in selecting the best model for our needs.

The superiority of the tuned Random Forest model highlights the critical role of hyperparameter optimization in machine learning workflows, particularly for complex, imbalanced datasets. Future efforts will focus on exploring additional ensemble methods and advanced anomaly detection algorithms to further enhance our ability to accurately identify fraudulent activities.


## Comparison of All Models

- Provide a comparison of all the models tested.
- Include metrics like accuracy, precision, recall, F1-score, etc., to support the comparison.

## Best Model Parameters

```plaintext
Best model: [Model Name]
Parameters:
- n_estimators: 200
- max_depth: None
- min_samples_split: 5
- min_samples_leaf: 1
