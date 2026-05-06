
---

# Breast Cancer Wisconsin Statistical Analysis

## Overview

This project performs a comprehensive statistical analysis on the Breast Cancer Wisconsin (Diagnostic) dataset to identify patterns, relationships, and key features that differentiate malignant and benign tumors.

The analysis focuses on applying statistical methods, feature engineering, and multiple programming paradigms to extract meaningful insights that can support medical diagnosis.

---

## Objective

* Analyze breast cancer dataset using statistical techniques
* Identify features that distinguish malignant and benign tumors
* Implement multiple programming approaches (iterative, functional, recursive)
* Develop a simple risk scoring system for classification

---

## Dataset Information

* Total Samples: 569
* Total Features: 32 (1 ID, 1 diagnosis, 30 numerical features)
* Source: UCI Machine Learning Repository / Kaggle
* Data Type: Digitized images of breast mass cell nuclei

### Diagnosis Distribution

* Malignant (M): 212 (37.3%)
* Benign (B): 357 (62.7%)

---

## Feature Categories

The dataset includes three groups of features:

### Mean Values

Average measurements of cell nuclei such as radius, texture, perimeter, and area.

### Standard Error Values

Represents variability in measurements.

### Worst Values

Largest observed values for each feature.

---

## Methodology

### 1. File Handling

* Implemented exception handling using try-except
* Managed errors such as file not found and permission issues
* Ensured safe and robust data loading

### 2. Data Exploration

* Inspected dataset structure and data types
* Identified patterns using conditional filtering
* Observed that larger radius and area are associated with malignancy

### 3. Missing Value Handling

* Checked for null values using pandas
* Dataset contained no missing values
* Demonstrated manual mean calculation for understanding

### 4. Feature Engineering

Created new features to improve analysis:

* Worst_Area_Mean = (area_mean + area_worst) / 2
* Radius_Texture_Index = radius_mean × texture_mean
* MalignancyRiskScore based on threshold conditions

### 5. Threshold Analysis

* Identified samples exceeding average radius and area
* Found strong correlation with malignant cases

### 6. Custom Statistical Calculations

* Implemented mean and median using reduce()
* Compared results with manual loops
* Verified accuracy with minimal deviation

### 7. Recursive Computation

* Designed recursive function to approximate fractal dimension
* Demonstrated recursion concepts and limitations

### 8. Statistical Analysis

* Calculated mean, median, mode, standard deviation
* Compared malignant vs benign distributions
* Identified significant differences across features

### 9. Report Generation

* Generated a structured text report
* Included key findings and feature comparisons

---

## Key Findings

### Most Important Features

* concavity_mean
* concave_points_mean
* area_worst
* radius_worst
* perimeter_mean

### Observations

* Malignant tumors are larger and more irregular
* Concavity-related features are the strongest indicators
* Significant statistical differences exist between classes

---

## Engineered Feature Performance

### Malignancy Risk Score

* Sensitivity: 92%

* Specificity: 88.5%

* Accuracy: ~90%

* Score ≥ 7 indicates high risk

* Score ≤ 5 indicates low risk

---

## Results Summary

* Malignant tumors show higher values across most features
* Strong correlations found between size-related attributes
* Statistical methods successfully distinguish tumor types

---

## Clinical Relevance

* Helps in early detection of breast cancer
* Supports development of diagnostic tools
* Enables feature-based screening systems

---

## Limitations

* Dataset is relatively old (1992–1995)
* Limited to binary classification (benign/malignant)
* No patient demographic or clinical data included

---

## Future Enhancements

* Apply machine learning models (SVM, Random Forest)
* Use deep learning with original images
* Perform multi-institutional data analysis
* Build real-time prediction system

---

## Technologies Used

* Python
* pandas
* numpy
* functools
* collections

---

## Skills Demonstrated

* Data analysis and preprocessing
* Feature engineering
* Statistical computation
* Functional and recursive programming
* Report generation

---

## Project Structure

```
├── data.csv
├── main.py
├── breast_cancer_analysis_report.txt
└── README.md
```

---

## Output

* Statistical analysis results
* Feature comparison insights
* Generated report file

---

## Conclusion

This project demonstrates that statistical analysis of cell nucleus features can effectively differentiate malignant and benign tumors. The results highlight the importance of feature selection and simple scoring systems in medical diagnosis.

---

## References

* UCI Machine Learning Repository
* Kaggle Breast Cancer Dataset
* Wolberg et al. (1995) Research Paper
* Python Documentation (pandas, numpy)

---

