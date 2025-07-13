# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
### Model Type: Supervised binary classifier
### Algorithms: Random Forest Classifier
### Framework: Scikit-learn
### Input Features: Demographic and employment-related fields (categorical and numerical)
### Output: Binary labels indicating is income is greater than or equal to 50k or less than 50k
### Hyperparamters: n_estimators = 100, random_state = 42

## Intended Use
### This model is intended to classify whether an individual earns more than $50K per year based on demographic and occupational data. The purpose is for educational and exploratory analysis. It is not intended for production use in deciosion-making activities. 

## Training Data
### Source: Census data from "data/census.csv"
### Preprocessing: Applied one-hot encoding to categorical features and label binarization to the target
### Split: 80% training / 20% test using stratified sampling

## Evaluation Data
### Evaluation performed on the 20% test set
### Stratified sampling used to preserve the income distribution
### Same preprocessing used on test data as training data via saved encoding

## Metrics
### Precision: 0.85
### Recall: 0.78
### F1 Score: 0.81
### Slice metrics: Model performamce evaluated across data sliced for categorical features

## Ethical Considerations
### Bias and Fairness: Income prediction models can reflect and amplify historical biases like gender, race, and educational access. Consideration to these factors should help inform model interpretation and model deployment
### Trasparency: Code and preprocessing should be accessible for audit and review
### Privacy: Dataset is publically available and anonymized. No personal data was used in the training or test model.

## Caveats and Recommendations
### Not suitable for decision making activities without consideration and review of potential biases and fairness audits
### Performance may degrade outside of the U.S census data distribution
### Model interpretation may be improved by methods like SHAP for feature attribution
