# Bank-Marketing-prediction
- By : **ABeer Al-Zebda | Machine Learning Engineer**
<img width="1024" height="536" alt="Digital-marketing-for-banking-industry-1024x536" src="https://github.com/user-attachments/assets/b76ac95d-bf09-44ca-ab38-9857354401e3" />

## Project Overview

This project analyzes a real-world dataset from direct marketing campaigns conducted by a Portuguese banking institution, where clients were contacted via phone calls to determine whether they would subscribe to a bank term deposit (yes or no). The goal was to build and evaluate advanced machine learning models capable of predicting client subscription decisions.

The analysis began with comprehensive **exploratory data analysis (EDA)**, visually examining feature distributions and their relationships with the subscription outcome. Key findings identified **call duration** and **number of campaign contacts** as the strongest predictors, while age and macroeconomic indicators showed minimal influence. Demographic patterns revealed university degree holders, single clients, retired individuals, and students as more likely subscribers.

**Four powerful classifiers** were implemented and rigorously evaluated—**Logistic Regression**, **Random Forest**, **XGBoost**, and **CatBoost**—each trained both **with and without feature scaling** using a structured **ColumnTransformer** and **Pipeline** for reproducible preprocessing.

All models achieved high test accuracy, though the dataset's significant **class imbalance** (few subscribers vs many non-subscribers) caused bias toward the majority class. To address this, **threshold tuning** was applied across all models to optimize the decision boundary, balancing precision and recall for better sensitivity toward identifying potential subscribers.

**XGBoost and CatBoost** emerged as top performers, demonstrating the most robust handling of the imbalanced data while maintaining excellent overall performance.

This pipeline showcases production-ready machine learning capabilities for optimizing marketing campaign targeting and client segmentation.
# Data Set
- The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed.

- Data Link : https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing
  
# Clean The data
- There was'nt any missing values
- Drop the duplicates

# Explore Data
<img width="759" height="553" alt="Screenshot 2026-04-17 082249" src="https://github.com/user-attachments/assets/7ce7d22a-7ebb-4aa2-84e1-ab5f86a8abc3" />

- The scatter plot reveals a clear negative relationship between the number of campaign contacts and call duration — clients contacted fewer times (1–5 contacts) show a wide spread of call durations, including very long calls exceeding 4,000 seconds, while those contacted more frequently tend to have progressively shorter and more uniform calls, suggesting that repeated outreach yields diminishing engagement from the client.
<img width="731" height="536" alt="image" src="https://github.com/user-attachments/assets/03e7aa24-b22a-4111-8d5c-8db442154ee6" />

-The last contact duration, in seconds (numeric) has a higher average for clients who subscribed a term deposit compared to those who did not , suggesting that this feature has some influence on the subscription decision
<img width="755" height="576" alt="Screenshot 2026-04-17 083453" src="https://github.com/user-attachments/assets/669b1ca1-2ef3-41e8-9b9a-880720063f65" />

- The most frequent education level for both clients who subscribed and those who did not is the university degree, and while the ranking of education levels is broadly similar across both groups, university degree holders appear slightly more represented among subscribers, suggesting that higher education may have a weak positive influence on the subscription decision.

# Preprocessing
- Prepare input features and target again
- Split the data using scikit-learn
- OneHotEncoder for categorical Features
- StandardScaler for numerical features
- ColumnTransformer
- Preprocessing without Scaling

# Building LogisticRegression Model
<img width="331" height="86" alt="image" src="https://github.com/user-attachments/assets/3a3dfd66-244f-4a67-bf92-461f5652791b" />

# Evaluate LogisticRegression Model
- Training accuracy: 0.9119
- Testing accuracy : 0.9113
  
  <img width="506" height="245" alt="image" src="https://github.com/user-attachments/assets/2b6e806b-42a9-4630-9c2e-03741c89572e" />

# Interpret the results for LogisticRegression Model
- Class 1 — "yes" (client subscribed)

   - Precision 0.67 → When the model predicts "yes", it's correct 67% of the time (33% are False Positives — predicted "yes" but actually "no")
   - Recall 0.43 → The model correctly catches 43% of all actual "yes" clients, missing 57% → those missed are False Negatives (predicted "no" but actually "yes")
- Class 0 — "no" (client did NOT subscribe)
   - Precision 0.93 → When the model predicts "no", it's correct 93% of the time (7% are False Positives — predicted "no" but actually "yes")
   - Recall 0.97 → The model correctly catches 97% of all actual "no" clients, missing only 3% → those missed are False Negatives (predicted "yes" but actually "no")

# Evaluate LogisticRegression Model without Scaling
- Training accuracy: 0.9103
- Testing accuracy : 0.9116

  <img width="527" height="255" alt="image" src="https://github.com/user-attachments/assets/ead96d77-eb81-4bbb-adef-6dc9dbfa0857" />

**The logistic regression model trained on scaled data achieved slightly higher accuracy compared to the unscaled version, which is expected behaviorexpected behavior since Logistic Regression is scale-sensitivescale-sensitive (stable coefficients/gradients). However, the overall performance remained nearly the same because:
Scaling provides modest gains for linear models when features aren't extremely skewed
Class imbalance affects BOTH versions equally - it doesn't explain the scaling differenc
Accuracy is misleading with imbalanced data regardless of scaling**

# Building RandomForestRegressor Model
<img width="382" height="98" alt="image" src="https://github.com/user-attachments/assets/828480ec-f7bf-43de-a04e-8f4d713fdef1" />

# Evaluate RandomForestRegressor Model
- Training accuracy: 1.0
- Testing accuracy : 0.9137
  
 <img width="505" height="245" alt="image" src="https://github.com/user-attachments/assets/206f913a-2776-4c00-af61-5deb274b55d7" />

# Evaluate RandomForestRegressor Model without Scaling
- Training accuracy: 1.0
- Testing accuracy : 0.9141
  
<img width="527" height="250" alt="image" src="https://github.com/user-attachments/assets/1d5a428f-1417-4eaf-8ff5-04d91c9d5766" />

**We can see That The accuracy increase a little bit , but overall the results for Random Forest with Scaling or Without is nearly the same efficiency**

# Change the threshold
## threshold = 0.3
<img width="557" height="242" alt="image" src="https://github.com/user-attachments/assets/92862e52-e04f-4cf7-b1c7-2f198ac54f66" />

## threshold = 0.7
<img width="535" height="252" alt="image" src="https://github.com/user-attachments/assets/a9b0e5d9-3d42-4029-a910-0699791e494f" />

**Threshold 0.5 gives better balance between catching 'yes' cases (higher recall) and maintaining precision. Macro F1 +11% improvement confirms it's superior for imbalanced data!**

# Building XGBoost Model 
# Evaluate XGBoost Model 
<img width="622" height="262" alt="image" src="https://github.com/user-attachments/assets/c1cf7c82-f91f-452f-b5bc-715f6954e5e8" />


# Building CatBoost Model 
# Evaluate CatBoost Model
<img width="538" height="326" alt="image" src="https://github.com/user-attachments/assets/19e56092-74fd-4f63-9a79-ad18647715a2" />

# Predict on new Data
the result of new data :
- Probability: 0.0503
- Predicted  : no
- Actual     : no

# Conclusion 
## Conclusion

This project successfully developed and evaluated a comprehensive machine learning pipeline to predict bank term deposit subscriptions using direct marketing campaign data. After thorough data cleaning and exploratory data analysis, **call duration** and **campaign contact frequency** emerged as the most predictive features, while age and macroeconomic indicators showed limited discriminatory power.

Four robust classifiers were implemented—**Logistic Regression**, **Random Forest**, **XGBoost**, and **CatBoost**—with and without feature scaling. All models achieved impressive **test accuracies around 91%**, demonstrating consistent performance across algorithms:

| Model | Test Accuracy | Yes Recall | Macro F1 |
|-------|---------------|------------|----------|
| **CatBoost** | **91.48%** | **53%** | **77%** |
| **XGBoost** | **91.48%** | **53%** | **77%** |
| **Random Forest** | **91.00%** | **44%** | **74%** |
| **Logistic Regression** | **91.00%** | **41%** | **73%** |

**Key Insights:**
- **High accuracy (91%)** was achieved across all models, but **class imbalance** (11.8% "yes" vs 88.2% "no") revealed limitations—excellent "no" prediction (97% recall) vs moderate "yes" detection (41-53% recall)
- **CatBoost** and **XGBoost** performed best with **macro F1-scores of 77%** and identical confusion matrices, correctly identifying **516-518 actual subscribers**
- Default **threshold=0.5** proved optimal after tuning, balancing precision/recall for imbalanced data

The pipeline successfully generalized to unseen test data, correctly predicting outcomes despite class imbalance challenges. **Future enhancements** could include **SMOTE oversampling**, **class weight adjustment**, or **ensemble methods** to boost sensitivity for identifying potential subscribers while maintaining high overall accuracy.

**The project demonstrates production-ready ML capabilities for business decision-making in marketing campaign optimization.** 🚀
