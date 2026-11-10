# Numpy and Table Display
import numpy as np
import matplotlib.pyplot as plt

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score

# Scikit-learn Dimensionality Reduction
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler, LabelEncoder

# XGBoost
from xgboost import XGBClassifier

# Data
from heart_data import x_axis, y_axis


# Convert class labels to numeric values
encoder = LabelEncoder()
y = encoder.fit_transform(y_axis)

# Split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(
    x_axis, y, test_size=0.2, random_state=42, stratify=y
)

###################################################
# =================== XGBoost =================== #
###################################################

model = XGBClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict with base XGBoost model
y_pred = model.predict(X_test)

###################################################
# ================ PCA + XGBoost ================ #
###################################################

pca_model = make_pipeline(
    StandardScaler(),
    PCA(n_components=1),
    XGBClassifier(random_state=42)
)

# Automatically fits scaler, PCA, and XGBoost
pca_model.fit(X_train, y_train)

# Automatically transforms X_test and predicts
y_pred_pca = pca_model.predict(X_test)

###################################################
# ================ LDA + XGBoost ================ #
###################################################
lda_model = make_pipeline(
    StandardScaler(),
    LDA(n_components=1),
    XGBClassifier(random_state=42)
)

# Automatically fits scaler, LDA, and XGBoost
lda_model.fit(X_train, y_train)

# Automatically transforms X_test and predicts
y_pred_lda = lda_model.predict(X_test)

# Evaluate into a table
results = [
    [accuracy_score(y_test, y_pred),    accuracy_score(y_test, y_pred_pca),     accuracy_score(y_test, y_pred_lda)],
    [precision_score(y_test, y_pred),   precision_score(y_test, y_pred_pca),    precision_score(y_test, y_pred_lda)],
    [recall_score(y_test, y_pred),      recall_score(y_test, y_pred_pca),       recall_score(y_test, y_pred_lda)],
    [f1_score(y_test, y_pred),          f1_score(y_test, y_pred_pca),           f1_score(y_test, y_pred_lda)]
]

# Rounds the results to 4 decimal places.
results = [[round(value, 4) for value in row] for row in results]

columns = ( 'XGBoost', 'PCA', 'LDA')
rows = ['accuracy', 'precision', 'recall', 'f1_score']

# Get some pastel shades for the colors
colors = plt.cm.BuPu(np.linspace(0, 0.5, len(rows)))

# Create figure
fig, ax = plt.subplots(figsize=(10, 3))
ax.axis('off')

# Create table
the_table = ax.table(
    cellText=results,
    rowLabels=rows,
    rowColours=colors,
    colLabels=columns,
    cellLoc='center',
    loc='center'
)

# Format table
the_table.auto_set_font_size(False)
the_table.set_fontsize(12)
the_table.scale(1, 1.5)

plt.title('Metrics for XGBoost, PCA + XGBoost, and LDA + XGBoost', fontsize=14)

# Evaluate
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=['No', 'Yes'],
    cmap='Blues'
)

plt.show()
