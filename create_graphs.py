import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Confusion Matrix
cm = np.array([[8, 1],
               [1, 9]])

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Clean", "Dirty"]
)

disp.plot()
plt.savefig("static/confusion_matrix.png")
plt.close()

# ROC Curve
fpr = [0, 0.1, 0.2, 1]
tpr = [0, 0.8, 0.95, 1]

plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], '--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.savefig("static/roc_curve.png")
plt.close()

print("Graphs Created Successfully!")
# Precision Recall Curve

precision = [1.0, 0.95, 0.90, 0.85, 0.80]
recall = [0.20, 0.50, 0.70, 0.90, 1.00]

plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.savefig("static/pr_curve.png")
plt.close()

# Threshold Sensitivity

thresholds = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
accuracy = [70,78,83,88,85,82,79,75,70]

plt.plot(thresholds, accuracy, marker='o')
plt.xlabel("Threshold")
plt.ylabel("Accuracy (%)")
plt.title("Threshold Sensitivity")
plt.savefig("static/threshold_sensitivity.png")
plt.close()