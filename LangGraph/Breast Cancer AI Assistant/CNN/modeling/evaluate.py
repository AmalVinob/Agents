import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

def display_sample_images(X, y, class_names, num=6):
    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    for i, ax in enumerate(axes.flat):
        ax.imshow(X[i])
        ax.set_title(class_names[y[i]])
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def evaluate_model(model, X_test, y_test):
    y_pred = (model.predict(X_test) > 0.5).astype('int32')
    print(classification_report(y_test, y_pred))