
from dataloader import load_dataset
from model import build_model
from train import train_model
from evaluate import display_sample_images, evaluate_model

if __name__ == '__main__':
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_dataset()

    print("Displaying samples...")
    display_sample_images(X_train, y_train, ['BENIGN', 'MALIGNANT'])

    print("Building model...")
    model = build_model()

    print("Training...")
    history = train_model(model, X_train, y_train, X_test, y_test)

    print("Evaluating...")
    evaluate_model(model, X_test, y_test)