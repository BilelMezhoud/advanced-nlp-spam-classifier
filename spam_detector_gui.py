
import sys
import os
from PyQt5 import QtWidgets, QtGui, QtCore
from spam_detection_app import preprocess_text, vectorize_text, predict_spam


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        # Set the window title and size
        self.setWindowTitle("Spam Detector")
        self.setFixedSize(800, 500)
        self.setWindowIcon(QtGui.QIcon("icon.png"))

        # Create a label for the text box
        self.label = QtWidgets.QLabel("Enter your message:")
        self.label.setStyleSheet("font-size: 20px; color: #3d3d3d; font-weight: bold;")

        # Create a text box for entering the message
        self.textbox = QtWidgets.QTextEdit()
        self.textbox.setStyleSheet("font-size: 16px;")

        # Create a combo box for selecting the model
        self.combobox = QtWidgets.QComboBox()
        self.combobox.addItem("Word2Vec + SVM")
        self.combobox.addItem("FastText + SVM")
        self.combobox.addItem("GloVe + SVM")
        self.combobox.addItem("BERT + SVM")
        self.combobox.setStyleSheet("font-size: 16px;")

        # Create a button for predicting the message
        self.button = QtWidgets.QPushButton("Predict")
        self.button.setStyleSheet(
            "font-size: 16px; background-color: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 5px;"
        )
        self.button.clicked.connect(self.predict_message)

        # Create a label for displaying the prediction result
        self.result_label = QtWidgets.QLabel()
        self.result_label.setStyleSheet("font-size: 20px;")

        # Load the image
        image = QtGui.QImage("spammer-icon.png")
        if image.isNull():
            print("Error loading image")
        else:
            image = image.scaled(350, 350, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            self.image_label = QtWidgets.QLabel(self)
            self.image_label.setPixmap(QtGui.QPixmap.fromImage(image))
            self.image_label.setAlignment(QtCore.Qt.AlignCenter)

        # Layouts
        input_layout = QtWidgets.QHBoxLayout()
        input_layout.addWidget(self.label)
        input_layout.addWidget(self.textbox)
        input_layout.addWidget(self.combobox)

        output_layout = QtWidgets.QHBoxLayout()
        output_layout.addWidget(self.button)
        output_layout.addWidget(self.result_label)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addStretch()
        layout.addLayout(input_layout)
        layout.addLayout(output_layout)

        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)
        self.setStyleSheet("background-color: #f0f0f0;")

    def predict_message(self):
        selected_model = self.combobox.currentText()
        message = self.textbox.toPlainText()
        prediction = predict_spam(message, selected_model)

        if prediction == 1:
            self.result_label.setText("Prediction: Spam")
            self.result_label.setStyleSheet("color: red; font-size: 20px; font-weight: bold;")
        else:
            self.result_label.setText("Prediction: Not Spam")
            self.result_label.setStyleSheet("color: green; font-size: 20px;")
            print("Prediction: Not Spam")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
