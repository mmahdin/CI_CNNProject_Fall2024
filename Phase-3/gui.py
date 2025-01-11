from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *
import sys

# CSS-like button styles for various functionalities
button_style_r = """
    QPushButton {
        border: 2px solid #8f8f91;
        border-radius: 10px;
        min-width: 50px;
        font-size: 24px;
        border-image: url(./imgs/play2b3.png);
    }
"""

button_style_si = """
    QPushButton {
        border: 2px solid #8f8f91;
        border-radius: 10px;
        min-width: 50px;
        font-size: 24px;
        border-image: url(./imgs/capb.png);
    }
    
    QPushButton:pressed {
        border-image: url(./imgs/capb2.png);
    }
"""

button_style_sv = """
    QPushButton {
        border: 2px solid #8f8f91;
        border-radius: 10px;
        min-width: 50px;
        font-size: 24px;
        border-image: url(./imgs/mic2b2.png);
    }
"""

button_style_sav = """
    QPushButton {
        border: 2px solid #8f8f91;
        border-radius: 10px;
        min-width: 50px;
        font-size: 24px;
        border-image: url(./imgs/dirb.png);
    }
    
    QPushButton:pressed {
        border-image: url(./imgs/dirb2.png);
    }
"""

button_style_send = """
    QPushButton {
        border: 2px solid #8f8f91;
        border-radius: 10px;
        min-width: 50px;
        font-size: 24px;
        border-image: url(./imgs/sv7.png);
    }
    
    QPushButton:pressed {
        border-image: url(./imgs/sv7b.png);
    }
"""

button_style_sub = """
    QPushButton {
        border: 2px solid #8f8f91;
        border-radius: 10px;
        font-size: 16px;
        border-radius: 10px;
        color: #2eff04;
        border: 1px solid #2eff04;
    }
    
    QPushButton:pressed {
        border-radius: 12px;
        color: #fffd0a;
        border: 1px solid #fffd0a;
    }
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set size of window
        self.setWindowTitle("TCP connection")
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.setFixedSize(1050, 670)

        # Create a central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Set background color for the window
        self.setStyleSheet("background-color: #000100;")

        # Create GUI layout
        self.create_layout()

    def create_layout(self):
        # Create main layout for central widget
        main_layout = QVBoxLayout()
        self.central_widget.setLayout(main_layout)

        # IP layout for entering IP address and port
        ip_widget = QWidget(self)
        ip_layout = QHBoxLayout()
        ip_layout.setObjectName("ipLayout")  # Set object name for styling

        # Label for displaying "IP : Port"
        desc = QLabel("IP : Port")
        desc.setStyleSheet(
            "color: #e500e8 ;background-color: None;font-size:20px;")

        # Line edit for entering IP address
        self.ip_entry1 = QLineEdit()
        self.ip_entry1.setPlaceholderText("  IP")
        self.ip_entry1.setStyleSheet(
            "color: white; background-color: black; border: 1px solid #ed1ee7; border-radius: 5px;")

        # Line edit for entering port number
        self.ip_entry2 = QLineEdit()
        self.ip_entry2.setStyleSheet(
            "color: white; background-color: black; border: 1px solid #ed1ee7; border-radius: 5px;")
        self.ip_entry2.setPlaceholderText("  Port")

        # Button for connecting
        ip_button = QPushButton("Connect")
        ip_button.setStyleSheet(button_style_sub)
        ip_button.clicked.connect(self.connect)

        # Set fixed sizes for the widgets
        desc.setFixedSize(80, 30)
        self.ip_entry1.setFixedSize(150, 30)
        self.ip_entry2.setFixedSize(70, 30)
        ip_button.setFixedSize(100, 30)

        # Add widgets to IP layout
        ip_layout.addWidget(desc, 1)
        ip_layout.addWidget(self.ip_entry1, 4)
        ip_layout.addWidget(self.ip_entry2, 1)
        ip_layout.addWidget(ip_button, 1)

        # Align IP layout horizontally
        ip_layout.setAlignment(Qt.AlignHCenter)

        # Set IP layout to IP widget
        ip_widget.setLayout(ip_layout)

        # Add IP widget to main layout
        main_layout.addWidget(ip_widget, 1)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
