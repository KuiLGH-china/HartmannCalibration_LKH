import sys
import numpy as np
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget, QFileDialog

# 导入计算函数
from main import final_calculate

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Calculate Lx and Ly")

        # 创建主布局
        layout = QVBoxLayout()

        # 创建标签和输入框
        self.label_img_path1 = QLabel("Image Path 1:")
        layout.addWidget(self.label_img_path1)
        self.entry_img_path1 = QLineEdit()
        layout.addWidget(self.entry_img_path1)
        self.button_browse1 = QPushButton("Browse")
        self.button_browse1.clicked.connect(self.browse_img1)
        layout.addWidget(self.button_browse1)

        self.label_img_path2 = QLabel("Image Path 2:")
        layout.addWidget(self.label_img_path2)
        self.entry_img_path2 = QLineEdit()
        layout.addWidget(self.entry_img_path2)
        self.button_browse2 = QPushButton("Browse")
        self.button_browse2.clicked.connect(self.browse_img2)
        layout.addWidget(self.button_browse2)

        self.label_param1 = QLabel("光源往x方向上移动位移:")
        layout.addWidget(self.label_param1)
        self.entry_param1 = QLineEdit()
        layout.addWidget(self.entry_param1)

        self.label_param2 = QLabel("光源往y方向上移动位移:")
        layout.addWidget(self.label_param2)
        self.entry_param2 = QLineEdit()
        layout.addWidget(self.entry_param2)

        self.label_param3 = QLabel("透镜焦距:")
        layout.addWidget(self.label_param3)
        self.entry_param3 = QLineEdit()
        layout.addWidget(self.entry_param3)

        # 创建按钮
        self.button_calculate = QPushButton("Calculate")
        self.button_calculate.clicked.connect(self.calculate_and_show_result)
        layout.addWidget(self.button_calculate)

        # 设置主布局
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def browse_img1(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image 1")
        if file_path:
            self.entry_img_path1.setText(file_path)

    def browse_img2(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image 2")
        if file_path:
            self.entry_img_path2.setText(file_path)

    def calculate_and_show_result(self):
        # 获取输入的图片路径和参数
        img_path1 = self.entry_img_path1.text()
        img_path2 = self.entry_img_path2.text()
        param1 = float(self.entry_param1.text())
        param2 = float(self.entry_param2.text())
        param3 = int(self.entry_param3.text())

        # 调用函数计算结果
        Lx, Ly = final_calculate(img_path1, img_path2, param1, param2, param3)

        self.show_result(Lx, Ly)

    def show_result(self, Lx, Ly):
        # 将 NaN 值替换为 0
        Lx = np.nan_to_num(Lx)
        Ly = np.nan_to_num(Ly)

        # 转换图像数据类型以适应 OpenCV 显示
        Lx_display = (Lx * 255).astype('uint8')
        Ly_display = (Ly * 255).astype('uint8')

        # 显示 Lx
        cv2.imshow('Lx', Lx_display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 显示 Ly
        cv2.imshow('Ly', Ly_display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        self.calculate_and_display(Lx, Ly)

    def calculate_and_display(self, Lx, Ly):
        # 创建新窗口
        self.result_window = ResultWindow(Lx, Ly)
        self.result_window.show()

class ResultWindow(QWidget):
    def __init__(self, Lx, Ly):
        super().__init__()
        self.setWindowTitle("计算结果")

        layout = QVBoxLayout()

        # 计算并显示光源在 x 方向上移动时 LH 的计算量和各个分量
        label1 = QLabel("光源在x方向上移动时，LH的计算量")
        layout.addWidget(label1)
        average_Lx = np.mean(Lx)
        label2 = QLabel(f"Lx的平均值: {average_Lx}")
        layout.addWidget(label2)
        label3 = QLabel("Lx的各个分量:")
        layout.addWidget(label3)
        label4 = QLabel(str(Lx))
        layout.addWidget(label4)

        # 计算并显示光源在 y 方向上移动时 LH 的计算量和各个分量
        label5 = QLabel("光源在y方向上移动时，LH的计算量")
        layout.addWidget(label5)
        average_Ly = np.mean(Ly)
        label6 = QLabel(f"Ly的平均值: {average_Ly}")
        layout.addWidget(label6)
        label7 = QLabel("Ly的各个分量:")
        layout.addWidget(label7)
        label8 = QLabel(str(Ly))
        layout.addWidget(label8)

        self.setLayout(layout)

def main():
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
