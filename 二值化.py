import sys
import numpy as np
from PIL import Image
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QMessageBox, QSplitter)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QSize

class FloydSteinbergDithering:
    """实现Floyd-Steinberg抖动算法"""
    
    @staticmethod
    def rgb_to_grayscale(image_array):
        """将RGB图像转换为灰度图像"""
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            # 使用标准的灰度转换公式
            gray = np.dot(image_array[..., :3], [0.299, 0.587, 0.114])
            return gray.astype(np.float32)
        return image_array.astype(np.float32)
    
    @staticmethod
    def floyd_steinberg_dithering(image_path, output_grayscale=False):
        """
        应用Floyd-Steinberg抖动算法
        Args:
            image_path: 图片路径
            output_grayscale: 是否输出灰度图像
        Returns:
            PIL Image对象
        """
        # 打开图片
        img = Image.open(image_path)
        
        # 转换为RGB模式（如果是RGBA，去掉alpha通道）
        if img.mode == 'RGBA':
            rgb_img = Image.new('RGB', img.size, (255, 255, 255))
            rgb_img.paste(img, mask=img.split()[3])
            img = rgb_img
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        # 获取图片数据
        img_array = np.array(img).astype(np.float32)
        height, width, _ = img_array.shape
        
        # 转换为灰度
        gray_array = FloydSteinbergDithering.rgb_to_grayscale(img_array)
        
        # 创建输出数组
        if output_grayscale:
            output = np.zeros((height, width), dtype=np.uint8)
        else:
            output = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Floyd-Steinberg抖动算法
        for y in range(height):
            for x in range(width):
                old_pixel = gray_array[y, x]
                
                # 量化到0或255
                new_pixel = 0 if old_pixel < 128 else 255
                
                # 保存结果
                if output_grayscale:
                    output[y, x] = new_pixel
                else:
                    output[y, x] = [new_pixel, new_pixel, new_pixel]
                
                # 计算量化误差
                quant_error = old_pixel - new_pixel
                
                # 扩散误差到相邻像素
                if x + 1 < width:
                    gray_array[y, x + 1] += quant_error * 7 / 16
                if y + 1 < height:
                    gray_array[y + 1, x] += quant_error * 5 / 16
                    if x - 1 >= 0:
                        gray_array[y + 1, x - 1] += quant_error * 3 / 16
                    if x + 1 < width:
                        gray_array[y + 1, x + 1] += quant_error * 1 / 16
        
        # 转换为PIL图像
        if output_grayscale:
            result_img = Image.fromarray(output, mode='L')
        else:
            result_img = Image.fromarray(output, mode='RGB')
        
        return result_img

class DitheringApp(QMainWindow):
    """主应用窗口"""
    
    def __init__(self):
        super().__init__()
        self.original_image = None
        self.processed_image = None
        self.init_ui()
        
    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle('Floyd-Steinberg 抖动图片处理')
        self.setGeometry(100, 100, 1200, 700)
        
        # 创建中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        
        # 控制面板
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel)
        
        # 图片显示区域
        image_display = self.create_image_display()
        main_layout.addWidget(image_display, 1)
        
        # 状态标签
        self.status_label = QLabel("请选择一张图片进行处理")
        self.status_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.status_label)
    
    def create_control_panel(self):
        """创建控制面板"""
        control_widget = QWidget()
        layout = QHBoxLayout(control_widget)
        
        # 按钮
        self.select_btn = QPushButton("选择图片")
        self.select_btn.clicked.connect(self.select_image)
        self.select_btn.setMinimumHeight(40)
        
        self.process_btn = QPushButton("处理图片")
        self.process_btn.clicked.connect(self.process_image)
        self.process_btn.setMinimumHeight(40)
        self.process_btn.setEnabled(False)
        
        self.export_btn = QPushButton("导出图片")
        self.export_btn.clicked.connect(self.export_image)
        self.export_btn.setMinimumHeight(40)
        self.export_btn.setEnabled(False)
        
        # 添加按钮到布局
        layout.addWidget(self.select_btn)
        layout.addWidget(self.process_btn)
        layout.addWidget(self.export_btn)
        
        return control_widget
    
    def create_image_display(self):
        """创建图片显示区域"""
        splitter = QSplitter(Qt.Horizontal)
        
        # 原始图片显示
        self.original_label = QLabel("原始图片")
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setStyleSheet("border: 2px solid #ccc; background-color: #f0f0f0;")
        self.original_label.setMinimumSize(400, 400)
        
        # 处理后图片显示
        self.processed_label = QLabel("处理后图片")
        self.processed_label.setAlignment(Qt.AlignCenter)
        self.processed_label.setStyleSheet("border: 2px solid #ccc; background-color: #f0f0f0;")
        self.processed_label.setMinimumSize(400, 400)
        
        # 添加到分割器
        splitter.addWidget(self.original_label)
        splitter.addWidget(self.processed_label)
        
        return splitter
    
    def select_image(self):
        """选择图片"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "选择图片", 
            "", 
            "图片文件 (*.png *.jpg *.jpeg *.bmp *.gif *.tiff)"
        )
        
        if file_path:
            try:
                # 显示原始图片
                pixmap = QPixmap(file_path)
                if pixmap.isNull():
                    QMessageBox.warning(self, "错误", "无法加载图片文件")
                    return
                
                # 缩放图片以适应显示区域
                scaled_pixmap = pixmap.scaled(
                    self.original_label.size() - QSize(20, 20),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                
                self.original_label.setPixmap(scaled_pixmap)
                self.original_label.setText("")
                
                # 保存图片路径
                self.image_path = file_path
                self.process_btn.setEnabled(True)
                self.status_label.setText(f"已选择图片: {file_path.split('/')[-1]}")
                
            except Exception as e:
                QMessageBox.warning(self, "错误", f"加载图片失败: {str(e)}")
    
    def process_image(self):
        """处理图片"""
        if not hasattr(self, 'image_path'):
            QMessageBox.warning(self, "错误", "请先选择一张图片")
            return
        
        try:
            # 显示处理中状态
            self.status_label.setText("正在处理图片...")
            QApplication.processEvents()
            
            # 应用Floyd-Steinberg抖动算法
            result = FloydSteinbergDithering.floyd_steinberg_dithering(self.image_path, output_grayscale=False)
            self.processed_image = result
            
            # 将PIL图像转换为QPixmap
            if result.mode == 'RGB':
                # 转换为numpy数组
                data = result.tobytes("raw", "RGB")
                qimg = QImage(data, result.width, result.height, QImage.Format_RGB888)
            else:  # 灰度图像
                data = result.tobytes("raw", "L")
                qimg = QImage(data, result.width, result.height, QImage.Format_Grayscale8)
            
            pixmap = QPixmap.fromImage(qimg)
            
            # 缩放图片以适应显示区域
            scaled_pixmap = pixmap.scaled(
                self.processed_label.size() - QSize(20, 20),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            
            self.processed_label.setPixmap(scaled_pixmap)
            self.processed_label.setText("")
            
            # 启用导出按钮
            self.export_btn.setEnabled(True)
            self.status_label.setText("图片处理完成！")
            
        except Exception as e:
            QMessageBox.warning(self, "处理错误", f"图片处理失败: {str(e)}")
            self.status_label.setText("处理失败，请重试")
    
    def export_image(self):
        """导出图片"""
        if self.processed_image is None:
            QMessageBox.warning(self, "错误", "没有可导出的图片")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存图片",
            "dithering_result.png",
            "PNG图片 (*.png);;JPEG图片 (*.jpg *.jpeg);;BMP图片 (*.bmp)"
        )
        
        if file_path:
            try:
                # 根据文件扩展名选择保存格式
                if file_path.endswith('.jpg') or file_path.endswith('.jpeg'):
                    format = 'JPEG'
                elif file_path.endswith('.bmp'):
                    format = 'BMP'
                else:
                    format = 'PNG'
                    if not file_path.endswith('.png'):
                        file_path += '.png'
                
                # 保存图片
                self.processed_image.save(file_path, format)
                QMessageBox.information(self, "成功", f"图片已保存到: {file_path}")
                self.status_label.setText(f"图片已保存: {file_path}")
                
            except Exception as e:
                QMessageBox.warning(self, "保存错误", f"保存图片失败: {str(e)}")
    
    def resizeEvent(self, event):
        """窗口大小改变时重新调整图片大小"""
        super().resizeEvent(event)
        
        # 重新调整原始图片
        if hasattr(self, 'image_path'):
            pixmap = QPixmap(self.image_path)
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(
                    self.original_label.size() - QSize(20, 20),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                self.original_label.setPixmap(scaled_pixmap)
        
        # 重新调整处理后图片
        if self.processed_image is not None:
            if self.processed_image.mode == 'RGB':
                data = self.processed_image.tobytes("raw", "RGB")
                qimg = QImage(data, self.processed_image.width, self.processed_image.height, QImage.Format_RGB888)
            else:
                data = self.processed_image.tobytes("raw", "L")
                qimg = QImage(data, self.processed_image.width, self.processed_image.height, QImage.Format_Grayscale8)
            
            pixmap = QPixmap.fromImage(qimg)
            scaled_pixmap = pixmap.scaled(
                self.processed_label.size() - QSize(20, 20),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.processed_label.setPixmap(scaled_pixmap)

def main():
    """主函数"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # 设置应用样式
    
    # 创建并显示窗口
    window = DitheringApp()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()