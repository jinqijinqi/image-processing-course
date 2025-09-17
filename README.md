# UESTC image processing course (2025-2026)
# Lecturer: Jin Qi
# You are welcome!

# 课程考核
1. 课堂表现及练习： 20%
2. 课后作业： 20%
3. 课程实验及设计： 30%
4. 期末考试： 30%

**评分标准：**

* **数学公式**：推导清晰、准确（20分）
* **代码**：规范，运行无误，有适当注释（20分）
* **结果与分析**：结果完整，分析透彻（20分）
* **心得与思考**：有独立思考和联系实际应用（20分）
* **格式**：排版规范，结构清晰美观（20分）
---

# 《数字图像处理》第3章实验报告范例

* **姓名：** 张三
* **学号：** 2023123456
* **实验题目：** 第3章 图像增强与直方图均衡化实验
* **实验时间：** 2024年9月20日



## 一、实验目的

本实验旨在理解和实现图像的基本增强方法，包括灰度线性变换、对数变换及直方图均衡化。通过对一幅简单灰度图像的处理，加深对数字图像增强原理和实际效果的认识。



## 二、实验步骤

1. **生成并归一化一幅4×4灰度图像**
2. **进行对数变换与直方图均衡化**
3. **可视化并对比处理结果**

**主要代码如下：**

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt

# 1. 生成并归一化灰度图像
img = np.array([[50,120,180,200],
                [80,130,170,195],
                [90,140,160,180],
                [70,110,150,175]], dtype=np.float32)
img_norm = (img - img.min()) / (img.max() - img.min())

# 2. 对数变换
img_log = np.log1p(img) / np.log1p(img.max())

# 3. 直方图均衡化
img_uint8 = (img_norm*255).astype(np.uint8)
img_eq = cv2.equalizeHist(img_uint8)

# 4. 可视化结果
plt.figure(figsize=(12,3))
plt.subplot(1,4,1); plt.imshow(img, cmap='gray'); plt.title('原始图像')
plt.subplot(1,4,2); plt.imshow(img_norm, cmap='gray'); plt.title('归一化')
plt.subplot(1,4,3); plt.imshow(img_log, cmap='gray'); plt.title('对数变换')
plt.subplot(1,4,4); plt.imshow(img_eq, cmap='gray'); plt.title('直方图均衡')
plt.show()
```



## 三、算法与数学推导

### 1. 归一化公式

$$
I' = \frac{I - I_{min}}{I_{max} - I_{min}}
$$

其中，$I_{min}$ 和 $I_{max}$ 分别为图像的最小和最大像素值。

### 2. 对数变换公式

$$
s = c \cdot \log(1 + r)
$$

本实验取 $c = 1/\log(1 + r_{max})$ 实现归一化。

### 3. 直方图均衡化公式

$$
s_k = T(r_k) = (L-1)\sum_{j=0}^{k} p_r(r_j)
$$

其中，$L$ 为灰度级数，$p_r(r_j)$ 为第$j$级的概率。



## 四、实验结果与分析

**运行结果：**

* 原始图像、归一化、对数变换、直方图均衡化均已显示。
* 处理后图像明显增强了对比度，尤其是直方图均衡化能拉伸并均衡灰度分布。

**分析：**

* 对数变换对低灰度区域提升更明显，高灰度区域压缩，对增强暗部信息有效。
* 直方图均衡化使图像整体对比度提升、细节更丰富，但可能产生噪声放大。
* 不同方法适用于不同图像特性，应按实际需求选择。



## 五、心得与思考

本次实验让我理解了数字图像增强的本质——利用数学变换提升图像可视性。掌握了归一化、对数变换和直方图均衡化的原理与代码实现。实际操作中感受到理论和实践结合的重要性，例如均衡化虽能提升对比，但对小尺寸图像会有“分块效应”，在大图像和医学等实际应用中需合理调整参数。

此外，通过查阅OpenCV等库资料，提高了自主查找与解决问题的能力。今后希望能在医学影像预处理等实际项目中，灵活应用这些图像增强方法。


---

# 课后作业(请用自己的人脸图像）[提交作业模板.docx](https://github.com/jinqijinqi/image-processing-course/blob/main/homework/%E4%BD%9C%E4%B8%9A%E6%A8%A1%E6%9D%BF.docx) <br/>



## **Week 1：数字图像基础（第2章）**

**主要内容：像素、灰度级、基本变换**
1. 图像归一化：
```python
import numpy as np
img = np.random.randint(50,200,(4,4)).astype(np.float32)
img_norm = (img - img.min())/(img.max() - img.min())
print("原始：\n", img)
print("归一化：\n", img_norm)
```

**公式**

$$
I' = \frac{I - I_{min}}{I_{max} - I_{min}}
$$

2. Gamma 变换（图像增强）

**数学公式**

$$
s = c\, r^{\gamma},\quad r\in[0,1],\ c=1
$$



```matlab
I = im2double(imread('cameraman.tif'));   % 灰度图
gammas = [0.5, 1.0, 2.2];
figure;
for k = 1:numel(gammas)
    J = I .^ gammas(k);
    subplot(1,3,k); imshow(J); title(sprintf('\\gamma=%.2f', gammas(k)));
end
```

3. 互相关图像块搜索（NCC 模板匹配）

**数学公式**

（互相关）

$$
C(u,v)=\sum_{x,y} T(x,y)\, I(x+u,y+v)
$$

（零均值归一化互相关，NCC）

![](https://github.com/jinqijinqi/image-processing-course/blob/main/%E5%85%AC%E5%BC%8Fcorrelation.PNG)

```matlab
I = im2double(imread('cameraman.tif'));   % 灰度图
r0 = 80; c0 = 90; h = 40; w = 40;
T  = I(r0:r0+h-1, c0:c0+w-1);             % 模板块

C = normxcorr2(T, I);                     % NCC 响应
[ypeak, xpeak] = find(C == max(C(:)));    % 峰值位置
yoff = ypeak - size(T,1);
xoff = xpeak - size(T,2);

figure; imshow(I); hold on;
rectangle('Position',[xoff, yoff, w, h], 'EdgeColor','r','LineWidth',2);
plot(xoff+w/2, yoff+h/2, 'r+'); title('Best NCC Match');
```



## **Week 2：图像增强（一）- 灰度变换与直方图处理（第3章）**

**内容：对数变换、直方图均衡化等**

```python
import cv2
img_log = np.log1p(img)/np.log1p(img.max())
img_uint8 = (img_norm*255).astype(np.uint8)
img_eq = cv2.equalizeHist(img_uint8)
```

**公式**
对数变换：

$$
s = c\cdot\log(1+r)
$$

直方图均衡化：

$$
s_k = T(r_k) = (L-1)\sum_{j=0}^{k} p_r(r_j)
$$

---

## **Week 3：空间滤波（第4章）**

**内容：均值滤波、中值滤波、锐化（Sobel、Laplacian）**

```python
img_med = cv2.medianBlur(img_uint8, 3)
sobel = cv2.Sobel(img_norm, cv2.CV_64F, 1, 1, ksize=3)
laplacian = cv2.Laplacian(img_norm, cv2.CV_64F)
```

**公式**
均值滤波器、Sobel算子、拉普拉斯算子矩阵。



## **Week 4：频域处理基础（第5章）**

**内容：傅里叶变换、低通/高通滤波**

```python
f = np.fft.fft2(img_norm)
fshift = np.fft.fftshift(f)
mask = np.zeros_like(img_norm); mask[1:3,1:3]=1
fshift_filtered = fshift * mask
img_back = np.fft.ifft2(np.fft.ifftshift(fshift_filtered)).real
```

**公式**
二维DFT：

$$
F(u,v) = \sum_{x=0}^{M-1}\sum_{y=0}^{N-1} f(x,y) e^{-j2\pi(ux/M+vy/N)}
$$



## **Week 5：图像复原（第5章）**

**内容：反卷积、维纳滤波（简单演示）**

```python
from scipy.signal import convolve2d
kernel = np.ones((3,3))/9
img_blur = convolve2d(img_norm, kernel, mode='same', boundary='symm')
img_deblur = img_blur / (kernel.sum() + 1e-8)
```

**公式**
维纳滤波基本表达式。



## **Week 6：色彩图像处理（第6章）**

**内容：RGB-Gray互转，通道处理，伪彩色**

```python
import matplotlib.pyplot as plt
img_rgb = np.stack([img_norm]*3,axis=-1)
img_gray = cv2.cvtColor((img_rgb*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
```

**公式**
灰度化公式：

$$
Gray = 0.299R + 0.587G + 0.114B
$$



## **Week 7：几何变换与插值（第7章）**

**内容：缩放、平移、旋转、仿射**

```python
M = cv2.getRotationMatrix2D((2,2), 45, 1)
img_rot = cv2.warpAffine(img_norm, M, (4,4))
```

**公式**
仿射/旋转矩阵推导。



## **Week 8：图像分割（第10章）**

**内容：全局阈值、Otsu法、边缘检测、区域生长**

```python
_, otsu = cv2.threshold(img_uint8,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
```

**公式**
Otsu法、梯度边缘公式。



## **Week 9：特征提取（第11章）**

**内容：角点、边缘、纹理（LBP）等**

```python
from skimage.feature import local_binary_pattern
lbp = local_binary_pattern(img_uint8, 8, 1)
```

**公式**
LBP模式提取公式。



## **Week 10：图像描述与模式识别（第12章）**

**内容：形状描述（矩）、特征向量、简单分类**

```python
moments = cv2.moments(img_uint8)
print("Hu矩：", cv2.HuMoments(moments).flatten())
```

**公式**
Hu不变矩公式。



## **Week 11：图像融合与综合应用（第13章/扩展应用）**

**内容：多模态图像融合（均值/PCA/小波）、综合处理**

```python
def fuse_average(ir, vis): return 0.5*ir + 0.5*vis
# 或见前PCA例子
```

**公式**
融合公式及原理简述。



## **Week 12：图像质量评价与主观评价（第14章）**

**内容：PSNR、SSIM、熵等指标**

```python
def psnr(img1, img2):
    mse = np.mean((img1-img2)**2)
    return 10*np.log10(1.0/(mse+1e-8))
```

**公式**
PSNR、SSIM表达式。




# 项目： 3 个 <br/>

1. (第六周截止)基于相位的图像对齐方法概述，需要理解如下论文中的算法和相应的MATLAB实现，并得到用自己的 _人脸照片_ 进行实验的结果 <br/>
   [提交项目模板1-基于相位的图像校准.docx](https://github.com/jinqijinqi/image-processing-course/blob/main/homework/%E9%A1%B9%E7%9B%AE%E6%A8%A1%E6%9D%BF1-%E5%9F%BA%E4%BA%8E%E7%9B%B8%E4%BD%8D%E7%9A%84%E5%9B%BE%E5%83%8F%E6%A0%A1%E5%87%86.docx) <br/>
   [MATLAB参考网页1-需要使用MATLAB调试命令进入函数内部理解其算法实现](https://www.mathworks.com/help/images/use-phase-correlation-as-preprocessing-step-in-registration.html) <br/>
   [MATLAB参考网页2-需要使用MATLAB调试命令进入函数内部理解其算法实现](https://www.mathworks.com/help/images/ref/imregcorr.html) <br/>
   算法理解论文参考<br/>
   [1] Reddy, B. S. and Chatterji, B. N., [An FFT-Based Technique for Translation, Rotation, and Scale-Invariant Image Registration](https://ieeexplore.ieee.org/document/506761), IEEE Transactions on Image Processing, Vol. 5, No. 8, August 1996
2. (第八周截止)图像盲去模糊方法概述，需要理解算法和相应的MATLAB实现，并得到用自己的 _人脸照片_ 进行实验的结果 <br/>
   [提交项目模板2-盲图像去卷积.docx](https://github.com/jinqijinqi/image-processing-course/blob/main/homework/%E9%A1%B9%E7%9B%AE%E6%A8%A1%E6%9D%BF2-%E7%9B%B2%E5%9B%BE%E5%83%8F%E5%8E%BB%E5%8D%B7%E7%A7%AF.docx) <br/>
   [MATLAB参考网页1-需要使用MATLAB调试命令进入函数内部理解其算法实现](https://www.mathworks.com/help/images/deblurring-images-using-the-blind-deconvolution-algorithm.html) <br/>
   [MATLAB参考网页2-需要使用MATLAB调试命令进入函数内部理解其算法实现](https://www.mathworks.com/help/images/ref/deconvblind.html) <br/>
   算法理解论文参考：<br/>
      [1] D.S.C. Biggs and M. Andrews, [Acceleration of iterative image restoration algorithms](https://opg.optica.org/ao/abstract.cfm?uri=ao-36-8-1766), Applied Optics, Vol. 36, No. 8, 1997.

      [2] R.J. Hanisch, R.L. White, and R.L. Gilliland, [Deconvolutions of Hubble Space Telescope Images and Spectra, Deconvolution of Images and Spectra](https://dl.acm.org/doi/10.5555/273488.273506), Ed. P.A. Jansson, 2nd ed., Academic Press, CA, 1997.

      [3] Timothy J. Holmes, et al, [Light Microscopic Images Reconstructed by Maximum Likelihood Deconvolution](https://link.springer.com/chapter/10.1007/978-1-4757-5348-6_24), Handbook of Biological Confocal Microscopy, Ed. James B. Pawley, Plenum Press, New York, 1995.

 3. (第十二周截止) 基于标志点watershed图像分割方法概述，需要理解算法和相应的MATLAB实现，并得到用自己的 _水果照片_ 进行实验的结果 <br/>
   [提交项目模板3-作业模板3-标志指导的watershed图像分割.docx](https://github.com/jinqijinqi/image-processing-course/blob/main/homework/%E9%A1%B9%E7%9B%AE%E6%A8%A1%E6%9D%BF3-%E6%A0%87%E5%BF%97%E6%8C%87%E5%AF%BC%E7%9A%84watershed%E5%9B%BE%E5%83%8F%E5%88%86%E5%89%B2.docx) <br/>
   [MATLAB参考网页1-需要使用MATLAB调试命令进入函数内部理解其算法实现](https://www.mathworks.com/help/images/marker-controlled-watershed-segmentation.html) <br/>
   [MATLAB参考网页2-需要使用MATLAB调试命令进入函数内部理解其算法实现](https://www.mathworks.com/help/images/ref/watershed.html) <br/>
   算法理解论文参考：<br/>
      [1] Meyer, Fernand, ["Topographic distance and watershed lines,”](https://www.sciencedirect.com/science/article/abs/pii/0165168494900604) Signal Processing , Vol. 38, July 1994, pp. 113-125.
