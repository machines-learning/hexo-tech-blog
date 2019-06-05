---
permalink: 1559642400
title: PyTorch 深度学习：60分钟速成
date: 2019-06-04 18:00:00
tags:
---

本教程是PyTorch官方教程Getting Started篇的第一篇教程，更多教程请点击下方链接查看：

- PyTorch深度学习：60分钟速成
- [数据加载与处理]()
- [迁移学习]()
- [使用混合前端部署Seq2Seq模型]()
- [保存与加载模型]()
- [torch.nn是什么]()

<!-- more -->

## 教学目标

- 高维度地理解PyTorch的Tensor库和神经网络
- 训练一个小的神经网络对图像进行分类

> 本教程假设您对 [NumPy](http://www.numpy.org) 已经有了基本的了解，请确保您已经安装了 [torch](https://github.com/pytorch/pytorch) 和 [torchvision](https://github.com/pytorch/vision) ，没有安装请前往 [安装向导页面](https://pytorch.org/get-started/locally/) 进行安装，也可以点击 [PyTorch 深度学习：60分钟速成](https://drive.google.com/open?id=1ZBnuiwwkmKLQ6Xkb9mIOHXOeuAaRwUGj) 前往Google Colab阅读教程并运行代码
> **注意**：本教程在原有教程的基础上增加了 Numpy 的对照代码，增加了部分补充说明与代码

## 目录

- [一、什么是PyTorch ？](#index1)
  - [1 开始](#index11)
  - [2 Numpy桥](#index12)
  - [3 CUDA张量](#index13)
- [二、Autograd：自动求导](#index2)
- [三、神经网络](#index3)
- [四、训练一个分类器](#index4)
- [五、选读：数据并行](#index5)

<a name="index1"></a>

## 什么是PyTorch ？

PyTorch是基于Python的科学计算包，它的特点如下：

- NumPy的替代品，可以使用GPU的强大功能
- 深度学习研究平台，提供最大的灵活性和速度
  <a name="index11"></a>

### 开始

#### 张量

Torch中的张量tensor与NumPy中的ndarray类似，另外tensor还可以使用 GPU 加速计算。

```python
from __future__ import print_function
import torch
import numpy as np
```

##### 构造一个未初始化的5x3的矩阵

```python
x = torch.empty(5, 3)
x
```

输出：
{% asset_img 61584c35.png 61584c35.png %}

使用NumPy构造一个未初始化的5x3的矩阵

```python
x = np.empty((5, 3))
x
```

输出：
{% asset_img 624451cc.png 624451cc.png %}

##### 构造一个随机初始化的矩阵

```python
x = torch.rand(5, 3)
x
```

输出：
{% asset_img 8e67cc1f.png 8e67cc1f.png %}

使用Numpy构造一个随机初始化的矩阵：

```python
x = np.random.rand(5, 3)
x
```

输出：
{% asset_img f2f5f7b6.png f2f5f7b6.png %}

##### 构造一个用0填充的矩阵，数据类型为long 

```python
x = torch.zeros(5, 3, dtype=torch.long)
x
```

输出：
{% asset_img a89ee470.png a89ee470.png %}

用NumPy构造一个用0填充的矩阵，数据类型为long 

```python
x = np.zeros((5, 3), dtype=np.long)
x
```

输出：
{% asset_img 84d4e2f5.png 84d4e2f5.png %}

##### 构造一个用已有数据初始化的tensor

```python
x = torch.tensor([5.5, 3])
x
```

输出：
{% asset_img db7a767b.png db7a767b.png %}

用NumPy构造一个用已有数据初始化的ndarray 

```python
x = np.array([5.5, 3])
x
```

输出：
{% asset_img d2dad8b5.png d2dad8b5.png %}

##### 或者基于已有的tensor创建新的tensor，这些方法将重用输入的tensor的属性，例如，dtype，除非用户提供新值

```python
x = torch.tensor([5.5, 3])
print(x)
x = x.new_ones(5, 3, dtype=torch.double)
print(x)
x = torch.randn_like(x, dtype=torch.float)
print(x)
```

输出：
{% asset_img 83edc98f.png 83edc98f.png %}

查看tensor的size：

```python
x.size()
```

```python
x.shape
```

> **注意**：
>
> - 官方教程并没有列出第二种写法，但是第二种写法确实存在并且能够正常运行。.shape 是 .size() 的别名，两种写法输出相同。具体可以看这两个issue：
>   - [.size() vs .shape, which one should be used? · Issue #5544 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/5544)
>   - [add shape alias by hughperkins · Pull Request #1983 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/pull/1983)
> - torch.Size 实际上是一个元祖，因此它支持所有元祖操作。

输出：
{% asset_img deb8d58b.png deb8d58b.png %}

使用 NumPy 基于已有的ndarray创造新的ndarray，

```python
x = np.array([5.5, 3])
print(x)
x = np.ones((5, 3), dtype=np.double)
print(x)
x = np.random.rand(*x.shape)
print(x)
```

输出：
{% asset_img 1bff7a87.png 1bff7a87.png %}

查看ndarray的shape ：

```python
x.shape
```

输出：
{% asset_img 8a7007d8.png 8a7007d8.png %}

#### 操作
操作有多种语法，在下面的示例中，我们将研究加法操作。

##### 加法：语法1

```python
x = torch.rand(5, 3)
print(x)
y = torch.rand(5, 3)
print(y)
print(x + y)
```

输出：
{% asset_img 7d036c4b.png 7d036c4b.png %}

NumPy 中的加法：语法1

```python
x = np.random.rand(5, 3)
print(x)
y = np.random.rand(5, 3)
print(y)
print(x + y)
```

输出：
{% asset_img c6ccba98.png c6ccba98.png %}

##### 加法：语法2

```python
x = torch.rand(5, 3)
print(x)
y = torch.rand(5, 3)
print(y)
print(torch.add(x, y))

```

输出：
{% asset_img 777b8966.png 777b8966.png %}

NumPy中的加法：语法2

```python
x = np.random.rand(5, 3)
print(x)
y = np.random.rand(5, 3)
print(y)
print(np.add(x, y))

```

输出：
{% asset_img 1d972dea.png 1d972dea.png %}

##### 加法：提供一个输出tensor作为参数

```python
x = torch.rand(5, 3)
print(x)
y = torch.rand(5, 3)
print(y)
result = torch.empty(5, 3)
print(result)
torch.add(x, y, out=result)
print(result)

```

输出：
{% asset_img 6f84d7d0.png 6f84d7d0.png %}

NumPy 加法：提供一个输出ndarray作为参数

```python
x = np.random.rand(5, 3)
print(x)
y = np.random.rand(5, 3)
print(y)
result = np.empty((5, 3))
print(result)
np.add(x, y, out=result)
print(result)

```

输出：
{% asset_img 36cb2728.png 36cb2728.png %}

##### 加法：in-place（就地，在原来的位置改变）

> **注意**：
>
> - 任何以“_”结尾的操作都会进行in-place操作，即在原来的tensor上操作并改变
> - 通过打印id可以看出，in-place是在原来的tensor上进行操作并改变，并不是生成一个新的tensor然后赋值给原有的变量，也就是说tensor没有发生变化但tensor内的数字发生了变化

```python
x = torch.rand(5, 3)
print(x)
y = torch.rand(5, 3)
print(y)
print(id(y))
y.add_(x)
print(y)
print(id(y))

```

输出：
{% asset_img beff2a6f.png beff2a6f.png %}

##### 您可以使用类似NumPy索引与切片的标准操作

```python
x = torch.rand(5, 3)
print(x)
print(x[:, 1])

```

输出：
{% asset_img 7d6ccc61.png 7d6ccc61.png %}

NumPy中的索引与切片操作

```python
x = np.random.rand(5, 3)
print(x)
print(x[:, 1])

```

输出：
{% asset_img b0eaf859.png b0eaf859.png %}

##### 调整大小：如果您想要调整大小/改变形状，你可以使用 torch.view

```python
x = torch.randn(4, 4)
print(x)
y = x.view(16)
print(y)
z = x.view(-1, 8)
print(z)
print(x.size(), y.size(), z.size())

```

```python
x = torch.randn(4, 4)
print(x)
y = x.reshape(16)
print(y)
z = x.reshape(-1, 8)
print(z)
print(x.size(), y.size(), z.size())

```

> **注意**：
>
> - Torch中tensor的view与reshape方法输出结果大部分情况下相同，view/reshape之后返回同样的数据与数据类型，只改变shape的值。
> - 二者区别：当tensor都为contiguous类型（邻近模式）时，两个函数并无差异，使用原来的数据内容，不会复制一份新的出来；如果tensor不是，例如经过了transpose或permute之后，需要contiguous然后再使用view。reshape其实就是contiguous+view，这样不会开辟新的空间存放tensor，而是共享原来的数据内存。

输出：
{% asset_img ff705f81.png ff705f81.png %}

##### 如果你的tensor只有一个元素，使用.item()得到一个Python Number

```python
x = torch.rand(1)
print(x)
print(x.item())

```

回答：
{% asset_img 21e8df91.png 21e8df91.png %}

NumPy .item() 示例：

```python
x = np.random.rand(1)
print(x)
print(x.item())

```

回答：
{% asset_img 38e46887.png 38e46887.png %}

##### **稍后阅读**

[这里](https://pytorch.org/docs/torch)有100多个Tensor操作，包括置换, 索引, 切片, 数学运算, 线性代数, 随机数等等。

<a name="index12"></a>

### NumPy 桥 

将Torch张量转换为NumPy数组是一件轻而易举的事情，反之亦然。
Torch张量和NumPy数组将共享它们的底层内存位置(如果Torch张量位于CPU上)，更改其中一个将会改变另一个。

##### 将Torch张量转换为NumPy数组

```python
a = torch.ones(5)
a

```

输出：
{% asset_img 4037d43b.png 4037d43b.png %}

```python
b = a.numpy()
b

```

输出：
{% asset_img b7f7314c.png b7f7314c.png %}

##### 查看NumPy数组的值是如何变化的：

```python
a.add_(1)
a

```

输出：
{% asset_img 7001b1a8.png 7001b1a8.png %}

```python
b

```

输出：
{% asset_img 4b42b954.png 4b42b954.png %}

> **注意**：
>
> - 只有Torch张量位于CPU上时，才会与NumPy数组共享内存地址，因为NumPy数组位于CPU上
> - 如果Torch张量位于GPU上，需要先将Torch张量移到CPU上，才能获取NumPy数组

```python
a = torch.ones(5, device='cuda')
a

```

输出：
{% asset_img 3000f0cd.png 3000f0cd.png %}

```python
b = a.numpy()

```

输出：
{% asset_img 28a69493.png 28a69493.png %}

```python
b = a.cpu().numpy()
b

```

输出：
{% asset_img 5884b5e9.png 5884b5e9.png %}

##### 查看NumPy数组的值是如何变化的：

```python
a.add_(1)
a

```

输出：
{% asset_img c717c186.png c717c186.png %}

```python
b

```

输出：
{% asset_img de0e4923.png de0e4923.png %}

##### 将NumPy数组转换为Torch张量

##### 查看如何改变NumPy数组自动改变Torch张量：

```python
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

```

输出：
{% asset_img a163b2f7.png a163b2f7.png %}

##### 查看改变NumPy数组，是否会改变位于GPU上的Torch张量：

```python
a = np.ones(5)
b = torch.from_numpy(a).cuda()
np.add(a, 1, out=a)
print(a)
print(b)

```

输出：
{% asset_img bf5c7507.png bf5c7507.png %}

<a name="index13"></a>

### CUDA 张量

张量可以使用.to方法移动到任何设备上。

> **注意**：
>
> - 这里的设备指的是CPU或者GPU，每台设备必有一块CPU，可能没有GPU，也可能有1块甚至多块GPU
> - 设备必须有一或多块GPU才能将tensor移到GPU上，然后使用GPU为tensor计算加速，使用torch.cuda.is_available() 判断当前设备是否有GPU
> - 因每台设备必有一块CPU，所以 torch.cuda.is_available() 为 False 时，可以直接在CPU上进行计算，也可以什么都不做。

```python
# let us run this cell only if CUDA is available
# We will use ``torch.device`` objects to move tensors in and out of GPU
x = torch.rand(2, 3)
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!

```

输出：
{% asset_img 7f78e18f.png 7f78e18f.png %}

<a name="index2"/></a>

## Autograd：自动求导

<a name="index3"></a>

## 神经网络

<a name="index4"></a>

## 训练一个分类器

<a name="index5"></a>

## 选读：数据并行