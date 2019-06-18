---
permalink: 1560232044
title: PyTorch 文档 > torch.Storage
date: 2019-06-11 13:47:24
tags:
---

> 本文在PyTorch官方文档的基础上，增加了描述性的文字与示例，对于一些文档没有写清楚的API的用法，参数的类型等进行了补充说明。最后一部分，深入源码的底层探索研究了Storage。您也可以前往 [Google Colab](https://colab.research.google.com/drive/1G38Gum83Dx-cMTR-rI2pD4yX_57-vxgg) 运行这篇文章中的代码。

<!-- more -->


## 什么是 torch.Storage

每一个张量（torch.Tensor）都有一个相关联的存储空间（torch.Storage），用来存储它的数据。所以张量实际数据不是直接保存在torch.Tensor中，而是保存在名为torch.Storage的数据结构上。

## 共有哪些存储空间

### 查看`_StorageBase`的子类

通过查看PyTorch的源码，我们可以找到一个名为[`_StorageBase`](https://github.com/pytorch/pytorch/blob/master/torch/storage.py)的类，它是所有Storage的父类。下面我们来看一下它的子类有哪些？

```python
torch._StorageBase.__subclasses__()
```

输出：
> [ torch.DoubleStorage,
>  torch.FloatStorage,
>  torch.HalfStorage,
>  torch.LongStorage,
>  torch.IntStorage,
>  torch.ShortStorage,
>  torch.CharStorage,
>  torch.ByteStorage,
>  torch.BoolStorage,
>  torch.cuda.DoubleStorage,
>  torch.cuda.FloatStorage,
>  torch.cuda.LongStorage,
>  torch.cuda.IntStorage,
>  torch.cuda.ShortStorage,
>  torch.cuda.CharStorage,
>  torch.cuda.ByteStorage,
>  torch.cuda.HalfStorage,
>  torch.cuda.BoolStorage ]

可以看到，一共有两大类Storage，一类存储在CPU上，这类Storage在torch包下；另一类存储在GPU上，这类Storage在torch.cuda上。我们知道，PyTorch中的张量可以在CPU上计算，也可以在GPU上计算，所以张量在CPU与GPU上时，持有不同包下的Storage。

### 示例1：创建张量并查看张量的存储空间的类别

```python
torch.tensor([1, 2]).storage()
```

输出：
> 1 
> 2 
> [ torch.LongStorage of size 2 ]

```python
torch.tensor([1., 2.]).storage()
```

输出：
> 1.0 
> 2.0 
> [torch.FloatStorage of size 2]

### 示例2：将张量移到GPU并查看张量的存储空间的类别

```python
torch.tensor([1, 2]).cuda().storage()
```

输出：
> 1 
> 2 
> [ torch.cuda.LongStorage of size 2 ]

```python
torch.tensor([1., 2.]).cuda().storage()
```

输出：
> 1.0 
> 2.0 
> [ torch.cuda.FloatStorage of size 2 ]


## 官方文档API介绍与简单示例

`torch.Storage`是一个单一数据类型的连续一维数组.

每一个[`torch.Tensor`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor "torch.Tensor")都有一个相同数据类型的对应存储.


> ##### <font color=red>CLASS</font> `torch.FloatStorage`

- **`bool()`**   
  - 将此存储类型转换为bool类型
  - 示例
    ```python
    torch.tensor([0, 2]).storage().bool()
    ```
    输出：
    > False 
    > True 
    > [ torch.BoolStorage of size 2 ]


- **`byte()`**
  - 将此存储类型转换为byte类型
  - 示例
    ```python
    torch.tensor([256, 257]).storage().byte()
    ```
    输出：
    > 0
    > 1
    > [ torch.ByteStorage of size 2 ]


- **`char()`**
  - 将此存储类型转换为char类型
  - 示例
    ```python
    torch.tensor([256, 257]).storage().char()
    ```
    输出：
    > 0 
    > 1 
    > [ torch.CharStorage of size 2 ]
    
  
- **`clone()`**
  - 返回此存储的副本
  - 示例
    ```python
    tensor_storage = torch.tensor([1, 2]).storage()
    tensor_storage_copy = tensor_storage.clone()
    tensor_storage_copy
    ```
    输出：
    > 1 
    > 2
    > [ torch.LongStorage of size 2 ] 
    
    ```python
    tensor_storage_copy[0] = 3 
    tensor_storage_copy
    ```
    输出：
    > 3 
    > 2 
    > [ torch.LongStorage of size 2 ]
    
    ```python
    tensor_storage
    ```
    输出：
    > 1 
    > 2 
    > [ torch.LongStorage of size 2 ]


- **`copy_()`**
  - 返回此存储的副本
  - 示例
    ```python
    tensor_storage = torch.tensor([1, 2]).storage()
    tensor_storage_copy = tensor_storage.copy_(tensor_storage)
    tensor_storage_copy
    ```
    输出：
    > 1  
    > 2  
    > [ torch.LongStorage of size 2 ]
    
    ```python
    tensor_storage_copy[0] = tensor_storage_copy
    ```
    输出：
    > 3  
    > 2  
    > [ torch.LongStorage of size 2 ]
    
    ```python
    tensor_storage
    ```
    输出：
    > 3  
    > 2  
    > [ torch.LongStorage of size 2 ]
  
  - <font color="red">注意</font>
    - `copy_()`方法的调用方式与文档不太一样，不加参数会报错
      ```python
      tensor_storage = torch.tensor([1, 2]).storage()
      tensor_storage.copy_()
      ```
      输出：
      <img width=70% src="https://raw.githubusercontent.com/machines-learning/image-repo/master/pytorch-doc-torch-storage/bc179ebf.png"/>
    - 调用`copy_()`方法后，改变副本的值，原本的值也发生改变
    - 个人建议用`clone()`方法代替`copy_()`方法，因为一般我们创建副本时不希望改变原来的对象或值，目前看来`copy_()`方法貌似违背设计初衷，而且这个方法实际调用与文档不一致，可能存在BUG。本人Google之后暂时未找到原因。


- **`cpu()`**
  - 如果此存储不在CPU上，返回此存储在CPU上的副本
  - 示例
    ```python
    torch.tensor([1, 2]).storage().cpu()
    ```
    输出：
    > 1 
    > 2 
    > [ torch.LongStorage of size 2 ]
    
    ```python
    torch.tensor([1, 2], device='cuda').storage().cpu()
    ```
    输出：
    > 1 
    > 2 
    > [ torch.LongStorage of size 2 ]


- <b>`cuda(device=None, non_blocking=False, **kwargs)`</b>
  
  - 返回此存储在CUDA内存上的副本。如果此存储已经在正确的设备上的CUDA内存上，不执行复制操作，返回原始对象。
  - 参数
    - **device**([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – 目标GPU的id，默认为当前设备
    - **non_blocking**([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – 如果将这个参数设为`True`并且数据源位于固定内存中，则副本相对于宿主是异步的。否则，这个参数不起任何作用
    - <b>**kwargs</b>– 为了兼容性，可以用关键字`async`替代参数`non_blocking`
  - 示例
    ```python
    torch.tensor([1, 2]).storage().cuda()
    ```
    输出：
    > 1 
    > 2 
    > [ torch.cuda.LongStorage of size 2 ]
    
    ```python
    torch.tensor([1, 2], device='cuda').storage().cuda()
    ```
    输出：
    > 1 
    > 2 
    > [ torch.cuda.LongStorage of size 2 ]
    
    ```python
    torch.tensor([1, 2]).storage().cuda(non_blocking=True)
    ```
    输出：
    > 1 
    > 2 
    > [ torch.cuda.LongStorage of size 2 ]
    
    ```python
    torch.tensor([1, 2]).storage().cuda(async=True)
    ```
    输出：
    > /usr/local/lib/python3.6/dist-packages/torch/_utils.py:85: UserWarning: 'async' is deprecated; use 'non_blocking'
    > warnings.warn("'async' is deprecated; use 'non_blocking'")
    > 1
    > 2
    > [ torch.cuda.LongStorage of size 2 ]
  
- **`data_ptr()`**
  - 返回数据所在的地址
  - 示例
    ```python
    torch.tensor([1, 2]).storage().data_ptr()
    ```
    输出：
    
  > 2991586816
  
- **`device`**
  - 存储所在的设备
  - 示例
    ```python
    torch.tensor([1, 2]).storage().device
    ```
    输出：
    > device(type='cpu')
    
    ```python
    torch.tensor([1, 2], device="cuda").storage().device
    ```
    输出：
    device(type='cuda', index=0)


- **`double()`**
  - 将此存储类型转换为doule类型
  - 示例
    ```python
    torch.tensor([1, 2]).storage().double()
    ```
    输出：
    > 1.0 
    > 2.0 
    > [ torch.DoubleStorage of size 2 ]


- **`dtype`**
  - 此存储中的数据类型
  - 示例
    ```python
    torch.tensor([1, 2]).storage().dtype
    ```
    输出：
    > torch.int64
    

- **`element_size()`**
  - 元素大小，以字节为单位
  - 示例
    ```python
    torch.tensor([1]).storage().int().element_size()
    ```
    输出：
    > 4
    
    ```python
    torch.tensor([1]).storage().long().element_size()
    ```
    输出：
    > 8
    
    ```python
    torch.tensor([1]).storage().byte().element_size()
    ```
    输出：
    > 1
    
    ```python
    torch.tensor([1]).storage().float().element_size()
    ```
    输出：
    > 4
    
    ```python
    torch.tensor([1]).storage().double().element_size()
    ```
    输出：
    > 8
    

- **`fill_()`**
  - 用整型或布尔类型填充
  - 示例
    ```python
    torch.tensor([1, 2]).storage().fill_(5)
    ```
    输出：
    > 5 
    > 5 
    > [ torch.LongStorage of size 2 ]
    
    ```python
    torch.tensor([True, True]).storage().fill_(False)
    ```
    输出：
    > 0 
    > 0 
    > [ torch.ByteStorage of size 2 ]


- **`float()`**
  - 将此数据类型转换为float类型
  - 示例：
    ```python
    torch.tensor([1, 2]).storage().float()
    ```
    输出：
    > 1.0 
    > 2.0 
    > [ torch.FloatStorage of size 2 ]
  
- <font color="red">STATIC</font> <b>`from_buffer()`</b>
  - 示例：
    ```python
    torch.FloatStorage.from_buffer(b"hell", 'big')
    ```
    输出：
    > 4.333687820629062e+24 
    > [ torch.FloatStorage of size 1 ]


- <font color="red">STATIC</font> <b>`from_file(filename,shared=False,size=0)`→ Storage</b>
	- 如果将 <i>shared</i> 设为 True, 那么会在所有进程之间共享内存。所有的改变都会写入文件中。如果将 <i>shared</i> 设为 False，那么在内存中做的改变不会影响到文件。<i>size</i> 是存储中元素的数量。 如果将 <i>shared</i> 设为 False, 那么文件必须包含至少sizeof(Type)字节（Type是存储的字节数）。如果将 <i>shared</i> 设为 True, 如果必要的话将创建文件
	- 参数
		- <b>filename</b>(<i>[str](https://docs.python.org/3/library/stdtypes.html#str)</i>) – 被映射的文件的名字
		- <b>shared</b>(<i>[bool](https://docs.python.org/3/library/functions.html#bool)</i>) – 是否共享内存
		- <b>size</b>(<i>[int](https://docs.python.org/3/library/functions.html#int)</i>) – 存储中元素的数量
	- 示例
    ```python
    !touch tensor_file & echo "1" > tensor_file
    !cat tensor_file
    ```
    输出：
    > 1
    
    ```python
    torch.FloatStorage.from_file('tensor_file')
    ```
    输出：
    >
    > [ torch.FloatStorage of size 0 ]
    
	- <font color="red">注意</font>
		
		- 文档中并没有写文件的格式是怎样的，扒了源码以及C层代码后依旧未发现，容我日后慢慢研究研究。。。


- **`int`**
  - 将此存储类型转换为int类型
  - 示例
    ```python
    torch.tensor([1., 2.]).storage().int()
    ```
    输出：
    >  1 
    >  2 
    >  [ torch.IntStorage of size 2 ]


- **`half()`**
  - 将此存储类型转换为half类型
  - 示例
    ```python
    torch.tensor([65519, 65520]).storage().half()
    ```
    输出：
    > 65504.0
    > inf 
    > [ torch.HalfStorage of size 2 ]
    
  - <font color="red">注意</font>
    
    - 半精度浮点数 [Half-precision floating-point format](https://en.wikipedia.org/wiki/Half-precision_floating-point_format) 是一种被计算机使用的二进制浮点数据类型。半精度浮点数使用2个字节（16位）来存储。在IEEE 754-2008中，它被称作binary16。这种数据类型只适合存储对精度要求不高的数字，不适合用来计算。


- <b>`is_cuda`= <font color="red">FALSE</font></b>
  - 是否存储在cuda上
  - 示例
    ```python
    torch.tensor([1, 2]).storage().is_cuda
    ```
    输出：
    > False
    
    ```python
    torch.tensor([1, 2], device="cuda").storage().is_cuda
    ```
    输出：
    
    > True


- **`is_pinned()`**
  - 是否位于固定内存
  - 示例：
    ```python
    torch.tensor([1, 2]).storage().is_pinned()
    ```
    输出：
    > False

    ```python
    torch.tensor([1, 2], device="cuda").storage().is_pinned()
    ```
    输出：
    > False
    
    ```python
    torch.tensor([1, 2]).storage().pin_memory().is_pinned()
    ```
    输出：
    > True
    

- **`is_shared()`**
  - 是否内存共享
  - 示例：
    ```python
    torch.tensor([1, 2]).storage().is_shared()
    ```
    输出：
    > False
    
    ```python
    torch.tensor([1, 2]).storage().share_memory_().is_shared()
    ```
    输出：
    > True
    

- **`is_sparse`**= <font color="red">FALSE</font>
  - 是否稀疏
  - 示例
    ```python
    torch.tensor([[1, 0], [0, 0]]).storage().is_sparse
    ```
    输出：
    > False
    
    ```python
    i = torch.LongTensor([[0, 1, 1], [2, 0, 2]])
    v = torch.FloatTensor([3, 4, 5])
    torch.sparse.FloatTensor(i, v, torch.Size([2,3])).storage().is_sparse
    ```
    输出：
    <img width=70% src="https://raw.githubusercontent.com/machines-learning/image-repo/master/pytorch-doc-torch-storage/408c0d88.png">
    
  - <font color="red">注意</font>
    - 按照稀疏矩阵的定义，如果矩阵中0元素占大多数，该矩阵即为稀疏矩阵。但根据输出来看，并不是单纯按照这个定义来判断的。
    - 通过查看文档，发现了一个名为[`torch.sparse`](https://pytorch.org/docs/stable/sparse.html)的包，目前这个包下的API是实验性的，未来可能会改变。
    - 根据官方示例，推测PyTorch为稀疏矩阵单独定义了一套数据的存储方式，其张量位于`torch.sparse`包下，例如`torch.sparse.FloatTensor`
    - 但是根据输出结果，`torch.sparse.FloatTensor`未持有Storage，所以这个字段可以暂时忽略，待官方完善后再学习如何使用


- **`long()`**
  - 将此存储类型转换为long类型
  - 示例
    ```python
    torch.tensor([1., 2.]).storage().long()
    ```
    输出：
    > 1 
    > 2 
    > [ torch.LongStorage of size 2 ]


- **`new()`**
  - 由已有的Storage创建出新的Storage，新的Storage的类型与已有的Storage的类型相同
  - 示例
    ```python
    torch.tensor([1, 2]).storage().new()
    ```
    输出：
    >
    >[ torch.LongStorage of size 0 ]
    
    ```python
    torch.tensor([1., 2.]).storage().new()
    ```
    输出：
    >
    >[ torch.FloatStorage of size 0 ]
    

- **`pin_memory()`**
  - 将存储复制到固定的内存(如果它还没有固定)
  - 示例  
    ```python
    torch.tensor([1, 2]).storage().pin_memory()
    ```
    输出：
    > 1 
    > 2 
    > [ torch.LongStorage of size 2 ]


- **`resize_()`**
  - 改变张量的形状
  - 示例
    ```python
    torch.tensor([[0, 0], [0, 0]]).resize_(1, 4)
    ```
    输出：
    
    > tensor([[0, 0, 0, 0]])


- **`share_memory_()`**
  - 将此存储移动到共享内存中
    对于已经在共享内存中的Storage和CUDA Storage，无需移动以在进程之间共享。共享内存中的Storage无法调整大小。
    返回：自己
  - 示例：
    ```python
    torch.tensor([1, 2]).storage().share_memory_()
    ```
    输出：
    > 1 
    > 2 
    > [ torch.LongStorage of size 2 ]


- **`short()`**
  - 将此存储类型转化为short类型
  - 示例
    ```python
    torch.tensor([1, 2]).storage().short()
    ```
    输出：
    > 1 
    > 2 
    > [ torch.ShortStorage of size 2 ]


- **`size()`**
  - 存储中元素的个数
  - 示例
    ```python
    torch.tensor([1, 2]).storage().size()
    ```
    输出：
    > 2
    
    ```python
    torch.zeros(2, 2).storage().size()
    ```
    输出：
    
    > 4


- **`tolist()`**
  - 返回一个包含了存储中所有元素的列表
  - 示例：
    ```python
    torch.zeros(2, 2).storage().tolist()
    ```
    输出：
    
    > [ 0.0, 0.0, 0.0, 0.0 ]


- <b>`type(dtype=None,non_blocking=False,**kwargs)`</b>
  - 如果没有提供_dtype_，返回此Storage的类型，否则将此Stroage的类型强制转换为指定的类型R。
    如果已经是正确的类型了，不执行复制操作，返回原始对象。
  - 参数
    - **dtype**([_type_](https://docs.python.org/3/library/functions.html#type "(in Python v3.7)")_or__string_) – 指定类型
    - **non_blocking**([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – 如果将这个参数设为`True`，并且数据源存储在固定内存中，目标在GPU中，反之亦然，则相对于宿主异步执行复制操作。否则，这个参数不起任何作用。
    - <b>**kwargs</b>– 为了兼容性，可以用关键字`async`替代参数`non_blocking`。`async`参数已经弃用.
  - 示例
    ```python
    torch.tensor([1., 2.]).storage().type(torch.LongStorage)
    ```
    输出：
    > 1 
    > 2 
    > [ torch.LongStorage of size 2 ]
    


## 深入理解张量存储空间

- 正在编写中



## 参考资料

- [torch.Tensor — PyTorch master documentation](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)
- [torch.Storage — PyTorch master documentation](https://pytorch.org/docs/stable/storage.html)
- [torch.sparse — PyTorch master documentation](https://pytorch.org/docs/stable/sparse.html)
- [PyTorch – Internal Architecture Tour \| Terra Incognita](http://blog.christianperone.com/2018/03/pytorch-internal-architecture-tour)
- [pytorch/storage.py at master · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/blob/master/torch/storage.py)
- [pytorch/aten/src/TH at master · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/tree/master/aten/src/TH)
- [pytorch/aten/src/THC at master · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/tree/master/aten/src/THC)
- [pytorch/torch/sparse at master · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/tree/master/torch/sparse)
- [Sparse matrix - Wikipedia](https://en.wikipedia.org/wiki/Sparse_matrix)