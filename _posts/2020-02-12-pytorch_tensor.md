# Pytorch Tensor 

Pytorch学习笔记系列。本文目录：

1. TOC
{:toc}

## pytorch是什么？
它是基于Python的，面向以下两类用户的科学计算程序包：

+ 可以替代 NumPy 以并在在 GPUs 上运行。
+ 提供最大的灵活性和速度的深度学习研究平台



```python
import torch, torchvision 
import numpy as np

print(torch.__version__)
print(torchvision.__version__)



```

    1.3.0
    0.4.1


## Tensors 张量 

Tensors 与 NumPy’s ndarrays 类似, 另外，可以用于 GPU上 加速计算。 Torch.tensor 与 numpy.array可以相互转换


```python
n=np.array([1,2,3,4])
t=torch.tensor([3,4,5,6])

t1=torch.from_numpy(n)
n1=t.numpy()

```

## 操作
同一操作可以有不同的实现方式。 更多操作可参考[PyTorch-Basic-operations](https://jhui.github.io/2018/02/09/PyTorch-Basic-operations/)


```python

a=torch.rand([5,3])
b=torch.ones([5,3])

# 1
c=a+b
# 2
c=torch.add(a,b)

# 3
d=torch.empty(a.size())
torch.add(a,b,out=d)

#4 inplace 操作， 所有的inplace操作均采用加` _`的形式
a.add_(b) 
```




    tensor([[1.8650, 1.0220, 1.0333],
            [1.7964, 1.5436, 1.0259],
            [1.2550, 1.8710, 1.4763],
            [1.3889, 1.8811, 1.5584],
            [1.0452, 1.4945, 1.3114]])



## 元素访问 
可以采用与np.array相同的方式进行元素的访问，index/slice


```python
t2=t1.reshape(2,-1)
t4=t2[:,1]
t5=t2[-1,0:2:1]

```

## shape 属性
与np.array的类似的 tensor的 shape属性


```python
n1.shape

t1.shape
t1.size()
```




    torch.Size([2, 2])



## 改变shape

1. `view()`

  与np.array的 `.reshape()` 功能相同，tensor 采用 `.view()` 进行。且不同的view指向同一物理存储，所以改变任一个，会影响另外的变量。所果要复制创建指向不同物理存府的变量使用`.clone()`


```python
t2=t1.view(-1,2)  # t2 与 t1 指向相同区域

t3=t1.clone() # t3 与 t1 指向不同物理区域
```

2. squeezing and unsqueezing

    + Squeezing a tensor: 移除长度为1的维度removes the dimensions or axes that have a length of one.
    + Unsqueezing a tensor： 增加一个长度为1的维度 adds a dimension with a length of one.


```python
t = torch.ones(4, 3)
print(t.reshape([1,12]))
print(t.reshape([1,12]).shape)

print(t.reshape([1,12]).squeeze())
print(t.reshape([1,12]).squeeze().shape)

print(t.reshape([1,12]).squeeze().unsqueeze(dim=0))
print(t.reshape([1,12]).squeeze().unsqueeze(dim=0).shape)
```

    tensor([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]])
    torch.Size([1, 12])
    tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
    torch.Size([12])
    tensor([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]])
    torch.Size([1, 12])


3. 拼接张量 Concatenating tensors
组合两个张量使用`cat()`


```python
t1 = torch.tensor([
    [1,2],
    [3,4]
])

t2 = torch.tensor([
    [5,6],
    [7,8]
])
print(torch.cat((t1, t2), dim=0).shape)
print(torch.cat((t1, t2), dim=1).shape)
    
    
```

    torch.Size([4, 2])
    torch.Size([2, 4])


### numpy.array的拼接操作

ar1 = np.array([[1,2,3], [4,5,6]])
ar2 = np.array([[7,8,9], [11,12,13]])

1. np.concatenate(a_tuple, axis=0, out=None)
"""
参数说明：
a_tuple:对需要合并的数组用元组的形式给出
axis: 沿指定的轴进行拼接，默认0，即第一个轴
"""

2. np.stack(arrays, axis=0, out=None)
"""
沿着指定的axis对arrays(每个array的shape必须一样)进行拼接，返回值的维度比原arrays的维度高1
axis：默认为0，即第一个轴
"""

3. np.hstack(),np.vstack(),np.dstack()
```
    >>>np.hstack((ar1,ar2))  # 水平拼接，沿着行的方向，对列进行拼接
    array([[ 1, 2, 3, 7, 8, 9],
        [ 4, 5, 6, 11, 12, 13]])
     
    >>>np.vstack((ar1,ar2))  # 垂直拼接，沿着列的方向，对行进行拼接
    array([[ 1, 2, 3],
        [ 4, 5, 6],
        [ 7, 8, 9],
        [11, 12, 13]])
         
    >>>np.dstack((ar1,ar2))  # 对于2维数组来说，沿着第三轴（深度方向）进行拼接, 效果相当于stack(axis=-1)
    array([[[ 1, 7],
        [ 2, 8],
        [ 3, 9]],
        [[ 4, 11],
        [ 5, 12],
        [ 6, 13]]])
```
对于两个shape一样的二维array来说:

+ 增加行（对行进行拼接）的方法有：

    + np.concatenate((ar1, ar2),axis=0)
    + np.append(ar1, ar2, axis=0)
    + np.vstack((ar1,ar2))
    + np.row_stack((ar1,ar2))
    + np.r_[ar1,ar2] # 垂直拼接，沿着列的方向，对行进行拼接

+ 增加列（对列进行拼接）的方法有：

    + np.concatenate((ar1, ar2),axis=1)
    + np.append(ar1, ar2, axis=1)
    + np.hstack((ar1,ar2))
    + np.column_stack((ar1,ar2))
    + np.c_[ar1,ar2] #水平拼接，沿着行的方向，对列进行拼接


## 单值tensor采用`.item()` 得到 pyhon number的值


```python
value = t2[0,0].item()
```

## tensor可以使用`.to()` 移动到任意设备上


```python
t5.to('cuda')
t4.to('cpu')

# We will use ``torch.device`` objects to move tensors in and out of GPU
x=t2
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!


```

    tensor([[6, 7],
            [8, 9]], device='cuda:0')
    tensor([[6., 7.],
            [8., 9.]], dtype=torch.float64)

