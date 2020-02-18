# 过拟合、欠拟合、梯度消失、梯度爆炸、RNN进阶

Pytorch学习笔记系列。

过拟合、欠拟合、梯度消失、梯度爆炸是机器学习训练中经常遇到的问题，本文前部分简单介绍其概念和解决方法。后一部分，介绍RNN高阶内容。本文目录：

1. TOC
{:toc}

##  一、欠拟合、过拟合

### （一）机器学习的三个数据集
机器学习中的误差是指机器学习算法的结果 与 真实目标值 之间的差异。往往可用机器学习中的损失函数（代价函数）的值的大小来表征。在任务相关数据收集后，我们往往将来自同一分布的数据集分为三个不相交的三个子集：

+ 训练集（Train set）： 用于机器学习算法进行训练。 （训练误差）
+ 验证集（validation set）： 可用于机器学习算法或超参数选择。（验证误差）
+ 测试集（Test set）：评估机器学习算法的泛化性能。（泛化误差、测试误差）

其中验证集、测试集也称为 hold set，即不参与算法的实际训练过程，只用作对算法的结果进行评估。模型在不同数据集上的误差分别称为 训练误差、验证误差、测试误差（泛化误差）。

+ **训练误差（training error）**：模型在训练数据集上表现出的误差，也称为经验误差（empirical error）。
+ **验证误差（validation error）**： 模型在验证数据集上表现出的误差
+ **泛化误差（generalization error）**：模型在任意一个测试数据样本上表现出的误差的期望，并常常通过测试数据集上的误差（**测试误差**）来近似。

机器学习关注的是最小化泛化误差，泛化误差的大小通常也称为算法的泛化性能。

### （二）欠拟合、过拟合
为了最小化泛化误差，我们应该从训练样本中尽可能学出适用于所有潜在样本的"普遍规律"，这样才能在遇到新样本时做出正确的判别.但存在下面两种我们不希望的情况：

+ **欠拟合(underfitting)** ：模型没有很好地捕捉到数据的一般性质，不能够很好地拟合数据. 表征是模型的训练误差和验证误差都很大 （bias大）。
+ **过拟合(overfitting)**：当学习器把训练样本学得"太好"了的时候，很可能巳经把训练样本自身的一些特点当作了所有潜在样本都会具有的一般性质，这样就会导致泛化性能下降这种现象在机器学习中称为"过拟合" (overfitting). 表征是模型的训练误差远小于它在测试数据集上的误差（variance 大）。

1. 欠拟合
    产生原因：
    + 对于数据来说，模型过于简单
    + 数据的特征信息不够丰富

    解决方法：
    + 尝试更复杂的模型
    + 设计具有更高预测能力的特征
    + 软小的正则项（后面讲，让模型受的约束小些）

2. 过拟合
    产生原因：
    + 在对模型进行训练时，有可能遇到训练数据不够，即训练数据无法对整个数据的分布进行估计。
    + 对模型进行过度训练（overtraining）。

    解决方法：
    + 降低模型复杂度
    + 降低数据集中样本特征的维数
    + 添加更多的训练数据
    + 提高正则化水平

### （三）正则化方法
正则化方法包含了迫使学习算法构建不那么复杂的模型的方法。最常用的两种正则化类型是**L1正则化**和**L2正则化**。

#### 1. L1和L2正则项 

以线性回归为例。线性回归的目标:

$$\min _ { \mathbf { w } , b } \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left( f _ { \mathbf { w } , b } \left( \mathbf { x } _ { i } \right) - y _ { i } \right) ^ { 2 }$$

加入L1正则化的目标(lasso)：

$$
\min _ { \mathbf { w } , b } \left[ C | \mathbf { w } | + \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left( f _ { \mathbf { w } , b } \left( \mathbf { x } _ { i } \right) - y _ { i } \right) ^ { 2 } \right] \\ \text{其中}| \mathbf { w } | \stackrel { \mathrm { def } } { = } \sum _ { j = 1 } ^ { D } \left| w ^ { ( j ) } \right|
$$

作用：使得那些原先处于零（即|w|≈0）附近的参数w往零移动，使得部分参数为零，从而降低模型的复杂度（模型的复杂度由参数决定），从而防止过拟合，提高模型的泛化能力。  
L1正则化得到的是一个稀疏模型，可增加模型的可解释性

加入L2正则化的目标(岭正则化（ridge regualization）)：

$$
\min _ { \mathbf { w } , b } \left[ C \| \mathbf { w } \| ^ { 2 } + \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left( f _ { \mathbf { w } , b } \left( \mathbf { x } _ { i } \right) - y _ { i } \right) ^ { 2 } \right] , \\ \text{ 其中 }\| \mathbf { w } \| ^ { 2 } \stackrel { \mathrm { def } } { = } \sum _ { j = 1 } ^ { D } \left( w ^ { ( j ) } \right) ^ { 2 }
$$

作用：更小的参数值w意味着模型的复杂度更低，对训练数据的拟合刚刚好（奥卡姆剃刀），不会过分拟合训练数据，从而使得不会过拟合，以提高模型的泛化能力。   
L2正则项会产生比较小的解； L2可微，可利用梯度下降优化。

pytorch中可以在定义优化器时使用 L2， 称为 权重衰减（weight decay）
如 使用torch.optim的优化器，可如下设置L2正则化,下面代码中`weight_decay=0.01中的`相当于上式中的系数$C$:

```python
optimizer = optim.Adam(model.parameters(),lr=learning_rate,weight_decay=0.01)
```
+ 弹性网络正则化（elastic net regularization）=L1正则项+L2正则项

#### 2. Dropout (用在DNN中)
一个单隐藏层的多层感知机。其中输入个数为4，隐藏单元个数为5，且隐藏单元$h_i$（$i=1, \ldots, 5$）的计算表达式为

$$
h_i = \phi\left(x_1 w_{1i} + x_2 w_{2i} + x_3 w_{3i} + x_4 w_{4i} + b_i\right)
$$

这里$\phi$是激活函数，$x_1, \ldots, x_4$是输入，隐藏单元$i$的权重参数为$w_{1i}, \ldots, w_{4i}$，偏差参数为$b_i$。当对该隐藏层使用丢弃法时，该层的隐藏单元将有一定概率被丢弃掉。设丢弃概率为$p$，那么有$p$的概率$h_i$会被清零，有$1-p$的概率$h_i$会除以$1-p$做拉伸。丢弃概率是丢弃法的超参数。具体来说，设随机变量$\xi_i$为0和1的概率分别为$p$和$1-p$。使用丢弃法时我们计算新的隐藏单元$h_i'$

$$
h_i' = \frac{\xi_i}{1-p} h_i
$$

由于在训练中隐藏层神经元的丢弃是随机的，即$h_1, \ldots, h_5$都有可能被清零，输出层的计算无法过度依赖$h_1, \ldots, h_5$中的任一个，从而在训练模型时起到正则化的作用，并可以用来应对过拟合。在测试模型时，我们为了拿到更加确定性的结果，一般不使用丢弃法。下面的代码展示了dropout的操作。

```python
def dropout(X, drop_prob):
X = X.float()
assert 0 <= drop_prob <= 1
keep_prob = 1 - drop_prob
# 这种情况下把全部元素都丢弃
if keep_prob == 0:
return torch.zeros_like(X)
mask = (torch.rand(X.shape) < keep_prob).float()

return mask * X / keep_prob
```
在Pytorch中 采用
```python
pytorch.nn.Dropout(p: float, inplace: bool) -> None` #for 1d pytorch.nn.Dropout(p: float, inplace: bool) -> None` #for 2d
pytorch.nn.Dropout(p: float, inplace: bool) -> None` #for 3d
```
#### 3. Batchnormal (用在DNN中)
批归一化处理。基本思想类似于对输入数据做了标准化处理，即处理后的任意一个特征在数据集中所有样本上的均值为0、标准差为1。标准化处理输入数据使各个特征的分布相近，使得更容易训练出有效的模型。
考虑一个由$m$个样本组成的小批量，仿射变换的输出为一个新的小批量$\mathcal{B} = \{\boldsymbol{x}^{(1)}, \ldots, \boldsymbol{x}^{(m)} \}$。它们正是批量归一化层的输入。对于小批量$\mathcal{B}$中任意样本$\boldsymbol{x}^{(i)} \in \mathbb{R}^d, 1 \leq  i \leq m$，批量归一化层的输出同样是$d$维向量

$$\boldsymbol{y}^{(i)} = \text{BN}(\boldsymbol{x}^{(i)}),$$

并由以下几步求得。首先，对小批量$\mathcal{B}$求均值和方差：

$$\boldsymbol{\mu}_\mathcal{B} \leftarrow \frac{1}{m}\sum_{i = 1}^{m} \boldsymbol{x}^{(i)},$$
$$\boldsymbol{\sigma}_\mathcal{B}^2 \leftarrow \frac{1}{m} \sum_{i=1}^{m}(\boldsymbol{x}^{(i)} - \boldsymbol{\mu}_\mathcal{B})^2,$$

其中的平方计算是按元素求平方。接下来，使用按元素开方和按元素除法对$\boldsymbol{x}^{(i)}$标准化：

$$\hat{\boldsymbol{x}}^{(i)} \leftarrow \frac{\boldsymbol{x}^{(i)} - \boldsymbol{\mu}_\mathcal{B}}{\sqrt{\boldsymbol{\sigma}_\mathcal{B}^2 + \epsilon}},$$

这里$\epsilon > 0$是一个很小的常数，保证分母大于0。在上面标准化的基础上，批量归一化层引入了两个可以学习的模型参数，**拉伸（scale）参数 $\boldsymbol{\gamma}$ 和偏移（shift）参数 $\boldsymbol{\beta}$**。这两个参数和$\boldsymbol{x}^{(i)}$形状相同，皆为$d$维向量。它们与$\boldsymbol{x}^{(i)}$分别做按元素乘法（符号$\odot$）和加法计算：

$${\boldsymbol{y}}^{(i)} \leftarrow \boldsymbol{\gamma} \odot \hat{\boldsymbol{x}}^{(i)} + \boldsymbol{\beta}.$$

至此，我们得到了$\boldsymbol{x}^{(i)}$的批量归一化的输出$\boldsymbol{y}^{(i)}$。

在训练时，$\boldsymbol{\mu}_\mathcal{B}$,$\boldsymbol{\sigma}_\mathcal{B}^2$由批数据计算得到，在测试时，使用训练时的其值的移动平均代替。对于(BCHW) 形状的数据，**每个通道都拥有独立的拉伸和偏移参数，并均为标量**。设小批量中有$m$个样本。在单个通道上，假设卷积计算输出的高和宽分别为$p$和$q$。我们需要对该通道中$m \times p \times q$个元素同时做批量归一化。

在测试时，一种常用的方法是通过移动平均估算整个训练数据集的样本均值和方差，并在预测时使用它们得到确定的输出。

下面我们自己实现批量归一化层。

``` python
import time
import torch
from torch import nn, optim
import torch.nn.functional as F

import sys
sys.path.append("..") 
import d2lzh_pytorch as d2l
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def batch_norm(is_training, X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # 判断当前模式是训练模式还是预测模式
    if not is_training:
        # 如果是在预测模式下，直接使用传入的移动平均所得的均值和方差
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # 使用全连接层的情况，计算特征维上的均值和方差
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # 使用二维卷积层的情况，计算通道维上（axis=1）的均值和方差。这里我们需要保持
            # X的形状以便后面可以做广播运算
            mean = X.mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
            var = ((X - mean) ** 2).mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        # 训练模式下用当前的均值和方差做标准化
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # 更新移动平均的均值和方差
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # 拉伸和偏移
    return Y, moving_mean, moving_var
```

接下来，我们自定义一个`BatchNorm`层。它保存参与求梯度和迭代的拉伸参数`gamma`和偏移参数`beta`，同时也维护移动平均得到的均值和方差，以便能够在模型预测时被使用。`BatchNorm`实例所需指定的`num_features`参数对于全连接层来说应为输出个数，对于卷积层来说则为输出通道数。该实例所需指定的`num_dims`参数对于全连接层和卷积层来说分别为2和4。

``` python
class BatchNorm(nn.Module):
    def __init__(self, num_features, num_dims):
        super(BatchNorm, self).__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # 参与求梯度和迭代的拉伸和偏移参数，分别初始化成0和1
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # 不参与求梯度和迭代的变量，全在内存上初始化成0
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.zeros(shape)

    def forward(self, X):
        # 如果X不在内存上，将moving_mean和moving_var复制到X所在显存上
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # 保存更新过的moving_mean和moving_var, Module实例的traning属性默认为true, 调用.eval()后设成false
        Y, self.moving_mean, self.moving_var = batch_norm(self.training, 
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.9)
        return Y
```
Pytorch中的实现：

```python
torch.nn.BatchNorm1d(num_features: int, eps: float, momentum: float, affine: bool, track_running_stats: bool)
torch.nn.BatchNorm2d(num_features: int, eps: float, momentum: float, affine: bool, track_running_stats: bool)
torch.nn.BatchNorm3d(num_features: int, eps: float, momentum: float, affine: bool, track_running_stats: bool)

```
## 二、梯度消失、梯度爆炸
**当神经网络的层数较多时，模型的数值稳定性容易变差。**

假设一个层数为$L$的多层感知机的第$l$层$\boldsymbol{H}^{(l)}$的权重参数为$\boldsymbol{W}^{(l)}$，输出层$\boldsymbol{H}^{(L)}$的权重参数为$\boldsymbol{W}^{(L)}$。为了便于讨论，不考虑偏差参数，且设所有隐藏层的激活函数为恒等映射（identity mapping）$\phi(x) = x$。给定输入$\boldsymbol{X}$，多层感知机的第$l$层的输出$\boldsymbol{H}^{(l)} = \boldsymbol{X} \boldsymbol{W}^{(1)} \boldsymbol{W}^{(2)} \ldots \boldsymbol{W}^{(l)}$。此时，如果层数$l$较大，$\boldsymbol{H}^{(l)}$的计算可能会出现衰减或爆炸。举个例子，假设输入和所有层的权重参数都是标量，如权重参数为0.2和5，多层感知机的第30层输出为输入$\boldsymbol{X}$分别与$0.2^{30} \approx 1 \times 10^{-21}$（**消失**）和$5^{30} \approx 9 \times 10^{20}$（**爆炸**）的乘积。当层数较多时，梯度的计算也容易出现消失或爆炸。

在很深的深度神经网络中通过加入 **跨层** 连接 如 ResNet，减少 梯度消失。

在RNN中通过加入 裁剪梯度（clip gradient） 应对梯度爆炸。假设我们把所有模型参数的梯度拼接成一个向量 $\boldsymbol{g}$，并设裁剪的阈值是$\theta$。裁剪后的梯度

$$
 \min\left(\frac{\theta}{\|\boldsymbol{g}\|}, 1\right)\boldsymbol{g}
$$
的$L_2$范数不超过$\theta$。

```python
# pytorch中的相关实现
#Clips gradient norm of an iterable of parameters.
torch.nn.utils.clip_grad_norm(parameters, max_norm, norm_type=2) 
# Clips gradient of an iterable of parameters at specified value.
torch.nn.utils.clip_grad_value_(parameters, clip_value) 
```

## 三、循环神经网络
###  （一）GRU
循环神经网络中的梯度计算方法。我们发现，当时间步数较大或者时间步较小时，循环神经网络的梯度较容易出现衰减或爆炸。虽然裁剪梯度可以应对梯度爆炸，但无法解决梯度衰减的问题。通常由于这个原因，循环神经网络在实际中较难捕捉时间序列中时间步距离较大的依赖关系。

门控循环神经网络（gated recurrent neural network）的提出，正是为了更好地捕捉时间序列中时间步距离较大的依赖关系。它通过可以学习的门来控制信息的流动。其中，门控循环单元（gated recurrent unit，GRU）是一种常用的门控循环神经网络.


下面将介绍门控循环单元的设计。它引入了重置门（reset gate）和更新门（update gate）的概念，从而修改了循环神经网络中隐藏状态的计算方式。

#### 1.  重置门和更新门

门控循环单元中的重置门和更新门的输入均为当前时间步输入$\boldsymbol{X}_t$与上一时间步隐藏状态$\boldsymbol{H}_{t-1}$，输出由激活函数为sigmoid函数的全连接层计算得到。

![ 门控循环单元中重置门和更新门的计算](/images/fit_gradient_rnn/gru_1.svg)

具体来说，假设隐藏单元个数为$h$，给定时间步$t$的小批量输入$\boldsymbol{X}_t \in \mathbb{R}^{n \times d}$（样本数为$n$，输入个数为$d$）和上一时间步隐藏状态$\boldsymbol{H}_{t-1} \in \mathbb{R}^{n \times h}$。重置门$\boldsymbol{R}_t \in \mathbb{R}^{n \times h}$和更新门$\boldsymbol{Z}_t \in \mathbb{R}^{n \times h}$的计算如下：

$$
\begin{aligned}
\boldsymbol{R}_t = \sigma(\boldsymbol{X}_t \boldsymbol{W}_{xr} + \boldsymbol{H}_{t-1} \boldsymbol{W}_{hr} + \boldsymbol{b}_r),\\
\boldsymbol{Z}_t = \sigma(\boldsymbol{X}_t \boldsymbol{W}_{xz} + \boldsymbol{H}_{t-1} \boldsymbol{W}_{hz} + \boldsymbol{b}_z),
\end{aligned}
$$

其中$\boldsymbol{W}_{xr}, \boldsymbol{W}_{xz} \in \mathbb{R}^{d \times h}$和$\boldsymbol{W}_{hr}, \boldsymbol{W}_{hz} \in \mathbb{R}^{h \times h}$是权重参数，$\boldsymbol{b}_r, \boldsymbol{b}_z \in \mathbb{R}^{1 \times h}$是偏差参数。3.8节（多层感知机）节中介绍过，sigmoid函数可以将元素的值变换到0和1之间。因此，重置门$\boldsymbol{R}_t$和更新门$\boldsymbol{Z}_t$中每个元素的值域都是$[0, 1]$。

####  2.  候选隐藏状态

接下来，门控循环单元将计算候选隐藏状态来辅助稍后的隐藏状态计算。如图6.5所示，我们将当前时间步重置门的输出与上一时间步隐藏状态做按元素乘法（符号为$\odot$）。如果重置门中元素值接近0，那么意味着重置对应隐藏状态元素为0，即丢弃上一时间步的隐藏状态。如果元素值接近1，那么表示保留上一时间步的隐藏状态。然后，将按元素乘法的结果与当前时间步的输入连结，再通过含激活函数tanh的全连接层计算出候选隐藏状态，其所有元素的值域为$[-1, 1]$。

![门控循环单元中候选隐藏状态的计算](/images/fit_gradient_rnn/gru_2.svg)

具体来说，时间步$t$的候选隐藏状态$\tilde{\boldsymbol{H}}_t \in \mathbb{R}^{n \times h}$的计算为

$$
\tilde{\boldsymbol{H}}_t = \text{tanh}(\boldsymbol{X}_t \boldsymbol{W}_{xh} + \left(\boldsymbol{R}_t \odot \boldsymbol{H}_{t-1}\right) \boldsymbol{W}_{hh} + \boldsymbol{b}_h),
$$

其中$\boldsymbol{W}_{xh} \in \mathbb{R}^{d \times h}$和$\boldsymbol{W}_{hh} \in \mathbb{R}^{h \times h}$是权重参数，$\boldsymbol{b}_h \in \mathbb{R}^{1 \times h}$是偏差参数。从上面这个公式可以看出，重置门控制了上一时间步的隐藏状态如何流入当前时间步的候选隐藏状态。而上一时间步的隐藏状态可能包含了时间序列截至上一时间步的全部历史信息。因此，重置门可以用来丢弃与预测无关的历史信息。

#### 3. 隐藏状态

最后，时间步$t$的隐藏状态$\boldsymbol{H}_t \in \mathbb{R}^{n \times h}$的计算使用当前时间步的更新门$\boldsymbol{Z}_t$来对上一时间步的隐藏状态$\boldsymbol{H}_{t-1}$和当前时间步的候选隐藏状态$\tilde{\boldsymbol{H}}_t$做组合：

$$
\boldsymbol{H}_t = \boldsymbol{Z}_t \odot \boldsymbol{H}_{t-1}  + (1 - \boldsymbol{Z}_t) \odot \tilde{\boldsymbol{H}}_t.
$$

![ 门控循环单元中隐藏状态的计算](/images/fit_gradient_rnn/gru_3.svg)

值得注意的是，更新门可以控制隐藏状态应该如何被包含当前时间步信息的候选隐藏状态所更新，如图6.6所示。假设更新门在时间步$t'$到$t$（$t' < t$）之间一直近似1。那么，在时间步$t'$到$t$之间的输入信息几乎没有流入时间步$t$的隐藏状态$\boldsymbol{H}_t$。实际上，这可以看作是较早时刻的隐藏状态$\boldsymbol{H}_{t'-1}$一直通过时间保存并传递至当前时间步$t$。这个设计可以应对循环神经网络中的梯度衰减问题，并更好地捕捉时间序列中时间步距离较大的依赖关系。

门控循环单元总结：
* 重置门有助于捕捉时间序列里短期的依赖关系；
* 更新门有助于捕捉时间序列里长期的依赖关系。

####  4. GRU的实现

```python
num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
print('will use', device)

def get_params():  
    def _one(shape):
        ts = torch.tensor(np.random.normal(0, 0.01, size=shape), device=device, dtype=torch.float32) #正态分布
        return torch.nn.Parameter(ts, requires_grad=True)
    def _three():
        return (_one((num_inputs, num_hiddens)),
                _one((num_hiddens, num_hiddens)),
                torch.nn.Parameter(torch.zeros(num_hiddens, device=device, dtype=torch.float32), requires_grad=True))
     
    W_xz, W_hz, b_z = _three()  # 更新门参数
    W_xr, W_hr, b_r = _three()  # 重置门参数
    W_xh, W_hh, b_h = _three()  # 候选隐藏状态参数
    
    # 输出层参数
    W_hq = _one((num_hiddens, num_outputs))
    b_q = torch.nn.Parameter(torch.zeros(num_outputs, device=device, dtype=torch.float32), requires_grad=True)
    return nn.ParameterList([W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q])

def init_gru_state(batch_size, num_hiddens, device):   #隐藏状态初始化
    return (torch.zeros((batch_size, num_hiddens), device=device), )
```

下面的代码定义隐藏状态初始化函数`init_gru_state`。它返回由一个形状为(批量大小, 隐藏单元个数)的值为0的`Tensor`组成的元组。

``` python
def init_gru_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )
```

下面根据门控循环单元的计算表达式定义模型。

``` python
def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        Z = torch.sigmoid(torch.matmul(X, W_xz) + torch.matmul(H, W_hz) + b_z)
        R = torch.sigmoid(torch.matmul(X, W_xr) + torch.matmul(H, W_hr) + b_r)
        H_tilda = torch.tanh(torch.matmul(X, W_xh) + torch.matmul(R * H, W_hh) + b_h)
        H = Z * H + (1 - Z) * H_tilda
        Y = torch.matmul(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, (H,)
```

### （二）LSTM 长短期记忆

LSTM 中引入了3个门，即输入门（input gate）、遗忘门（forget gate）和输出门（output gate），以及与隐藏状态形状相同的记忆细胞（某些文献把记忆细胞当成一种特殊的隐藏状态），从而记录额外的信息。

#### 1.  输入门、遗忘门和输出门

与门控循环单元中的重置门和更新门一样，如图6.7所示，长短期记忆的门的输入均为当前时间步输入$\boldsymbol{X}_t$与上一时间步隐藏状态$\boldsymbol{H}_{t-1}$，输出由激活函数为sigmoid函数的全连接层计算得到。如此一来，这3个门元素的值域均为$[0,1]$。
![长短期记忆中输入门、遗忘门和输出门的计算](/images/fit_gradient_rnn/lstm_0.svg)

具体来说，假设隐藏单元个数为$h$，给定时间步$t$的小批量输入$\boldsymbol{X}_t \in \mathbb{R}^{n \times d}$（样本数为$n$，输入个数为$d$）和上一时间步隐藏状态$\boldsymbol{H}_{t-1} \in \mathbb{R}^{n \times h}$。
时间步$t$的输入门$\boldsymbol{I}_t \in \mathbb{R}^{n \times h}$、遗忘门$\boldsymbol{F}_t \in \mathbb{R}^{n \times h}$和输出门$\boldsymbol{O}_t \in \mathbb{R}^{n \times h}$分别计算如下：

$$
\begin{aligned}
\boldsymbol{I}_t &= \sigma(\boldsymbol{X}_t \boldsymbol{W}_{xi} + \boldsymbol{H}_{t-1} \boldsymbol{W}_{hi} + \boldsymbol{b}_i),\\
\boldsymbol{F}_t &= \sigma(\boldsymbol{X}_t \boldsymbol{W}_{xf} + \boldsymbol{H}_{t-1} \boldsymbol{W}_{hf} + \boldsymbol{b}_f),\\
\boldsymbol{O}_t &= \sigma(\boldsymbol{X}_t \boldsymbol{W}_{xo} + \boldsymbol{H}_{t-1} \boldsymbol{W}_{ho} + \boldsymbol{b}_o),
\end{aligned}
$$

其中的$\boldsymbol{W}_{xi}, \boldsymbol{W}_{xf}, \boldsymbol{W}_{xo} \in \mathbb{R}^{d \times h}$和$\boldsymbol{W}_{hi}, \boldsymbol{W}_{hf}, \boldsymbol{W}_{ho} \in \mathbb{R}^{h \times h}$是权重参数，$\boldsymbol{b}_i, \boldsymbol{b}_f, \boldsymbol{b}_o \in \mathbb{R}^{1 \times h}$是偏差参数。


#### 2. 候选记忆细胞

接下来，长短期记忆需要计算候选记忆细胞$\tilde{\boldsymbol{C}}_t$。它的计算与上面介绍的3个门类似，但使用了值域在$[-1, 1]$的tanh函数作为激活函数，如图6.8所示。

![长短期记忆中候选记忆细胞的计算](/images/fit_gradient_rnn/lstm_1.svg)

具体来说，时间步$t$的候选记忆细胞$\tilde{\boldsymbol{C}}_t \in \mathbb{R}^{n \times h}$的计算为

$$
\tilde{\boldsymbol{C}}_t = \text{tanh}(\boldsymbol{X}_t \boldsymbol{W}_{xc} + \boldsymbol{H}_{t-1} \boldsymbol{W}_{hc} + \boldsymbol{b}_c),
$$

其中$\boldsymbol{W}_{xc} \in \mathbb{R}^{d \times h}$和$\boldsymbol{W}_{hc} \in \mathbb{R}^{h \times h}$是权重参数，$\boldsymbol{b}_c \in \mathbb{R}^{1 \times h}$是偏差参数。


#### 3. 记忆细胞

我们可以通过元素值域在$[0, 1]$的输入门、遗忘门和输出门来控制隐藏状态中信息的流动，这一般也是通过使用按元素乘法（符号为$\odot$）来实现的。当前时间步记忆细胞$\boldsymbol{C}_t \in \mathbb{R}^{n \times h}$的计算组合了上一时间步记忆细胞和当前时间步候选记忆细胞的信息，并通过遗忘门和输入门来控制信息的流动：

$$\boldsymbol{C}_t = \boldsymbol{F}_t \odot \boldsymbol{C}_{t-1} + \boldsymbol{I}_t \odot \tilde{\boldsymbol{C}}_t.$$


如图6.9所示，遗忘门控制上一时间步的记忆细胞$\boldsymbol{C}_{t-1}$中的信息是否传递到当前时间步，而输入门则控制当前时间步的输入$\boldsymbol{X}_t$通过候选记忆细胞$\tilde{\boldsymbol{C}}_t$如何流入当前时间步的记忆细胞。如果遗忘门一直近似1且输入门一直近似0，过去的记忆细胞将一直通过时间保存并传递至当前时间步。这个设计可以应对循环神经网络中的梯度衰减问题，并更好地捕捉时间序列中时间步距离较大的依赖关系。

![长短期记忆中记忆细胞的计算](/images/fit_gradient_rnn/lstm_2.svg)

#### 4.  隐藏状态

有了记忆细胞以后，接下来我们还可以通过输出门来控制从记忆细胞到隐藏状态$\boldsymbol{H}_t \in \mathbb{R}^{n \times h}$的信息的流动：

$$\boldsymbol{H}_t = \boldsymbol{O}_t \odot \text{tanh}(\boldsymbol{C}_t).$$

这里的tanh函数确保隐藏状态元素值在-1到1之间。需要注意的是，当输出门近似1时，记忆细胞信息将传递到隐藏状态供输出层使用；当输出门近似0时，记忆细胞信息只自己保留。图6.10展示了长短期记忆中隐藏状态的计算。

![长短期记忆中隐藏状态的计算](/images/fit_gradient_rnn/lstm_3.svg)

#### 5. LSTM实现
```python
## 定义参数
def get_params():
    def _one(shape):
        ts = torch.tensor(np.random.normal(0, 0.01, size=shape), device=device, dtype=torch.float32)
        return torch.nn.Parameter(ts, requires_grad=True)
    def _three():
        return (_one((num_inputs, num_hiddens)),
                _one((num_hiddens, num_hiddens)),
                torch.nn.Parameter(torch.zeros(num_hiddens, device=device, dtype=torch.float32), requires_grad=True))
    
    W_xi, W_hi, b_i = _three()  # 输入门参数
    W_xf, W_hf, b_f = _three()  # 遗忘门参数
    W_xo, W_ho, b_o = _three()  # 输出门参数
    W_xc, W_hc, b_c = _three()  # 候选记忆细胞参数
    
    # 输出层参数
    W_hq = _one((num_hiddens, num_outputs))
    b_q = torch.nn.Parameter(torch.zeros(num_outputs, device=device, dtype=torch.float32), requires_grad=True)
    return nn.ParameterList([W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q])
```
在初始化函数中，长短期记忆的隐藏状态需要返回额外的形状为(批量大小, 隐藏单元个数)的值为0的记忆细胞。

``` python
def init_lstm_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), 
            torch.zeros((batch_size, num_hiddens), device=device))
```

定义lstm

```python
def lstm(inputs, state, params):
    [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q] = params
    (H, C) = state
    outputs = []
    for X in inputs:
        I = torch.sigmoid(torch.matmul(X, W_xi) + torch.matmul(H, W_hi) + b_i)
        F = torch.sigmoid(torch.matmul(X, W_xf) + torch.matmul(H, W_hf) + b_f)
        O = torch.sigmoid(torch.matmul(X, W_xo) + torch.matmul(H, W_ho) + b_o)
        C_tilda = torch.tanh(torch.matmul(X, W_xc) + torch.matmul(H, W_hc) + b_c)
        C = F * C + I * C_tilda
        H = O * C.tanh()
        Y = torch.matmul(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, (H, C)
```

pytorch中的实现：
```python
# Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.
torch.nn.LSTM(input_size: int, hidden_size: int, num_layers: int, bias: bool, batch_first: bool, dropout: float, bidirectional: bool, nonlinearity: str) -> None 

```

## 四、现代RNN的语言模型的pytorch实现

### （一） 数据准备

#### 1. 建立字符索引

将每个字符映射成一个从0开始的连续整数，又称索引，来方便之后的数据处理。为了得到索引，我们将数据集里所有不同字符取出来，然后将其逐一映射到索引来构造词典。接着，打印`vocab_size`，即词典中不同字符的个数，又称词典大小。

``` python
#假设读入的字符数据在 corpus_chars 中
idx_to_char = list(set(corpus_chars)) #利用集合建立索引->字符，也称词典
char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)]) ##字符->索引
vocab_size = len(char_to_idx) ##词典的大小

```

之后，将训练数据集中每个字符转化为索引。

``` python
# 训练数据集中每个字符转化为索引 在 corpus_indices 中
corpus_indices = [char_to_idx[char] for char in corpus_chars] 

## 测试，并打印前20个字符及其对应的索引。
sample = corpus_indices[:20]
print('chars:', ''.join([idx_to_char[idx] for idx in sample]))
print('indices:', sample)
```

#### 2. 时序数据采样

+ 随机采样

  每次从数据里随机采样一个小批量。其中批量大小`batch_size`指每个小批量的样本数，`num_steps`为每个样本所包含的时间步数。
  在随机采样中，每个样本是原始序列上任意截取的一段序列。相邻的两个随机小批量在原始序列上的位置不一定相毗邻。在训练模型时，每次随机采样前都需要重新初始化隐藏状态。

  ```python
  def data_iter_random(corpus_indices, batch_size, num_steps, device=None):
      # 减1是因为输出的索引x是相应输入的索引y加1
      num_examples = (len(corpus_indices) - 1) // num_steps
      epoch_size = num_examples // batch_size
      example_indices = list(range(num_examples))
      random.shuffle(example_indices)
  
      # 返回从pos开始的长为num_steps的序列
      def _data(pos):
          return corpus_indices[pos: pos + num_steps]
      if device is None:
          device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      
      for i in range(epoch_size):
          # 每次读取batch_size个随机样本
          i = i * batch_size
          batch_indices = example_indices[i: i + batch_size]
          X = [_data(j * num_steps) for j in batch_indices]
          Y = [_data(j * num_steps + 1) for j in batch_indices]
          yield torch.tensor(X, dtype=torch.float32, device=device), torch.tensor(Y, dtype=torch.float32, device=device)
  ```

+ 相邻采样
 令相邻的两个随机小批量在原始序列上的位置相毗邻。这时候，我们就可以用一个小批量最终时间步的隐藏状态来初始化下一个小批量的隐藏状态，从而使下一个小批量的输出也取决于当前小批量的输入，并如此循环下去。

  ```python
  def data_iter_consecutive(corpus_indices, batch_size, num_steps, device=None):
      if device is None:
          device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      corpus_indices = torch.tensor(corpus_indices, dtype=torch.float32, device=device)
      data_len = len(corpus_indices)
      batch_len = data_len // batch_size
      indices = corpus_indices[0: batch_size*batch_len].view(batch_size, batch_len)
      epoch_size = (batch_len - 1) // num_steps
      for i in range(epoch_size):
          i = i * num_steps
          X = indices[:, i: i + num_steps]
          Y = indices[:, i + 1: i + num_steps + 1]
          yield X, Y
  ```

  #### 3. one-hot向量

  为了将词表示成向量输入到神经网络，一个简单的办法是使用one-hot向量。假设词典中不同字符的数量为$N$（即词典大小`vocab_size`），每个字符已经同一个从0到$N-1$的连续整数值索引一一对应。如果一个字符的索引是整数$i$, 那么我们创建一个全0的长为$N$的向量，并将其位置为$i$的元素设成1。该向量就是对原字符的one-hot向量。

  ```python
  # pytorch实现
  torch.nn.functional.one_hot(tensor, num_classes=[...])
  ##基本原理类似以下的实现
  def one_hot(x, n_class, dtype=torch.float32): 
      # X shape: (batch), output shape: (batch, n_class)
      x = x.long()
      res = torch.zeros(x.shape[0], n_class, dtype=dtype, device=x.device)
      res.scatter_(1, x.view(-1, 1), 1)
      return res
  
  ```


  从步骤2采样的得到小批量的训练数据$[X,y]$中$X$的形状是*(batch, seq_len )*。下面的函数将这样的小批量变换成数个可以输入进网络的形状为 seq_len 个(batch, vocab_size)的矩阵。也就是说，每个时间步$t$的输入为$\boldsymbol{X}_t \in \mathbb{R}^{n \times d}$，其中$n$为批量大小，$d$为输入个数，即one-hot向量长度（词典大小）。

  ``` python
  def to_onehot(X, n_class):  
      # X shape: (batch, seq_len), output: seq_len elements of (batch, n_class)
      return [one_hot(X[:, i], n_class) for i in range(X.shape[1])]
  
  ```

  经过这样处理后，就可以作为RNN的输入了。

### （二）基本框架

1. 一个完整的完整的基于循环神经网络的语言模型。**

```python
class RNNModel(nn.Module):
    def __init__(self, rnn_layer, vocab_size):
        super(RNNModel, self).__init__()
        self.rnn = rnn_layer
        self.hidden_size = rnn_layer.hidden_size * (2 if rnn_layer.bidirectional else 1) 
        self.vocab_size = vocab_size
        self.dense = nn.Linear(self.hidden_size, vocab_size)

    def forward(self, inputs, state):
        # inputs.shape: (batch_size, num_steps)
        X = to_onehot(inputs, vocab_size)
        X = torch.stack(X)  # X.shape: (num_steps, batch_size, vocab_size)
        hiddens, state = self.rnn(X, state)
        hiddens = hiddens.view(-1, hiddens.shape[-1])  # hiddens.shape: (num_steps * batch_size, hidden_size)
        output = self.dense(hiddens)
        return output, state
```

2. 训练和预测函数

   ```python
   def train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device,corpus_indices, idx_to_char, char_to_idx,
    num_epochs, num_steps, lr, clipping_theta,batch_size, pred_period, pred_len, prefixes):
       loss = nn.CrossEntropyLoss()
       optimizer = torch.optim.Adam(model.parameters(), lr=lr)
       model.to(device)
       for epoch in range(num_epochs):
           l_sum, n, start = 0.0, 0, time.time()
           data_iter = d2l.data_iter_consecutive(corpus_indices, batch_size, num_steps, device) # 相邻采样
           state = None
           for X, Y in data_iter:
               if state is not None:
                   # 使用detach函数从计算图分离隐藏状态
                   if isinstance (state, tuple): # LSTM, state:(h, c)  
                       state[0].detach_()
                       state[1].detach_()
                   else: 
                       state.detach_()
               (output, state) = model(X, state) # output.shape: (num_steps * batch_size, vocab_size)
               y = torch.flatten(Y.T)
               l = loss(output, y.long())
               
               optimizer.zero_grad()
               l.backward()
               grad_clipping(model.parameters(), clipping_theta, device)
               optimizer.step()
               l_sum += l.item() * y.shape[0]
               n += y.shape[0]
           
   		## 预测 （如果不需要，可以用作验证训练结果）
           if (epoch + 1) % pred_period == 0:
               print('epoch %d, perplexity %f, time %.2f sec' % (epoch + 1, math.exp(l_sum / n), time.time() - start))
               for prefix in prefixes:
                   print(' -', predict_rnn_pytorch(
                       prefix, pred_len, model, vocab_size, device, idx_to_char, char_to_idx))
   ```

   进行训练：

   ```python
   num_epochs, batch_size, lr, clipping_theta = 250, 32, 1e-3, 1e-2
   pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']
   train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device,corpus_indices, idx_to_char, char_to_idx,num_epochs, num_steps, lr, clipping_theta, batch_size, pred_period, pred_len, prefixes)
   ```

   

3. 预测函数

   实现一个预测函数，与前面的模型的区别在于前向计算和初始化隐藏状态。

   ```python
   def predict_rnn_pytorch(prefix, num_chars, model, vocab_size, device, idx_to_char,
                         char_to_idx):
       state = None
       output = [char_to_idx[prefix[0]]]  # output记录prefix加上预测的num_chars个字符
       for t in range(num_chars + len(prefix) - 1):
           X = torch.tensor([output[-1]], device=device).view(1, 1)
           (Y, state) = model(X, state)  # 前向计算不需要传入模型参数
           if t < len(prefix) - 1:
               output.append(char_to_idx[prefix[t + 1]])
           else:
               output.append(Y.argmax(dim=1).item())
       return ''.join([idx_to_char[i] for i in output])
   ```

   使用方法：

   ```python
   model = RNNModel(rnn_layer, vocab_size).to(device) 
   # 在此可以调入已训练的模型
   predict_rnn_pytorch('分开', 10, model, vocab_size, device, idx_to_char, char_to_idx)
   ```
### （三）现代RNN的实现的语言模型
#### 1. GRU

```python
num_hiddens=256
num_epochs, num_steps, batch_size, lr, clipping_theta = 160, 35, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 40, 50, ['分开', '不分开']

lr = 1e-2 # 注意调整学习率

## 注意此处理不同-------------
gru_layer = torch.nn.GRU(input_size=vocab_size, hidden_size=num_hiddens)
model = RNNModel(gru_layer, vocab_size)
## --------------------------

train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device,corpus_indices, idx_to_char, char_to_idx,num_epochs, num_steps, lr, clipping_theta,batch_size, pred_period, pred_len, prefixes)
```



#### 2. LSTM

```python
num_hiddens=256
num_epochs, num_steps, batch_size, lr, clipping_theta = 160, 35, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 40, 50, ['分开', '不分开']

lr = 1e-2 # 注意调整学习率

## 注意此处理不同-------------
lstm_layer = torch.nn.LSTM(input_size=vocab_size, hidden_size=num_hiddens)
model = RNNModel(lstm_layer, vocab_size)
## --------------------------

train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device,corpus_indices, idx_to_char, char_to_idx,num_epochs, num_steps, lr, clipping_theta,batch_size, pred_period, pred_len, prefixes)
```



#### 3.  深度循环神经网络

由多层lstm或 gru组成。只需添加下式中的`num_lauers`参数，并指明深度的层数即可。

```
## gru多层
gru_layer = torch.nn.GRU(input_size=vocab_size, hidden_size=num_hiddens,num_layers=6)
## lstm多层
lstm_layer = torch.nn.LSTM(input_size=vocab_size, hidden_size=num_hiddens,num_layers=2)
```

#### 4. 双向循环神经网络 

由同层的循环神经单元之间的输出同时具有前向和后向连接。只面添加下式中的`bidirectional=True`参数即可。

```python
## gru双向
gru_layer = nn.GRU(input_size=vocab_size, hidden_size=num_hiddens,bidirectional=True)
## lstm双向
lstm_layer = torch.nn.LSTM(input_size=vocab_size,bidirectional=True)

```

