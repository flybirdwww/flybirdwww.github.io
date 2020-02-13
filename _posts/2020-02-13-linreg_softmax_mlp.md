# linreg-softmax-mlp_Pytorch

Pytorch学习笔记系列。本文目录：

1. TOC
{:toc}


## 线性回归

### 原理
线性回归是机器学习解决回归问题，即有监督学习，预测实数值的问题的最简单方法。它以数据中的特征（原始或变化后的）的线性组合建立机器学习模型。广义上讲，当数据样本数为 $n$，特征数为 $d$ 时，线性回归的矢量计算表达式为
$$
\boldsymbol{\hat{y}} = \boldsymbol{X} \boldsymbol{w} + b
$$
其中 批量数据样本特征 $\boldsymbol{X} \in \mathbb{R}^{n \times d}$，批量数据样本标签 $\boldsymbol{y} \in \mathbb{R}^{n \times 1}$，权重 $\boldsymbol{w} \in \mathbb{R}^{d \times 1}$， 偏差 $b \in \mathbb{R}$。相应地，模型输出 $\boldsymbol{\hat{y}} \in \mathbb{R}^{n \times 1}$ 。设模型参数 $\boldsymbol{\theta} = [w_1, w_2, b]^\top$，我们可以重写损失函数一般采用 **平方误差损失函数(Mean Square Error, MSE)**, 即
$$
\ell(\boldsymbol{\theta})=\frac{1}{2n}(\boldsymbol{\hat{y}}-\boldsymbol{y})^\top(\boldsymbol{\hat{y}}-\boldsymbol{y})
$$

这类问题一般有解析解（即最小二乘法），也可以用数值方法求解（如梯度下降方法），采用小批量随机梯度下降的迭代步骤将相应地改写为
$$
\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} -   \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}}   \nabla_{\boldsymbol{\theta}} \ell^{(i)}(\boldsymbol{\theta}),
$$
其中$|\mathcal{B}|$表示批处理的大小，$\eta$ 表示学习率。
### Pytorch实现
表达式$\boldsymbol{\hat{y}} = \boldsymbol{X} \boldsymbol{w} + b$ 中在Pytorch中可以用一个有d维输入，只有一个具有线性输出的神经节点组成的神经网络表示。$\boldsymbol{w}$ 表示d个输入到输出节点的权重值，$b$ 则为输出节点的‘bias’。因此可直接采用 `torch.nn.Linear` 实现。

```python
class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, 1)
    # forward 定义前向传播
    def forward(self, x):
        y = self.linear(x)
        return y
net = LinearNet(num_inputs)
```
或者

```python
net = nn.Sequential(
    nn.Linear(num_inputs, 1)
    # 此处还可以传入其他层
    )
```

MSEloss：

```python
torch.nn.MSELoss()
```

## softmax问题 

**用于解决多个分类的预测问题**。假设有三个类别的输出，分别对应 $o_1, o_2, o_3$。softmax运算符（softmax operator）通过下式将输出值变换成值为正且和为1的概率分布：
$$
\hat{y}_1, \hat{y}_2, \hat{y}_3 = \text{softmax}(o_1, o_2, o_3)
$$

其中

$$
\hat{y}_1 = \frac{ \exp(o_1)}{\sum_{i=1}^3 \exp(o_i)},\quad
\hat{y}_2 = \frac{ \exp(o_2)}{\sum_{i=1}^3 \exp(o_i)},\quad
\hat{y}_3 = \frac{ \exp(o_3)}{\sum_{i=1}^3 \exp(o_i)}.
$$

容易看出$\hat{y}_1 + \hat{y}_2 + \hat{y}_3 = 1$且$0 \leq \hat{y}_1, \hat{y}_2, \hat{y}_3 \leq 1$，因此$\hat{y}_1, \hat{y}_2, \hat{y}_3$是一个合法的概率分布。

### softmax 回归问题--用于线性多分类：

给定一个小批量样本，其批量大小为$n$，输入个数（特征数）为$d$，输出个数（类别数）为$q$。设批量特征为$\boldsymbol{X} \in \mathbb{R}^{n \times d}$。假设softmax回归的权重和偏差参数分别为$\boldsymbol{W} \in \mathbb{R}^{d \times q}$和$\boldsymbol{b} \in \mathbb{R}^{1 \times q}$。softmax回归的矢量计算表达式为
$$
\begin{aligned}
\boldsymbol{O} &= \boldsymbol{X} \boldsymbol{W} + \boldsymbol{b},\\
\boldsymbol{\hat{Y}} &= \text{softmax}(\boldsymbol{O}),
\end{aligned}
$$

其中的加法运算使用了广播机制，$\boldsymbol{O}, \boldsymbol{\hat{Y}} \in \mathbb{R}^{n \times q}$且这两个矩阵的第$i$行分别为样本$i$的输出$\boldsymbol{o}^{(i)}$和概率分布$\boldsymbol{\hat{y}}^{(i)}$。softmax回归可以看作是一个单层神经网络，输出个数等于分类问题中的类别个数。

$\boldsymbol{O} = \boldsymbol{X} \boldsymbol{W} + \boldsymbol{b}$的实现：

```python
class LinearNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)
    def forward(self, x): # x shape: (batch, 1, 28, 28)
        y = self.linear(x.view(x.shape[0], -1))
        return y
    
net = LinearNet(num_inputs, num_outputs)

```

$\boldsymbol{\hat{Y}} = \text{softmax}(\boldsymbol{O})$的实现：

```python
torch.nn.Softmax(net)
```

### 损失函数：交叉熵损失函数

交叉熵（cross entropy）是一个常用的衡量两个概率分布的差异的方法。

$$H\left(\boldsymbol y^{(i)}, \boldsymbol {\hat y}^{(i)}\right ) = -\sum_{j=1}^q y_j^{(i)} \log \hat y_j^{(i)}= -\log \hat y_{y^{(i)}}^{(i)},$$

其中带下标的$y_j^{(i)}$是向量$\boldsymbol y^{(i)}$中非0即1的元素，需要注意将它与样本$i$类别的离散数值，即不带下标的$y^{(i)}$区分。在上式中，我们知道向量$\boldsymbol y^{(i)}$中只有第$y^{(i)}$个元素$y^{(i)}_{y^{(i)}}$为1，其余全为0，于是$H(\boldsymbol y^{(i)}, \boldsymbol {\hat y}^{(i)}) = -\log \hat y_{y^{(i)}}^{(i)}$。也就是说，交叉熵只关心对正确类别的预测概率，因为只要其值足够大，就可以确保分类结果正确。当然，遇到一个样本有多个标签时，例如图像里含有不止一个物体时，我们并不能做这一步简化。但即便对于这种情况，交叉熵同样只关心对图像中出现的物体类别的预测概率。

假设训练数据集的样本数为$n$，交叉熵损失函数定义为
$$\ell(\boldsymbol{\Theta}) = \frac{1}{n} \sum_{i=1}^n H\left(\boldsymbol y^{(i)}, \boldsymbol {\hat y}^{(i)}\right ),$$

其中$\boldsymbol{\Theta}$代表模型参数。同样地，如果每个样本只有一个标签，那么交叉熵损失可以简写成$\ell(\boldsymbol{\Theta}) = -(1/n)  \sum_{i=1}^n \log \hat y_{y^{(i)}}^{(i)}$。从另一个角度来看，我们知道最小化$\ell(\boldsymbol{\Theta})$等价于最大化$\exp(-n\ell(\boldsymbol{\Theta}))=\prod_{i=1}^n \hat y_{y^{(i)}}^{(i)}$，即最小化交叉熵损失函数等价于最大化训练数据集所有标签类别的联合预测概率。

由于分开定义softmax运算和交叉熵损失函数可能会造成数值不稳定。因此，PyTorch提供了一个包括**softmax运算和交叉熵损失计算的函数。它的数值稳定性更好。**那么下面的函数可以即可用于前面的 softmax 回归问题（线性多分类），也可以 前接深度网络实现 深度非线性多分类问题，**后一种用得更广泛**。

``` python
loss = torch.nn.CrossEntropyLoss() #包含`torch.nn.Softmax(net)`在内
```

## mlp多层感知机
单个或多个隐层的全连接神经网络。

```pyton
net = nn.Sequential(
        d2l.FlattenLayer(),
        nn.Linear(num_inputs, num_hiddens), 
        nn.ReLU(),                          
        nn.Linear(num_hiddens, num_outputs), 
        )
  ##重复以下多次，可以得到多个隐层的mlp.         
		    nn.Linear(num_inputs, num_hiddens), 
        nn.ReLU(),
```

### 隐层的激活函数可以有多种选择：

+ **ReLU（rectified linear unit）**函数提供了一个很简单的非线性变换。给定元素$x$，该函数定义为

  $$\text{ReLU}(x) = \max(x, 0).$$  其导数为： 当输入为负数时，ReLU函数的导数为0；当输入为正数时，ReLU函数的导数为1。

  ![](/images/linreg_softmax_mlp/relu.png)

+ sigmoid函数

  sigmoid函数可以将元素的值变换到0和1之间：

  $$\text{sigmoid}(x) = \frac{1}{1 + \exp(-x)}.$$

  依据链式法则，sigmoid函数的导数

  $$\text{sigmoid}'(x) = \text{sigmoid}(x)\left(1-\text{sigmoid}(x)\right).$$

  ![](/images/linreg_softmax_mlp/sigmoid.png)


+ tanh函数

  tanh（双曲正切）函数可以将元素的值变换到-1和1之间：

  $$\text{tanh}(x) = \frac{1 - \exp(-2x)}{1 + \exp(-2x)}.$$

  依据链式法则，tanh函数的导数

  $$\text{tanh}'(x) = 1 - \text{tanh}^2(x).$$

  ![](/images/linreg_softmax_mlp/tanh.png)

  

  