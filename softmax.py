import torch
import numpy as np

def my_softmax(x):
    """
    softmax原理：
    输入任意实数x，输出[0, 1]代表概率，且概率之和为1,
    本质上是在做0-1概率分布的归一化
    这里利用了 非负实数集合/sum 的值域为[0,1]来做计算
    这样不仅能限定值域为[0, 1]，还能让输出值之和为1,
    而sigmoid只能做到前者

    为什么softmax需要exp？softmax和max的关系？
    要判断x属于哪个类别，首先能想到的是x属于哪个类的概率最高，我就认为它属于哪个类，概率可以用[0,1]表示，
    所以我遍历一下概率向量，直接把最大的概率置为1，其他置为0即可，这是很直接的max想法，但max函数是不可
    导的，不能求梯度做反向传播来学习。怎么办呢？我们还是再回顾一下目标
    1. 希望总的概率和为1
    2. 希望只有一个概率为1，其他都为0
    3. 函数要可导

    第一个目标很好实现，做一个归一化即可，即把x变成非负实数，再除以sum，即可得到。
    比较简单的让x平移使得最小值变为0，满足非负实数的条件，再除以sum： (x - min(x)) / sum((x - min(x)))
    假如x是[0.3, 0.8, 0.9]，那么计算后得到[0, 0.456, 0.544],
    问题来了，0.456和0.544比较接近，我们能不能让0.544变得更接近1， 而0.456更接近0？
    也就是说，我们怎么能让微小的变化变得更大？我们应该马上就能想到指数函数可以做到！
    这样，用极限的思维来看，最大的概率会扩大得比其他概率大得多，其他概率相当于无穷小，
    再除以sum，最大的概率就会接近1，其他相对无穷小的概率就会接近0。
    而用e做底数的指数函数，在结合交叉熵做反向传播时，很方便算导数，因此使用了exp，
    这样一来，我们达到了上面3个目标，使得max值尽量接近1，而且是连续可导的，相比max
    会很平滑，所以是soft的
    """

    # 由于对任意实数x取指数函数很容易导致类型上溢，
    # 而且softmax(x) 和 softmax(x + c)是等价的（softmax主要是用x向量内各个值的相对差来做计算，所以整体平移相对差不变，结果也不变）
    # 因此先整体减去一个最大值做预处理，让x变成(-∞, 0]
    x = x - np.max(x)

    # 指数函数可以把x变成正实数，此时值域变成了(0, 1]，和不为1
    e = np.exp(x)
    # 除以sum后结果还是(0, 1]，但是和为1
    # 虽然概率永远不可能为0，但是可以无限接近0，我们认为这样就够了
    # 这也是exp的小弊端
    return e / np.sum(e)


def my_softmax_grad(y, loss_value):
    # softmax要分n类（这里我们分10类）的求导有两种情况
    # 第一种情况：
    #    假如有一个yj，代表x分到第j类的概率，那么yj对yj的偏导等于yj - yj ** 2
    # 第二种情况：
    #    假如有一个yj，代表x分到第j类的概率，那么除了第j个y值，其他n-1个y值，对这个yj的偏导等于 0 -yi * yj    (i从0到n-1，i等于j除外)

    # n个y值，生成对角矩阵，第j行第j列，代表x被分到第j类的概率yj
    diag_matrix = np.diag(y)
    # np.outer用来生成双层for循环，生成n*n的矩阵，矩阵的i,j处位置代表yi * yj的结果
    multi_matrix = np.outer(y, y)
    # 对角矩阵 - 相乘矩阵 = 偏导矩阵，在对角部分等于yj - yj * yj, 在非对角部分等于0 - yi * yj
    partial_matrix = diag_matrix - multi_matrix

    # partial_matrix的第j行，代表x的分类概率向量（y，n个值）中每一个分量（概率）对x被分在第j类的偏导
    # loss_value有n个值，代表x被分为这n类的误差,所以-loss_value * partial_matrix[j]等于
    # x被分到第j类的梯度
    return np.dot(partial_matrix, loss_value)


if __name__ == '__main__':
    class_num = 10  # 分10类
    input = np.random.rand(class_num)  # 1x10 vector
    loss = np.random.rand(class_num)

    print("torch的softmax:")
    print(torch.softmax(torch.tensor(input), dim=0).numpy())
    print("自己写的softmax:")
    print(my_softmax(input))

    t = torch.tensor(input, requires_grad=True)
    s = torch.softmax(t, dim=0)
    s.backward(torch.tensor(loss))
    print("torch的softmax grad:")
    print(t.grad.numpy())
    print("自己写的softmax grad:")
    print(my_softmax_grad(my_softmax(input), loss))