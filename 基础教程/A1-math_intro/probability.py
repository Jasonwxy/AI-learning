import numpy as np
import matplotlib
import pandas as pd
import random
from scipy import stats
from scipy.stats import binom
from scipy import special as sps
from tabulate import tabulate
from matplotlib import pyplot as plt

matplotlib.use('TkAgg')

# df = pd.DataFrame({'name': ['Dan', 'Joann', 'Pedro', 'Rosie', 'Ethan', 'Vicky', 'Frederic'],
#                    'salary': [50000, 54000, 50000, 189000, 55000, 40000, 59000],
#                    'hours': [41, 40, 36, 30, 35, 39, 40],
#                    'grade': [50, 50, 46, 95, 50, 5, 57]})

df = pd.DataFrame({'name': ['Dan', 'Joann', 'Pedro', 'Rosie', 'Ethan', 'Vicky', 'Frederic', 'Jimmie', 'Rhonda',
                            'Giovanni', 'Francesca', 'Rajab', 'Naiyana', 'Kian', 'Jenny'],
                   'salary': [50000, 54000, 50000, 189000, 55000, 40000, 59000, 42000, 47000, 78000, 119000, 95000,
                              49000, 29000, 130000],
                   'hours': [41, 40, 36, 17, 35, 39, 40, 45, 41, 35, 30, 33, 38, 47, 24],
                   'grade': [50, 50, 46, 95, 50, 5, 57, 42, 26, 72, 78, 60, 40, 17, 85]})

df = df[(df['grade'] > 5) & (df['grade'] < 95)]  # 去掉异常

salary = df['salary']
hours = df['hours']
grade = df['grade']
cols = ['salary', 'hours', 'grade']


def demo1():
    print(tabulate([['最小值', '众数', '中间值', '均值', '最大值'],
                    [salary.min(), salary.mode()[0], salary.median(), salary.mean(), salary.max()]]))

    density = stats.gaussian_kde(salary)
    n, x, _ = plt.hist(salary, histtype='step', density=True, bins=25)
    plt.plot(x, density(x) * 5)  # 画制密度曲线  分布线可以看出salary数据集为 右偏态分布
    # salary.plot.hist(title='Salary Distribution', color='lightblue', bins=25)  # 制作柱状图
    plt.axvline(salary.mean(), color='m', linestyle='dashed', linewidth=2)  # 均值线
    plt.axvline(salary.median(), color='g', linestyle='dashed', linewidth=2)  # 中值线
    plt.show()

    density = stats.gaussian_kde(hours)
    n, x, _ = plt.hist(hours, histtype='step', density=True, bins=25)
    plt.plot(x, density(x) * 7)  # 画制密度曲线  分布线可以看出hours数据集为 左偏态分布
    # hours.plot.hist(title='hours Distribution', color='lightblue', bins=25)  # 制作柱状图
    plt.axvline(hours.mean(), color='m', linestyle='dashed', linewidth=2)  # 均值线
    plt.axvline(hours.median(), color='g', linestyle='dashed', linewidth=2)  # 中值线
    plt.show()

    density = stats.gaussian_kde(grade)
    n, x, _ = plt.hist(grade, histtype='step', density=True, bins=25)
    plt.plot(x, density(x) * 7.5)  # 画制密度曲线  分布线可以看出grade数据集为 正态分布
    # grade.plot.hist(title='grade Distribution', color='lightblue', bins=25)  # 制作柱状图
    plt.axvline(grade.mean(), color='m', linestyle='dashed', linewidth=2)  # 均值线
    plt.axvline(grade.median(), color='g', linestyle='dashed', linewidth=2)  # 中值线
    plt.show()


def demo2():
    for col in cols:
        print(df[col].name + " range:" + str(df[col].max() - df[col].min()))  # 求范围，最大值-最小值

    # 求给定值在样本数据中的百分比
    print(stats.percentileofscore(df['grade'], 50, 'rank'))  # rank 排序 4/7
    print(stats.percentileofscore(df['grade'], 50, 'weak'))  # weak 弱匹配 小于等于score 5/7
    print(stats.percentileofscore(df['grade'], 50, 'strict'))  # strict 强匹配 小于score 2/7
    print(stats.percentileofscore(df['grade'], 50, 'mean'))  # mean strict和weak的平均值


def demo3():
    for col in cols:
        print(df[col].quantile([0.25, 0.5, 0.75]))
        df[col].plot(kind='box', title='Weekly %s Distribution' % col, figsize=(10, 8))  # showfliers=False 忽略例外值
        # 箱线图：
        # 矩形(箱) 代表 1/4->1/2 + 1/2->3/4部分
        # 箱上、下线表示 0->1/4 和 3/4->1区间
        # 中间绿线代表中值

        plt.show()


def demo4():
    # for col in cols:
    #     print(col, '方差为:', df[col].var(), '标准差为:', df[col].std(), '均值为:', df[col].mean())
    mean = grade.mean()
    sum_g = 0
    for g in grade:
        sum_g = (g - mean) ** 2 + sum_g
    var = sum_g / (grade.__len__() - 1)
    std = np.sqrt(var)
    print(var, std)
    print('grade', '方差为:', grade.var(), '标准差为:', grade.std())

    print(df.describe())  # 总结数据分布


def demo5():
    plt.figure()
    df['grade'].plot(kind='box', title='Grade Distribution')
    plt.figure()
    df['grade'].hist(bins=9)
    plt.show()
    print(df.describe())
    print('median:', str(df['grade'].median()))


def demo6():
    # 归一化
    df['salary'], df['hours'], df['grade'] = tuple(map(min_max_scale, [df['salary'], df['hours'], df['grade']]))
    df.plot(kind='box', title='distribution', figsize=(10, 8))
    plt.show()


def min_max_scale(v):
    min_value = v.min()
    max_value = v.max()
    scale = max_value - min_value
    return [(i - min_value) / scale for i in v]


def demo7():
    # 计算相关性
    # print(df['grade'].corr(df['salary']))
    # 双变量数据画离散点
    df.plot(kind='scatter', title='Grade vs Salary', x='grade', y='salary')
    plt.plot(np.unique(df['grade']), np.poly1d(np.polyfit(df['grade'], df['salary'], 1))(np.unique(df['grade'])))
    plt.show()


def demo8():
    # 最小二乘拟合
    df['x2'] = df['grade'] ** 2
    df['xy'] = df['grade'] * df['salary']
    x = df['grade'].sum()
    y = df['salary'].sum()
    x2 = df['x2'].sum()
    xy = df['xy'].sum()
    n = df['grade'].count()
    m = ((n * xy) - (x * y)) / ((n * x2) - x ** 2)
    b = (y - m * x) / n
    df['fx'] = m * df['grade'] + b
    df['error'] = df['fx'] - df['salary']
    print('斜率为：', str(m))
    print('y轴截距为：', str(b))
    df.plot(kind='scatter', title='Grade vs Salary Regression', x='grade', y='salary', color='r')
    plt.plot(df['grade'], df['fx'])
    plt.show()


def demo9():
    # 线性回归方法 等同于demo8 的130行~138行
    m, b, r, p, se = stats.linregress(df['grade'], df['salary'])
    df['fx'] = m * df['grade'] + b
    df['error'] = df['fx'] - df['salary']
    print('斜率为：', str(m))
    print('y轴截距为：', str(b))
    df.plot(kind='scatter', title='Grade vs Salary Regression', x='grade', y='salary', color='r')
    plt.plot(df['grade'], df['fx'])
    plt.show()


def demo10():
    heads_tails = [0, 0]
    h3 = 0
    results = []
    trials = 10000

    # 模拟抛硬币 10000次
    for i in range(trials):
        toss = random.randint(0, 1)
        heads_tails[toss] = heads_tails[toss] + 1
        result = ['H' if random.randint(0, 1) else 'T',
                  'H' if random.randint(0, 1) else 'T',
                  'H' if random.randint(0, 1) else 'T']
        results.append(result)
        h3 = h3 + int(result == ['H', 'H', 'H'])  # 连续扔3次都是正面
    print(heads_tails)

    print('%.2f%%' % ((h3 / trials) * 100))
    plt.figure(figsize=(6, 6))
    plt.pie(heads_tails, labels=['heads', 'tails'])
    plt.legend()
    plt.show()


def demo11():
    # 抛3次硬币，正面朝上的概率分布
    trials = 3
    possibilities = 2 ** trials
    x = np.array(range(0, trials + 1))
    p = np.array([sps.comb(trials, i, exact=True) / possibilities for i in x])
    # print(p)

    plt.xlabel('x')
    plt.ylabel('Possibility')
    plt.bar(x, p)
    plt.show()


def demo12():
    h = [0, 0, 0, 0]
    trials = 10000

    # 模拟抛硬币 10000次
    for i in range(trials):
        result = ['H' if random.randint(0, 1) else 'T',
                  'H' if random.randint(0, 1) else 'T',
                  'H' if random.randint(0, 1) else 'T']
        count = result.count('H')  # 三次正面的次数
        h[count] = h[count] + 1

    res = np.array(['%.2f%%' % ((x / trials) * 100) for x in h])

    print(res)


def demo13():
    n = 100
    p = 0.25
    x = np.array(range(0, n + 1))

    prob = np.array([binom.pmf(k, n, p) for k in x])

    print(binom.mean(n, p))
    print(binom.var(n, p))
    print(binom.std(n, p))

    plt.xlabel('x')
    plt.ylabel('Possibility')
    plt.bar(x, prob)
    plt.show()


def demo14():
    searches = np.array([0.1875, 0.25, 0.3125, 0.1875, 0.125, 0.375, 0.25, 0.1875, 0.3125, 0.25, 0.25, 0.3125])
    plt.xlabel('search results')
    plt.ylabel('frequency')
    plt.hist(searches)
    plt.show()


def demo15():
    n, p, s = 1000, 0.5, 10000
    df = pd.DataFrame(np.random.binomial(n, p, s) / n, columns=['p-hat'])
    means = df['p-hat']
    m = means.mean()
    sd = means.std()
    moe11 = m - sd
    moe12 = m + sd
    moe21 = m - (sd * 2)
    moe22 = m + (sd * 2)
    moe31 = m - (sd * 3)
    moe32 = m + (sd * 3)

    means.plot.hist(title='Simulate Sampling Distribution')

    plt.axvline(m, color='r', linestyle='dashed', linewidth=2)
    plt.axvline(moe11, color='m', linestyle='dashed', linewidth=2)
    plt.axvline(moe12, color='m', linestyle='dashed', linewidth=2)
    plt.axvline(moe21, color='y', linestyle='dashed', linewidth=2)
    plt.axvline(moe22, color='y', linestyle='dashed', linewidth=2)
    plt.axvline(moe31, color='b', linestyle='dashed', linewidth=2)
    plt.axvline(moe32, color='b', linestyle='dashed', linewidth=2)

    plt.show()


def demo16():
    mu, sigma, n = 3.2, 1.2, 500
    data = np.array([])
    sampling = np.array([])

    for s in range(0, 10000):
        sample = np.random.normal(mu, sigma, n)
        data = np.append(data, sample)
        sampling = np.append(sampling, sample.mean())

    df = pd.DataFrame(sampling, columns=['mean'])

    means = df['mean']
    m = means.mean()
    std = means.std()
    ci = stats.norm.interval(0.95, m, std)  # 置信区间 95%  ±1.96倍标准差

    # means.plot.hist(title='Simulated Sampling Distribution', bins=100)
    # plt.axvline(m, color='r', linestyle='dashed', linewidth=2)
    # plt.axvline(ci[0], color='m', linestyle='dashed', linewidth=2)
    # plt.axvline(ci[1], color='m', linestyle='dashed', linewidth=2)
    # plt.show()

    print('Sampling Mean: ' + str(m))
    print('Sampling StdErr: ' + str(std))
    print('99% Confidence Interval: ' + str(ci))


def demo17():
    np.random.seed(123)
    lo = np.random.randint(-5, -1, 6)
    mid = np.random.randint(0, 3, 38)
    hi = np.random.randint(4, 6, 6)
    sample = np.append(lo, np.append(mid, hi))
    pop = np.random.normal(0, 1.15, 100000)  # 生成正态分布

    t, p = stats.ttest_1samp(sample, 0)

    print('t-statistic:', str(t))
    print('p-value:', str(p))

    ci = stats.norm.interval(0.9, 0, 1.15)
    plt.hist(pop, bins=100)

    plt.axvline(pop.mean(), color='y', linestyle='dashed', linewidth=2)
    plt.axvline(ci[1], color='r', linestyle='dashed', linewidth=2)
    plt.axvline(ci[0], color='r', linestyle='dashed', linewidth=2)
    plt.axvline(pop.mean() + t * pop.std(), color='m', linestyle='dashed', linewidth=2)
    plt.axvline(pop.mean() - t * pop.std(), color='m', linestyle='dashed', linewidth=2)

    # plt.hist(sample)
    plt.show()


if __name__ == '__main__':
    demo17()
