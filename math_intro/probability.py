import numpy as np
import matplotlib
import pandas as pd
from scipy import stats
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

salary = df['salary']
hours = df['hours']
grade = df['grade']
cols = ['salary', 'hours', 'grade']


def demo1():
    # print(tabulate([['最小值', '众数', '中间值', '均值', '最大值'],
    #                 [salary.min(), salary.mode()[0], salary.median(), salary.mean(), salary.max()]]))

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
    print(df['grade'].corr(df['salary']))
    # 双变量数据画离散点
    df.plot(kind='scatter', title='Grade vs Salary', x='grade', y='salary')
    plt.plot(np.unique(df['grade']), np.poly1d(np.polyfit(df['grade'], df['salary'], 1))(np.unique(df['grade'])))
    plt.show()

def demo8():
    # 最小二乘拟合
    pass



if __name__ == '__main__':
    demo7()
    # z = np.poly1d(np.polyfit(df['grade'],df['salary'],2))
    # print(z)
