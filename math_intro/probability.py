import numpy as np
import matplotlib
import pandas as pd
from scipy import stats
from tabulate import tabulate
from matplotlib import pyplot as plt

matplotlib.use('TkAgg')

df = pd.DataFrame({'name': ['Dan', 'Joann', 'Pedro', 'Rosie', 'Ethan', 'Vicky', 'Frederic'],
                   'salary': [50000, 54000, 50000, 189000, 55000, 40000, 59000],
                   'hours': [41, 40, 36, 30, 35, 39, 40],
                   'grade': [50, 50, 46, 95, 50, 5, 57]})
salary = df['salary']
hours = df['hours']
grade = df['grade']


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


if __name__ == '__main__':
    demo1()
