import numpy as np
import matplotlib as mtl
from IPython.core.interactiveshell import InteractiveShell
from matplotlib import pyplot as plt

mtl.use('TkAgg')
InteractiveShell.ast_node_interactivity = 'last_expr'

x = np.array(range(-10, 11))  # 从 -10 到10取21个数据点
y = (3 * x - 4) / 2  # 对于函数值

plt.plot(x, y, color='blue')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
# 画出坐标轴
plt.axhline()
plt.axvline()
# 画出截距
x0 = 4 / 3
y0 = -4 / 2
m = 1.5
mx = [0, x0]
my = [y0, y0 + m * x0]
plt.annotate('x', (x0, 0))
plt.annotate('y', (0, y0))
plt.plot(mx,my,color='red')
plt.show()
