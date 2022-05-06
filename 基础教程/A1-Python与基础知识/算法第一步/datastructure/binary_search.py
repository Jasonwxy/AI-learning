if __name__ == '__main__':
    print("输入目标数(0~100):")
    target = int(input())
    start = 0
    end = 100
    middle = int((start + end) / 2)
    while middle != target:
        if target > middle:
            start = middle
        else:
            end = middle
        middle = int((start + end) / 2)
    print(middle)
