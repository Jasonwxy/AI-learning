import random


# 顾客参加一个抽奖活动，三个关闭的门后面只有一个有奖品，顾客选择一个门之后，主持人会打开一个没有奖品的门，并给顾客一次改变选择的机会。
# 此时，改选另外一个门会得到更大的获奖几率么？
def door_and_prize(switch, loop_num):
    win = 0
    for loop in range(loop_num):
        prize = random.randint(0, 2)  # 随机生成奖品门
        init_choice = random.randint(0, 2)  # 初始选择的门
        doors = [0, 1, 2]  # 设置三个门id
        doors.remove(prize)  # 将奖品门移除，后续去除无奖品的门

        if init_choice in doors:  # 如果选择的门在剩下的门中，将doors列表中选择的门去掉
            doors.remove(init_choice)

        open_door = doors[random.randint(0, len(doors) - 1)]

        if switch:  # 如果更换选择，init_choice+open_door+second_choice=3，全部门的总和
            second_choice = 3 - open_door - init_choice
        else:
            second_choice = init_choice

        if second_choice == prize:
            win += 1
    return win / loop_num


if __name__ == '__main__':
    p1 = door_and_prize(1, 100000)
    p2 = door_and_prize(0, 100000)
    print('switching  winning rate is:', p1)
    print('not switching  winning rate is:', p2)
