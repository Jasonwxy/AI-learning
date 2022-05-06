def now_time():
    from datetime import datetime
    return datetime.now()


# 计算函数运行时间
def get_run_time(func):
    def new_func(*args):
        import time
        # start_time = now_time()
        start_time = time.process_time()
        func(*args)
        # end_time = now_time()
        end_time = time.process_time()
        print('function run time is ', end_time - start_time)

    return new_func
