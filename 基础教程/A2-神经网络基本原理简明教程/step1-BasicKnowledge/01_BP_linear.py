error = 1e-5


def target_function(w, b):
    x = 2 * w + 3 * b
    y = 2 * b + 1
    z = x * y
    print(w, b, x, y, z)
    return x, y, z


def single_variable(w, b, t):
    print("\nsingle variable: b ----- ")
    while True:
        x, y, z = target_function(w, b)
        delta_z = z - t
        print("w=%f,b=%f,z=%f,delta_z=%f" % (w, b, z, delta_z))
        if abs(delta_z) < error:
            break
        delta_b = delta_z / 63
        print("delta_b=%f" % delta_b)
        b = b - delta_b
    print("done!")
    print("final b=%f" % b)


def single_variable_new(w, b, t):
    print("\nsingle variable_new: b ----- ")
    while True:
        x, y, z = target_function(w, b)
        delta_z = z - t
        print("w=%f,b=%f,z=%f,delta_z=%f" % (w, b, z, delta_z))
        if abs(delta_z) < error:
            break
        factor_b, factor_w = calculate_wb_factor(x, y)
        delta_b = delta_z / factor_b
        print("delta_b=%f" % delta_b)
        b = b - delta_b
    print("done!")
    print("final b=%f" % b)


def double_variable(w, b, t):
    print("\ndouble variable: w,b ----- ")
    while True:
        x, y, z = target_function(w, b)
        delta_z = z - t
        print("w=%f,b=%f,z=%f,delta_z=%f" % (w, b, z, delta_z))
        if abs(delta_z) < error:
            break
        delta_b = delta_z / 63 / 2
        delta_w = delta_z / 18 / 2
        print("delta_b=%f, delta_w=%f" % (delta_b, delta_w))
        b = b - delta_b
        w = w - delta_w
    print("done!")
    print("final b=%f, final w=%f" % (b, w))


def double_variable_new(w, b, t):
    print("\ndouble variable_new: w,b ----- ")
    while True:
        x, y, z = target_function(w, b)
        delta_z = z - t
        print("w=%f,b=%f,z=%f,delta_z=%f" % (w, b, z, delta_z))
        if abs(delta_z) < error:
            break
        factor_b, factor_w = calculate_wb_factor(x, y)
        delta_b = delta_z / factor_b / 2
        delta_w = delta_z / factor_w / 2
        print("delta_b=%f, delta_w=%f" % (delta_b, delta_w))
        b = b - delta_b
        w = w - delta_w
    print("done!")
    print("final b=%f, final w=%f" % (b, w))


def calculate_wb_factor(x, y):
    return 2 * x + 3 * y, 2 * y


if __name__ == '__main__':
    b = 4
    w = 3
    t = 150
    # single_variable(w, b, t)
    # single_variable_new(w, b, t)
    # double_variable(w, b, t)
    double_variable_new(w, b, t)
