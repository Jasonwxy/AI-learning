class StateMachine:
    start_state = 0

    def __init__(self):
        self.state = self.start_state

    def get_next_values(self, state, inp):
        return None, None

    def step(self, inp):
        (s, o) = self.get_next_values(self.state, inp)
        self.state = s
        return o

    def feeder(self, inputs):
        return [self.step(inp) for inp in inputs]


class TextSeq(StateMachine):
    start_state = 0

    def get_next_values(self, state, inp):
        if state == 0 and inp == 'A':
            return 1, True
        elif state == 1 and inp == 'G':
            return 2, True
        elif state == 2 and inp == 'C':
            return 0, True
        else:
            return 3, False


if __name__ == "__main__":
    InSeq = TextSeq()

    x = InSeq.feeder(['A', 'G', 'C'])

    y = InSeq.feeder(['A', 'G', 'C', 'A', 'G', 'C', 'G'])

    print(x, y)
