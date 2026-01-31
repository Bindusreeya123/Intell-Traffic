STOP_LINE_Y = 420
SIGNAL_RED = True

def check_violation(cx, cy, tid):
    if SIGNAL_RED and cy > STOP_LINE_Y:
        return True
    return False
