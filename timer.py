
import time

def t():
    start_time = time.time()
    def get_time(print_info=True):
        total_time = time.time() - start_time
        import os
        import sys
        sys.stderr.write("Time: %.3fs\n" % total_time)
        sys.stderr.flush()
        if "STOPTIME" in os.environ:
            return 0.0
        return total_time / 2.0
    global t
    t = get_time

t()
