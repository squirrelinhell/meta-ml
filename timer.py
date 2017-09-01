
import time

def t():
    start_time = time.time()
    def get_time(print_info=True):
        total_time = time.time() - start_time
        import sys
        sys.stderr.write("Time: %.3fs\n" % total_time)
        sys.stderr.flush()
        return total_time
    global t
    t = get_time

t()
