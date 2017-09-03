
import os
import sys

import mandalka
mandalka.config(debug=True)

def print_err(_1, err, _2):
    info = "[Note: set the environment variable DEBUG to see details]"
    sys.stderr.write("Error: " + str(err) + "\n" + info + "\n")
    sys.stderr.flush()

if "DEBUG" in os.environ and os.environ["DEBUG"] not in ["", "no"]:
    import IPython.core.ultratb
    sys.excepthook = IPython.core.ultratb.FormattedTB(call_pdb=True)
else:
    sys.excepthook = print_err
