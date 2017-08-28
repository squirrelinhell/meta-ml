
import os
import sys
import mandalka
import tensorflow as tf

from . import Model

class TFModel(Model):
    def _build_session(self, problem, **kwargs): # -> session
        raise NotImplementedError("_build_session")

    def __init__(self, problem, seed, **kwargs):
        path = ("__cache__/" + self.__class__.__name__.lower()
            + "_" + mandalka.unique_id(self))

        def print_info(i):
            sys.stderr.write("%s: %s\n" % (i, mandalka.describe(self)))
            sys.stderr.flush()

        if os.path.exists(path):
            print_info("Loading model from cache")
            saver = tf.train.import_meta_graph(path + "/graph.meta")
            sess = tf.Session()
            saver.restore(sess, path + "/model")

        else:
            print_info("Training a new model")
            tf.set_random_seed(seed)
            sess = self._build_session(problem, **kwargs)

            print_info("Saving model to cache")
            saver = tf.train.Saver()
            os.makedirs(path + ".tmp", exist_ok=True)
            saver.export_meta_graph(filename=path + ".tmp/graph.meta")
            saver.save(sess, path + ".tmp/model")
            with open(path + ".tmp/info.txt", "w") as f:
                f.write(mandalka.describe(self, -1))
            os.rename(path + ".tmp", path)

        sess.names = dict()
        for op in sess.graph.get_operations():
            sess.names[op.name] = op
            for t in op.outputs:
                sess.names[t.name.split(":")[0]] = t

        tf.Session.__getattr__ = (
            lambda self, name: self.names.get(name)
        )

        self.get_session = lambda: sess
