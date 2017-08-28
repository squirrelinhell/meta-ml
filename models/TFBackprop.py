
import mandalka
import numpy as np
import tensorflow as tf
import tqdm

from .TFModel import TFModel

class TFBackprop(TFModel):
    def _predict_batch(self, input_batch, output_shape, **kwargs):
        raise NotImplementedError("_predict_batch")

    def _build_session(self, problem,
            steps=100000,
            batch_size=512,
            lr=0.01,
            eps=0.0001,
            **kwargs):
        lr = float(lr)
        eps = float(eps)

        input_batch = tf.placeholder(
            tf.float32,
            (None,) + problem.input_shape,
            name="input_batch"
        )

        output_batch = tf.reshape(
            self._predict_batch(
                input_batch,
                problem.output_shape,
                **kwargs
            ),
            (-1,) + problem.output_shape,
            name="output_batch"
        )

        reward_grad_in = tf.placeholder(tf.float32, output_batch.shape)
        intermediate_reward = tf.reduce_sum(tf.reduce_mean(
            tf.multiply(reward_grad_in, output_batch),
            axis=0
        ))

        train_op = tf.train.AdamOptimizer(
            learning_rate = lr,
            epsilon = eps
        ).minimize(-intermediate_reward)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        def learn_batch(ep, inputs):
            outputs = sess.run(
                output_batch,
                feed_dict={input_batch: inputs}
            )

            reward_grad = [ep.next_reward(o)[1] for o in outputs]

            sess.run(
                train_op,
                feed_dict={
                    input_batch: inputs,
                    reward_grad_in: reward_grad
                }
            )

        def train():
            ep = problem.start_episode()
            inputs = []
            for _ in tqdm.trange(steps, unit="steps"):
                try:
                    inputs.append(ep.next_input())
                except StopIteration:
                    if len(inputs) >= 1:
                        learn_batch(ep, inputs)
                    ep = problem.start_episode()
                    inputs = [ep.next_input()]
                if len(inputs) == batch_size:
                    learn_batch(ep, inputs)
                    inputs = []
            if len(inputs) >= 1:
                learn_batch(ep, inputs)

        train()
        return sess

    def predict(self, inp):
        sess = self.get_session()
        return sess.run(
            sess.output_batch,
            feed_dict={sess.input_batch: [inp]}
        )[0]
