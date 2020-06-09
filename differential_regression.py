import tensorflow as tf

import numpy as np

from bingo.evaluation.fitness_function import VectorBasedFunction

tf.config.set_visible_devices([], 'GPU')


class DifferentialRegression_TF(VectorBasedFunction):

    def __init__(self,
                 X, U, X_df, df_err,
                 df_order=1,
                 differential_weight=1,
                 metric="mae",
                 detect_const_solutions=True):

        super().__init__(None, metric)

        self.X = tf.convert_to_tensor(X, dtype=tf.float64)
        self.U = U  # Keep as a numpy array
        self.X_df = tf.convert_to_tensor(X_df, dtype=tf.float64)

        self.persistent = df_order > 1

        self.differential_weight = differential_weight
        self.detect_const_solutions = detect_const_solutions

        self.df_err = df_err

    def build_tf_graph_from_agraph(self, individual):

        commands = individual._short_command_array
        constants = individual.constants

        def evaluate(X):
            tf_stack = [None] * commands.shape[0]

            for i in range(commands.shape[0]):

                node = commands[i, 0]
                if node == 0:
                    column_idx = commands[i, 1]
                    tf_stack[i] = X[:, column_idx]
                elif node == 1:
                    const_idx = commands[i, 1]
                    tf_stack[i] = tf.ones_like(X[:, 0]) * constants[const_idx]
                elif node == 2:
                    t1_idx, t2_idx = commands[i, 1], commands[i, 2]
                    tf_stack[i] = tf_stack[t1_idx] + tf_stack[t2_idx]
                elif node == 3:
                    t1_idx, t2_idx = commands[i, 1], commands[i, 2]
                    tf_stack[i] = tf_stack[t1_idx] - tf_stack[t2_idx]
                elif node == 4:
                    t1_idx, t2_idx = commands[i, 1], commands[i, 2]
                    tf_stack[i] = tf_stack[t1_idx] * tf_stack[t2_idx]
                elif node == 5:
                    t1_idx, t2_idx = commands[i, 1], commands[i, 2]
                    tf_stack[i] = tf_stack[t1_idx] / tf_stack[t2_idx]
                elif node == 6:
                    t1_idx = commands[i, 1]
                    tf_stack[i] = tf.sin(tf_stack[t1_idx])
                elif node == 7:
                    t1_idx = commands[i, 1]
                    tf_stack[i] = tf.cos(tf_stack[t1_idx])
                elif node == 8:
                    t1_idx = commands[i, 1]
                    tf_stack[i] = tf.exp(tf_stack[t1_idx])
                elif node == 9:
                    t1_idx = commands[i, 1]
                    tf_stack[i] = tf.math.log(tf.abs(tf_stack[t1_idx]))
                elif node == 10:
                    t1_idx, t2_idx = commands[i, 1], commands[i, 2]
                    tf_stack[i] = tf.math.pow(
                        tf.abs(tf_stack[t1_idx]), tf_stack[t2_idx])
                elif node == 11:
                    t1_idx = commands[i, 1]
                    tf_stack[i] = tf.abs(tf_stack[t1_idx])
                elif node == 12:
                    t1_idx = commands[i, 1]
                    tf_stack[i] = tf.sqrt(tf.abs(tf_stack[t1_idx]))
                else:
                    raise IndexError(f"Node value {node} unrecognized")

            return tf_stack[-1]

        return evaluate

    def evaluate_fitness_vector(self, individual):

        self.eval_count += 1
        tf_graph_function = self.build_tf_graph_from_agraph(individual)
        U_hat = tf_graph_function(self.X)
        error_fit = self.U[:, 0] - U_hat.numpy()

        if self.X_df is not None:
            # Use persistent gradients in the case that we need to take more than one derivative.
            with tf.GradientTape(persistent=self.persistent) as g:
                g.watch(self.X_df)

                U_df = tf_graph_function(self.X_df)

                error_df = self.df_err(self.X_df, U_df, g).numpy()

            # g is not cleaned up automatically if set it as persistent
            if self.persistent:
                del g

            fitness = self._metric(error_fit) + \
                self.differential_weight * self._metric(error_df)
        else:
            fitness = self._metric(error_fit)

        if self.detect_const_solutions and not np.isinf(self._metric(error_df)) and self.X_df is not None:
            random_idx = np.random.choice(list(range(U_df.shape[0])), 6)
            U_df_npy = U_df.numpy()

            vals = U_df_npy[random_idx]

            residual = vals[0] - vals[1] + \
                vals[2] - vals[3] + vals[4] - vals[5]

            if np.abs(residual) < 1e-6:
                fitness = np.inf

        return fitness
