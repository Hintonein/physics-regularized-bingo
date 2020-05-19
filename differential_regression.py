import tensorflow as tf

import numpy as np

from bingo.symbolic_regression.implicit_regression import ImplicitRegression

class ImplicitRegression_TF(ImplicitRegression):

    def __init__(self, training_data, df_err, bcs, n, persistent=False):

        super().__init__(training_data)

        self.n = n

        self.persistent = persistent

        self.x_bcs = tf.convert_to_tensor( np.asarray(bcs[0], dtype=np.float64 ).reshape((-1,1)))
        self.u_bcs = np.asarray(bcs[1], dtype=np.float64).reshape((-1,1))

        self.x_df = tf.convert_to_tensor( np.linspace(bcs[0][0], bcs[0][1], self.n).reshape((-1,1))[1:-1] )

        self.df_err = df_err

    def build_tf_graph_from_agraph(self,individual):
        
        commands = individual._short_command_array
        constants = individual.constants

        def evaluate(X):
            tf_stack = [None] * commands.shape[0]

            for i in range(commands.shape[0]):

                node = commands[i,0]
                if node == 0:
                    X_used = True
                    column_idx = commands[i,1]
                    tf_stack[i] = X[:,column_idx]
                elif node == 1:
                    const_idx = commands[i,1]
                    tf_stack[i] = tf.ones_like(X[:,0]) * constants[const_idx]
                elif node == 2:
                    t1_idx,t2_idx = commands[i,1],commands[i,2]
                    tf_stack[i] = tf_stack[t1_idx] + tf_stack[t2_idx]
                elif node == 3:
                    t1_idx,t2_idx = commands[i,1],commands[i,2]
                    tf_stack[i] = tf_stack[t1_idx] - tf_stack[t2_idx]
                elif node == 4:
                    t1_idx,t2_idx = commands[i,1],commands[i,2]
                    tf_stack[i] = tf_stack[t1_idx] * tf_stack[t2_idx]
                elif node == 5:
                    t1_idx,t2_idx = commands[i,1],commands[i,2]
                    tf_stack[i] = tf_stack[t1_idx] / tf_stack[t2_idx]
                elif node == 6:
                    t1_idx = commands[i,1]
                    tf_stack[i] = tf.sin(tf_stack[t1_idx])
                elif node == 7:
                    t1_idx = commands[i,1]
                    tf_stack[i] = tf.cos(tf_stack[t1_idx])
                elif node == 8:
                    t1_idx = commands[i,1]
                    tf_stack[i] = tf.exp(tf_stack[t1_idx])
                elif node == 9:
                    t1_idx = commands[i,1]
                    tf_stack[i] = tf.math.log(tf.abs(tf_stack[t1_idx]) )
                elif node == 10:
                    t1_idx,t2_idx = commands[i,1],commands[i,2]
                    tf_stack[i] = tf.math.pow( tf.abs(tf_stack[t1_idx]), tf_stack[t2_idx] )
                elif node == 11:
                    t1_idx = commands[i,1]
                    tf_stack[i] = tf.abs(tf_stack[t1_idx])
                elif node == 12:
                    t1_idx = commands[i,1]
                    tf_stack[i] = tf.sqrt(tf.abs(tf_stack[t1_idx]))
                else:
                    raise IndexError(f"Node value {node} unrecognized")

        
            return tf_stack[-1]
        
        return evaluate

    def evaluate_fitness_vector(self, individual):

        #print(individual)
        self.eval_count += 1
        tf_graph_function = self.build_tf_graph_from_agraph(individual)
        u = tf_graph_function(self.x_bcs)
        bcs_model = u.numpy()
        bcs_err = self.u_bcs[:,0] - bcs_model

        with tf.GradientTape(persistent=self.persistent) as g:
            g.watch(self.x_df)

            u_df = tf_graph_function(self.x_df)

            df_err = self.df_err(self.x_df,u_df,g).numpy()

        if self.persistent:
            del g

        fitness = np.empty([self.n], dtype=float)

        fitness[[0,-1]] = bcs_err
        fitness[1:-1] = df_err

        #print(fitness)
        return fitness

