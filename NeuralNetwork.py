import numpy as np
import scipy as sc
import NeuralLayer as Nl


class NeuralNetwork:
    def __init__(self, inputs, outputs=1, lr=0.1, reg=1e-2, drop_out=0., alpha=0.5, lr_decay=0.9):
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.alpha = alpha
        self.learning_rate = lr
        self.lr_decay = lr_decay
        self.reg = reg
        self.drop_out = drop_out

        self.layers = []
        self.add_layer(inputs)
        self.add_layer(outputs)

    def add_layer(self, num_neuron):
        layer = Nl.NeuralLayer(num_neuron)
        self.layers.append(layer)

    def get_weights(self):
        weights = []
        bias = []
        for i in range(len(self.layers)-2):
            w = self.layers[i].weights
            b = self.layers[i].bias
            weights.append(w)
            bias.append(b)

        return weights, bias

    def train(self, x_train, y_train, x_valid, y_valid, epochs, mini_batch=10):

        self.create_model()

        reg_loss = 0
        t_loss = 0
        old_accuracy = 0.0

        # plt.ion()
        # plt.grid(True)
        # plt.xlim(0, epochs*2)
        # plt.ylim(0., 1.)
        # plt.xlabel('Epochs')
        # plt.ylabel('Accuracy')
        for i in range(epochs):
            # if self.drop_out > 0.:
            #     for k in xrange(1, len(self.layers)-1):
            #         mask = np.random.binomial(1, self.drop_out, size=(self.layers[k].neurons, 1))
            #         self.layers[k].set_drop_out_mask(mask)

            for j in range(x_train.shape[0]/mini_batch):
                batch_num = j*mini_batch
                x = x_train[batch_num:batch_num+mini_batch]
                y = y_train[batch_num:batch_num+mini_batch]

                aug = np.random.randint(0, 1)
                if aug == 1:
                    x = np.fliplr(x)
                # elif aug == 2:
                #     angle = np.random.randint(-10, 10)
                #     x = sc.ndimage.interpolation.rotate(x, angle, axes=(1, 2), mode='nearset', reshape=False, order=0)

                self.feed_forward(x)
                t_loss, reg_loss = self.propagate_back(y)
                self.update_weights()

            if i % 5 == 0 or i == epochs-1:
                out_train = self.feed_forward(x_train, t='test')
                out_valid = self.feed_forward(x_valid, t='test')

                t_accuracy = self.get_accuracy(out_train, y_train)
                v_accuracy = self.get_accuracy(out_valid, y_valid)

                _, v_loss = self.calculate_loss_soft_max(out_valid, y_valid)

                print('\t==> epoch:', i)
                print(', Loss Train:', t_loss)
                print(', Reg Loss:', reg_loss)
                print(', Accuracy Train:', t_accuracy)
                print(', Loss Valid: ', v_loss)
                print(', Accuracy Valid:', v_accuracy)

                x = (v_accuracy - old_accuracy) / v_accuracy
                if abs(x) <= 0.05:
                    if self.learning_rate >= 1e-6:
                        self.learning_rate *= self.lr_decay
                        print('\t==>Decaying learning rate =>', self.learning_rate)

                old_accuracy = v_accuracy

            # if i == epochs-1:
            #     print '==> Loss Train:', t_loss,
            #     print ', Reg Loss:', reg_loss,
            #     print ', Accuracy Train:', t_accuracy,
            #     print ', Loss Valid: ', v_loss,
            #     print ', Accuracy Valid:', v_accuracy
            #     # plt.scatter(i, t_accuracy, c='r')
            #     # plt.scatter(i, v_accuracy, c='g')
            #     # plt.pause(0.00001)

    @staticmethod
    def pre_process_data(x_data):
        x_data -= np.mean(x_data, axis=0)
        x_data /= np.std(x_data, axis=0)
        # cov = np.dot(x_data.T, x_data) / x_data.shape[0]
        # U, S, V = np.linalg.svd(cov)
        # Xrot = np.dot(x_data, U)  # decorrelate the data
        # # Xrot_reduced = np.dot(x_data, U[:, :1024])  # Xrot_reduced becomes [N x 1024]
        # Xwhite = Xrot / np.sqrt(S + 1e-5)
        return x_data

    def create_model(self):
        self.layers += [self.layers.pop(1)]  # Move the output layer to be the last layer
        self.seed_weights()

    def seed_weights(self):
        for i in range(0, len(self.layers)-1):
            fan_in = self.layers[i].neurons  # +1 for the bias
            fan_out = self.layers[i+1].neurons
            w = np.random.randn(fan_in, fan_out)/np.sqrt(fan_in/2.0)
            # w = total_weights[0:fan_in-1,:]
            # b = total_weights[[-1]]
            b = np.random.randn(1, fan_out)/np.sqrt(1/2.0)

            self.layers[i].set_weights(w, b)

    def update_weights(self):
        for i in range(len(self.layers)-1):
            m = self.layers[i].momentum
            m_b = self.layers[i].momentum_b
            # accumulated = self.layers[i].accumulated
            # accumulated_b = self.layers[i].accumulated_b

            dl_dw = self.layers[i].dl_dw
            dl_db = self.layers[i].dl_db

            momentum = self.alpha*m - self.learning_rate*dl_dw
            momentum_b = self.alpha*m_b - self.learning_rate*dl_db

            # m = self.beta1 * m + (1 - self.beta1)*dl_dw
            # accumulated = self.beta2*accumulated + (1-self.beta2)*dl_dw*dl_dw
            #
            # m_b = self.beta1 * m_b + (1 - self.beta1) * dl_db
            # accumulated_b = self.beta2 * accumulated_b + (1 - self.beta2) * dl_db * dl_db

            # SGD
            # delta_w = -self.learning_rate*dl_dw
            # delta_b = -self.learning_rate*dl_db

            # SGD Momentum
            delta_w = momentum
            delta_b = momentum_b

            # ADAM
            # delta_w = -self.learning_rate * m/(np.sqrt(accumulated + 1e-7))
            # delta_b = -self.learning_rate * m_b/(np.sqrt(accumulated_b + 1e-7))

            new_weights = self.layers[i].weights + delta_w
            new_bias = self.layers[i].bias + delta_b

            self.layers[i].set_weights(new_weights, new_bias)
            # self.layers[i].momentum = m
            # self.layers[i].accumulated = accumulated
            #
            # self.layers[i].momentum_b = m_b
            # self.layers[i].accumulated_b = accumulated_b

    def feed_forward(self, x_train, t='train'):
        self.layers[0].a = x_train  # input layer

        for i in range(1, len(self.layers)-1):
            w = self.layers[i-1].weights
            b = self.layers[i-1].bias
            a = self.layers[i-1].a

            result = Nl.NeuralLayer.compute(w, a, b)
            result = Nl.NeuralLayer.activate_relu(result)

            if t == 'train' and self.drop_out > 0.:
                # mask = (np.random.rand(*result.shape) < drop_out) / drop_out
                # self.layers[i].set_drop_out_mask(mask[i])
                # mask = self.layers[i].drop_out_mask
                mask = (np.random.rand(*result.shape) < self.drop_out) / self.drop_out
                self.layers[i].drop_out_mask = mask
                result *= mask

            self.layers[i].a = result

        w = self.layers[-2].weights
        b = self.layers[-2].bias
        a = self.layers[-2].a

        result = Nl.NeuralLayer.compute(w, a, b)
        result = Nl.NeuralLayer.activate_soft_max(result)
        self.layers[-1].a = result

        return result

    def propagate_back(self, y):
        # From hidden to output layer
        o = self.layers[-1].a
        delta, loss = self.calculate_loss_soft_max(o, y)
        reg_loss = self.calculate_reg_loss()

        a = self.layers[-2].a

        # if self.drop_out > 0.:
        #     mask = self.layers[-2].drop_out_mask
        #     delta *= mask

        dl_dw = np.dot(a.transpose(), delta)
        dl_dw += self.reg*self.layers[-2].weights
        dl_db = np.sum(delta, axis=0, keepdims=True)

        self.layers[-1].delta = delta
        self.layers[-2].dl_dw = dl_dw
        self.layers[-2].dl_db = dl_db
        ################################

        # input to hl, or hl to hl without bias
        for i in reversed(range(len(self.layers)-2)):
            weights = self.layers[i+1].weights
            a = self.layers[i+1].a

            delta = np.dot(delta, weights.transpose())
            activation_prime = Nl.NeuralLayer.diff_relu(a)
            delta = np.multiply(activation_prime, delta)

            x = self.layers[i].a

            if self.drop_out > 0.:
                mask = self.layers[i+1].drop_out_mask
                delta *= mask

            dl_dw = np.dot(x.transpose(), delta)
            dl_dw += self.reg * self.layers[i].weights
            dl_db = np.sum(delta, axis=0, keepdims=True)

            self.layers[i].dl_dw = dl_dw
            self.layers[i].dl_db = dl_db
            self.layers[i+1].delta = delta
        ##################################

        return loss, reg_loss

    @staticmethod
    def calculate_loss_soft_max(o, t):
        num_examples = o.shape[0]
        correct_probability = -np.log(o[range(num_examples), t])
        data_loss = np.sum(correct_probability)
        data_loss = data_loss / num_examples

        # Soft_max prime
        d_error = o
        d_error[range(num_examples), t] -= 1
        d_error /= num_examples

        return d_error, data_loss

    def calculate_reg_loss(self):
        reg_loss = 0
        for i in range(len(self.layers) - 1):
            w = self.layers[i].weights
            reg_loss += 0.5 * self.reg * np.sum(w * w)

        return reg_loss

    def predict(self, x_test, y_test):
        scores = self.feed_forward(x_test, 'test')
        print('accuracy: %.2f' % (self.get_accuracy(scores, y_test)))

    @staticmethod
    def get_accuracy(scores, y_test):
        predicted_class = np.argmax(scores, axis=1)
        accuracy = np.mean(predicted_class == y_test)

        return accuracy
