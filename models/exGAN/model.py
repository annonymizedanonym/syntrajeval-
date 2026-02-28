import os
import random

import numpy as np
import tensorflow as tf
from keras.initializers import he_uniform
from keras.layers import LSTM, Concatenate, Dense, Input, Lambda, TimeDistributed
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.regularizers import l1

from data.csv2npy import data_conversion
from eval import evaluate_on_batch
from exGAN_module import exGAN
from losses import d_bce_loss, trajLoss


class LSTM_Traj_exGAN:
    def __init__(
        self,
        latent_dim,
        keys,
        vocab_size,
        max_length,
        lat_centroid,
        lon_centroid,
        scale_factor,
        SEED=None,
    ):
        if SEED is not None:
            random.seed(SEED)
            np.random.seed(SEED)
            tf.random.set_random_seed(SEED)
        else:
            print("SEED is not fixed")

        self.latent_dim = latent_dim
        self.max_length = max_length

        self.keys = keys
        self.vocab_size = vocab_size

        self.lat_centroid = lat_centroid
        self.lon_centroid = lon_centroid
        self.scale_factor = scale_factor
        self.SEED = SEED

        self.x_train = None

        # Define the optimizer
        self.optimizer = Adam(0.001, 0.5)

        # Build the trajectory generator
        self.generator = self.build_generator()

        # The trajectory generator takes real trajectories and noise as inputs
        noise = Input(shape=(self.latent_dim,), name="input_noise")
        inputs = []
        for idx, key in enumerate(self.keys):
            i = Input(
                shape=(self.max_length, self.vocab_size[key]), name="input_" + key
            )
            inputs.append(i)
        inputs.append(noise)

        # The trajectory generator generates synthetic trajectories
        gen_trajs = self.generator(inputs)

        # Build and compile the discriminator (by default trainable is True)
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(
            loss=d_bce_loss(gen_trajs[4]),
            optimizer=self.optimizer,
            metrics=["accuracy"],
        )

        # The combined model only trains the trajectory generator
        self.discriminator.trainable = False

        # The discriminator takes generated trajectories as input and makes predictions
        pred = self.discriminator(gen_trajs[:4] + [gen_trajs[5]])

        # The combined model (combining the generator and the discriminator)
        self.combined = Model(inputs, pred)
        self.combined.compile(
            loss=trajLoss(inputs, gen_trajs), optimizer=self.optimizer
        )

        os.makedirs("params", exist_ok=True)
        C_model_json = self.combined.to_json()
        with open("params/C_model.json", "w") as json_file:
            json_file.write(C_model_json)

        G_model_json = self.generator.to_json()
        with open("params/G_model.json", "w") as json_file:
            json_file.write(G_model_json)

        D_model_json = self.discriminator.to_json()
        with open("params/D_model.json", "w") as json_file:
            json_file.write(D_model_json)

    def build_discriminator(self):
        # Input Layer
        inputs = []

        # Embedding Layer
        embeddings = []
        for idx, key in enumerate(self.keys):
            if key == "mask":
                continue
            if key == "lat_lon":
                i = Input(
                    shape=(self.max_length, self.vocab_size[key]), name="input_" + key
                )

                unstacked = Lambda(lambda x: tf.unstack(x, axis=1))(i)
                d = Dense(
                    units=64,
                    use_bias=True,
                    activation="relu",
                    kernel_initializer=he_uniform(seed=1),
                    name="emb_" + key,
                )
                dense_latlon = [d(x) for x in unstacked]
                e = Lambda(lambda x: tf.stack(x, axis=1))(dense_latlon)

            else:
                i = Input(
                    shape=(self.max_length, self.vocab_size[key]), name="input_" + key
                )
                unstacked = Lambda(lambda x: tf.unstack(x, axis=1))(i)
                d = Dense(
                    units=self.vocab_size[key],
                    use_bias=True,
                    activation="relu",
                    kernel_initializer=he_uniform(seed=1),
                    name="emb_" + key,
                )
                dense_attr = [d(x) for x in unstacked]
                e = Lambda(lambda x: tf.stack(x, axis=1))(dense_attr)
            inputs.append(i)
            embeddings.append(e)

        # Feature Fusion Layer
        concat_input = Concatenate(axis=2)(embeddings)
        unstacked = Lambda(lambda x: tf.unstack(x, axis=1))(concat_input)
        d = Dense(
            units=100,
            use_bias=True,
            activation="relu",
            kernel_initializer=he_uniform(seed=1),
            name="emb_trajpoint",
        )
        dense_outputs = [d(x) for x in unstacked]
        emb_traj = Lambda(lambda x: tf.stack(x, axis=1))(dense_outputs)

        # LSTM Modeling Layer (many-to-one)
        lstm_cell = LSTM(units=100, recurrent_regularizer=l1(0.02))(emb_traj)

        # Output
        sigmoid = Dense(1, activation="sigmoid")(lstm_cell)

        return Model(inputs=inputs, outputs=sigmoid)

    def build_generator(self):
        # Input Layer
        inputs = []

        # Embedding Layer
        embeddings = []
        noise = Input(shape=(self.latent_dim,), name="input_noise")
        mask = Input(shape=(self.max_length, 1), name="input_mask")
        for idx, key in enumerate(self.keys):
            if key == "mask":
                inputs.append(mask)
                continue
            elif key == "lat_lon":
                i = Input(
                    shape=(self.max_length, self.vocab_size[key]), name="input_" + key
                )
                unstacked = Lambda(lambda x: tf.unstack(x, axis=1))(i)
                d = Dense(
                    units=64,
                    activation="relu",
                    use_bias=True,
                    kernel_initializer=he_uniform(seed=1),
                    name="emb_" + key,
                )
                dense_latlon = [d(x) for x in unstacked]
                e = Lambda(lambda x: tf.stack(x, axis=1))(dense_latlon)
            else:
                i = Input(
                    shape=(self.max_length, self.vocab_size[key]), name="input_" + key
                )
                unstacked = Lambda(lambda x: tf.unstack(x, axis=1))(i)
                d = Dense(
                    units=self.vocab_size[key],
                    activation="relu",
                    use_bias=True,
                    kernel_initializer=he_uniform(seed=1),
                    name="emb_" + key,
                )
                dense_attr = [d(x) for x in unstacked]
                e = Lambda(lambda x: tf.stack(x, axis=1))(dense_attr)
            inputs.append(i)
            embeddings.append(e)
        inputs.append(noise)

        # Feature Fusion Layer
        concat_input = Concatenate(axis=2)(embeddings)
        unstacked = Lambda(lambda x: tf.unstack(x, axis=1))(concat_input)
        d = Dense(
            units=100,
            use_bias=True,
            activation="relu",
            kernel_initializer=he_uniform(seed=1),
            name="emb_trajpoint",
        )
        dense_outputs = [d(Concatenate(axis=1)([x, noise])) for x in unstacked]
        emb_traj = Lambda(lambda x: tf.stack(x, axis=1))(dense_outputs)

        # LSTM Modeling Layer (many-to-many)
        lstm_cell = LSTM(
            units=100,
            batch_input_shape=(None, self.max_length, 100),
            return_sequences=True,
            recurrent_regularizer=l1(0.02),
        )(emb_traj)

        # Outputs
        outputs = []
        for idx, key in enumerate(self.keys):
            if key == "mask":
                output_mask = Lambda(lambda x: x)(mask)
                outputs.append(output_mask)
            elif key == "lat_lon":
                output = TimeDistributed(
                    Dense(2, activation="tanh"), name="output_latlon"
                )(lstm_cell)
                scale_factor = self.scale_factor
                output_stratched = Lambda(lambda x: x * scale_factor)(output)
                outputs.append(output_stratched)
            else:
                output = TimeDistributed(
                    Dense(self.vocab_size[key], activation="softmax"),
                    name="output_" + key,
                )(lstm_cell)
                outputs.append(output)

        return Model(inputs=inputs, outputs=outputs)

    def train(
        self,
        train_norm,
        val_norm,
        epochs=200,
        batch_size=256,
        sample_interval=10,
        start_epoch=1,
        path=None,
        tensorboard=False,
        log_name="no_name",
    ):
        if tensorboard:
            log_dir = os.path.join("./logs/", log_name)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

            writer = tf.summary.FileWriter(log_dir)
            if start_epoch == 1:
                summary = tf.Summary(
                    value=[
                        tf.Summary.Value(
                            tag="hparams/latent_dim", simple_value=self.latent_dim
                        ),
                        tf.Summary.Value(
                            tag="hparams/max_length", simple_value=self.max_length
                        ),
                        # tf.Summary.Value(tag="hparams/num_categories", simple_value=self.vocab_size["category"]),
                    ]
                )
                writer.add_summary(summary)
                writer.flush()

        # Training data
        x_train = data_conversion(train_norm, "tid")
        # Padding zero to reach the maxlength
        X_train = [
            pad_sequences(f, self.max_length, padding="pre", dtype="float64")
            for f in x_train
        ]

        self.X_train = X_train

        x_val = data_conversion(val_norm, "tid")
        X_val = [
            pad_sequences(f, self.max_length, padding="pre", dtype="float64")
            for f in x_val
        ]

        self.X_val = X_val

        val_trajs_bc = []
        val_trajs_bc.append(X_val[0])  # latlon
        val_trajs_bc.append(X_val[1])  # day
        val_trajs_bc.append(X_val[2])  # hour
        val_trajs_bc.append(X_val[3])  # category
        val_trajs_bc.append(X_val[4])  # mask

        sensitive_label = 0

        except_label_cond = np.zeros(
            (self.X_val[0].shape[0], self.vocab_size["category"])
        )
        except_label_cond[:, sensitive_label] = 1
        except_label_cond = np.repeat(
            except_label_cond[:, np.newaxis, :], self.max_length, axis=1
        )  # repeat across time steps

        val_trajs_bc.append(except_label_cond)

        noise = np.random.normal(0, 1, (self.X_val[0].shape[0], self.latent_dim))
        val_trajs_bc.append(noise)  # noise
        val_real_bc = np.ones((self.X_val[0].shape[0], 1))
        val_syn_bc = np.zeros((self.X_val[0].shape[0], 1))
        exgan = exGAN()

        # Originally, we refer to a Keras implementation of GAN that trains on one random batch per epoch.
        # This is a bit confusing as an epoch here is more like an iteration rather than going through all the samples.
        # Doing this helps converge the model too, but the number of epochs it takes would be naturally large.
        # More details and explanations regarding this implementation:
        # https://github.com/eriklindernoren/Keras-GAN/issues/9
        # https://github.com/eriklindernoren/Keras-GAN/blob/master/gan/gan.py

        # We also implement a regular training loop that trains on all samples (multiple batches) per epoch.
        # Please commment the first for-loop and uncomment the second for-loop to switch to the regular training loop.
        # The regular one usually takes fewer epochs to converge (around 200 when the batch size is 256).

        # Training the model
        # The original training loop that trains on one random batch per epoch

        for epoch in range(1, epochs + 1):
            # Select a random batch of real trajectories
            idx = np.random.randint(0, X_train[0].shape[0], batch_size)

            # Ground truths for real trajectories and synthetic trajectories
            real_bc = np.ones((batch_size, 1))
            syn_bc = np.zeros((batch_size, 1))

            real_trajs_bc = []
            real_trajs_bc.append(X_train[0][idx])  # latlon
            real_trajs_bc.append(X_train[1][idx])  # day
            real_trajs_bc.append(X_train[2][idx])  # hour
            real_trajs_bc.append(X_train[3][idx])  # category
            real_trajs_bc.append(X_train[4][idx])  # mask
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Randomly select sensitive category
            cond = np.random.randint(0, self.vocab_size["category"])

            # Compute average values for each category in the batch
            category_index = exgan.average_value_by_category(
                real_trajs_bc, mask=real_trajs_bc[4]
            )
            avg_latlon = category_index[cond][0]

            if np.isnan(avg_latlon[0]) or np.isnan(avg_latlon[1]):
                print("there is nan")
                continue

            else:
                # Replace sensitive category points with average values
                #    (returns new latlon, day, hour arrays)
                new_latlon, new_day, new_hour = exgan.conditional_label(
                    real_trajs_bc,
                    real_trajs_bc[3],
                    category_index,
                    real_trajs_bc[4],
                    cond=cond,
                )
                # Replace the batch with the new values
                real_trajs_bc[0] = pad_sequences(
                    new_latlon, self.max_length, padding="pre", dtype="float64"
                )
                real_trajs_bc[1] = pad_sequences(
                    new_day, self.max_length, padding="pre", dtype="float64"
                )
                real_trajs_bc[2] = pad_sequences(
                    new_hour, self.max_length, padding="pre", dtype="float64"
                )

            # Add exept-label one-hot vector as input
            except_label_cond = np.zeros((batch_size, self.vocab_size["category"]))
            except_label_cond[:, cond] = 1
            except_label_seq = np.repeat(
                except_label_cond[:, np.newaxis, :], self.max_length, axis=1
            )  # repeat across time steps
            real_trajs_bc.append(except_label_seq.astype(np.float32))

            real_trajs_bc.append(noise)  # add noise at last

            # Generate a batch of synthetic trajectories
            gen_trajs_bc = self.generator.predict(real_trajs_bc)

            # Train the discriminator
            # No mask and noise are used
            d_loss_real = self.discriminator.train_on_batch(
                real_trajs_bc[:4] + [real_trajs_bc[5]], real_bc
            )
            d_loss_syn = self.discriminator.train_on_batch(
                gen_trajs_bc[:4] + [gen_trajs_bc[5]], syn_bc
            )
            d_loss = 0.5 * np.add(d_loss_real, d_loss_syn)

            # Train the generator
            # Mask and noise are used
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            real_trajs_bc[-1] = noise
            self.discriminator.trainable = True
            g_loss = self.combined.train_on_batch(real_trajs_bc, real_bc)

            haus = evaluate_on_batch(real_trajs_bc, gen_trajs_bc, batch_size)

            print(
                "\n[%d/%d] D Loss: %f | G Loss: %f "
                % (epoch, epochs, d_loss[0], g_loss),
                end="",
            )

            # Validation phase
            except_label_cond = np.zeros(
                (self.X_val[0].shape[0], self.vocab_size["category"])
            )
            except_label_cond[:, cond] = 1
            except_label_cond = np.repeat(
                except_label_cond[:, np.newaxis, :], self.max_length, axis=1
            )  # repeat across time steps

            gen_traj_val = self.generator.predict(val_trajs_bc)
            d_loss_real_val = self.discriminator.test_on_batch(
                val_trajs_bc[:4] + [val_trajs_bc[5]], val_real_bc
            )
            d_loss_syn_val = self.discriminator.test_on_batch(
                gen_traj_val[:4] + [gen_traj_val[5]], val_syn_bc
            )
            d_loss_val = 0.5 * np.add(d_loss_real_val, d_loss_syn_val)

            g_loss_val = self.combined.test_on_batch(val_trajs_bc, val_real_bc)
            if tensorboard:
                summary = tf.Summary(
                    value=[
                        tf.Summary.Value(
                            tag="loss/discriminator/real: TPR = 1 - FNR",
                            simple_value=d_loss_real[0],
                        ),
                        tf.Summary.Value(
                            tag="loss/discriminator/syn: TNR = 1 - FPR",
                            simple_value=d_loss_syn[0],
                        ),
                        tf.Summary.Value(
                            tag="accuracy/discriminator/real",
                            simple_value=d_loss_real[1],
                        ),
                        tf.Summary.Value(
                            tag="accuracy/discriminator/syn", simple_value=d_loss_syn[1]
                        ),
                        tf.Summary.Value(tag="loss/generator", simple_value=g_loss),
                        tf.Summary.Value(
                            tag="metric/hausdorff_train/mean",
                            simple_value=np.mean(haus),
                        ),
                        tf.Summary.Value(
                            tag="val/loss/generator", simple_value=g_loss_val
                        ),
                        tf.Summary.Value(
                            tag="val/loss/discriminator", simple_value=d_loss_val[0]
                        ),
                    ]
                )
                writer.add_summary(summary, epoch)
                writer.flush()
            print("Validation  D Loss: %f | G Loss: %f" % (d_loss_val[0], g_loss_val))

            # Print and save the losses/params
            if epoch % sample_interval == 0:
                self.save_checkpoint(epoch, path)
                print("Model params saved to the disk.")

                # Make predictions
                # test_haus, mean_acc_day, mean_acc_hour, mean_acc_cat, recall = (
                #     evaluate_on_test(self, val_trajs_bc)
                # )

                # if tensorboard:
                #     summary = tf.Summary(
                #         value=[
                #             tf.Summary.Value(
                #                 tag="val/metric/hausdorff/mean(denorm)",
                #                 simple_value=np.mean(test_haus),
                #             ),
                #             tf.Summary.Value(
                #                 tag="val/metric/hausdorff/max(denorm)",
                #                 simple_value=np.max(test_haus),
                #             ),
                #             tf.Summary.Value(
                #                 tag="val/metric/accuracy/day",
                #                 simple_value=mean_acc_day,
                #             ),
                #             tf.Summary.Value(
                #                 tag="val/metric/accuracy/hour",
                #                 simple_value=mean_acc_hour,
                #             ),
                #             tf.Summary.Value(
                #                 tag="val/metric/accuracy/category",
                #                 simple_value=mean_acc_cat,
                #             ),
                #             tf.Summary.Value(
                #                 tag="val/metric/recall/unique_location (fraction of real unique locations covered by generated within a 100m radius)",
                #                 simple_value=recall,
                #             ),
                #         ]
                #     )
                #     writer.add_summary(summary, epoch)
                #     writer.flush()

        if tensorboard:
            writer.close()

        # Training the model
        # The regular training loop that trains on all samples (multiple batches) per epoch

    #         for epoch in range(1,epochs+1):

    #             random_indices = np.random.permutation(X_train[0].shape[0])

    #             num_batches = np.ceil(random_indices.shape[0]/batch_size).astype(np.int)

    #             for i in range(num_batches):

    #                 # Select a random batch of real trajectories
    #                 idx = random_indices[batch_size*i:batch_size*(i+1)]

    #                 # Ground truths for real trajectories and synthetic trajectories
    #                 real_bc = np.ones((idx.shape[0], 1))
    #                 syn_bc = np.zeros((idx.shape[0], 1))

    #                 # Random noise
    #                 noise = np.random.normal(0, 1, (idx.shape[0], self.latent_dim))

    #                 real_trajs_bc = []
    #                 real_trajs_bc.append(X_train[0][idx]) # latlon
    #                 real_trajs_bc.append(X_train[1][idx]) # day
    #                 real_trajs_bc.append(X_train[2][idx]) # hour
    #                 real_trajs_bc.append(X_train[3][idx]) # category
    #                 real_trajs_bc.append(X_train[4][idx]) # mask
    #                 real_trajs_bc.append(noise) # random noise

    #                 # Generate a batch of synthetic trajectories
    #                 gen_trajs_bc = self.generator.predict(real_trajs_bc)

    #                 # Train the discriminator
    #                 # No mask and noise are used
    #                 d_loss_real = self.discriminator.train_on_batch(real_trajs_bc[:4], real_bc)
    #                 d_loss_syn = self.discriminator.train_on_batch(gen_trajs_bc[:4], syn_bc)
    #                 d_loss = 0.5 * np.add(d_loss_real, d_loss_syn)

    #                 # Train the generator
    #                 # Mask and noise are used
    #                 noise = np.random.normal(0, 1, (idx.shape[0], self.latent_dim))
    #                 real_trajs_bc[5] = noise
    #                 g_loss = self.combined.train_on_batch(real_trajs_bc, real_bc)

    #                 # Print the losses
    #                 print("[Epoch %d/%d] [Batch %d/%d] D Loss: %f | G Loss: %f" % (epoch, epochs, i+1, num_batches, d_loss[0], g_loss))
    #             # Save the params
    #             if epoch % sample_interval == 0:
    #                 self.save_checkpoint(epoch)
    #                 print('Model params saved to the disk.')

    def save_checkpoint(self, epoch, path):
        if path is None:
            path = "training_params"

        os.makedirs(path, exist_ok=True)
        self.combined.save_weights(path + "/C_model_" + str(epoch) + ".h5")
        self.generator.save_weights(path + "/G_model_" + str(epoch) + ".h5")
        self.discriminator.save_weights(path + "/D_model_" + str(epoch) + ".h5")
        print("Training Params Saved")
