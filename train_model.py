import glob
import os
import random
from datetime import datetime
from functools import lru_cache
from math import floor

import tensorflow.keras.callbacks as callbacks
import tensorflow.keras.layers as layers
import tensorflow.keras.metrics
import tensorflow.keras.models as models
import tensorflow.keras.optimizers as optimizers
import numpy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import Sequence

import data_utils
from scored_fen_to_dataset import MAX_DATASET_FILE_SIZE, calc_batch_size


class MultiFileEvalGenerator(Sequence):
    def __init__(self, training_files, for_validation=False, file_parts=1, ratio=0.7):
        self.file_parts = file_parts
        self.ratio = ratio
        if for_validation:
            self.training_files = training_files[int(len(training_files) * self.ratio):]
        else:
            self.training_files = training_files[:int(len(training_files) * self.ratio)]
        self.for_validation = for_validation
        if for_validation:
            print(f"Loading {len(self.training_files)} Validation files: {self.training_files}")
        else:
            print(f"Loading {len(self.training_files)} Training files: {self.training_files}")
        self.total_length = len(self.training_files) * self.file_parts
        self.file_count = len(self.training_files)
        self.indices = numpy.arange(self.total_length)
        self.inner_indices = numpy.arange(MAX_DATASET_FILE_SIZE)

    def __len__(self):
        return self.total_length

    @lru_cache
    def load_data(self, file_index):
        current_file = self.training_files[file_index]
        container = numpy.load(current_file)
        b, v1, v2 = container['arr_0'], container['arr_1'], container['arr_2']
        return b, v1, v2

    def on_epoch_end(self):
        numpy.random.shuffle(self.indices)
        numpy.random.shuffle(self.inner_indices)

    def __getitem__(self, index):
        return self._get_real_item(self.indices[index])

    def _get_real_item(self, index):
        file_index = floor((index / self.total_length) * self.file_count)
        b, v1, v2 = self.load_data(file_index)
        original_length = len(b)
        if original_length != len(self.inner_indices):
            raise Exception(f'Data is {original_length} but random indices are {len(self.inner_indices)} long.')
        if self.file_parts == len(b):
            offset = (index % self.file_parts)
            end_offset = offset + 1
            random_indices = self.inner_indices[offset:end_offset]
            b = b[random_indices]
            v1 = v1[random_indices]
            v2 = v2[random_indices]
            return b, [v1, v2]
        array_start_percent = (index % self.file_parts) / self.file_parts
        array_end_percent = array_start_percent + (1.0 / self.file_parts)
        starting_offset = floor(original_length * array_start_percent)
        ending_offset = floor(original_length * array_end_percent)
        random_indices = self.inner_indices[starting_offset:ending_offset]
        b = b[random_indices]
        v1 = v1[random_indices]
        v2 = v2[random_indices]
        return b, [v1, v2]


def train(file_parts=1, ratio=0.7, limit_files=None, residual_blocks=20, residual_layers=2, use_bias=True,
          kernel_regularizer=None, do_batch_norm=True,
          eval_out_loss="mean_squared_error", top_moves_loss="categorical_crossentropy",
          eval_weight=1.0, top_moves_weight=1.25, learn_rate=1e-3, epochs=1000, bias_regularizer=None,
          filter_count=256):
    date_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    model = compile_model(residual_blocks, residual_layers, use_bias, kernel_regularizer, do_batch_norm,
                          eval_out_loss, top_moves_loss,
                          eval_weight, top_moves_weight, learn_rate, bias_regularizer, filter_count)
    training_files = []
    for filename in glob.iglob(f'dataset\\scored_fen_dataset_*.npz', recursive=False):
        training_files.append(filename)

    random.shuffle(training_files)
    if (limit_files is not None) and (len(training_files) >= limit_files):
        training_files = training_files[0:limit_files]
        print(f"reduced training files to: {len(training_files)}")
    if len(training_files) < 2:
        print("Need at least 2 training files.")
        return

    training_generator = MultiFileEvalGenerator(training_files, False, file_parts, ratio)
    validation_generator = MultiFileEvalGenerator(training_files, True, file_parts, ratio)
    print(f"all: {training_files}")
    fit_and_save_model_batch(date_time, model, training_generator, validation_generator, epochs)


def fit_and_save_model_batch(date_time, model, training_generator, validation_generator, epochs=1000):
    model.fit(x=training_generator,
              validation_data=validation_generator,
              epochs=epochs,
              verbose=1,
              callbacks=fit_callbacks())
    model.save(f'models\\leg_model_{date_time}.h5')


def fit_callbacks():
    # slow down rate of changes to fine-tune after no improvement and prevent over fit
    return [callbacks.ReduceLROnPlateau(monitor='loss', patience=5, min_lr=1e-10, min_delta=1e-1, factor=0.1),  # 10?
            callbacks.EarlyStopping(monitor='loss', patience=3, min_delta=5e-5, restore_best_weights=True),
            callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=3, restore_best_weights=True)]


def compile_model(residual_blocks, residual_layers=2, use_bias=True, kernel_regularizer=None, do_batch_norm=True,
                  eval_out_loss="mean_squared_error", top_moves_loss="categorical_crossentropy",
                  eval_weight=1.0, top_moves_weight=1.25, learn_rate=1e-3, bias_regularizer=None, filter_count=256):
    new_model = build_residual_eval_model(residual_blocks, residual_layers, use_bias, kernel_regularizer,
                                          do_batch_norm, bias_regularizer, filter_count)  # l2(1e-4)
    losses = {"eval_out": eval_out_loss, "top_moves_out": top_moves_loss}
    loss_weights = {"eval_out": eval_weight, "top_moves_out": top_moves_weight}
    new_model.compile(optimizer=Adam(learn_rate), loss=losses, loss_weights=loss_weights)
    return new_model


def build_residual_eval_model(residual_blocks, residual_layers=2, use_bias=True, kernel_regularizer=None,
                              do_batch_norm=True, bias_regularizer=None, filter_count=256):
    encoded_board = layers.Input(shape=data_utils.BOARD_SHAPE)
    x = encoded_board
    x = layers.Conv2D(filters=filter_count, kernel_size=5, padding='same', use_bias=use_bias,
                      data_format='channels_last', kernel_regularizer=kernel_regularizer,
                      bias_regularizer=bias_regularizer)(x)
    if do_batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    for _ in range(residual_blocks):
        x = build_residual_skip_block(x, layer_count=residual_layers, use_bias=use_bias,
                                      kernel_regularizer=kernel_regularizer, do_batch_norm=do_batch_norm,
                                      bias_regularizer=bias_regularizer, filter_count=filter_count)
    residual_out = x

    x = layers.Conv2D(filters=64, kernel_size=1, data_format="channels_last",
                      use_bias=use_bias, kernel_regularizer=kernel_regularizer,
                      bias_regularizer=bias_regularizer)(residual_out)
    if do_batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(data_utils.get_uci_labels_length() * 16, activation="relu", kernel_regularizer=kernel_regularizer,
                     bias_regularizer=bias_regularizer)(x)
    top_moves_out = layers.Dense(data_utils.get_uci_labels_length(), name="top_moves_out",
                                 activation="softmax", kernel_regularizer=kernel_regularizer,
                                 bias_regularizer=bias_regularizer)(x)

    x = layers.Conv2D(filters=64, kernel_size=1, data_format="channels_last",
                      use_bias=use_bias, kernel_regularizer=kernel_regularizer,
                      bias_regularizer=bias_regularizer)(residual_out)
    if do_batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(16, activation="relu", kernel_regularizer=kernel_regularizer,
                     bias_regularizer=bias_regularizer)(x)
    eval_out = layers.Dense(3, activation="softmax", name="eval_out",
                            kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer)(x)
    # softmax vs sigmoid both are 0 to 1, but softmax is a probability distributed amongst full array
    return models.Model(inputs=encoded_board, outputs=[eval_out, top_moves_out])


def build_residual_skip_block(x, layer_count=2, use_bias=True, kernel_regularizer=None,
                              do_batch_norm=True, bias_regularizer=None, filter_count=256):
    in_x = x
    remaining = layer_count
    for _ in range(layer_count):
        x = layers.Conv2D(filters=filter_count, kernel_size=3, padding='same', use_bias=use_bias,
                          data_format='channels_last', kernel_regularizer=kernel_regularizer,
                          bias_regularizer=bias_regularizer)(x)
        if do_batch_norm:
            x = layers.BatchNormalization()(x)
        if remaining > 1:
            x = layers.Activation("relu")(x)
        remaining -= 1
    x = layers.Add()([in_x, x])
    x = layers.Activation("relu")(x)
    return x


if __name__ == '__main__':
    # disable use of GPU:
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    # async allocation
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

    # roughly 20 million positions:
    # Epoch 23/80
    # 77796/77796 [==============================] - 23581s 303ms/step - loss: 1.1039 - eval_out_loss: 0.1009 - top_moves_out_loss: 0.7079 - val_loss: 1.5230 - val_eval_out_loss: 0.1507 - val_top_moves_out_loss: 1.0794 - lr: 1.0000e-06

    # roughly 10 million positions:
    # loss: 1.7945 -
    # eval_out_loss: 0.1708 - top_moves_out_loss: 1.1187 -
    # val_loss: 2.0801 -
    # val_eval_out_loss: 0.2100 - val_top_moves_out_loss: 1.3677 - lr: 1.0000e-05

    # roughly 7 million positions:
    # loss: 2.0496 -
    # eval_out_loss: 0.1793 - top_moves_out_loss: 1.1708 -
    # val_loss: 2.4687 -
    # val_eval_out_loss: 0.2291 - val_top_moves_out_loss: 1.5471 - lr: 1.0000e-05
    # 1500 samples. 1.00/1 (moves: 1055/1500 70% 1.00 avg rank predictions: 1429/1500 95% wcp: 95% lcp: 4% - difficult predictions: 1479 - confident good predictions: 1422)
    # 1500 samples. 5.00/4 (moves: 1146/1500 76% 1.46 avg rank predictions: 1360/1500 90% wcp: 94% lcp: 12% - difficult predictions: 1475 - confident good predictions: 1342)
    # 1500 samples. 5.00/3 (moves: 1225/1500 81% 1.27 avg rank predictions: 1352/1500 90% wcp: 95% lcp: 17% - difficult predictions: 1456 - confident good predictions: 1338)
    # 1500 samples. 20.00/3 (moves: 1215/1500 81% 1.41 avg rank predictions: 1318/1500 87% wcp: 94% lcp: 17% - difficult predictions: 1454 - confident good predictions: 1306)
    # 1500 samples. 20.00/2 (moves: 1263/1500 84% 1.19 avg rank predictions: 1422/1500 94% wcp: 96% lcp: 9% - difficult predictions: 1491 - confident good predictions: 1415)
    # 1500 samples. 10.00/4 (moves: 1309/1500 87% 1.40 avg rank predictions: 1357/1500 90% wcp: 94% lcp: 12% - difficult predictions: 1450 - confident good predictions: 1337)
    # 1500 samples. 20.00/4 (moves: 1367/1500 91% 1.45 avg rank predictions: 1365/1500 91% wcp: 96% lcp: 15% - difficult predictions: 1472 - confident good predictions: 1352)
    # 1500 samples. 20.00/5 (moves: 1393/1500 92% 1.54 avg rank predictions: 1401/1500 93% wcp: 94% lcp: 6% - difficult predictions: 1475 - confident good predictions: 1394)

    # added roughly 500k fens
    # loss: 2.5025 -
    # eval_out_loss: 0.1675 - top_moves_out_loss: 1.1128 -
    # val_loss: 3.5745 -
    # val_eval_out_loss: 0.3140 - val_top_moves_out_loss: 2.0526 - lr: 1.0000e-05
    # 1500 samples. 1.00/1 (moves: 936/1500 62% 1.00 avg rank predictions: 1407/1500 93% wcp: 91% lcp: 8% - difficult predictions: 1444 - confident good predictions: 1396)
    # 1500 samples. 1.50/2 (moves: 961/1500 64% 1.08 avg rank predictions: 1276/1500 85% wcp: 88% lcp: 22% - difficult predictions: 1416 - confident good predictions: 1256)
    # 1500 samples. 1.50/15 (moves: 1011/1500 67% 1.13 avg rank predictions: 1258/1500 83% wcp: 87% lcp: 23% - difficult predictions: 1420 - confident good predictions: 1231)
    # 1500 samples. 5.00/2 (moves: 1106/1500 73% 1.19 avg rank predictions: 1397/1500 93% wcp: 93% lcp: 11% - difficult predictions: 1466 - confident good predictions: 1385)
    # 1500 samples. 10.00/2 (moves: 1214/1500 80% 1.16 avg rank predictions: 1390/1500 92% wcp: 93% lcp: 14% - difficult predictions: 1470 - confident good predictions: 1380)
    # 1500 samples. 15.00/2 (moves: 1223/1500 81% 1.20 avg rank predictions: 1337/1500 89% wcp: 90% lcp: 16% - difficult predictions: 1446 - confident good predictions: 1323)
    # 1500 samples. 20.00/2 (moves: 1282/1500 85% 1.16 avg rank predictions: 1433/1500 95% wcp: 93% lcp: 6% - difficult predictions: 1477 - confident good predictions: 1428)
    # 1500 samples. 20.00/3 (moves: 1286/1500 85% 1.33 avg rank predictions: 1312/1500 87% wcp: 92% lcp: 18% - difficult predictions: 1454 - confident good predictions: 1300)
    # 1500 samples. 10.00/5 (moves: 1330/1500 88% 1.43 avg rank predictions: 1383/1500 92% wcp: 93% lcp: 13% - difficult predictions: 1468 - confident good predictions: 1371)
    # 1500 samples. 15.00/5 (moves: 1329/1500 88% 1.55 avg rank predictions: 1400/1500 93% wcp: 93% lcp: 11% - difficult predictions: 1480 - confident good predictions: 1388)
    # 1500 samples. 25.00/5 (moves: 1363/1500 90% 1.56 avg rank predictions: 1400/1500 93% wcp: 92% lcp: 11% - difficult predictions: 1451 - confident good predictions: 1390)
    # 1500 samples. 20.00/4 (moves: 1367/1500 91% 1.35 avg rank predictions: 1423/1500 94% wcp: 93% lcp: 7% - difficult predictions: 1485 - confident good predictions: 1409)
    # 1500 samples. 20.00/5 (moves: 1408/1500 93% 1.37 avg rank predictions: 1413/1500 94% wcp: 92% lcp: 11% - difficult predictions: 1486 - confident good predictions: 1405)
    # 1500 samples. 50.00/5 (moves: 1381/1500 92% 1.63 avg rank predictions: 1307/1500 87% wcp: 89% lcp: 19% - difficult predictions: 1431 - confident good predictions: 1288)
    # 1500 samples. 100.00/5 (moves: 1432/1500 95% 1.45 avg rank predictions: 1399/1500 93% wcp: 94% lcp: 12% - difficult predictions: 1487 - confident good predictions: 1397)

    # After adding 1.5 million more fens through create_missing_scored_fens
    # loss: 2.6581 -
    # eval_out_loss: 0.1563 - top_moves_out_loss: 1.0641 -
    # val_loss: 4.1347 -
    # val_eval_out_loss: 0.4024 - val_top_moves_out_loss: 2.3109 - lr: 1.0000e-05
    # 1500 samples. 1.00/1 (moves: 959/1500 63% 1.00 avg rank predictions: 1360/1500 90% wcp: 83% lcp: 6% - difficult predictions: 1467 - confident good predictions: 1354)

    # loss: 4.9977 - eval_out_loss: 0.2397 - top_moves_out_loss: 1.7433 - val_loss: 5.8677 - val_eval_out_loss: 0.4035 - val_top_moves_out_loss: 2.5298 - lr: 1.0000e-05
    # 1500 samples. 1.00/1 (moves: 859/1500 57% 1.00 avg rank predictions: 1414/1500 94% wcp: 91% lcp: 9% - difficult predictions: 1473 - confident good predictions: 1405)

    # * more data
    # /computer restarted overnight, don't know the exact loss, but it was roughly on par with previous run/
    # 1500 samples. 1.00/1    (moves:  855/1500 56% 1.00 avg rank predictions: 1383/1500 92% wcp: 90% lcp: 11% - difficult predictions: 1463 - confident good predictions: 1374)
    # 1500 samples. 1.50/5    (moves:  946/1500 63% 1.14 avg rank predictions: 1348/1500 89% wcp: 90% lcp: 17% - difficult predictions: 1455 - confident good predictions: 1332)
    # 1500 samples. 1000.00/5 (moves: 1322/1500 88% 1.61 avg rank predictions: 1394/1500 92% wcp: 91% lcp: 14% - difficult predictions: 1465 - confident good predictions: 1377)
    train(calc_batch_size(256), 0.98, None, 23, 2, False, l2(1e-3), True,
          "categorical_crossentropy", "categorical_crossentropy", 1.0, 1.0, 1e-5, 80, None, 256)

    # * more data
    # * scored_fen_to_dataset will only keep moves not impacted by the 50 move clock when looking for duplicates
    # * position encoding for fifty_move_percent will now only be greater than zero if the clock is 99 or 100 (and we are only sending moves below 99)
    # My rationale is that we'd need millions of moves to train the model on the true impact of the 50 move clock
    # when the situation shouldn't be that common if we play well. Better to have a draw than make bad moves and lose.
    # loss: 4.5073 -
    # eval_out_loss: 0.1938 - top_moves_out_loss: 1.4796 -
    # val_loss: 6.0164 -
    # val_eval_out_loss: 0.4961 - val_top_moves_out_loss: 2.7400 - lr: 1.0000e-05
    # 1500 samples. 1.10/1    (moves:  847/1500 56% 1.00 avg rank predictions: 1349/1500 89% wcp: 87% lcp: 12% - difficult predictions: 1440 - confident good predictions: 1341)
    # 1500 samples. 1.10/3    (moves:  873/1500 58% 1.03 avg rank predictions: 1399/1500 93% wcp: 90% lcp: 9% - difficult predictions: 1465 - confident good predictions: 1393)
    # 1500 samples. 1000.00/5 (moves: 1070/1500 71% 1.97 avg rank predictions: 1190/1500 79% wcp: 81% lcp: 24% - difficult predictions: 1432 - confident good predictions: 1167)
    # train(dt, calc_batch_size(256), 0.98, None, 23, 2, False, l2(1e-3), True,
    #       "categorical_crossentropy", "categorical_crossentropy", 1.0, 1.0, 1e-5, 80, None, 256)

    # * more data
    # * best move Dense 7/3 -> 16
    # loss: 4.3522 -
    # eval_out_loss: 0.1710 - top_moves_out_loss: 1.3701 -
    # val_loss: 6.1695 -
    # val_eval_out_loss: 0.5509 - val_top_moves_out_loss: 2.8558 - lr: 1.0000e-05
    # 1500 samples. 1.00/1    (moves:  980/1500 65% 1.00 avg rank predictions: 1451/1500 96% wcp: 89% lcp: 2% - difficult predictions: 1461 - confident good predictions: 1447)
    # 1500 samples. 1000.00/3 (moves: 1275/1500 85% 1.36 avg rank predictions: 1356/1500 90% wcp: 78% lcp: 5% - difficult predictions: 1447 - confident good predictions: 1340)
    # 1500 samples. 1000.00/5 (moves: 1386/1500 92% 1.57 avg rank predictions: 1368/1500 91% wcp: 79% lcp: 3% - difficult predictions: 1433 - confident good predictions: 1351)
    # train(dt, calc_batch_size(256), 0.98, None, 23, 2, False, l2(1e-3), True,
    #       "categorical_crossentropy", "categorical_crossentropy", 1.0, 1.0, 1e-5, 80, None, 256)

    # * more data
    # * changed shape (8, 8, 18) -> (8, 8, 40)
    # loss: 5.7347 -
    # eval_out_loss: 0.2311 - top_moves_out_loss: 1.7882 -
    # val_loss: 6.8116 -
    # val_eval_out_loss: 0.5163 - val_top_moves_out_loss: 2.6807 - lr: 1.0000e-05
    # train(dt, calc_batch_size(256), 0.98, None, 23, 2, False, l2(1e-3), True,
    #       "categorical_crossentropy", "categorical_crossentropy", 1.0, 1.0, 1e-5, 80, None, 256)

    # * l2(1e-5) -> l2(1e-3)
    # * dense 16/1 -> 7/3
    # loss: 5.1603 -
    # eval_out_loss: 0.1755 - top_moves_out_loss: 1.5311 -
    # val_loss: 6.8331 -
    # val_eval_out_loss: 0.4479 - val_top_moves_out_loss: 2.9939 - lr: 1.0000e-05
    # 1500 samples. 1.00/1 (moves: 759/1500 50% 1.00 avg rank predictions: 1359/1500 90% wcp: 87% lcp: 10% - difficult predictions: 1449 - confident good predictions: 1350)
    # train(dt, calc_batch_size(256), 0.98, None, 23, 2, False, l2(1e-3), True,
    #       "categorical_crossentropy", "categorical_crossentropy", 1.0, 1.0, 1e-5, 80, None, 256)

    # * Weights 0.5:1 -> 1:1
    # * l2(1e-3) -> l2(1e-5)
    # * dense 16 -> 16/1
    # loss: 2.7602 -
    # eval_out_loss: 0.3852 - top_moves_out_loss: 2.1447 -
    # val_loss: 3.4291 -
    # val_eval_out_loss: 0.5087 - val_top_moves_out_loss: 2.6893 - lr: 1.0000e-05
    # 1500 samples. 1.00/1 (moves: 643/1500 42% 1.00 avg rank predictions: 1267/1500 84% wcp: 88% lcp: 25% - difficult predictions: 1480 - confident good predictions: 1261)
    # train(dt, calc_batch_size(256), 0.98, None, 23, 2, False, l2(1e-5), True,
    #       "categorical_crossentropy", "categorical_crossentropy", 1.0, 1.0, 1e-5, 80, None, 256)

    # * more data
    # * Weights 1:1 -> 0.5:1
    # * layers 24/2 -> 23/2 (we had an OOM error)
    # loss: 4.8574 -
    # eval_out_loss: 0.3154 - top_moves_out_loss: 1.6446 -
    # val_loss: 6.0673 -
    # val_eval_out_loss: 0.4532 - val_top_moves_out_loss: 2.8548 - lr: 1.0000e-05
    # 1500 samples. 1.00/1 (moves: 759/1500 50% 1.00 avg rank predictions: 1269/1500 84% wcp: 79% lcp: 21% - difficult predictions: 1446 - confident good predictions: 1247)
    # 1500 samples. 1000.00/4 (moves: 1214/1500 80% 1.61 avg rank predictions: 1295/1500 86% wcp: 82% lcp: 17% - difficult predictions: 1449 - confident good predictions: 1282)
    # 1500 samples. 1000.00/15 (moves: 1404/1500 93% 3.11 avg rank predictions: 1282/1500 85% wcp: 81% lcp: 19% - difficult predictions: 1465 - confident good predictions: 1266)
    # train(dt, calc_batch_size(256), 0.98, None, 23, 2, False, l2(1e-3), True,
    #       "categorical_crossentropy", "categorical_crossentropy", 0.5, 1.0, 1e-5, 80, None, 256)

    # * more data
    # * changed shape (8, 8, 40) -> (8, 8, 18)
    # * added layers 23 -> 24
    # loss: 5.0384 -
    # eval_out_loss: 0.1662 - top_moves_out_loss: 1.5013 -
    # val_loss: 6.8750 -
    # val_eval_out_loss: 0.5725 - val_top_moves_out_loss: 2.9904 - lr: 1.0000e-05
    # 1500 samples. 1.00/1 (moves: 734/1500 48% 1.00 avg rank predictions: 1438/1500 95% wcp: 91% lcp: 4% - difficult predictions: 1492 - confident good predictions: 1433)
    # 1500 samples. 1.00/1 (moves: 832/1500 55% 1.00 avg rank predictions: 1383/1500 92% wcp: 87% lcp: 10% - difficult predictions: 1436 - confident good predictions: 1370)
    # 1500 samples. 1000.00/4 (moves: 1245/1500 83% 1.54 avg rank predictions: 1375/1500 91% wcp: 86% lcp: 10% - difficult predictions: 1428 - confident good predictions: 1362)
    # 1500 samples. 1000.00/15 (moves: 1455/1500 97% 2.69 avg rank predictions: 1431/1500 95% wcp: 91% lcp: 6% - difficult predictions: 1467 - confident good predictions: 1425)
    # 1500 samples. 1000.00/15 (moves: 1463/1500 97% 2.35 avg rank predictions: 1398/1500 93% wcp: 88% lcp: 9% - difficult predictions: 1457 - confident good predictions: 1383)>
    # train(dt, calc_batch_size(256), 0.98, None, 24, 2, False, l2(1e-3), True,
    #       "categorical_crossentropy", "categorical_crossentropy", 1.0, 1.0, 1e-5, 80, None, 256)

    # * more data
    # * Weights 0.5:1 -> 1:1
    # loss: 4.4801 -
    # eval_out_loss: 0.1525 - top_moves_out_loss: 1.2235 -
    # val_loss: 6.6592 -
    # val_eval_out_loss: 0.6022 - val_top_moves_out_loss: 2.9987 - lr: 1.0000e-05
    # 1/1     (moves: 339/515 65% 1 avg rank predictions: 483/515 93% wcp: 83% lcp: 3% - difficult predictions: 495 - confident good predictions: 476)
    # train(dt, calc_batch_size(256), 0.98, None, 23, 2, False, l2(1e-3), True,
    #       "categorical_crossentropy", "categorical_crossentropy", 1.0, 1.0, 1e-5, 80, None, 256)

    # * more data
    # * Weights 1:1 -> 0.5:1
    # loss: 4.4342 -
    # eval_out_loss: 0.3205 - top_moves_out_loss: 1.3410 -
    # val_loss: 6.0346 -
    # val_eval_out_loss: 0.5171 - val_top_moves_out_loss: 2.8959 - lr: 1.0000e-05
    # 1.1/2   (moves: 430/683 62% 1 avg rank predictions: 537/683 78% wcp: 85% lcp: 31% - difficult predictions: 638 - confident good predictions: 518)
    # 1000/3  (moves: 660/836 78% 1 avg rank predictions: 653/836 78% wcp: 85% lcp: 32% - difficult predictions: 782 - confident good predictions: 637)
    # train(dt, calc_batch_size(256), 0.98, None, 23, 2, False, l2(1e-3), True,
    #       "categorical_crossentropy", "categorical_crossentropy", 1.0, 1.0, 1e-5, 80, None, 256)

    # * more data
    # * Weights 0.1:1 -> 1:1
    # loss: 5.0186 -
    # eval_out_loss: 0.1817 - top_moves_out_loss: 1.4850 -
    # val_loss: 6.7101 -
    # val_eval_out_loss: 0.5429 - val_top_moves_out_loss: 2.8774 - lr: 1.0000e-05
    # 1.1/2   (moves: 340/506 67% 1 avg rank predictions: 470/506 92% wcp: 85% lcp: 6% - difficult predictions: 482 - confident good predictions: 465)
    # train(dt, calc_batch_size(256), 0.98, None, 23, 2, False, l2(1e-3), True,
    #       "categorical_crossentropy", "categorical_crossentropy", 1.0, 1.0, 1e-5, 80, None, 256)

    # * Weights 1:1 -> 0.1:1
    # loss: 4.6358 -
    # eval_out_loss: 0.5096 - top_moves_out_loss: 1.6184 -
    # val_loss: 5.6559 -
    # val_eval_out_loss: 0.5919 - val_top_moves_out_loss: 2.7017 - lr: 1.0000e-05
    # 1.1/2   (moves: 330/622 53% 1 avg rank predictions: 439/622 70% wcp: 51% lcp: 20% - difficult predictions: 591 - confident good predictions: 428)
    # 1000/2  (moves: 800/1069 74% 1 avg rank predictions: 819/1069 76% wcp: 60% lcp: 17% - difficult predictions: 1040 - confident good predictions: 799)
    # 1000/3  (moves: 401/507 79% 1 avg rank predictions: 356/507 70% wcp: 54% lcp: 22% - difficult predictions: 493 - confident good predictions: 350)
    # train(dt, calc_batch_size(256), 0.98, None, 23, 2, False, l2(1e-3), True,
    #       "categorical_crossentropy", "categorical_crossentropy", 0.1, 1.0, 1e-5, 80, None, 256)

    # * more data
    # * Weights 0.5:1 -> 1:1
    # loss: 5.3046 -
    # eval_out_loss: 0.2100 - top_moves_out_loss: 1.6084 -
    # val_loss: 6.7502 -
    # val_eval_out_loss: 0.5652 - val_top_moves_out_loss: 2.7714 - lr: 1.0000e-05
    # train(dt, calc_batch_size(256), 0.98, None, 23, 2, False, l2(1e-3), True,
    #       "categorical_crossentropy", "categorical_crossentropy", 1.0, 1.0, 1e-5, 80, None, 256)

    # * Weights 1:2 -> 0.5:1
    # loss: 4.4725 -
    # eval_out_loss: 0.3164 - top_moves_out_loss: 1.3351 -
    # val_loss: 6.0506 -
    # val_eval_out_loss: 0.4960 - val_top_moves_out_loss: 2.8766 - lr: 1.0000e-05
    # 1.1/2  (moves: 355/515 68% 1 avg rank predictions: 429/515 83% wcp: 88% lcp: 27% - difficult predictions: 489 - confident good predictions: 425)
    # 1000/2 (moves: 403/518 77% 1 avg rank predictions: 439/518 84% wcp: 84% lcp: 22% - difficult predictions: 493 - confident good predictions: 433)
    # 2/3    (moves: 382/509 75% 1 avg rank predictions: 414/509 81% wcp: 86% lcp: 27% - difficult predictions: 478 - confident good predictions: 409)
    # 1000/3 (moves: 596/673 88% 1 avg rank predictions: 557/673 82% wcp: 84% lcp: 25% - difficult predictions: 640 - confident good predictions: 552)
    # train(dt, calc_batch_size(256), 0.98, None, 23, 2, False, l2(1e-3), True,
    #       "categorical_crossentropy", "categorical_crossentropy", 0.5, 1.0, 1e-5, 80, None, 256)

    # * more data
    # * lr  5e-6 -> 1e-5
    # * Weights 1:1.5 -> 1:2
    # # layers: 20/2 -> 23/2
    # loss: 9.6268 -
    # eval_out_loss: 0.3638 - top_moves_out_loss: 1.5253 -
    # val_loss: 11.9674 -
    # val_eval_out_loss: 0.4939 - val_top_moves_out_loss: 2.6912 - lr: 1.0000e-05
    # 1.1/2   (moves: 406/660 61% 1 avg rank predictions: 593/660 89% wcp: 79% lcp: 6% - difficult predictions: 655 - confident good predictions: 592)
    # 1.1/3   (moves: 284/455 62% 1 avg rank predictions: 368/455 80% wcp: 67% lcp: 16% - difficult predictions: 431 - confident good predictions: 361)
    # 1000/15 (moves: 569/580 98% 2 avg rank predictions: 459/580 79% wcp: 67% lcp: 13% - difficult predictions: 545 - confident good predictions: 454)
    # train(dt, calc_batch_size(256), 0.98, None, 23, 2, False, l2(1e-3), True,
    #       "categorical_crossentropy", "categorical_crossentropy", 1.0, 2.0, 1e-5, 80, None, 256)

    # * more data
    # * lr 1e-5 -> 5e-6
    # loss: 8.8602 -
    # eval_out_loss: 0.3080 - top_moves_out_loss: 1.4619 -
    # val_loss: 11.0560 -
    # val_eval_out_loss: 0.5579 - val_top_moves_out_loss: 2.8068 - lr: 5.0000e-06
    # 1.1/2  (moves: 298/599 49% 1 avg rank predictions: 517/599 86% wcp: 82% lcp: 12% - difficult predictions: 595 - confident good predictions: 514)
    # train(dt, calc_batch_size(256), 0.98, None, 20, 2, False, l2(1e-3), True,
    #       "categorical_crossentropy", "categorical_crossentropy", 1.0, 1.5, 5e-6, 200, None, 256)

    # * more data
    # loss: 7.6013 -
    # eval_out_loss: 0.3157 - top_moves_out_loss: 1.6550 -
    # val_loss: 9.2107 -
    # val_eval_out_loss: 0.5019 - val_top_moves_out_loss: 2.6717 - lr: 1.0000e-05
    # 1.1/2  (moves: 297/565 52% 1 avg rank predictions: 518/565 91% wcp: 84% lcp: 8% - difficult predictions: 549 - confident good predictions: 513)
    # train(dt, calc_batch_size(256), 0.98, None, 20, 2, False, l2(1e-3), True,
    #       "categorical_crossentropy", "categorical_crossentropy", 1.0, 1.5, 1e-5, 80, None, 256)

    # * more data
    # loss: 7.6858 -
    # eval_out_loss: 0.3069 - top_moves_out_loss: 1.6654 -
    # val_loss: 9.3087 -
    # val_eval_out_loss: 0.5361 - val_top_moves_out_loss: 2.6634 - lr: 1.0000e-05
    # 1.1/2    (moves: 475/772 61% 1 avg rank predictions: 690/772 89% wcp: 80% lcp: 6% - difficult predictions: 765 - confident good predictions: 689)
    # train(dt, calc_batch_size(256), 0.98, None, 20, 2, False, l2(1e-3), True,
    #       "categorical_crossentropy", "categorical_crossentropy", 1.0, 1.5, 1e-5, 80, None, 256)

    # * more data
    # * only have a draw eval if the centipawn eval is exactly 0
    # * disabled bias for the early layers, which likely doesn't matter for anywhere but the last dense layers.
    # * l2(5e-6) -> l2(1e-3)
    # * lr 5e-6 -> lr 1e-5
    # * weights 1:1 -> 1:1.5
    # * Early stopping to 5 from 3
    # loss: 7.3293 -
    # eval_out_loss: 0.2915 - top_moves_out_loss: 1.5104 -
    # val_loss: 9.4343 -
    # val_eval_out_loss: 0.5605 - val_top_moves_out_loss: 2.7900 - lr: 1.0000e-05
    # 1.1/2   (moves: 312/500 62% 1 avg rank predictions: 445/500 89% wcp: 84% lcp: 15% - difficult predictions: 484 - confident good predictions: 440)
    # 100/2   (moves: 431/557 77% 1 avg rank predictions: 497/557 89% wcp: 84% lcp: 13% - difficult predictions: 533 - confident good predictions: 490)
    # 2/3     (moves: 486/687 70% 1 avg rank predictions: 600/687 87% wcp: 82% lcp: 17% - difficult predictions: 663 - confident good predictions: 588)
    # 100/3   (moves: 414/505 81% 1 avg rank predictions: 481/505 95% wcp: 91% lcp: 6% - difficult predictions: 492 - confident good predictions: 479)
    # 2/5     (moves: 370/515 71% 1 avg rank predictions: 453/515 87% wcp: 84% lcp: 18% - difficult predictions: 497 - confident good predictions: 446)
    # train(dt, calc_batch_size(256), 0.98, None, 20, 2, False, l2(1e-3), True,
    #       "categorical_crossentropy", "categorical_crossentropy", 1.0, 1.5, 1e-5, 80, None, 256)

    # * more data
    # * No bias regularizer
    # * l2(5e-6)
    # loss: 2.5804 -
    # eval_out_loss: 0.5683 - top_moves_out_loss: 1.9017 -
    # val_loss: 3.3748 -
    # val_eval_out_loss: 0.6384 - val_top_moves_out_loss: 2.6257 - lr: 5.0000e-06
    # 1.1/3  (moves: 236/506 46% 1 avg rank predictions: 376/506 74% wcp: 59% lcp: 14% - difficult predictions: 406 - confident good predictions: 336)
    # 2/3    (moves: 332/584 56% 1 avg rank predictions: 486/584 83% wcp: 78% lcp: 12% - difficult predictions: 561 - confident good predictions: 471)
    # 100/3  (moves: 368/512 71% 1 avg rank predictions: 366/512 71% wcp: 57% lcp: 16% - difficult predictions: 401 - confident good predictions: 321)
    # 100/5  (moves: 415/503 82% 1 avg rank predictions: 345/503 68% wcp: 56% lcp: 17% - difficult predictions: 376 - confident good predictions: 304)
    # train(dt, calc_batch_size(256), 0.98, None, 20, 2, True, l2(5e-6), True,
    #       "categorical_crossentropy", "categorical_crossentropy", 1.0, 1.0, 5e-6, 80, None, 256)

    # * more data
    # * lr changed to 5e-6.
    # * filter count parameterized
    # * fixed an issue with call to
    #     build_residual_skip_block (batch normalization may have been true or not and
    #     bias regularizer was probably not set)
    # * added dense layer for moves:
    #   layers.Dense(data_utils.get_uci_labels_length()*16, activation="relu", kernel_regularizer=kernel_regularizer,
    #                      bias_regularizer=bias_regularizer)(x)
    # loss: 2.4664 -
    # eval_out_loss: 0.5542 - top_moves_out_loss: 1.9120 -
    # val_loss: 3.3077 -
    # val_eval_out_loss: 0.6611 - val_top_moves_out_loss: 2.6464 - lr: 5.0000e-06
    # 1.1/2 (moves: 245/563 43% 1 avg rank predictions: 501/563 88% wcp: 95% lcp: 19% - difficult predictions: 547 - confident good predictions: 488)
    # 2/2   (moves: 349/557 62% 1 avg rank predictions: 401/557 71% wcp: 82% lcp: 34% - difficult predictions: 443 - confident good predictions: 370)
    # 3/2   (moves: 448/711 63% 1 avg rank predictions: 643/711 90% wcp: 94% lcp: 17% - difficult predictions: 666 - confident good predictions: 622)
    # 100/2 (moves: 376/602 62% 1 avg rank predictions: 394/602 65% wcp: 79% lcp: 38% - difficult predictions: 416 - confident good predictions: 348)
    # 2/3   (moves: 319/504 63% 1 avg rank predictions: 353/504 70% wcp: 83% lcp: 34% - difficult predictions: 389 - confident good predictions: 323)
    # 3/3   (moves: 362/511 70% 1 avg rank predictions: 366/511 71% wcp: 81% lcp: 34% - difficult predictions: 386 - confident good predictions: 328)
    # 5/3   (moves: 374/522 71% 1 avg rank predictions: 369/522 70% wcp: 83% lcp: 35% - difficult predictions: 412 - confident good predictions: 344)
    # 10/3  (moves: 397/563 70% 1 avg rank predictions: 377/563 66% wcp: 80% lcp: 37% - difficult predictions: 405 - confident good predictions: 334)
    # 100/3 (moves: 371/501 74% 1 avg rank predictions: 378/501 75% wcp: 83% lcp: 31% - difficult predictions: 401 - confident good predictions: 344)
    # 2/5   (moves: 435/668 65% 1 avg rank predictions: 491/668 73% wcp: 81% lcp: 33% - difficult predictions: 541 - confident good predictions: 439)
    # 3/5   (moves: 485/692 70% 1 avg rank predictions: 508/692 73% wcp: 83% lcp: 33% - difficult predictions: 549 - confident good predictions: 453)
    # 100/5 (moves: 438/533 82% 1 avg rank predictions: 388/533 72% wcp: 82% lcp: 33% - difficult predictions: 430 - confident good predictions: 349)
    # 100/7 (moves: 455/503 90% 2 avg rank predictions: 355/503 70% wcp: 82% lcp: 36% - difficult predictions: 388 - confident good predictions: 329)
    # 10/10 (moves: 434/510 85% 2 avg rank predictions: 410/510 80% wcp: 89% lcp: 27% - difficult predictions: 473 - confident good predictions: 392)
    # 100/10(moves: 469/501 93% 2 avg rank predictions: 370/501 73% wcp: 83% lcp: 33% - difficult predictions: 405 - confident good predictions: 340)
    # 100/15(moves: 634/666 95% 2 avg rank predictions: 589/666 88% wcp: 93% lcp: 21% - difficult predictions: 643 - confident good predictions: 567)
    # train(dt, calc_batch_size(256), 0.98, None, 20, 2, True, l2(1e-8), True,
    #       "categorical_crossentropy", "categorical_crossentropy", 1.0, 1.0, 5e-6, 80, l2(1e-8), 256)

    # more data.
    # loss: 2.5427 -
    # eval_out_loss: 0.4146 - top_moves_out_loss: 2.1280 -
    # val_loss: 3.6421 -
    # val_eval_out_loss: 0.7326 - val_top_moves_out_loss: 2.9093 - lr: 1.0000e-06
    # 1.5/3 (moves: 287/526 54% 1 avg rank predictions: 426/526 80% wcp: 84% lcp: 27% - difficult predictions: 389 - confident good predictions: 395)
    # 100/3(moves: 329/539 61% 1 avg rank predictions: 416/539 77% wcp: 84% lcp: 31% - difficult predictions: 427 - confident good predictions: 392)
    # train(dt, calc_batch_size(256), 0.98, None, 20, 2, True, l2(1e-8), True,
    #       "categorical_crossentropy", "categorical_crossentropy", 1.0, 1.0, 1e-6, 80, l2(1e-8))

    # more data.
    # loss: 2.6174 -
    # eval_out_loss: 0.4705 - top_moves_out_loss: 2.1467 -
    # val_loss: 3.6805 -
    # val_eval_out_loss: 0.7441 - val_top_moves_out_loss: 2.9363 - lr: 1.0000e-06
    # TOP_MOVE_BEST_TO_WORST_SCORE_RATIO = 1.5 (moves: 275/544 50% 1 avg rank predictions: 431/544 79% wcp: 84% lcp: 24% - difficult predictions: 433 - confident good predictions: 406)
    # TOP_MOVE_BEST_TO_WORST_SCORE_RATIO = 100/3 moves (moves: 285/449 63% 1 avg rank predictions: 383/449 85% wcp: 90% lcp: 20% - difficult predictions: 402 - confident good predictions: 361)
    # train(dt, calc_batch_size(256), 0.98, None, 20, 2, True, l2(1e-8), True,
    #       "categorical_crossentropy", "categorical_crossentropy", 1.0, 1.0, 1e-6, 80, l2(1e-8))

    # loss: 2.6104 -
    # eval_out_loss: 0.4467 - top_moves_out_loss: 2.1636 -
    # val_loss: 3.6986 -
    # val_eval_out_loss: 0.7472 - val_top_moves_out_loss: 2.9513 - lr: 1.0000e-06
    # TOP_MOVE_BEST_TO_WORST_SCORE_RATIO = 1.5; (moves: 225/515 43% 1 avg rank predictions: 465/515 90% wcp: 93% lcp: 20% - difficult predictions: 480 - confident good predictions: 453)
    # train(dt, calc_batch_size(256), 0.98, None, 20, 2, True, l2(1e-8), True,
    #       "categorical_crossentropy", "categorical_crossentropy", 1.0, 1.0, 1e-6, 80, l2(1e-8))

    # more data, but also deduped fens by excluding turn
    # 1st train:
    # with TOP_MOVE_BEST_TO_WORST_SCORE_RATIO=2? (moves: 116/505 22% 1 avg rank predictions: 418/505 82% wcp: 87% lcp: 20% - difficult predictions: 471 - confident good predictions: 404)
    # with TOP_MOVE_BEST_TO_WORST_SCORE_RATIO=1.5? (moves: 119/392 30% 1 avg rank predictions: 265/392 67% wcp: 88% lcp: 41% - difficult predictions: 298 - confident good predictions: 245)
    # 2nd train (more data):
    # loss: 5.4544 -
    # eval_out_loss: 0.0818 - top_moves_out_loss: 1.2568 -
    # val_loss: 8.8346 -
    # val_eval_out_loss: 1.0277 - val_top_moves_out_loss: 2.7879 - lr: 1.0000e-05
    # with TOP_MOVE_BEST_TO_WORST_SCORE_RATIO=10 (moves: 417/453 92% 1 avg rank predictions: 408/453 90% wcp: 92% lcp: 10% - difficult predictions: 428 - confident good predictions: 406)
    # with TOP_MOVE_BEST_TO_WORST_SCORE_RATIO=1.5 (moves: 146/407 35% 1 avg rank predictions: 286/407 70% wcp: 82% lcp: 24% - difficult predictions: 323 - confident good predictions: 282)
    # train(dt, calc_batch_size(256), 0.98, None, 20, 2, True, l2(1e-3), True,
    #       "categorical_crossentropy", "categorical_crossentropy", 2.0, 1.0, 1e-5, 80, l2(1e-3))

    # more data
    # l2 1e-8 -> 1e-3, lr 1e6 -> 1e5, weights 1:1 -> 2:1, patience on early stop from 4 -> 5
    # loss: 11.3060 -
    # eval_out_loss: 0.4699 - top_moves_out_loss: 3.2288 -
    # val_loss: 11.9450 -
    # val_eval_out_loss: 0.7064 - val_top_moves_out_loss: 3.3428 - lr: 1.0000e-05
    # (moves: 216/317 68% 5 avg rank predictions: 269/317 84% wcp: 76% lcp: 8% - difficult predictions: 299 - confident good predictions: 258)
    # train(dt, calc_batch_size(256), 0.98, None, 20, 2, True, l2(1e-3), True,
    #       "categorical_crossentropy", "categorical_crossentropy", 2.0, 1.0, 1e-5, 80, l2(1e-3))

    # more data
    # lr 5e-6 -> 1e-6, weight 3:1 to 1:1
    # 32->16 on last dense layer
    # added back 1/0 to from and number of attacker %
    # loss: 3.8470 -
    # eval_out_loss: 0.6037 - top_moves_out_loss: 3.2432 -
    # val_loss: 4.3047 -
    # val_eval_out_loss: 0.7800 - val_top_moves_out_loss: 3.5246 - lr: 1.0000e-06
    # (moves: 297/477 62% 5 avg rank predictions: 398/477 83% wcp: 78% lcp: 14% - difficult predictions: 455 - confident good predictions: 382)
    # train(dt, calc_batch_size(256), 0.98, None, 20, 2, True, l2(1e-8), True,
    #       "categorical_crossentropy", "categorical_crossentropy", 1.0, 1.0, 1e-6, 80, l2(1e-8))

    # more data
    # changed weights 1:1.25 to 3:1
    # loss: 5.1868 -
    # eval_out_loss: 0.5208 - top_moves_out_loss: 3.6241 -
    # val_loss: 5.9561 -
    # val_eval_out_loss: 0.7166 - val_top_moves_out_loss: 3.8060 - lr: 5.0000e-06
    # (moves: 245/345 71% 6 avg rank predictions: 252/345 73% wcp: 60% lcp: 15% - difficult predictions: 271 - confident good predictions: 235)
    # train(dt, calc_batch_size(256), 0.98, None, 20, 2, True, l2(1e-8), True,
    #       "categorical_crossentropy", "categorical_crossentropy", 3.0, 1.0, 5e-6, 80, l2(1e-8))

    # more data. changed the to/from move encoding. changed square attackers encoding.
    # changed weights from 2:1 to 1:1.25
    # loss: 4.6784 -
    # eval_out_loss: 0.6241 - top_moves_out_loss: 3.2434 -
    # val_loss: 5.0365 -
    # val_eval_out_loss: 0.7129 - val_top_moves_out_loss: 3.4587 - lr: 5.0000e-06
    # (moves: 456/575 79% 5 avg rank predictions: 387/575 67% wcp: 60% lcp: 19% - difficult predictions: 420 - confident good predictions: 341)
    # train(dt, calc_batch_size(256), 0.98, None, 20, 2, True, l2(1e-8), True,
    #       "categorical_crossentropy", "categorical_crossentropy", 1.0, 1.25, 5e-6, 80, l2(1e-8))

    # more data
    # switched how attackers of a squares are measured, removed 0/1 flags for attackers
    # loss: 4.4467 -
    # eval_out_loss: 0.5517 - top_moves_out_loss: 3.3432 -
    # val_loss: 4.9890 -
    # val_eval_out_loss: 0.7407 - val_top_moves_out_loss: 3.5074 - lr: 5.0000e-06
    # (moves: 264/300 88% predictions: 201/300 67% - difficult predictions: 219 - confident good predictions: 176)
    # 1 best move only (rather than 10 best moves):
    # (moves: 141/569 24% predictions: 392/569 68% wcp: 62% lcp: 23 - difficult predictions: 414 - confident good predictions: 344)
    # 2 best move only (rather than 10 best moves):
    # (moves: 231/560 41% predictions: 402/560 71% wcp: 62% lcp: 17% - difficult predictions: 411 - confident good predictions: 354)
    # 20 best move only (rather than 10 best moves):
    # (moves: 1153/1203 95% predictions: 933/1203 77% wcp: 68% lcp: 17% - difficult predictions: 1002 - confident good predictions: 845)
    # train(dt, calc_batch_size(256), 0.98, None, 20, 2, True, l2(1e-8), True,
    #       "categorical_crossentropy", "categorical_crossentropy", 2.0, 1.0, 5e-6, 80, l2(1e-8))

    # more data
    # added attackers for each side as a percent.
    # loss: 4.4003 -
    # eval_out_loss: 0.5498 - top_moves_out_loss: 3.3005 -
    # val_loss: 4.8392 -
    # val_eval_out_loss: 0.7056 - val_top_moves_out_loss: 3.4279 - lr: 5.0000e-06
    # (moves: 206/300 68% predictions: 217/300 72% - difficult predictions: 232 - confident good predictions: 202)
    # train(dt, calc_batch_size(256), 0.98, None, 20, 2, True, l2(1e-8), True,
    #       "categorical_crossentropy", "categorical_crossentropy", 2.0, 1.0, 5e-6, 80, l2(1e-8))

    # more data
    # added a 1/0 flag if white pieces are attacked by black and a separate 1/0 flag if black pieces are attacked by white
    # [callbacks.ReduceLROnPlateau(monitor='loss', patience=7, min_lr=1e-10, min_delta=1e-1, factor=0.1),  # 10?
    #             callbacks.EarlyStopping(monitor='loss', patience=4, min_delta=5e-5, restore_best_weights=True),
    #             callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=4, restore_best_weights=True)]
    # loss: 4.3332 -
    # eval_out_loss: 0.5282 - top_moves_out_loss: 3.2767 -
    # val_loss: 4.9139 -
    # val_eval_out_loss: 0.7066 - val_top_moves_out_loss: 3.5006 - lr: 5.0000e-06
    # changed the entire prediction algorithm to be about confidence of winning and only looking at best moves
    # this is different from the old algorithm that tried to evaluate winning without adjusting for low
    # confidence.
    # train(dt, calc_batch_size(256), 0.98, None, 20, 2, True, l2(1e-8), True,
    #       "categorical_crossentropy", "categorical_crossentropy", 2.0, 1.0, 5e-6, 80, l2(1e-8))

    # more data
    # loss: 4.4494 -
    # eval_out_loss: 0.5514 - top_moves_out_loss: 3.3465 -
    # val_loss: 4.9894 -
    # val_eval_out_loss: 0.7301 - val_top_moves_out_loss: 3.5291 - lr: 5.0000e-06
    # (moves: 214/300 71% predictions: 239/300 79% - difficult predictions: 244 - confident good predictions: 225)
    # train(dt, calc_batch_size(256), 0.9, None, 20, 2, True, l2(1e-8), True,
    #       "categorical_crossentropy", "categorical_crossentropy", 2.0, 1.0, 5e-6, 80, l2(1e-8))

    # previous input data may have noise in the form of simple eval
    # removed the simple eval from the input as it might be noise.
    # replace the data with the legal_move.from locations to help eliminate impossible best moves
    # Conv2D before final dense layers back to filters=64 from 512
    # 40/2 -> 20/2
    # eval dense layer from 256->32
    # kept bias regulator on dense layer
    # ReduceLROnPlateau mindelta from 1e-3 to 1e-1, patience from 10 to 8, min_lr from 1e-9 to 1e-10
    # ReduceLROnPlateau(monitor='loss', patience=8, min_lr=1e-10, min_delta=1e-1, factor=0.1)
    # loss: 4.2772 -
    # eval_out_loss: 0.5126 - top_moves_out_loss: 3.2518 -
    # val_loss: 4.9183 -
    # val_eval_out_loss: 0.7099 - val_top_moves_out_loss: 3.4984 - lr: 5.0000e-06
    # (moves: 184/300 61% predictions: 231/300 77% - difficult predictions: 245 - confident good predictions: 217)
    # train(dt, calc_batch_size(256), 0.9, None, 20, 2, True, l2(1e-8), True,
    #       "categorical_crossentropy", "categorical_crossentropy", 2.0, 1.0, 5e-6, 80, l2(1e-8))

    # more data. 20/2 -> 40/2
    # 50/2 = oom
    # loss: 4.5986 -
    # eval_out_loss: 0.5269 - top_moves_out_loss: 3.5437 -
    # val_loss: 5.0564 -
    # val_eval_out_loss: 0.7388 - val_top_moves_out_loss: 3.5777 - lr: 5.0000e-06
    # looks worse (slower to train and slightly worse results than we had.)
    # train(dt, calc_batch_size(256), 0.9, None, 40, 2, True, l2(1e-8), True,
    #       "categorical_crossentropy", "categorical_crossentropy", 2.0, 1.0, 5e-6, 80, l2(1e-8))

    # more data.
    # added bias regularizers to dense layer
    # increased convolution layers before final dense layers 64->512
    # dense layer before eval out 64->256
    # loss: 4.8073 -
    # eval_out_loss: 0.6586 - top_moves_out_loss: 3.4900 -
    # val_loss: 5.1168 -
    # val_eval_out_loss: 0.7259 - val_top_moves_out_loss: 3.6648 - lr: 5.0000e-06
    # moves: 147/300 49% predictions: 252/300 84% - difficult predictions: 291 - confident good predictions: 244
    # train(dt, calc_batch_size(256), 0.9, None, 20, 2, True, l2(1e-8), True,
    #       "categorical_crossentropy", "categorical_crossentropy", 2.0, 1.0, 5e-6, 80, l2(1e-8))

    # more data. added extra eval dense layer, 20/2 layers
    # loss: 4.2594 -
    # eval_out_loss: 0.4945 - top_moves_out_loss: 3.2702 -
    # val_loss: 4.9003 -
    # val_eval_out_loss: 0.6833 - val_top_moves_out_loss: 3.5335 - lr: 5.0000e-06
    # result: 95/100 top moves 81/100 evals
    # train(dt, calc_batch_size(256), 0.9, None, 20, 2, True, l2(1e-8), True,
    #       "categorical_crossentropy", "categorical_crossentropy", 2.0, 1.0, 5e-6, 80, l2(1e-8))

    # last eval dense layer:  32->16, 30/3 layers
    # loss: 4.2387 -
    # eval_out_loss: 0.4335 - top_moves_out_loss: 3.3714 -
    # val_loss: 5.2677 -
    # val_eval_out_loss: 0.7766 - val_top_moves_out_loss: 3.7142 - lr: 5.0000e-06
    # in 100 moves checked with "scan_dataset", 79 best moves were found and the eval was right 79 times.
    # for comparison, simple eval would have only been right 27 times.
    # train(dt, calc_batch_size(256), 0.9, None, 30, 3, True, l2(1e-8), True,
    #       "categorical_crossentropy", "categorical_crossentropy", 2.0, 1.0, 5e-6, 80, l2(1e-8))

    # more data. last eval dense layer:  16->32
    # no real difference
    # loss: 3.9960 -
    # eval_out_loss: 0.4049 - top_moves_out_loss: 3.1861 -
    # val_loss: 4.8330 -
    # val_eval_out_loss: 0.6742 - val_top_moves_out_loss: 3.4846 - lr: 5.0000e-06
    # train(dt, calc_batch_size(256), 0.9, None, 20, 2, True, l2(1e-8), True,
    #       "categorical_crossentropy", "categorical_crossentropy", 2.0, 1.0, 5e-6, 80, l2(1e-8))

    # ~75% perfect evals, even with a loss of 0.70! best moves frequently(? %) found
    # removed ability to predict win/loss/draw. We already detect that heuristically.
    # This means, remove the last move of every game. There was no top move for these
    # positions, so it'll help train that scenario, and it simplifies the number of classifications
    # for evaluation to 3, where the most common are winning and losing.
    # I may also want to tighten "even" to be +/- 0.25, as I'd prefer most predictions were
    # winning or losing.
    # loss: 3.9752 -
    # eval_out_loss: 0.3999 - top_moves_out_loss: 3.1753 -
    # val_loss: 4.9194 -
    # val_eval_out_loss: 0.7051 - val_top_moves_out_loss: 3.5091 - lr: 5.0000e-06
    # train(dt, calc_batch_size(256), 0.9, None, 20, 2, True, l2(1e-8), True,
    #       "categorical_crossentropy", "categorical_crossentropy", 2.0, 1.0, 5e-6, 80, l2(1e-8))

    # file size 1024, more data, 30/2 layers, and 2:1 weights
    # loss: 3.9821 -
    # eval_out_loss: 0.3235 - top_moves_out_loss: 3.3350 -
    # val_loss: 5.0989 -
    # val_eval_out_loss: 0.7694 - val_top_moves_out_loss: 3.5600 - lr: 5.0000e-06
    # train(dt, calc_batch_size(256), 0.9, None, 30, 2, True, l2(1e-8), True,
    #       "categorical_crossentropy", "categorical_crossentropy", 2.0, 1.0, 5e-6, 80, l2(1e-8))

    # ReduceLROnPlateau(monitor='loss', patience=10, min_lr=1e-9, min_delta=1e-3, factor=0.1)
    # changed when eval is 1.0ds and 0.0
    # loss: 3.7202 -
    # eval_out_loss: 0.5450 - top_moves_out_loss: 3.1750 -
    # val_loss: 4.0204 -
    # val_eval_out_loss: 0.6852 - val_top_moves_out_loss: 3.3350 - lr: 5.0000e-06
    # train(dt, calc_batch_size(256), 0.9, None, 20, 2, True, l2(1e-8), True,
    #       "categorical_crossentropy", "categorical_crossentropy", 1.0, 1.0, 5e-6, 80, l2(1e-8))

    # switched to categorized eval
    # train(dt, calc_batch_size(256), 0.9, None, 20, 2, True, l2(1e-8), True,
    #       "categorical_crossentropy", "categorical_crossentropy", 1.0, 1.0, 5e-6, 80, l2(1e-8))

    # change to stepped (0, 0.25, 0.5, 0.75, and 1.0) evaluation with the goal of having fewer
    # errors around 0.5. The steps are lost, losing, undecided, winning, and won.
    # may need to switch to categorical classification on this.
    # ReduceLROnPlateau(monitor='val_eval_out_loss', patience=10, min_lr=1e-9, min_delta=1e-3, factor=0.1)
    # changed last eval dense layer from 256 to 16... this might be good or bad.
    # train(dt, calc_batch_size(256), 0.9, None, 20, 2, True, l2(1e-8), True,
    #       "mean_squared_error", "categorical_crossentropy", 50.0, 1.0, 5e-6, 80, l2(1e-8))

    # ReduceLROnPlateau(monitor='val_eval_out_loss', patience=4, min_lr=1e-9, min_delta=1e-3, factor=0.1)
    # loss: 9.5120 - eval_out_loss: 8.5109e-04 - top_moves_out_loss: 5.2563 - val_loss: 16.5540 -
    # val_eval_out_loss: 0.0022 - val_top_moves_out_loss: 5.3402 - lr: 5.0000e-08
    # train(dt, calc_batch_size(256), 0.9, None, 30, 3, True, l2(1e-8), True,
    #       "mean_squared_error", "categorical_crossentropy", 5000.0, 1.0, 5e-6, 80, l2(1e-8))

    # changed from loss: ReduceLROnPlateau(monitor='val_eval_out_loss', patience=4, min_lr=1e-3, factor=0.1)
    # reduced l2s and increased layers
    # was osculating between 24 and 25 eval out loss, stopped at:
    # loss: 4.3829 - eval_out_loss: 0.0015 - top_moves_out_loss: 3.6104 - val_loss: 4.9566 -
    # val_eval_out_loss: 0.0024 - val_top_moves_out_loss: 3.7787 - lr: 5.0000e-06
    # train(dt, calc_batch_size(256), 0.9, None, 30, 2, True, l2(1e-7), True,
    #       "mean_squared_error", "categorical_crossentropy", 500.0, 1.0, 5e-6, 80, l2(1e-7))

    # try bias again, but add l2 regularizer
    # canceled, on target for:
    # loss: 4.2187 - eval_out_loss: 0.0012 - top_moves_out_loss: 3.4948 - val_loss: 5.0391
    # val_eval_out_loss: 0.0025 - val_top_moves_out_loss: 3.6701 - lr: 5.0000e-06
    # train(dt, calc_batch_size(256), 0.9, None, 20, 2, True, l2(1e-5), True,
    #       "mean_squared_error", "categorical_crossentropy", 500.0, 1.0, 5e-6, 80, l2(1e-5))

    # deeper with less l2, data file size down to 2048, more data
    # canceled, but on track for:
    # loss: 4.7309 - eval_out_loss: 0.0017 - top_moves_out_loss: 3.8211 - val_loss: 5.5477
    # val_eval_out_loss: 0.0031 - val_top_moves_out_loss: 3.9810 - lr: 5.0000e-06
    # train(dt, calc_batch_size(256), 0.9, None, 30, 3, False, l2(1e-5), True,
    #       "mean_squared_error", "categorical_crossentropy", 500.0, 1.0, 5e-6, 80, None)

    # loss: 4.0147 - eval_out_loss: 0.0010 - top_moves_out_loss: 3.5026 - val_loss: 5.0585
    # val_eval_out_loss: 0.0027 - val_top_moves_out_loss: 3.6974 - lr: 5.0000e-06
    # train(dt, calc_batch_size(256), 0.9, None, 30, 2, False, l2(1e-7), True,
    #       "mean_squared_error", "categorical_crossentropy", 500.0, 1.0, 5e-6, 80, None)

    # changed MAX_DATASET_FILE_SIZE from 60,000 to 4096.
    # we will have fewer missed training records, and it's a power of 2
    # also validation set size will more accurately line up with goals.
    # gave 400:1 weight to eval, changed batch size to 256, set layers to 24/3, learning rate to 5e-6, added more data.
    # Trained fast and hit reasonably low losses, but over-fit and didn't plateau.
    # train(dt, calc_batch_size(256), 0.9, None, 24, 3, False, l2(1e-6), True, "mean_squared_error",
    #       "categorical_crossentropy", 400.0, 1.0, 5e-6, 1000, None)

    # models/leg_model_2022_07_02_11_43_59.h5
    # took 159 epochs at around 21 minutes each... basically forever and the end results were on par
    # with learn rate 1e-6 and batches of 200.
    # small batch size 100 (500 parts) and slower learn rate (1e-7)
    # loss: 4.1665 - eval_out_loss: 0.0097 - top_moves_out_loss: 3.3222
    # val_loss: 4.6996 - val_eval_out_loss: 0.0044 - val_top_moves_out_loss: 3.7530 - lr: 1.0000e-07
    # train(dt, calc_batch_size(100), 0.9, None, 20, 2, False, l2(1e-6), True, "mean_squared_error", "categorical_crossentropy", 1.0, 1.25,
    #       1e-7, 1000, None)

    # early stopping increased from 4 -> 6, plus flipped black evals properly?
    # batch size to 600 (200 parts), learning rate to 1e-3 for speed
    # Wild results after first epoch, like other times I've used a high learn rate
    # train(dt, calc_batch_size(600), 0.9, None, 20, 2, False, l2(1e-6), True, "mean_squared_error", "categorical_crossentropy", 1.0, 1.25,
    #       1e-3, 1000, None)

    # Good results! Not perfect, but very good: models/leg_model_2022_07_01_05_57_04.h5
    # loss: 3.8705 - eval_out_loss: 0.0097 - top_moves_out_loss: 3.0852
    # val_loss: 4.3691 - val_eval_out_loss: 0.0135 - val_top_moves_out_loss: 3.4810 - lr: 1.0000e-06
    # even with a learn rate of 1e-6, gets a decent eval/top move on 1st epoch. Good 2nd epoch so far.
    # train(dt, calc_batch_size(200), 0.9, None, 20, 2, False, l2(1e-6), True, "mean_squared_error", "categorical_crossentropy", 1.0, 1.25,
    #       1e-6, 1000, None)

    # High learn rate means a decent 1st epoch and a rough 2nd epoch. patience won't give enough time to stabilize
    # train(dt, calc_batch_size(200), 0.9, None, 20, 2, True, l2(1e-6), True, "mean_squared_error", "categorical_crossentropy", 1.0, 1.25,
    #       1e-3, 1000, None)
