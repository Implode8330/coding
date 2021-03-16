import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
from library.encoder.transmission_characteristics   import Transmission_Channel
from library.data_manipulation.create_data          import data_generator, accuracy_measuring
from library.decoder.destillation                   import Teacher_Student_Training
from library.decoder.BCJR   import BCJR_Decoder, Turbo_Encoder
from library.decoder.NN     import MyModel
import numpy as np
import random as r
import keras


code_gen = data_generator(32,SNRdb = 4, seed = 0)
# code_gen.generate_20_bit()
# code_gen.generate_20_bit()
# code_gen.generate_20_bit()
# corr_y   = code_gen.generate_16_bit()
# corr_y   = code_gen.generate_32_bit()
# corr_y   = code_gen.generate_64_bit()
code_gen.generate_64_bit()
code_gen.generate_64_bit()
code_gen.generate_64_bit()
code_gen.generate_64_bit()
code_gen.generate_64_bit()
code_gen.generate_64_bit()
code_gen.generate_64_bit()
code_gen.generate_64_bit()
code_gen.generate_64_bit()
# exit()
max_datasets = code_gen.next_number("data_x_") - 1
(test_x , test_y , test_p), v_test  = code_gen.mini_load_function(max_datasets)


m = MyModel([test_x.shape[1],1024,512,256,128,test_y.shape[1]])
# m.load()
m.compile()
m.set_training_settings(epochs=10, vali_split=0)
m.eval(test_x, test_y)

distiller = Teacher_Student_Training(student=m)
distiller.compile(
    optimizer=keras.optimizers.Adam(),
    metrics=[keras.metrics.CategoricalAccuracy()],
    # distillation_loss_fn=keras.losses.KLDivergence(),
    distillation_loss_fn=keras.losses.CategoricalCrossentropy(),
    student_loss_fn=keras.losses.CategoricalCrossentropy(),
    alpha = 0.2,
)
accuracy_measuring(test_p, test_y)


for i in range(30):
    data, v_train = code_gen.mini_load_function(i%max_datasets)
    print(f"Andel likadana värden i test: {np.sum(v_test == v_train)/v_test.size}  --- ")
    distiller.fit(data, epochs=10, batch_size=2)
    # distiller.fit(data, epochs=10, batch_size=2)
    m.eval(test_x, test_y)
    m.save("default.h5")
# for i in range(30):
#     data_x, corr_y, pred_y = code_gen.mini_load_function(0)#i%max_datasets)
#     m.train(data_x, pred_y)
#     m.eval(test_x, test_y)
#     m.save("default.h5")
#
