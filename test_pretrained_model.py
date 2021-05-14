import keras_ocr
import string
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn.model_selection
from ro_text_generator import RoTextGenerator

#use models for inference



diacritics = "şţăâî"
alphabet_ro = string.digits + string.ascii_letters + diacritics + '!?. '
recognizer_alphabet = ''.join(sorted(set(alphabet_ro.lower())))


checkpoint_path = 'model3_saved/recognizer_2021-05-14T11_23_54.434784.h5'


#incercat cu full recognizer model saved!

# checkpoint_path = 'data/recognizer_2021-05-14T08:22:08.807105.h5'
# checkpoint_path = 'data/recognizer_2021-05-14T11:23:54.434784.h5'

# recog_loaded  = tf.keras.models.load_model('model3_saved/my_recognizer3.h5', compile='False')
# trebuie sau nu compile???
# recog_loaded.compile()


recognizer3 = keras_ocr.recognition.Recognizer(
    alphabet=recognizer_alphabet,
    # alphabet=alphabet_ro,
    weights='kurapan',
    # build_params = 
)


# recognizer3.backbone,recognizer3.model, recognizer3.training_model, recognizer3.prediction_model = keras_ocr.recognition.build_model(
#             # alphabet=recognizer_alphabet, **build_params)
#             alphabet=recognizer_alphabet, **DEFAULT_BUILD_PARAMS)


# recognizer3.model = recog_loaded
# recognizer3.model.load_weights(checkpoint_path)

# recognizer3.training_model = recog_loaded
# recognizer3.training_model.load_weights(checkpoint_path)

# recognizer3.prediction_model = recog_loaded
# recognizer3.prediction_model.load_weights(checkpoint_path)

# recognizer3.compile()


# recognizer3.model.load_weights('data/recognizer_2021-05-14T11:23:54.434784.h5')
recognizer3.model.load_weights(checkpoint_path)

detector = keras_ocr.detection.Detector(weights='clovaai_general')


# pipeline = keras_ocr.pipeline.Pipeline(detector=detector, recognizer=recog_loaded)
pipeline = keras_ocr.pipeline.Pipeline(detector=detector, recognizer=recognizer3)


################ test on sample image

# Get a set of three example images
images = [
    keras_ocr.tools.read(url) for url in [
        # 'https://upload.wikimedia.org/wikipedia/commons/b/bd/Army_Reserves_Recruitment_Banner_MOD_45156284.jpg',
        # 'https://upload.wikimedia.org/wikipedia/commons/e/e8/FseeG2QeLXo.jpg',
        # 'https://upload.wikimedia.org/wikipedia/commons/b/b4/EUBanana-500x112.jpg'
        'page0_small.jpg'
    ]
]

# Each list of predictions in prediction_groups is a list of
# (word, box) tuples.
prediction_groups, pred_chars = pipeline.recognize(images)

print("pred groups:", prediction_groups)
# print("pred chars:", pred_chars)



# Plot the predictions
# fig, axs = plt.subplots(nrows=len(images), figsize=(20, 20))
# for ax, image, predictions in zip(axs, images, prediction_groups):
    # keras_ocr.tools.drawAnnotations(image=image, predictions=predictions, ax=ax)

fig, axs = plt.subplots(nrows=len(images), figsize=(10, 10))
if(len(images) == 1):
  for image, prediction in zip(images, prediction_groups):
    keras_ocr.tools.drawAnnotations(image=image, predictions=prediction, ax=axs)
else:
  for ax, image, prediction in zip(axs, images, prediction_groups):
    keras_ocr.tools.drawAnnotations(image=image, predictions=prediction, ax=ax)
plt.show()
# fig.show()
# # Test on Generated images:
# #######################3

# ro_txt_generator = RoTextGenerator(alphabet=alphabet_ro)
# text_generator = ro_txt_generator.generate()



# def get_train_val_test_split(arr):
#     train, valtest = sklearn.model_selection.train_test_split(arr, train_size=0.8, random_state=42)
#     val, test = sklearn.model_selection.train_test_split(valtest, train_size=0.5, random_state=42)
#     return train, val, test

# data_dir = 'data'

# fonts = keras_ocr.data_generation.get_fonts(
#     alphabet=alphabet_ro,
#     cache_dir=data_dir
# )
# backgrounds = keras_ocr.data_generation.get_backgrounds(cache_dir=data_dir)


# background_splits = get_train_val_test_split(backgrounds)
# font_splits = get_train_val_test_split(fonts)

# image_generators = [
#     keras_ocr.data_generation.get_image_generator(
#         height=640,
#         width=640,
#         text_generator=text_generator,
#         font_groups={
#             alphabet_ro: current_fonts
#         },
#         backgrounds=current_backgrounds,
#         font_size=(60, 120),
#         margin=50,
#         rotationX=(-0.05, 0.05),
#         rotationY=(-0.05, 0.05),
#         rotationZ=(-15, 15)
#     )  for current_fonts, current_backgrounds in zip(
#         font_splits,
#         background_splits
#     )
# ]





# for i in range(0,5):
#   image, lines = next(image_generators[0])
#   # print("image:", image)
#   # print("lines:", lines)
#   # predictions = pipeline.recognize(images=[image])[0]
#   predictions = pipeline.recognize(images=[image])[0]
#   drawn = keras_ocr.tools.drawBoxes(
#       image=image, boxes=predictions, boxes_format='predictions'
#   )
#   print(
#       'Actual:', '\n'.join([' '.join([character for _, character in line]) for line in lines]),
#       'Predicted:', [text for text, box in predictions]
#   )
#   plt.imshow(drawn)
#   plt.show()


# # detector = keras_ocr.detection.Detector(weights='clovaai_general')



# # pipeline = keras_ocr.pipeline.Pipeline(detector=detector, recognizer=recognizer3)



# # ro_txt_generator = RoTextGenerator(alphabet=alphabet_ro)
# # text_generator = ro_txt_generator.generate()


# # def get_train_val_test_split(arr):
# #     train, valtest = sklearn.model_selection.train_test_split(arr, train_size=0.8, random_state=42)
# #     val, test = sklearn.model_selection.train_test_split(valtest, train_size=0.5, random_state=42)
# #     return train, val, test

# # data_dir = 'data'

# # fonts = keras_ocr.data_generation.get_fonts(
# #     # alphabet=alphabet_ro,
# #     alphabet=alphabet_en,
# #     cache_dir=data_dir
# # )
# # backgrounds = keras_ocr.data_generation.get_backgrounds(cache_dir=data_dir)




# # background_splits = get_train_val_test_split(backgrounds)
# # font_splits = get_train_val_test_split(fonts)

# # image_generators = [
# #     keras_ocr.data_generation.get_image_generator(
# #         height=640,
# #         width=640,
# #         text_generator=text_generator,
# #         font_groups={
# #             # alphabet_ro: current_fonts
# #             alphabet_en: current_fonts
# #         },
# #         backgrounds=current_backgrounds,
# #         font_size=(60, 120),
# #         margin=50,
# #         rotationX=(-0.05, 0.05),
# #         rotationY=(-0.05, 0.05),
# #         rotationZ=(-15, 15)
# #     )  for current_fonts, current_backgrounds in zip(
# #         font_splits,
# #         background_splits
# #     )
# # ]




# # for i in range(0,20):
# #   image, lines = next(image_generators[0])
# #   predictions = pipeline.recognize(images=[image])[0]
# #   drawn = keras_ocr.tools.drawBoxes(
# #       image=image, boxes=predictions, boxes_format='predictions'
# #   )
# #   print(
# #       'Actual:', '\n'.join([' '.join([character for _, character in line]) for line in lines]),
# #       'Predicted:', [text for text, box in predictions]
# #   )
# #   plt.imshow(drawn)
# #   plt.show()