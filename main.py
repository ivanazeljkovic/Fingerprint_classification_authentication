import predict
import predict_alexNet
import dataset_preprocessing
import matching

if __name__ == "__main__":
    # Preprocessing images from dataset
    # classes = ['arch', 'left_loop', 'right_loop', 'tented_arch', 'whorl']
    # dataset_preprocessing.preprocess_images_from_dataset('data/training', classes, 'processed_data/training')
    # dataset_preprocessing.preprocess_images_from_dataset('data/testing', classes, 'processed_data/testing')

    # Prediction with VGG-19 model
    # predict.predict()

    # Prediction with AlexNet or ZFNet model
    # predict_alexNet.predict()

    image_path = 'data/testing/whorl/f1996_04.png'
    image_path2 = 'data/testing/whorl/s1996_04.png'
    processed_image_path = 'processed_data/testing/whorl/f1996_04.png'
    processed_image_path2 = 'processed_data/testing/whorl/s1996_04.png'

    class_1 = predict_alexNet.predict_class_for_image(image_path)
    print('Fingerprint 1 successfully predict: ' + str(class_1))
    class_2 = predict_alexNet.predict_class_for_image(image_path2)
    print('Fingerprint 2 successfully predict: ' + str(class_2))

    if class_1 == class_2:
        matching.check_matching(processed_image_path, processed_image_path2)
