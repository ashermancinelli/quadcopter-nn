
def use(model_file, images):
    """
    Takes the filename of a .h5 file with a fully compiled
    keras model, and an array of file names to be predicted
    """
    from keras.models import load_model
    from keras.preprocessing import image
    import numpy as np
    model = load_model(model_file)

    cleaned_images = []
    for im in images:
        img = image.load_img(im, target_size=(150, 150))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        cleaned_images.append(x)

    results = []
    for im in cleaned_images:
        results.append(model.predict(im)[0])

    return results 



