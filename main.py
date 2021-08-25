import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import vgg19
import time
from PIL import Image
import glob

        
Menu=['Home','De-Blurify','Neural Filters']

choice=st.sidebar.selectbox('Get Started Now!',Menu)

if choice=='Home':

    st.image('/Users/vananhle/streamlit_demo/CHECK OUT SOME examples! (4).png')
    st.image('/Users/vananhle/streamlit_demo/Brown Black Cutout Collage Personal Facebook Cover.png')
    st.image('/Users/vananhle/streamlit_demo/CHECK OUT SOME examples! (5).png')
    st.image('/Users/vananhle/streamlit_demo/Copy of CHECK OUT SOME examples!.png')
    st.image('/Users/vananhle/streamlit_demo/examples.png')  
    st.image('/Users/vananhle/streamlit_demo/examples (1).png') 
    st.image('/Users/vananhle/streamlit_demo/Copy of examples.png')

elif choice=='De-Blurify':
    st.image('/Users/vananhle/Documents/web_app/streamlit_demo/texts/Copy of CHECK OUT SOME examples! (4).png')
    mode=st.selectbox("De-Blurify Mode", ['Light','Strong'])
    st.header('Upload a photo to get started!')
    photo=st.file_uploader('Upload photo',['png','jpg','jpeg','bmp'],key='1')
    if mode=='Light':
            if photo!=None:
                image_np=np.asarray(bytearray(photo.read()),dtype=np.uint8)
                img=cv2.imdecode(image_np,1)
                cv2.imwrite('/Users/vananhle/Documents/web_app/streamlit_demo/media/original.png',img)
                image_path='/Users/vananhle/Documents/web_app/streamlit_demo/media/original.png'

                
                model = keras.models.load_model('/Users/vananhle/Documents/web_app/streamlit_demo/superresolution3.h5')
                target = cv2.imread('/Users/vananhle/Documents/web_app/streamlit_demo/media/original.png', cv2.IMREAD_COLOR)
                target = cv2.cvtColor(target, cv2.COLOR_BGR2YCrCb)
                shape = target.shape

                Y_img = cv2.resize(target[:, :, 0], (int(shape[1] / 1), int(shape[0] / 1)), cv2.INTER_CUBIC)

                # Resize up to orignal image
                Y_img = cv2.resize(Y_img, (shape[1], shape[0]), cv2.INTER_CUBIC)
                target[:, :, 0] = Y_img
                target = cv2.cvtColor(target, cv2.COLOR_YCrCb2BGR)

                Y = np.zeros((1, target.shape[0], target.shape[1], 1), dtype=np.float32)

                # Normalize
                Y[0, :, :, 0] = Y_img.astype(np.float32) / 255.

                # Predict
                pre = model.predict(Y, batch_size=1) * 255.

                # Post process output
                pre[pre[:] > 255] = 255
                pre[pre[:] < 0] = 0
                pre = pre.astype(np.uint8)

                # Copy y channel back to image and convert to BGR
                output = cv2.cvtColor(target, cv2.COLOR_BGR2YCrCb)
                output[6: -6, 6: -6, 0] = pre[0, :, :, 0]
                output = cv2.cvtColor(output, cv2.COLOR_YCrCb2BGR)

                # Save image
                cv2.imwrite('/Users/vananhle/Documents/web_app/streamlit_demo/media/enhanced.png', output)

                st.header('Original Image')
                st.image('/Users/vananhle/Documents/web_app/streamlit_demo/media/original.png')
                st.header('Enhanced Image')
                st.image('/Users/vananhle/Documents/web_app/streamlit_demo/media/enhanced.png')
    if mode=='Strong':
            if photo!=None:
                image_np=np.asarray(bytearray(photo.read()),dtype=np.uint8)
                img=cv2.imdecode(image_np,1)
                cv2.imwrite('/Users/vananhle/Documents/web_app/streamlit_demo/media/original.png',img)
                image_path='/Users/vananhle/Documents/web_app/streamlit_demo/media/original.png'

                
                model = keras.models.load_model('/Users/vananhle/Documents/web_app/streamlit_demo/superresolutionxx42.h5')
                target = cv2.imread('/Users/vananhle/Documents/web_app/streamlit_demo/media/original.png', cv2.IMREAD_COLOR)
                target = cv2.cvtColor(target, cv2.COLOR_BGR2YCrCb)
                shape = target.shape

                Y_img = cv2.resize(target[:, :, 0], (int(shape[1] / 1), int(shape[0] / 1)), cv2.INTER_CUBIC)

                # Resize up to orignal image
                Y_img = cv2.resize(Y_img, (shape[1], shape[0]), cv2.INTER_CUBIC)
                target[:, :, 0] = Y_img
                target = cv2.cvtColor(target, cv2.COLOR_YCrCb2BGR)

                Y = np.zeros((1, target.shape[0], target.shape[1], 1), dtype=np.float32)

                # Normalize
                Y[0, :, :, 0] = Y_img.astype(np.float32) / 255.

                # Predict
                pre = model.predict(Y, batch_size=1) * 255.

                # Post process output
                pre[pre[:] > 255] = 255
                pre[pre[:] < 0] = 0
                pre = pre.astype(np.uint8)

                # Copy y channel back to image and convert to BGR
                output = cv2.cvtColor(target, cv2.COLOR_BGR2YCrCb)
                output[6: -6, 6: -6, 0] = pre[0, :, :, 0]
                output = cv2.cvtColor(output, cv2.COLOR_YCrCb2BGR)

                # Save image
                cv2.imwrite('/Users/vananhle/Documents/web_app/streamlit_demo/media/enhanced.png', output)

                st.header('Original Image')
                st.image('/Users/vananhle/Documents/web_app/streamlit_demo/media/original.png')
                st.header('Enhanced Image')
                st.image('/Users/vananhle/Documents/web_app/streamlit_demo/media/enhanced.png')
          
    
elif choice=='Neural Filters':
    st.image('/Users/vananhle/streamlit_demo/Copy of CHECK OUT SOME examples! (2).png')
    st.image('/Users/vananhle/streamlit_demo/Copy of Copy of Copy of examples.png')
    st.image('/Users/vananhle/streamlit_demo/Copy of Copy of Copy of examples (2).png')

    st.header('Choose a Style!')
    style=st.selectbox("Pick a recommended style or upload your own", ['Candy', 'Mosaic', 'Wave','The Scream','Starry Night','Painted Wood','Upload Your Own Style'])
    if style=='Upload Your Own Style':
        photo=st.file_uploader('Upload style',['png','jpg','jpeg','bmp'],key='2')
        if photo!=None:
            image_np=np.asarray(bytearray(photo.read()),dtype=np.uint8)
            img=cv2.imdecode(image_np,1)
            cv2.imwrite('/Users/vananhle/Documents/web_app/streamlit_demo/media/styleimage.png',img)
            style_reference_image_path='/Users/vananhle/Documents/web_app/streamlit_demo/media/styleimage.png'
    elif style=='Candy':
        style_reference_image_path='/Users/vananhle/Documents/web_app/streamlit_demo/neural_style/candy.jpeg'
    
    elif style=='Mosaic':
        style_reference_image_path='/Users/vananhle/Documents/web_app/streamlit_demo/neural_style/mosaic.jpeg'

    elif style=='Wave':
        style_reference_image_path='/Users/vananhle/Documents/web_app/streamlit_demo/neural_style/wave.jpeg'

    elif style=='The Scream':
        style_reference_image_path='/Users/vananhle/Documents/web_app/streamlit_demo/neural_style/the_scream.jpeg'

    elif style=='Starry Night':
        style_reference_image_path='/Users/vananhle/Documents/web_app/streamlit_demo/neural_style/18-15360351387451840638669.jpeg'

    elif style=='Painted Wood':
        style_reference_image_path='/Users/vananhle/Documents/web_app/streamlit_demo/neural_style/Screen Shot 2021-08-10 at 10.21.27 PM.png'

    st.header('Upload a photo to get started!')
    photo=st.file_uploader('Upload photo',['png','jpg','jpeg','bmp'])
    if photo!=None:
            image_np=np.asarray(bytearray(photo.read()),dtype=np.uint8)
            img=cv2.imdecode(image_np,1)
            cv2.imwrite('/Users/vananhle/Documents/web_app/streamlit_demo/media/base.png',img)
            base_image_path='/Users/vananhle/Documents/web_app/streamlit_demo/media/base.png'
            result_prefix = "generated_image"

            # Weights of the different loss components
            total_variation_weight = 1e-6
            style_weight = 1e-6
            content_weight = 2.5e-8

            # Dimensions of the generated picture.
            width, height = keras.preprocessing.image.load_img(base_image_path).size
            img_nrows = 400
            img_ncols = int(width * img_nrows / height)

            def preprocess_image(image_path):
                # Util function to open, resize and format pictures into appropriate tensors
                img = keras.preprocessing.image.load_img(
                    image_path, target_size=(img_nrows, img_ncols)
                )
                img = keras.preprocessing.image.img_to_array(img)
                img = np.expand_dims(img, axis=0)
                img = vgg19.preprocess_input(img)
                return tf.convert_to_tensor(img)


            def deprocess_image(x):
                # Util function to convert a tensor into a valid image
                x = x.reshape((img_nrows, img_ncols, 3))
                # Remove zero-center by mean pixel
                x[:, :, 0] += 103.939
                x[:, :, 1] += 116.779
                x[:, :, 2] += 123.68
                # 'BGR'->'RGB'
                x = x[:, :, ::-1]
                x = np.clip(x, 0, 255).astype("uint8")
                return x

            def gram_matrix(x):
                x = tf.transpose(x, (2, 0, 1))
                features = tf.reshape(x, (tf.shape(x)[0], -1))
                gram = tf.matmul(features, tf.transpose(features))
                return gram


# The "style loss" is designed to maintain
# the style of the reference image in the generated image.
# It is based on the gram matrices (which capture style) of
# feature maps from the style reference image
# and from the generated image


            def style_loss(style, combination):
                S = gram_matrix(style)
                C = gram_matrix(combination)
                channels = 3
                size = img_nrows * img_ncols
                return tf.reduce_sum(tf.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))


# An auxiliary loss function
# designed to maintain the "content" of the
# base image in the generated image


            def content_loss(base, combination):
                return tf.reduce_sum(tf.square(combination - base))


            # The 3rd loss function, total variation loss,
            # designed to keep the generated image locally coherent


            def total_variation_loss(x):
                a = tf.square(
                    x[:, : img_nrows - 1, : img_ncols - 1, :] - x[:, 1:, : img_ncols - 1, :]
                )
                b = tf.square(
                    x[:, : img_nrows - 1, : img_ncols - 1, :] - x[:, : img_nrows - 1, 1:, :]
                )
                return tf.reduce_sum(tf.pow(a + b, 1.25))

                        # Build a VGG19 model loaded with pre-trained ImageNet weights
            model = vgg19.VGG19(weights="imagenet", include_top=False)

            # Get the symbolic outputs of each "key" layer (we gave them unique names).
            outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

            # Set up a model that returns the activation values for every layer in
            # VGG19 (as a dict).
            feature_extractor = keras.Model(inputs=model.inputs, outputs=outputs_dict)

                        # List of layers to use for the style loss.
            style_layer_names = [
                "block1_conv1",
                "block2_conv1",
                "block3_conv1",
                "block4_conv1",
                "block5_conv1",
            ]
            # The layer to use for the content loss.
            content_layer_name = "block5_conv2"


            def compute_loss(combination_image, base_image, style_reference_image):
                input_tensor = tf.concat(
                    [base_image, style_reference_image, combination_image], axis=0
                )
                features = feature_extractor(input_tensor)

                # Initialize the loss
                loss = tf.zeros(shape=())

                # Add content loss
                layer_features = features[content_layer_name]
                base_image_features = layer_features[0, :, :, :]
                combination_features = layer_features[2, :, :, :]
                loss = loss + content_weight * content_loss(
                    base_image_features, combination_features
                )
                # Add style loss
                for layer_name in style_layer_names:
                    layer_features = features[layer_name]
                    style_reference_features = layer_features[1, :, :, :]
                    combination_features = layer_features[2, :, :, :]
                    sl = style_loss(style_reference_features, combination_features)
                    loss += (style_weight / len(style_layer_names)) * sl

                # Add total variation loss
                loss += total_variation_weight * total_variation_loss(combination_image)
                return loss

            @tf.function
            def compute_loss_and_grads(combination_image, base_image, style_reference_image):
                with tf.GradientTape() as tape:
                    loss = compute_loss(combination_image, base_image, style_reference_image)
                grads = tape.gradient(loss, combination_image)
                return loss, grads

            optimizer = keras.optimizers.SGD(
                keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=100.0, decay_steps=100, decay_rate=0.96
                )
            )

            base_image = preprocess_image(base_image_path)
            style_reference_image = preprocess_image(style_reference_image_path)
            combination_image = tf.Variable(preprocess_image(base_image_path))
            
            iterations = 20
            bar = st.progress(0)
            for i in range(1, iterations + 1):  
                bar.progress(5*i)   
                loss, grads = compute_loss_and_grads(
                    combination_image, base_image, style_reference_image
                )
                optimizer.apply_gradients([(grads, combination_image)])

                if i % 20 == 0:
                    img = deprocess_image(combination_image.numpy())
                    fname = '/Users/vananhle/Documents/web_app/streamlit_demo/media/generated.png'
                    keras.preprocessing.image.save_img(fname, img)

            st.text("Done generating image!")
            st.image(fname)
