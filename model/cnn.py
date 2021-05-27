from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Activation, BatchNormalization, Conv3DTranspose, \
    concatenate, Dropout, ELU
from tensorflow.keras.regularizers import l2

def CNN(size=(192, 192, 64), regularizer_scale=1e-5,base_n_filter = 16, output_activation_name='sigmoid', is_training = True,checkpoint_path = None):
    input_size = (size[0], size[1], size[2], 1)
    input = Input(input_size)
    conv1 = getConvolutionlayer(input,base_n_filter,3,regularizer_scale)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    conv2 = getConvolutionlayer(pool1,base_n_filter*2,3,regularizer_scale)
    conv2 = getConvolutionlayer(conv2,base_n_filter*2,3,regularizer_scale)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    conv3 = getConvolutionlayer(pool2,base_n_filter*4,3,regularizer_scale)
    conv3 = getConvolutionlayer(conv3,base_n_filter*4,3,regularizer_scale)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
    conv4 = getConvolutionlayer(pool3,base_n_filter*8,3,regularizer_scale)
    conv4 = getConvolutionlayer(conv4,base_n_filter*8,3,regularizer_scale)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)
    conv5 = getConvolutionlayer(pool4,base_n_filter*16,3,regularizer_scale)
    conv5 = getConvolutionlayer(conv5,base_n_filter*16,3,regularizer_scale)
    
    deconv5 = Conv3DTranspose(filters=base_n_filter*8, kernel_size=(2, 2, 2), strides=(2, 2, 2))(conv5)
    deconv5 = Conv3DTranspose(filters=base_n_filter*4, kernel_size=(2, 2, 2), strides=(2, 2, 2))(deconv5)
    deconv5 = Conv3DTranspose(filters=base_n_filter*2, kernel_size=(2, 2, 2), strides=(2, 2, 2))(deconv5)
    deconv5 = Conv3DTranspose(filters=base_n_filter, kernel_size=(2, 2, 2), strides=(2, 2, 2))(deconv5)
    score_main = Conv3D(1, (1, 1, 1), activation=output_activation_name,name = 'score_main')(deconv5)
    
    #aux_deconvolution layer
    deconv2 = Conv3DTranspose(filters=base_n_filter, kernel_size=(2, 2, 2), strides=(2, 2, 2))(conv2)
    score_aux1 = Conv3D(1, (1, 1, 1), activation=output_activation_name,name = 'score_aux1')(deconv2)

    deconv3 = Conv3DTranspose(filters=base_n_filter*4, kernel_size=(2, 2, 2), strides=(2, 2, 2))(conv3)
    deconv3 = Conv3DTranspose(filters=base_n_filter*2, kernel_size=(2, 2, 2), strides=(2, 2, 2))(deconv3)
    score_aux2 = Conv3D(1, (1, 1, 1), activation=output_activation_name,name = 'score_aux2')(deconv3)
    
    deconv4 = Conv3DTranspose(filters=base_n_filter*4, kernel_size=(2, 2, 2), strides=(2, 2, 2))(conv4)
    deconv4 = Conv3DTranspose(filters=base_n_filter*2, kernel_size=(2, 2, 2), strides=(2, 2, 2))(deconv4)
    deconv4 = Conv3DTranspose(filters=base_n_filter, kernel_size=(2, 2, 2), strides=(2, 2, 2))(deconv4)
    score_aux3 = Conv3D(1, (1, 1, 1), activation=output_activation_name,name = 'score_aux3')(deconv4)
    #output
    model_training = Model(input, outputs=[score_main, score_aux1, score_aux2, score_aux3])
    model_predict = Model(input, outputs=score_main)
    if checkpoint_path!=None:
        model_training.load_weights(checkpoint_path)
    if is_training: 
        return model_training
    else:
        return model_predict

def getConvolutionlayer(input_layer, n_filter , kernel_size, regularizer_scale):
    if regularizer_scale>0.:
        conv = Conv3D(n_filter, (kernel_size, kernel_size, kernel_size), padding='same', kernel_regularizer=l2(regularizer_scale),use_bias = True)(input_layer)
    else:
        conv = Conv3D(n_filter, (kernel_size, kernel_size, kernel_size), padding='same',use_bias = True)(input_layer)
    conv = BatchNormalization(axis=-1)(conv)
    conv = ELU()(conv)
    return conv
def getUpsampleLayer(input_layer, n_filter, regularizer_scale , kernel_size = 2, strides = 2,use_Conv3DTranspose = False):
    if use_Conv3DTranspose:
        upsample = Conv3DTranspose(filters=n_filter, kernel_size=(kernel_size, kernel_size, kernel_size), strides=(strides, strides, strides), kernel_regularizer=l2(regularizer_scale))(input_layer)
    else:
        upsample = UpSampling3D(size=(strides, strides, strides))(input_layer)
        upsample = Conv3D(n_filter, (kernel_size, kernel_size, kernel_size), padding='same', kernel_regularizer=l2(regularizer_scale))(upsample)
    return upsample
    
def Unet3D(size=(192, 192, 64),base_n_filter = 16, output_activation_name='sigmoid', CNN_weights_path = None):
    input_size = (size[0], size[1], size[2], 1)
    input = Input(input_size)
    regularizer_scale = 0.
    conv1 = getConvolutionlayer(input,base_n_filter,3,regularizer_scale)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    conv2 = getConvolutionlayer(pool1,base_n_filter*2,3,regularizer_scale)
    conv2 = getConvolutionlayer(conv2,base_n_filter*2,3,regularizer_scale)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    conv3 = getConvolutionlayer(pool2,base_n_filter*4,3,regularizer_scale)
    conv3 = getConvolutionlayer(conv3,base_n_filter*4,3,regularizer_scale)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
    conv4 = getConvolutionlayer(pool3,base_n_filter*8,3,regularizer_scale)
    conv4 = getConvolutionlayer(conv4,base_n_filter*8,3,regularizer_scale)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)
    conv5 = getConvolutionlayer(pool4,base_n_filter*16,3,regularizer_scale)
    conv5 = getConvolutionlayer(conv5,base_n_filter*16,3,regularizer_scale)
    
    deconv5 = Conv3DTranspose(filters=base_n_filter*8, kernel_size=(2, 2, 2), strides=(2, 2, 2))(conv5)
    deconv5 = Conv3DTranspose(filters=base_n_filter*4, kernel_size=(2, 2, 2), strides=(2, 2, 2))(deconv5)
    deconv5 = Conv3DTranspose(filters=base_n_filter*2, kernel_size=(2, 2, 2), strides=(2, 2, 2))(deconv5)
    deconv5 = Conv3DTranspose(filters=base_n_filter, kernel_size=(2, 2, 2), strides=(2, 2, 2))(deconv5)
    score_main = Conv3D(1, (1, 1, 1), activation=output_activation_name,name = 'score_main')(deconv5)
    
    #aux_deconvolution layer
    deconv2 = Conv3DTranspose(filters=base_n_filter, kernel_size=(2, 2, 2), strides=(2, 2, 2))(conv2)
    score_aux1 = Conv3D(1, (1, 1, 1), activation=output_activation_name,name = 'score_aux1')(deconv2)

    deconv3 = Conv3DTranspose(filters=base_n_filter*4, kernel_size=(2, 2, 2), strides=(2, 2, 2))(conv3)
    deconv3 = Conv3DTranspose(filters=base_n_filter*2, kernel_size=(2, 2, 2), strides=(2, 2, 2))(deconv3)
    score_aux2 = Conv3D(1, (1, 1, 1), activation=output_activation_name,name = 'score_aux2')(deconv3)
    
    deconv4 = Conv3DTranspose(filters=base_n_filter*4, kernel_size=(2, 2, 2), strides=(2, 2, 2))(conv4)
    deconv4 = Conv3DTranspose(filters=base_n_filter*2, kernel_size=(2, 2, 2), strides=(2, 2, 2))(deconv4)
    deconv4 = Conv3DTranspose(filters=base_n_filter, kernel_size=(2, 2, 2), strides=(2, 2, 2))(deconv4)
    score_aux3 = Conv3D(1, (1, 1, 1), activation=output_activation_name,name = 'score_aux3')(deconv4)
    #output
    model_CNN = Model(input, outputs=[score_main, score_aux1, score_aux2, score_aux3])
    model_CNN.load_weights(CNN_weights_path)
    for layer in model_CNN.layers:
        layer.trainable=False
    conv1_c1 = Conv3D(base_n_filter//4, (5, 5, 5), padding='same', activation = 'elu')(conv1)
    conv2_c1 = Conv3D(base_n_filter//2, (5, 5, 5), padding='same', activation = 'elu')(conv2)
    conv3_c1 = Conv3D(base_n_filter, (5, 5, 5), padding='same', activation = 'elu')(conv3)
    conv4_c1 = Conv3D(base_n_filter*2, (5, 5, 5), padding='same', activation = 'elu')(conv4)
    
    up4 = Conv3DTranspose(filters=base_n_filter*6, kernel_size=(2, 2, 2), strides=(2, 2, 2))(conv5)
    conv4_concat = concatenate([conv4_c1, up4], axis=-1)
    conv4_c2 = Conv3D(base_n_filter*8, (3, 3, 3), padding='same', activation = 'elu')(conv4_concat)
    conv4_c2 = Conv3D(base_n_filter*8, (3, 3, 3), padding='same', activation = 'elu')(conv4_c2)
    
    up3 = Conv3DTranspose(filters=base_n_filter*3, kernel_size=(2, 2, 2), strides=(2, 2, 2))(conv4_c2)
    conv3_concat = concatenate([conv3_c1, up3], axis=-1)
    conv3_c2 = Conv3D(base_n_filter*4, (3, 3, 3), padding='same', activation = 'elu')(conv3_concat)
    conv3_c2 = Conv3D(base_n_filter*4, (3, 3, 3), padding='same', activation = 'elu')(conv3_c2)
    
    up2 = Conv3DTranspose(filters=base_n_filter//2*3, kernel_size=(2, 2, 2), strides=(2, 2, 2))(conv3_c2)
    conv2_concat = concatenate([conv2_c1, up2], axis=-1)
    conv2_c2 = Conv3D(base_n_filter*2, (3, 3, 3), padding='same', activation = 'elu')(conv2_concat)
    conv2_c2 = Conv3D(base_n_filter*2, (3, 3, 3), padding='same', activation = 'elu')(conv2_c2)
    
    up1 = Conv3DTranspose(filters=base_n_filter//4*3, kernel_size=(2, 2, 2), strides=(2, 2, 2))(conv2_c2)
    conv1_concat = concatenate([conv1_c1, up1], axis=-1)
    conv1_c2 = Conv3D(base_n_filter, (3, 3, 3), padding='same', activation = 'elu')(conv1_concat)
    conv1_c2 = Conv3D(base_n_filter, (3, 3, 3), padding='same', activation = 'elu')(conv1_c2)
    
    score_Unet = Conv3D(1, (1, 1, 1), activation=output_activation_name,name = 'score_Unet')(conv1_c2)
    
    model_3D_Unet = Model(input, score_Unet)
    return model_3D_Unet
    
def Unet3D_release(size=(192, 192, 64),base_n_filter = 16, regularizer_scale = 1e-5, output_activation_name='sigmoid', Unet3D_weights_path = None):
    input_size = (size[0], size[1], size[2], 1)
    input = Input(input_size)
    conv1 = getConvolutionlayer(input,base_n_filter,3,regularizer_scale)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    conv2 = getConvolutionlayer(pool1,base_n_filter*2,3,regularizer_scale)
    conv2 = getConvolutionlayer(conv2,base_n_filter*2,3,regularizer_scale)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    conv3 = getConvolutionlayer(pool2,base_n_filter*4,3,regularizer_scale)
    conv3 = getConvolutionlayer(conv3,base_n_filter*4,3,regularizer_scale)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
    conv4 = getConvolutionlayer(pool3,base_n_filter*8,3,regularizer_scale)
    conv4 = getConvolutionlayer(conv4,base_n_filter*8,3,regularizer_scale)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)
    conv5 = getConvolutionlayer(pool4,base_n_filter*16,3,regularizer_scale)
    conv5 = getConvolutionlayer(conv5,base_n_filter*16,3,regularizer_scale)
    
    conv1_c1 = Conv3D(base_n_filter//4, (5, 5, 5), padding='same', activation = 'elu')(conv1)
    conv2_c1 = Conv3D(base_n_filter//2, (5, 5, 5), padding='same', activation = 'elu')(conv2)
    conv3_c1 = Conv3D(base_n_filter, (5, 5, 5), padding='same', activation = 'elu')(conv3)
    conv4_c1 = Conv3D(base_n_filter*2, (5, 5, 5), padding='same', activation = 'elu')(conv4)
    
    up4 = Conv3DTranspose(filters=base_n_filter*6, kernel_size=(2, 2, 2), strides=(2, 2, 2))(conv5)
    conv4_concat = concatenate([conv4_c1, up4], axis=-1)
    conv4_c2 = Conv3D(base_n_filter*8, (3, 3, 3), padding='same', activation = 'elu')(conv4_concat)
    conv4_c2 = Conv3D(base_n_filter*8, (3, 3, 3), padding='same', activation = 'elu')(conv4_c2)
    
    up3 = Conv3DTranspose(filters=base_n_filter*3, kernel_size=(2, 2, 2), strides=(2, 2, 2))(conv4_c2)
    conv3_concat = concatenate([conv3_c1, up3], axis=-1)
    conv3_c2 = Conv3D(base_n_filter*4, (3, 3, 3), padding='same', activation = 'elu')(conv3_concat)
    conv3_c2 = Conv3D(base_n_filter*4, (3, 3, 3), padding='same', activation = 'elu')(conv3_c2)
    
    up2 = Conv3DTranspose(filters=base_n_filter//2*3, kernel_size=(2, 2, 2), strides=(2, 2, 2))(conv3_c2)
    conv2_concat = concatenate([conv2_c1, up2], axis=-1)
    conv2_c2 = Conv3D(base_n_filter*2, (3, 3, 3), padding='same', activation = 'elu')(conv2_concat)
    conv2_c2 = Conv3D(base_n_filter*2, (3, 3, 3), padding='same', activation = 'elu')(conv2_c2)
    
    up1 = Conv3DTranspose(filters=base_n_filter//4*3, kernel_size=(2, 2, 2), strides=(2, 2, 2))(conv2_c2)
    conv1_concat = concatenate([conv1_c1, up1], axis=-1)
    conv1_c2 = Conv3D(base_n_filter, (3, 3, 3), padding='same', activation = 'elu')(conv1_concat)
    conv1_c2 = Conv3D(base_n_filter, (3, 3, 3), padding='same', activation = 'elu')(conv1_c2)
    
    score_Unet = Conv3D(1, (1, 1, 1), activation=output_activation_name,name = 'score_Unet')(conv1_c2)
    
    model_3D_Unet = Model(input, score_Unet)
    if Unet3D_weights_path:
        model_3D_Unet.load_weights(Unet3D_weights_path)
    return model_3D_Unet