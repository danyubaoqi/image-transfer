import vgg

import tensorflow as tf
import numpy as np

from sys import stderr
import time

from PIL import Image

CONTENT_LAYERS = ('relu4_2', 'relu5_2')
STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')

try:
    reduce
except NameError:
    from functools import reduce


def stylize(network, initial, initial_noiseblend, content, styles, preserve_colors, iterations,
        content_weight, content_weight_blend, style_weight, style_layer_weight_exp, style_blend_weights, tv_weight,
        learning_rate, beta1, beta2, epsilon, pooling,
        print_iterations=None, checkpoint_iterations=None):
    """
    Stylize images.
    This function yields tuples (iteration, image); `iteration` is None
    if this is the final image (the last iteration).  Other tuples are yielded
    every `checkpoint_iterations` iterations.
    :rtype: iterator[tuple[int|None,image]]
    """
    shape = (1,) + content.shape
    style_shapes = [(1,) + style.shape for style in styles]
    content_features = {}
    style_features = [{} for _ in styles]

    vgg_weights, vgg_mean_pixel = vgg.load_net(network)#加载vgg19与训练模型


    layer_weight = 1.0#layer权重默认为1
    style_layers_weights = {}
    for style_layer in STYLE_LAYERS:
        style_layers_weights[style_layer] = layer_weight#style_layer_weight_exp为图像风格的权重默认为1，否则为指数级增长

        layer_weight *= style_layer_weight_exp

    # normalize style layer weights
    layer_weights_sum = 0
    for style_layer in STYLE_LAYERS:
        layer_weights_sum += style_layers_weights[style_layer]
    for style_layer in STYLE_LAYERS:
        style_layers_weights[style_layer] /= layer_weights_sum

    # compute content features in feedforward mode
    g = tf.Graph()

    with g.as_default(), g.device('/gpu:0'), tf.Session() as sess:#使用gpu训练，cpu训练大约2个小时，gpu5分钟

        image = tf.placeholder('float', shape=shape)
        net = vgg.net_preloaded(vgg_weights, image, pooling)
        content_pre = np.array([vgg.preprocess(content, vgg_mean_pixel)])
        for layer in CONTENT_LAYERS:
            content_features[layer] = net[layer].eval(feed_dict={image: content_pre})

    # compute style features in feedforward mode
    for i in range(len(styles)):
        g = tf.Graph()
        with g.as_default(), g.device('/gpu:0'), tf.Session() as sess:
            image = tf.placeholder('float', shape=style_shapes[i])
            net = vgg.net_preloaded(vgg_weights, image, pooling)#池化层默认为max规则
            style_pre = np.array([vgg.preprocess(styles[i], vgg_mean_pixel)])
            for layer in STYLE_LAYERS:
                features = net[layer].eval(feed_dict={image: style_pre})
                features = np.reshape(features, (-1, features.shape[3]))#将一维数组根据图像大小转为三维
                gram = np.matmul(features.T, features) / features.size#计算gram矩阵
                style_features[i][layer] = gram

    initial_content_noise_coeff = 1.0 - initial_noiseblend

    # make stylized image using backpropogation
    with tf.Graph().as_default():
        if initial is None:
            noise = np.random.normal(size=shape, scale=np.std(content) * 0.1)
            initial = tf.random_normal(shape) * 0.256#得到一个随机白噪音
        else:
            initial = np.array([vgg.preprocess(initial, vgg_mean_pixel)])
            initial = initial.astype('float32')
            noise = np.random.normal(size=shape, scale=np.std(content) * 0.1)
            initial = (initial) * initial_content_noise_coeff + (tf.random_normal(shape) * 0.256) * (1.0 - initial_content_noise_coeff)
        image = tf.Variable(initial)#将得到的白噪音转为tensorflow对象
        net = vgg.net_preloaded(vgg_weights, image, pooling)

        # content loss
        content_layers_weights = {}
        #网络的高层特征一般是关于输入图像的物体和布局等信息，低层特征一般表达输入图像的像素信息
        #最终选择conv4_2
        content_layers_weights['relu4_2'] = content_weight_blend
        content_layers_weights['relu5_2'] = 1.0 - content_weight_blend

        content_loss = 0
        content_losses = []
        for content_layer in CONTENT_LAYERS:
            content_losses.append(content_layers_weights[content_layer] * content_weight * (2 * tf.nn.l2_loss(
                    net[content_layer] - content_features[content_layer]) /
                    content_features[content_layer].size))
        content_loss += reduce(tf.add, content_losses)
        #计算content loss

        # style loss
        style_loss = 0
        for i in range(len(styles)):
            style_losses = []
            for style_layer in STYLE_LAYERS:
                layer = net[style_layer]
                _, height, width, number = map(lambda i: i.value, layer.get_shape())
                size = height * width * number
                feats = tf.reshape(layer, (-1, number))
                gram = tf.matmul(tf.transpose(feats), feats) / size
                style_gram = style_features[i][style_layer]
                style_losses.append(style_layers_weights[style_layer] * 2 * tf.nn.l2_loss(gram - style_gram) / style_gram.size)
            style_loss += style_weight * style_blend_weights[i] * reduce(tf.add, style_losses)

        # total variation denoising
        tv_y_size = _tensor_size(image[:,1:,:,:])
        tv_x_size = _tensor_size(image[:,:,1:,:])
        tv_loss = tv_weight * 2 * (
                (tf.nn.l2_loss(image[:,1:,:,:] - image[:,:shape[1]-1,:,:]) /
                    tv_y_size) +
                (tf.nn.l2_loss(image[:,:,1:,:] - image[:,:,:shape[2]-1,:]) /
                    tv_x_size))
        # overall loss
        #总loos为loss相加
        loss = content_loss + style_loss + tv_loss

        # optimizer setup
        train_step = tf.train.AdamOptimizer(learning_rate, beta1, beta2, epsilon).minimize(loss)

        def print_progress():#输出相关信息
            stderr.write('  content loss: %g\n' % content_loss.eval())
            stderr.write('    style loss: %g\n' % style_loss.eval())
            stderr.write('       tv loss: %g\n' % tv_loss.eval())
            stderr.write('    total loss: %g\n' % loss.eval())

        # optimization
        best_loss = float('inf')
        best = None
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            stderr.write('Optimization started...\n')
            if (print_iterations and print_iterations != 0):
                print_progress()
            iteration_times = []
            start = time.time()
            for i in range(iterations):
                iteration_start = time.time()
                if i > 0:
                    elapsed = time.time() - start
                    # take average of last couple steps to get time per iteration
                    remaining = np.mean(iteration_times[-10:]) * (iterations - i)
                    stderr.write('Iteration %4d/%4d (%s elapsed, %s remaining)\n' % (
                        i + 1,
                        iterations,
                        hms(elapsed),
                        hms(remaining)
                    ))
                else:
                    stderr.write('Iteration %4d/%4d\n' % (i + 1, iterations))
                train_step.run()

                last_step = (i == iterations - 1)
                if last_step or (print_iterations and i % print_iterations == 0):
                    print_progress()

                if (checkpoint_iterations and i % checkpoint_iterations == 0) or last_step:
                    this_loss = loss.eval()
                    if this_loss < best_loss:
                        best_loss = this_loss
                        best = image.eval()

                    img_out = vgg.unprocess(best.reshape(shape[1:]), vgg_mean_pixel)

                    if preserve_colors and preserve_colors == True:
                        original_image = np.clip(content, 0, 255)
                        styled_image = np.clip(img_out, 0, 255)

                        # Luminosity transfer steps:
                        # 1. 将风格图像的RGB转为gray
                        # 2. 将风格图像gray转为ycrcb
                        # 3. 将事物图像转为ycrcb
                        # 4. 将图像重组
                        # 5. 最后转为RGB
                        # 1
                        styled_grayscale = rgb2gray(styled_image)
                        styled_grayscale_rgb = gray2rgb(styled_grayscale)

                        # 2
                        styled_grayscale_yuv = np.array(Image.fromarray(styled_grayscale_rgb.astype(np.uint8)).convert('YCbCr'))

                        # 3
                        original_yuv = np.array(Image.fromarray(original_image.astype(np.uint8)).convert('YCbCr'))

                        # 4
                        w, h, _ = original_image.shape
                        combined_yuv = np.empty((w, h, 3), dtype=np.uint8)
                        combined_yuv[..., 0] = styled_grayscale_yuv[..., 0]
                        combined_yuv[..., 1] = original_yuv[..., 1]
                        combined_yuv[..., 2] = original_yuv[..., 2]

                        # 5
                        img_out = np.array(Image.fromarray(combined_yuv, 'YCbCr').convert('RGB'))


                    yield (
                        (None if last_step else i),
                        img_out
                    )

                iteration_end = time.time()
                iteration_times.append(iteration_end - iteration_start)


def _tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d.value for d in tensor.get_shape()), 1)

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def gray2rgb(gray):
    w, h = gray.shape
    rgb = np.empty((w, h, 3), dtype=np.float32)
    rgb[:, :, 2] = rgb[:, :, 1] = rgb[:, :, 0] = gray
    return rgb

def hms(seconds):
    seconds = int(seconds)
    hours = (seconds // (60 * 60))
    minutes = (seconds // 60) % 60
    seconds = seconds % 60
    #判断剩余时间的
    if hours > 0:
        return '%d hr %d min' % (hours, minutes)
    elif minutes > 0:
        return '%d min %d sec' % (minutes, seconds)
    else:
        return '%d sec' % seconds