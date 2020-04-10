'''VGGFace models for Keras.

# Reference:
- [Deep Face Recognition](http://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf)
- [VGGFace2: A dataset for recognising faces across pose and age](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/vggface2.pdf)

'''
from __future__ import print_function
from forkcnn.vgg import VGG16
from forkcnn.resnet import RESNET50
from forkcnn.senet import SENET50


# from keras_vggface.models import VGG16_two_stream


def get_model(include_top=False, model='vgg16', weights=None, stream=1,
              input_1_tensor=None, input_2_tensor=None, input_shape=None, pooling=None, classes=None, merge_point=None,
              merge_style=None):
    """Instantiates the VGGFace architectures.
    Optionally loads weights pre-trained
    on VGGFace datasets. Note that when using TensorFlow,
    for best performance you should set
    `image_data_format="channels_last"` in your Keras config
    at ~/.keras/keras.json.
    The model and the weights are compatible with both
    TensorFlow and Theano. The data format
    convention used by the model is the one
    specified in your Keras config file.
    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization)
            or "vggface" (pre-training on VGGFACE datasets).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        model: selects the one of the available architectures
            vgg16, resnet50 or senet50 default is vgg16.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 244)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 48.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """

    if weights not in {'vggface', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `vggface`'
                         '(pre-training on VGGFace Datasets).')

    if model == 'vgg16':

        if classes is None:
            classes = 29

        if weights == 'vggface' and include_top and classes != 2622:
            raise ValueError(
                'If using `weights` as vggface original with `include_top`'
                ' as true, `classes` should be 2622')

        return VGG16(include_top=include_top, input_1_tensor=input_1_tensor, input_2_tensor=input_2_tensor,
                     stream=stream, merge_style=merge_style, merge_point=merge_point, input_shape=input_shape,
                     pooling=pooling, weights=weights, classes=classes)

    # if model == 'vgg16_two_stream':
    #     if classes in None:
    #         classes = 2622  # TODO
    #
    #     if merge is None:
    #         raise ValueError(
    #             'If using `VGG16_two_stream`, please specify the method to merge'
    #             'with `concatenate` or `addition`'
    #         )
    #
    #
    #     if weights == 'vggface' and include_top and classes != 2622:
    #         raise ValueError(
    #             'If using `weights` as vggface original with `include_top`'
    #             ' as true, `classes` should be 2622')
    #
    #     return VGG16_two_stream(include_top=include_top, input_tensor=input_tensor,
    #                             input_shape=input_shape, pooling=pooling,
    #                             weights=weights,
    #                             classes=classes, merge=merge)

    if model == 'resnet50':

        if classes is None:
            classes = 29

        if weights == 'vggface' and include_top and classes != 8631:
            raise ValueError(
                'If using `weights` as vggface original with `include_top`'
                ' as true, `classes` should be 8631')

        return RESNET50(include_top=include_top, input_1_tensor=input_1_tensor, input_2_tensor=input_2_tensor,
                        input_shape=input_shape, pooling=pooling, stream=stream, merge_style=merge_style,
                        merge_point=merge_point, weights=weights, classes=classes)

    if model == 'senet50':

        if classes is None:
            classes = 29

        if weights == 'vggface' and include_top and classes != 8631:
            raise ValueError(
                'If using `weights` as vggface original with `include_top`'
                ' as true, `classes` should be 8631')

        return SENET50(include_top=include_top, input_1_tensor=input_1_tensor, input_2_tensor=input_2_tensor,
                        input_shape=input_shape, pooling=pooling, stream=stream, merge_style=merge_style,
                        merge_point=merge_point, weights=weights, classes=classes)
