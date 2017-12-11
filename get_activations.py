
# Keras code for retrieving activations


import keras.backend as K

# Optimised and simplified based on 
# https://stackoverflow.com/questions/41711190/keras-how-to-get-the-output-of-each-layer
def get_activations(model, model_inputs, layer_name=None):
    # layer_name=None -> return all layer outputs
    outputs = [layer.output
               for layer in model.layers
               if layer.name == layer_name or layer_name is None]
     # evaluation function
    functor = K.function([model.input] + [K.learning_phase()], outputs)
    # Learning phase 0.0 = Test mode (no dropout or batch normalization)
    activations = functor([model_inputs, 0.0])[0]
    return activations


## ORIGINAL -> https://github.com/philipperemy/keras-visualize-activations
def get_activations(model, model_inputs, print_shape_only=False, layer_name=None):
    print('----- activations -----')
    activations = []
    inp = model.input

    model_multi_inputs_cond = True
    if not isinstance(inp, list):
        # only one input! let's wrap it in a list.
        inp = [inp]
        model_multi_inputs_cond = False

    outputs = [layer.output for layer in model.layers if
               layer.name == layer_name or layer_name is None]  # all layer outputs

    functors = [K.function(inp + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions

    if model_multi_inputs_cond:
        list_inputs = []
        list_inputs.extend(model_inputs)
        list_inputs.append(0.)
    else:
        list_inputs = [model_inputs, 0.]

    # Learning phase. 0 = Test mode (no dropout or batch normalization)
    # layer_outputs = [func([model_inputs, 0.])[0] for func in functors]
    layer_outputs = [func(list_inputs)[0] for func in functors]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
        if print_shape_only:
            print(layer_activations.shape)
        else:
            print(layer_activations)
    return activations
