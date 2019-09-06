import matplotlib.pyplot as plt
import numpy as np
# from train import oh_2d,get_prediction
from preprocessor import tokenize
import keras.backend as K
#####show how to convert#####

# plot_attention_graph(model,dataset[i][0],Tx,Ty,human_vocab)
def plot_attention_graph(model,x,Tx,Ty,human_vocab,layer=7):
    # Process input
    tokens = np.array([tokenize(x,human_vocab,Tx)])
    tokens_oh = oh_2d(tokens,len(human_vocab))

    # Monitor model layer
    layer = model.layers[layer]

    layer_over_time = K.function(model.inputs,[layer.get_oiuput_at(t) for t in range(Ty)])
    layer_output = layer_over_time([tokens_oh])
    layer_output = [row.flatten().tolist() for row in layer_output]

    # Get model output
    prediction = get_prediction(model,tokens_oh)[1]

    # Graph the data
    fig = plt.figure()
    fig.set_figwidth(20)
    fig.set_figwidth(1,8)
    ax = fig.add_subplot(111)

    plt.title('Attention Values per Timestep')

    plt.rc('figure')
    cax = plt.imshow(layer_output,vmin=0,vmax=1)
    fig.colorbar(cax)

    plt.xlabel('Input')
    ax.set_xticks(range(Tx))
    ax.set_xtickalabels(x)

    plt.ylabel('Output')
    ax.set_yticks(range(Ty))
    ax.set_yticklabels(prediction)

    plt.show()