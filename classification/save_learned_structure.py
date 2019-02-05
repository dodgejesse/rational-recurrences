import torch
import train_classifier
import numpy as np

THRESHOLD = 0.1

def to_file(model, filepath, args):
    #print("Writing model to", of)
    #torch.save(model.state_dict(), of)
    import pdb; pdb.set_trace()
    extract_learned_structure(model, args)


def extract_learned_structure(model, args):
    states = train_classifier.get_states_weights(model, args)
    #embed_dim = model.emb_layer.n_d
    #num_edges_in_wfsa = model.encoder.rnn_lst[0].cells[0].k
    num_wfsas = int(args.d_out)
    
    #reshaped_weights = model.encoder.rnn_lst[0].cells[0].weight.view(embed_dim, num_wfsas, num_edges_in_wfsa)
    # pretty sure these are transitions
    #first_weights = reshaped_weights[...,0:int(num_edges_in_wfsa/2)]
    # pretty sure these are self loops
    #second_weights = reshaped_weights[...,int(num_edges_in_wfsa/2):num_edges_in_wfsa]

    #states = torch.cat((reshaped_weights[...,0:int(num_edges_in_wfsa/2)],
    #                    reshaped_weights[...,int(num_edges_in_wfsa/2):num_edges_in_wfsa]),0)


    num_ngrams, num_of_each_ngram = find_num_ngrams(states, num_wfsas)

    new_model = create_new_model(num_of_each_ngram, args, model)

    update_new_model_weights(model, new_model, num_ngrams, args)


def update_new_model_weights(model, new_model, num_ngrams, args):
    embed_dim = model.emb_layer.n_d
    num_edges_in_wfsa = model.encoder.rnn_lst[0].cells[0].k
    num_wfsas = int(args.d_out)
    reshaped_weights = model.encoder.rnn_lst[0].cells[0].weight.view(embed_dim, num_wfsas, num_edges_in_wfsa)

    
    for i in range(len(new_model.encoder.rnn_lst[0].cells)):
        # to get the indices of the wfsas of length i in reshaped_weights
        if args.gpu:
            wfsa_indices = torch.autograd.Variable(torch.cuda.LongTensor(np.where(num_ngrams == i)[0]))
        else:
            wfsa_indices = torch.autograd.Variable(torch.LongTensor(np.where(num_ngrams == i)[0]))
        # DEBUG: check this is getting the correct thing
        cur_model_weights = torch.index_select(reshaped_weights, 1, wfsa_indices)
        # to get only the non-zero states
        cur_model_weights = cur_model_weights[:,:,0:(i+1)*2]
        #cur_model_weights = reshaped_weights[:,np.where(num_ngrams == i),0:(i+1)*2]
        cur_new_model_weights = new_model.encoder.rnn_lst[0].cells[i].weight
        cur_new_model_weights = cur_new_model_weights.view(embed_dim, cur_model_weights.shape[1], (i+1)*2)
        if args.gpu:
            cur_new_model_weights = cur_new_model_weights.cuda()

        cur_new_model_weights_data = cur_new_model_weights.data
        cur_model_weights_data = cur_model_weights.data

        cur_new_model_weights_data.add_(-cur_new_model_weights_data)
        cur_new_model_weights_data.add_(cur_model_weights_data)

        # do something with the bias term.
        model.encoder.rnn_lst[0].cells[0].bias.view(8, 1, 24)


def create_new_model(num_of_each_ngram, args, model):
    # to store the current d_out and pattern
    tmp_d_out = args.d_out
    tmp_pattern = args.pattern

    # to generate the new learned d_out
    new_d_out = ""
    for i in range(len(num_of_each_ngram)):
        new_d_out += "{},".format(num_of_each_ngram[i])
    new_d_out = new_d_out[:-1]
    new_pattern = "1-gram,2-gram,3-gram,4-gram"

    # setting the new d_out and pattern in args
    args.d_out = new_d_out
    args.pattern = new_pattern

    # creating the new model
    new_model = train_classifier.Model(args, model.emb_layer, model.out.out_features)

    # putting the d_out and pattern back in args
    args.d_out = tmp_d_out
    args.pattern = tmp_pattern
    return new_model
    
    
def find_num_ngrams(states, num_wfsas):
    # a list which stores the n-gram of each wfsa (e.g. 0,1,2 etc.) 
    num_ngrams = []
    
    num_states = states.shape[2]    

    # to find the largest state which is above the threshold
    for i in range(num_wfsas):
        cur_max_state = -1
        for j in range(num_states):

            cur_group = states[:,i,j].data

            if cur_group.norm(2) > THRESHOLD and cur_max_state == j - 1:
                cur_max_state = j
        num_ngrams.append(cur_max_state)

    num_ngrams = np.asarray(num_ngrams)
    num_of_each_ngram = []
    for i in range(num_states):
        num_of_each_ngram.append(sum(num_ngrams == i))

    return num_ngrams, num_of_each_ngram
    
    
