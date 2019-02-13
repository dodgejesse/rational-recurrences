import torch
import train_classifier
import numpy as np
import os

THRESHOLD = 0.1
ALLWFSA = None

def to_file(model, filepath, args, data_x, data_y):
    new_model, new_d_out = extract_learned_structure(model, args)
    check_new_model_predicts_same(model, new_model, data_x, data_y)

    reduced_model_path = os.path.join(args.output_dir, "best_model_dout={}.pth".format(new_d_out))
    print("Writing model to", reduced_model_path)
    torch.save(new_model.state_dict(), reduced_model_path)

    full_model_path = os.path.join(args.output_dir, "best_model.pth")
    #print("Writing model to", reduced_model_path)
    torch.save(model.state_dict(), full_model_path)


# a method used to make sure the extracted structure behaves similarly to the learned model
def check_new_model_predicts_same(model, new_model, data_x, data_y):
    # can manually look at feats vs new_model_feats, should be close (and identical for max-length WFSAs)
    #if check == "manually check features from wfsa":
    if True:
        cur_x = (data_x[0])
        model_wfsakeep_pred = predict_one_example(model, ALLWFSA, cur_x)

        indices_in_new_model = torch.autograd.Variable(torch.arange(ALLWFSA.shape[0]).type(torch.cuda.LongTensor))
        new_model_pred = predict_one_example(new_model, indices_in_new_model, cur_x)
        
        # the features which didn't make it into the smaller model:
        model_indices_not_in_new = [int(x) for x in torch.arange(24) if not x in ALLWFSA.data]
        model_indices_not_in_new = torch.autograd.Variable(torch.cuda.LongTensor(model_indices_not_in_new))
        
        model_wfsadiscard_pred = predict_one_example(model, model_indices_not_in_new, cur_x, add_bias = False)


        # DEBUG stuff here
        model_feat = encoder_fwd(model, cur_x)
        model_feat = model.drop(model_feat)
        selected_feats = torch.index_select(model_feat, 1, ALLWFSA)[0,:]

        new_model_feat = encoder_fwd(new_model, cur_x)
        new_model_feat = model.drop(new_model_feat)
        
        # this shows that the 14th WFSA (seen in ALLWFSA) has the largest gap, of 0.2152, at epoch 37
        new_model_feat[0,:] - selected_feats
        

        
        print(model_wfsakeep_pred, new_model_pred,model_wfsadiscard_pred)
    if True:
        predict_all_train(model, new_model, data_x)
    if True:
        compare_err(model, new_model, data_x, data_y)

def predict_one_example(model, indices, cur_x, add_bias = True):

    model_feat = encoder_fwd(model, cur_x)
    model_feat = model.drop(model_feat)
    selected_feats = torch.index_select(model_feat, 1, indices)[0,:]
    # to see what the contribution of the existing wfsas is, and what the contribution is of the ones that were removed

    if add_bias:
        model_b = model.out.bias
    else:
        model_b = 0
    selected_weights = torch.index_select(model.out.weight, 1, indices)
    #selected_weights = torch.index_select(model_w, 1, indices)
    return selected_weights.matmul(selected_feats) + model_b
    

        
def compare_err(model, new_model, data_x, data_y):
    model_err = round(train_classifier.eval_model(None, model, data_x, data_y), 4)
    new_model_err = round(train_classifier.eval_model(None, new_model, data_x, data_y), 4)
    print("difference: {}, model err: {}, extracted structure model err: {}".format(
        round(new_model_err - model_err, 4), model_err, new_model_err))
    
# a method used to make sure the extracted structure behaves similarly to the learned model
def predict_all_train(model, new_model, data_x):
    model.eval()
    new_model.eval()
    total_examples = 0
    total_same_pred = 0
    for i in range(len(data_x)):
        cur_x = (data_x[i])
        total_same_pred += sum(new_model(cur_x).data.max(1)[1] == model(cur_x).data.max(1)[1])
        total_examples += cur_x[0].shape[0]
    print("total: {}, same pred: {}, frac: {}".format(total_examples, total_same_pred, round(total_same_pred * 1.0 / total_examples, 4)))
    #print("same pred: {}".format(total_same_pred))
    #assert total_same_pred * 1.0 / total_examples > .90

# a method used to make sure the extracted structure behaves similarly to the learned model
def encoder_fwd(model, cur_x):
    model.eval()
    emb_fwd = model.emb_layer(cur_x)
    emb_fwd = model.drop(emb_fwd)

    out_fwd, hidden_fwd, _ = model.encoder(emb_fwd)
    batch, length = emb_fwd.size(-2), emb_fwd.size(0)
    out_fwd = out_fwd.view(length, batch, 1, -1)
    feat = out_fwd[-1,:,0,:]
    return feat

def extract_learned_structure(model, args, epoch = 0):
    
    states = train_classifier.get_states_weights(model, args)
    num_wfsas = int(args.d_out)
    num_ngrams, num_of_each_ngram = find_num_ngrams(states, num_wfsas)
   
    if max(num_ngrams) == -1:
        return None, 0

    new_model, new_d_out = create_new_model(num_of_each_ngram, args, model)

    update_new_model_weights(model, new_model, num_ngrams, args)
    return new_model, new_d_out

# all weights are either "model" weights or "new_model" weights, as denoted in the name of the local variable
def update_new_model_weights(model, new_model, num_ngrams, args):
    embed_dim = model.emb_layer.n_d
    num_edges_in_wfsa = model.encoder.rnn_lst[0].cells[0].k
    num_wfsas = int(args.d_out)
    reshaped_model_weights = model.encoder.rnn_lst[0].cells[0].weight.view(embed_dim, num_wfsas, num_edges_in_wfsa)

    cur_cell_num = 0
    all_wfsa_indices = []
    for i in range(int(num_edges_in_wfsa/2)):
        # if there are no ngrams of this length, continue
        if sum(num_ngrams == i) == 0:
            #import pdb; pdb.set_trace()
            continue
        # to get the indices of the wfsas of length i in reshaped_model_weights
        if args.gpu:
            wfsa_indices = torch.autograd.Variable(torch.cuda.LongTensor(np.where(num_ngrams == i)[0]))
        else:
            wfsa_indices = torch.autograd.Variable(torch.LongTensor(np.where(num_ngrams == i)[0]))
        

        update_mult_weights(model, reshaped_model_weights, new_model, wfsa_indices, args, i, cur_cell_num)
        update_bias_weights(num_edges_in_wfsa, num_wfsas, model, new_model, wfsa_indices, args, i, cur_cell_num)
        all_wfsa_indices.append(wfsa_indices)
        cur_cell_num += 1


    # DEBUG: have to fix the Linear layer at the end
    update_linear_output_layer(model, new_model, num_ngrams, torch.cat(all_wfsa_indices))
    # ignore the bias_final term, and check we don't use it. it's for language modeling.
    assert not model.encoder.bidirectional
    assert not args.use_rho


def update_linear_output_layer(model, new_model, num_ngrams, all_wfsa_indices):
    model_weights = model.out.weight
    model_bias = model.out.bias.data

    new_model_weights = new_model.out.weight.data
    new_model_bias = new_model.out.bias.data

    cur_model_weights = torch.index_select(model_weights, 1, all_wfsa_indices).data

    # DEBUG
    if not new_model_weights.shape == cur_model_weights.shape:
        import pdb; pdb.set_trace()
    
    
    new_model_weights.copy_(cur_model_weights)
    new_model_bias.copy_(model_bias)
    
    
    # DEBUG
    global ALLWFSA
    ALLWFSA = all_wfsa_indices
    
    
    
def update_bias_weights(num_edges_in_wfsa, num_wfsas, model, new_model, wfsa_indices, args, i, cur_cell_num):
    # do something with the bias term.
    # it looks like we don't actually use the first half of the first dim of the bias anywhere.
    # DEBUG fix these magic numbers
    model_bias = model.encoder.rnn_lst[0].cells[0].bias.view(8, 1, 24)
    cur_model_bias = torch.index_select(model_bias, 2, wfsa_indices)
    # to get the parts of the bias that are actually used
    model_start_index = int(cur_model_bias.shape[0]/2)
    # [model_start_index - (i+1) : model_start_index + (i+1)] is the middle set of params from this matrix
    cur_model_bias = cur_model_bias[model_start_index - (i+1) : model_start_index + (i+1), ...]
    
    cur_new_model_bias = new_model.encoder.rnn_lst[0].cells[cur_cell_num].bias.view((i+1)*2, 1, wfsa_indices.shape[0])
    
    cur_new_model_bias_data = cur_new_model_bias.data
    cur_model_bias_data = cur_model_bias.data
    
    cur_new_model_bias_data.add_(cur_model_bias_data)
        
        
# updates the multiplicative weights in new_model to be the same as in model, for the patterns of length i+1
def update_mult_weights(model, reshaped_model_weights, new_model, wfsa_indices, args, i, cur_cell_num):

    cur_model_weights = torch.index_select(reshaped_model_weights, 1, wfsa_indices)
    # to get only the non-zero states
    cur_model_weights = cur_model_weights[:,:,0:(i+1)*2]
    
    cur_new_model_weights = new_model.encoder.rnn_lst[0].cells[cur_cell_num].weight
    cur_new_model_weights = cur_new_model_weights.view(cur_model_weights.shape[0], cur_model_weights.shape[1], (i+1)*2)
    #if args.gpu:
    #    cur_new_model_weights = cur_new_model_weights.cuda()
        
    cur_new_model_weights_data = cur_new_model_weights.data
    cur_model_weights_data = cur_model_weights.data
    
    cur_new_model_weights_data.add_(-cur_new_model_weights_data)
    cur_new_model_weights_data.add_(cur_model_weights_data)
        
        
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
    if args.gpu:
        new_model.cuda()

    # putting the d_out and pattern back in args
    args.d_out = tmp_d_out
    args.pattern = tmp_pattern
    return new_model, new_d_out
    
    
def find_num_ngrams(states, num_wfsas):
    # a list which stores the n-gram of each wfsa (e.g. 0,1,2 etc.) 
    num_ngrams = []
    
    num_states = states.shape[2]

    # to find the largest state which is above the threshold
    for i in range(num_wfsas):
        cur_max_state = -1
        prev_group_norm = -1
        for j in range(num_states):

            cur_group = states[:,i,j].data

            if cur_group.norm(2) > THRESHOLD and cur_max_state == j - 1:
                # and prev_group_norm * .1 < cur_group.norm(2)
                cur_max_state = j
            prev_group_norm = cur_group.norm(2)
        num_ngrams.append(cur_max_state)

    num_ngrams = np.asarray(num_ngrams)
    num_of_each_ngram = []
    for i in range(num_states):
        num_of_each_ngram.append(sum(num_ngrams == i))

    return num_ngrams, num_of_each_ngram
    
    
