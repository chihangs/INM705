import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_history(history):
    '''
    history: dictionary containing loss, val_loss, acc, val_acc, each is list by epoch
    plot from epoch 0 to (num_epoch + 1), if no value for epoch 0, pls initialize list with None before appending loss of epoch 1 etc...
    '''
    num_e = len(history['loss'])
    plt.plot(history['loss'], label='loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.xticks(range(0, num_e+1))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training vs Validation')
    plt.legend()
    plt.show()
    plt.plot(history['acc'], label='acc')
    plt.plot(history['val_acc'], label='val_acc')
    plt.xticks(range(0, num_e+1))
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training vs Validation')
    plt.legend()
    plt.show()
    
def para_num(model):       #no of learnable parameters
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


#________________slight error if last batch is not full batch (i.e. data not divisible by batch size)_________________________

#   assume neural network output has no sigmoid or softmax activation, just plain output from FC

def accuracy(outputs, labels, binaryloss=True):
    """ calculate percent of true labels """
    # predicted labels
    if not binaryloss:
        _, preds = torch.max(outputs, dim = 1)
    else:
        preds = torch.where(outputs>0,1,0)
    return torch.sum(preds == labels).item() / len(preds)


def evaluate(model, loader, loss_fn, metric_fn, device, binaryloss=True):
    """ Evaluate trained weights using calculate loss and metrics """
    # Evaluate model
    model.eval()
    losses = 0.0
    metrics = 0.0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs.t())
            if binaryloss:
                loss = loss_fn(outputs, labels.to(torch.float32))
            else: loss = loss_fn(outputs, labels)
            losses += loss
            metrics += metric_fn(outputs, labels, binaryloss)
    return losses.item() / len(loader), metrics / len(loader)    
    

#________________no last batch error, but need to concat first to predict___________________________________

#   assume neural network output has no sigmoid or softmax activation, just plain output from FC
    
def confusion_matrix(predicted, label):
    #label format: either 1 or 0 for each item
    c_m=np.zeros((2,2))
    for idx, pred in enumerate(predicted):
        true_or_false = 1 if pred == label[idx] else 0
        pos_or_neg = 1 if pred == 1 else 0
        c_m[true_or_false][pos_or_neg] +=1
    return c_m

def predict(model, dataloader, binaryloss=True, show_sample=0):
    #pred_all, out_all, label_all = torch.tensor([]).to(device), torch.tensor([]).to(device), torch.tensor([]).to(device)
    pred_all, out_all, label_all = [], [], []
    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs.t())
            if not binaryloss:
                _, preds = torch.max(outputs, dim = 1)
            else:
                preds = torch.where(outputs>0,1,0)
            #pred_all = torch.cat((pred_all, preds))
            #out_all = torch.cat((out_all, outputs))
            #label_all = torch.cat((label_all, labels))
            pred_all.append(preds)
            out_all.append(outputs)
            label_all.append(labels)
    pred_all = torch.cat(pred_all)
    out_all = torch.cat(out_all)
    label_all = torch.cat(label_all)
    if show_sample != 0:
        print('samples of predictions, outputs and true labels:')
        print('prediction: ', pred_all[0:show_sample])
        print('model output: ', out_all[0:show_sample])
        print('true label: ', label_all[0:show_sample])
    return pred_all, out_all, label_all

def performance(model, dataloader, binaryloss=True):
    #_, cm = evaluate(model, dataloader, loss_fn, confusion_matrix, device, binaryloss=False) * len(dataloader)
    pred_all, out_all, label_all = predict(model, dataloader, binaryloss=binaryloss)
    cm = confusion_matrix(pred_all, label_all)
    true, false = 1, 0
    print('---------------------------------------------------')
    print('              Predicted')
    print('              +ve    -ve')
    print('Actual  +ve  {}  {}'.format(str(round(cm[true][1])).rjust(5), str(round(cm[false][0])).rjust(5)))
    print('        -ve  {}  {}'.format(str(round(cm[false][1])).rjust(5), str(round(cm[true][0])).rjust(5)))
    accuracy = sum(cm[true]) / (sum(cm[true])+sum(cm[false]))
    precision = cm[true][1] / (cm[true][1] + cm[false][1])
    recall = cm[true][1] / (cm[true][1] + cm[false][0])
    f1 = 2*precision*recall/(precision+recall)      #harmonic mean
    print('---------------------------------------------------')
    print('precision:{:.4f}, recall:{:.4f}'.format(precision, recall))
    print('F1-score :{:.4f}'.format(f1))
    print('accuracy :{:.4f}'.format(accuracy))
    print('---------------------------------------------------')
    return cm, accuracy, precision, recall


#_________________________________________________________________________

def confidence_interval(acc, test_size):
    #binomial distribution variance, for metrics such as accuracy that is either right or wrong for each data point
    Var_X = test_size*acc*(1-acc)
    Var_p = Var_X/(test_size**2)
    sigma = Var_p**0.5
    c_i95 = sigma*1.96
    c_i90 = sigma*1.645
    print('95% confidence interval: +/-{:.2f}%'.format(c_i95*100))
    print('90% confidence interval: +/-{:.2f}%'.format(c_i90*100))