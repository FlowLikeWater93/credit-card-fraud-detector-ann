import numpy as np
import pandas as pd
import torch
import math
import pickle
import fraudDetector
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix



def train_model(X, Y, model, optimizer, loss_function):
    '''
    Parameters :
    X : batch of training data
    Y : true labels
    model : ML model
    optimizer :
    loss_function:

    - Pass a batch of training data to our model
    - Compare to true labels
    - Calculate loss
    - Update model parameters

    Returns :
    training loss value
    '''

    y_preds = model(X)
    loss = loss_function(y_preds, Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate_model(X, Y, model):
    '''
    Parameters :
    X : batch of training data
    Y : true labels
    model : ML model

    - Pass a batch of test data to our model
    - Compare to true labels
    - Calculate evaluation metrics

    Returns :
    F1 score
    Recall
    Precision
    '''

    y_preds = model(X)
    rounded_tensor = torch.round(y_preds)
    # calculate confusion matrix and F1 score
    tn, fp, fn, tp = confusion_matrix(Y.detach().numpy(), rounded_tensor.detach().numpy()).ravel()
    print('TP = {}, FP = {}, TN = {}, FN = {}'.format(tp, fp, tn, fn))
    if (fp+tp) == 0 :
        precision = 0
    else :
        precision = tp / (tp + fp)
    if (tp+fp) == 0 :
        recall = 0
    else :
        recall = tp / (tp + fn)
    if recall + precision == 0 :
        f1 = 0
    else :
        f1 = 2 * (precision*recall) / (precision+recall)
    print('F1 score : ', f1)
    print('precision : ', precision)
    print('recall : ', recall)
    print()
    return precision, recall, f1



# Load dataset
df = pd.read_csv('creditcard.csv')

# Split into X and Y, omitting 'Time'
x = df.iloc[:,1:-1]
y = df.iloc[:,-1]

# Use oversampling techniques to workaround class imbalance
## Unbalanced dataset = Oversample using SMOTE
sm = SMOTE(random_state=42)
x_smote, y_smote = sm.fit_resample(x, y)

# Split into train and test (80/20)
X_train, X_test, y_train, y_test = train_test_split(x_smote, y_smote, test_size = 0.2, random_state = 42)

# Standardize numeric features
## Standardization in machine learning is crucial because it ensures features contribute equally to model training,
## leading to improved accuracy, faster convergence, and reduced bias, especially for algorithms that rely on distance or gradient calculations
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert x and y into torch tensor
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

# Initialze model, optimizer and loss function
model = fraudDetector.FraudClassifier(X_train_tensor.shape[1])
loss_function = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


## Start training and evaluating the model
epochs = 10
# update parameters in batches of 1024
batches = math.ceil(X_train_tensor.shape[0]/1024.0)
# save tarining and evaltion metrics in a dictionary
model_metrics = {'training_loss': [], 'f1': [], 'precision': [], 'recall': []}
# repeat for each epoch
for epoch in range(epochs) :
    losses = []
    model.train()
    print('-- EPOCH ',(epoch+1),' --')
    # repeat for each batch
    for batch in range(batches) :
        n = batch * 1024
        # training batch
        if batch == (batches - 1) :
            loss = train_model(X_train_tensor[n:,:], y_train_tensor[n:,:], model, optimizer, loss_function)
        else :
            loss = train_model(X_train_tensor[n:(n+1024),:], y_train_tensor[n:(n+1024),:], model, optimizer, loss_function)
        # Save batch loss
        losses.append(loss)
    # print mean loss of epoch
    print('Training Loss : {}'.format(np.array(losses).mean()))
    # save epoch loss
    model_metrics['training_loss'].append(np.array(losses).mean())
    # Evaluating the model after each epoch
    model.eval()
    with torch.no_grad():
        # pass test data model and calculate evaluation metrics
        precision, recall, f1 = evaluate_model(X_test_tensor, y_test_tensor, model)
        # save evaluation metrics
        model_metrics['f1'].append(f1)
        model_metrics['recall'].append(recall)
        model_metrics['precision'].append(precision)

# save model parameters and metrics
## model parameters
torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, 'model.pth')
## model metrics
with open('model_metrics.pickle', 'wb') as handle :
    pickle.dump(model_metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)

# END
print('COMPLETED SUCCESSFULLY ...')
