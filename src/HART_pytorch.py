import torch
import torch.nn as nn
import torch.optim as optim
from src.stepbystep.v2 import StepByStep
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from models import HART, HART_encoder
from collections import OrderedDict





processedDataX = np.load('x.npy')
processedDataY = np.load('y.npy')
processedDataY = processedDataY.reshape(-1)
processedDataUser = np.load('user.npy')
processedDataUser = processedDataUser.reshape(-1)


x_tensor = torch.tensor(processedDataX, dtype=torch.float32)
y_tensor = torch.tensor(processedDataY, dtype=torch.long)

dataset = TensorDataset(x_tensor, y_tensor)


###All users, 80-20
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


HART = HART(activityCount=6, projection_dim=192, patchSize=16, timeStep=16, patchCount=8)
#total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#print("-------------------: " + str(total_params))
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(HART.parameters(), lr=0.0001)
HART_architecture = StepByStep(HART, loss, optimizer)
HART_architecture.set_loaders(train_loader, test_loader)
#HART_architecture.load_checkpoint('HART_divided_by_activity')
#sbs_seq.train(50)
#sbs_seq.save_checkpoint('HART_divided_by_activity')
#HART_architecture.plot_losses()
#plt.show()

HART_encoder = HART_encoder(activityCount=6, projection_dim=192, patchSize=16, timeStep=16, patchCount=8)



#faccio il loading dei pesi di HART allenato in HART_encoder
checkpoint = torch.load('HART_divided_by_activity')


def restringi_a_sottoinsieme_chiavi(dizionario, chiavi_da_mantenere):
    return OrderedDict((k, dizionario[k]) for k in chiavi_da_mantenere if k in dizionario)

keys = ['sensorPatches.accProjection.weight',
        'sensorPatches.accProjection.bias',
        'sensorPatches.gyroProjection.weight',
        'sensorPatches.gyroProjection.bias',
        'patchEncoder.position_embedding.weight',
        'AttentionCombinedList.0.layerNormalizer1.norm.weight',
        'AttentionCombinedList.0.layerNormalizer1.norm.bias',
        'AttentionCombinedList.0.accMHA.MHA.in_proj_weight',
        'AttentionCombinedList.0.accMHA.MHA.in_proj_bias',
        'AttentionCombinedList.0.accMHA.MHA.out_proj.weight',
        'AttentionCombinedList.0.accMHA.MHA.out_proj.bias',
        'AttentionCombinedList.0.lightConv.depthwise_conv.weight',
        'AttentionCombinedList.0.gyroMHA.MHA.in_proj_weight',
        'AttentionCombinedList.0.gyroMHA.MHA.in_proj_bias',
        'AttentionCombinedList.0.gyroMHA.MHA.out_proj.weight',
        'AttentionCombinedList.0.gyroMHA.MHA.out_proj.bias',
        'AttentionCombinedList.0.layerNormalizer2.norm.weight',
        'AttentionCombinedList.0.layerNormalizer2.norm.bias',
        'AttentionCombinedList.0.mlp2.fc1.weight',
        'AttentionCombinedList.0.mlp2.fc1.bias',
        'AttentionCombinedList.0.mlp2.fc2.weight',
        'AttentionCombinedList.0.mlp2.fc2.bias',
        'AttentionCombinedList.1.layerNormalizer1.norm.weight',
        'AttentionCombinedList.1.layerNormalizer1.norm.bias',
        'AttentionCombinedList.1.accMHA.MHA.in_proj_weight',
        'AttentionCombinedList.1.accMHA.MHA.in_proj_bias',
        'AttentionCombinedList.1.accMHA.MHA.out_proj.weight',
        'AttentionCombinedList.1.accMHA.MHA.out_proj.bias',
        'AttentionCombinedList.1.lightConv.depthwise_conv.weight',
        'AttentionCombinedList.1.gyroMHA.MHA.in_proj_weight',
        'AttentionCombinedList.1.gyroMHA.MHA.in_proj_bias',
        'AttentionCombinedList.1.gyroMHA.MHA.out_proj.weight',
        'AttentionCombinedList.1.gyroMHA.MHA.out_proj.bias',
        'AttentionCombinedList.1.layerNormalizer2.norm.weight', 'AttentionCombinedList.1.layerNormalizer2.norm.bias', 'AttentionCombinedList.1.mlp2.fc1.weight', 'AttentionCombinedList.1.mlp2.fc1.bias', 'AttentionCombinedList.1.mlp2.fc2.weight', 'AttentionCombinedList.1.mlp2.fc2.bias', 'AttentionCombinedList.2.layerNormalizer1.norm.weight', 'AttentionCombinedList.2.layerNormalizer1.norm.bias', 'AttentionCombinedList.2.accMHA.MHA.in_proj_weight', 'AttentionCombinedList.2.accMHA.MHA.in_proj_bias', 'AttentionCombinedList.2.accMHA.MHA.out_proj.weight', 'AttentionCombinedList.2.accMHA.MHA.out_proj.bias', 'AttentionCombinedList.2.lightConv.depthwise_conv.weight', 'AttentionCombinedList.2.gyroMHA.MHA.in_proj_weight', 'AttentionCombinedList.2.gyroMHA.MHA.in_proj_bias', 'AttentionCombinedList.2.gyroMHA.MHA.out_proj.weight', 'AttentionCombinedList.2.gyroMHA.MHA.out_proj.bias', 'AttentionCombinedList.2.layerNormalizer2.norm.weight', 'AttentionCombinedList.2.layerNormalizer2.norm.bias', 'AttentionCombinedList.2.mlp2.fc1.weight', 'AttentionCombinedList.2.mlp2.fc1.bias', 'AttentionCombinedList.2.mlp2.fc2.weight', 'AttentionCombinedList.2.mlp2.fc2.bias', 'AttentionCombinedList.3.layerNormalizer1.norm.weight', 'AttentionCombinedList.3.layerNormalizer1.norm.bias', 'AttentionCombinedList.3.accMHA.MHA.in_proj_weight', 'AttentionCombinedList.3.accMHA.MHA.in_proj_bias', 'AttentionCombinedList.3.accMHA.MHA.out_proj.weight', 'AttentionCombinedList.3.accMHA.MHA.out_proj.bias', 'AttentionCombinedList.3.lightConv.depthwise_conv.weight', 'AttentionCombinedList.3.gyroMHA.MHA.in_proj_weight', 'AttentionCombinedList.3.gyroMHA.MHA.in_proj_bias', 'AttentionCombinedList.3.gyroMHA.MHA.out_proj.weight', 'AttentionCombinedList.3.gyroMHA.MHA.out_proj.bias', 'AttentionCombinedList.3.layerNormalizer2.norm.weight', 'AttentionCombinedList.3.layerNormalizer2.norm.bias', 'AttentionCombinedList.3.mlp2.fc1.weight', 'AttentionCombinedList.3.mlp2.fc1.bias', 'AttentionCombinedList.3.mlp2.fc2.weight', 'AttentionCombinedList.3.mlp2.fc2.bias', 'AttentionCombinedList.4.layerNormalizer1.norm.weight', 'AttentionCombinedList.4.layerNormalizer1.norm.bias', 'AttentionCombinedList.4.accMHA.MHA.in_proj_weight', 'AttentionCombinedList.4.accMHA.MHA.in_proj_bias', 'AttentionCombinedList.4.accMHA.MHA.out_proj.weight', 'AttentionCombinedList.4.accMHA.MHA.out_proj.bias', 'AttentionCombinedList.4.lightConv.depthwise_conv.weight', 'AttentionCombinedList.4.gyroMHA.MHA.in_proj_weight', 'AttentionCombinedList.4.gyroMHA.MHA.in_proj_bias', 'AttentionCombinedList.4.gyroMHA.MHA.out_proj.weight', 'AttentionCombinedList.4.gyroMHA.MHA.out_proj.bias', 'AttentionCombinedList.4.layerNormalizer2.norm.weight', 'AttentionCombinedList.4.layerNormalizer2.norm.bias', 'AttentionCombinedList.4.mlp2.fc1.weight', 'AttentionCombinedList.4.mlp2.fc1.bias', 'AttentionCombinedList.4.mlp2.fc2.weight', 'AttentionCombinedList.4.mlp2.fc2.bias', 'AttentionCombinedList.5.layerNormalizer1.norm.weight', 'AttentionCombinedList.5.layerNormalizer1.norm.bias', 'AttentionCombinedList.5.accMHA.MHA.in_proj_weight', 'AttentionCombinedList.5.accMHA.MHA.in_proj_bias', 'AttentionCombinedList.5.accMHA.MHA.out_proj.weight', 'AttentionCombinedList.5.accMHA.MHA.out_proj.bias', 'AttentionCombinedList.5.lightConv.depthwise_conv.weight', 'AttentionCombinedList.5.gyroMHA.MHA.in_proj_weight', 'AttentionCombinedList.5.gyroMHA.MHA.in_proj_bias', 'AttentionCombinedList.5.gyroMHA.MHA.out_proj.weight', 'AttentionCombinedList.5.gyroMHA.MHA.out_proj.bias', 'AttentionCombinedList.5.layerNormalizer2.norm.weight', 'AttentionCombinedList.5.layerNormalizer2.norm.bias', 'AttentionCombinedList.5.mlp2.fc1.weight', 'AttentionCombinedList.5.mlp2.fc1.bias', 'AttentionCombinedList.5.mlp2.fc2.weight', 'AttentionCombinedList.5.mlp2.fc2.bias', 'layerNormalizer.norm.weight', 'layerNormalizer.norm.bias']


dict = restringi_a_sottoinsieme_chiavi(checkpoint['model_state_dict'], keys)
checkpoint['model_state_dict'] = dict
HART_encoder.load_state_dict(checkpoint['model_state_dict'])


#t-SNE representation of encoder output

from sklearn.manifold import TSNE

def extract_features(dataloader, model):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for inputs, label in dataloader:
            outputs = model(inputs)
            outputs = torch.flatten(outputs, 1)
            #outputs = torch.mean(outputs, dim=1)
            features.append(outputs)
            labels.append(label)
    features = torch.cat(features).cpu().numpy()
    labels = torch.cat(labels).cpu().numpy()
    return features, labels


features, labels = extract_features(test_loader, HART_encoder)

for perplexity in range(1, 50, 10):
    tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity, n_iter=5000)
    tsne_results = tsne.fit_transform(features)

    plt.figure(figsize=(10, 5))
    for label in np.unique(labels):
        indices = labels == label
        plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], label=label, alpha=0.5)
    plt.legend(markerscale=2)
    plt.title('t-SNE of Submodel Features')
    plt.show()



from sklearn.metrics import accuracy_score

HART.eval()  # Set the model to evaluation mode
y_true = []
y_pred = []

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = HART(inputs)
        _, predicted = torch.max(outputs, 1)

        y_true.extend(labels.numpy())
        y_pred.extend(predicted.numpy())

accuracy = accuracy_score(y_true, y_pred)
print(accuracy)

#(32, 128, 6) -> (32, 128, 3),(32, 128, 3) -> (32, 8, 96), (32, 8, 96) -> (32, 8, 192) -> (32, 8, 48), (32, 8, 96), (32, 8, 48) -> (32, 8, 192) -> (32, 192) -> (32, 1024) -> (32, 10)
#                                                                                             MHA x 3     LC x 6      MHA x 3