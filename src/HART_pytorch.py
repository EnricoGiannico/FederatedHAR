import torch
import torch.nn as nn
import torch.nn.functional as F
#from dataHHAR import processedDataX, processedDataY
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from src.stepbystep.v2 import StepByStep
from torch.utils.data import TensorDataset, DataLoader, random_split

import numpy as np
import matplotlib.pyplot as plt


class SensorPatches(nn.Module):
    def __init__(self, projection_dim, patchSize, timeStep):
        super(SensorPatches, self).__init__()
        self.patchSize = patchSize
        self.timeStep = timeStep
        self.projection_dim = projection_dim
        self.accProjection = nn.Conv1d(in_channels=3, out_channels=int(projection_dim/2), kernel_size=patchSize, stride=timeStep)
        self.gyroProjection = nn.Conv1d(in_channels=3, out_channels=int(projection_dim/2), kernel_size=patchSize, stride=timeStep)

    def forward(self, inputData):
        #(N,L,F)
        inputData_acc_permuted = inputData[:,:,:3].permute(0,2,1)
        inputData_gyro_permuted = inputData[:,:,3:].permute(0,2,1)
        accProjections = self.accProjection(inputData_acc_permuted)
        gyroProjections = self.accProjection(inputData_gyro_permuted)
        accProjections_permuted = accProjections.permute(0,2,1)
        gyroProjections_permuted = gyroProjections.permute(0,2,1)
        output = torch.cat((accProjections_permuted, gyroProjections_permuted), dim=2)
        return output



class PatchEncoder(nn.Module):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.position_embedding = nn.Embedding(num_embeddings=num_patches, embedding_dim=projection_dim)

    def forward(self, patch):
        positions = torch.arange(0, patch.shape[1]).unsqueeze(0).expand(patch.shape[0], -1)  # Adjust based on the actual batch size in patch
        encoded = patch + self.position_embedding(positions)
        return encoded

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.norm = nn.LayerNorm(normalized_shape, eps=eps)

    def forward(self, x):
        return self.norm(x)


class LightConvLayer(nn.Module):
    def __init__(self, kernelSize=16, attentionHead=4):
        super(LightConvLayer, self).__init__()
        self.kernelSize = kernelSize
        self.attentionHead = attentionHead

        # Initialize depthwise convolution layers
        # Assuming each attention head is treated as a separate channel
        self.depthwise_conv = nn.Conv1d(in_channels=attentionHead, out_channels=attentionHead,
                                        kernel_size=kernelSize, stride=1, padding=kernelSize // 2, groups=attentionHead, bias=False)

    def forward(self, input):
        input_reshaped = input.view(-1, self.attentionHead, input.shape[2])
        depthwise_output = self.depthwise_conv(input_reshaped)
        output = depthwise_output.view(input.shape)
        return output


class SensorWiseMHA(nn.Module):
    def __init__(self, projectionQuarter, num_heads, startIndex, stopIndex, dropout_rate):
        super(SensorWiseMHA, self).__init__()
        self.projectionQuarter = projectionQuarter
        self.dropout_rate = dropout_rate
        self.num_heads = num_heads
        self.startIndex = startIndex
        self.stopIndex = stopIndex
        self.MHA = nn.MultiheadAttention(embed_dim=projectionQuarter, num_heads=num_heads, dropout=dropout_rate)

    def forward(self, inputData, return_attention_scores=False):
        # Assumendo che inputData sia di dimensione [batch_size, seq_len, features]
        extractedInput = inputData[:, :, self.startIndex:self.stopIndex].permute(1, 0, 2)
        # Per MultiheadAttention in PyTorch, l'input deve essere di dimensione [seq_len, batch_size, features]

        MHA_Outputs, attentionScores = self.MHA(extractedInput, extractedInput, extractedInput)
        MHA_Outputs = MHA_Outputs.permute(1, 0, 2)

        if return_attention_scores:
            return MHA_Outputs, attentionScores
        else:
            return MHA_Outputs


class MLP2(nn.Module):
    def __init__(self, input_features, hidden_units, dropout_rate):
        super(MLP2, self).__init__()
        self.fc1 = nn.Linear(input_features, hidden_units[0])
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_units[0], hidden_units[1])
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        return x

class AttentionCombined(nn.Module):
    def __init__(self, kernel_size):
        #[32, 8, 196]
        super(AttentionCombined, self).__init__()
        self.layerNormalizer1 = LayerNorm(normalized_shape=192)
        self.accMHA = SensorWiseMHA(projectionQuarter=48, num_heads=3, startIndex=0, stopIndex=48, dropout_rate=0.01)
        self.lightConv = LightConvLayer(kernelSize=kernel_size, attentionHead=4)
        self.gyroMHA = SensorWiseMHA(projectionQuarter=48, num_heads=3, startIndex=96, stopIndex=144, dropout_rate=0.01)
        self.layerNormalizer2 = LayerNorm(normalized_shape=192)
        self.mlp2 = MLP2(input_features=192, hidden_units=[384, 192], dropout_rate=0.01)


    def forward(self, input):
        x = self.layerNormalizer1(input)
        x_lightconv = x[:, :, 48:144]
        x_lightconv = self.lightConv(x_lightconv)
        x_acc = self.accMHA(x)
        x_gyro = self.gyroMHA(x)
        x = torch.cat((x_acc, x_lightconv, x_gyro), dim=2)
        x1 = x + input
        tt = self.layerNormalizer2(x1)
        tt = self.mlp2(tt)
        output = x1 + tt
        return output


class Classifier(nn.Module):
    def __init__(self, input_features, hidden_features, output_features):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, output_features)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

class HART(nn.Module):
    def __init__(self, activityCount, projection_dim, patchSize, timeStep, patchCount=None, convKernels = [3, 7, 15, 31, 31, 31]):
        super(HART, self).__init__()
        self.activityCount = activityCount
        self.convKernels = convKernels
        self.sensorPatches = SensorPatches(projection_dim, patchSize, timeStep)
        self.patchEncoder = PatchEncoder(num_patches=patchCount, projection_dim=projection_dim)
        self.AttentionCombinedList = nn.ModuleList([AttentionCombined(kernel_size=i) for i in convKernels])
        self.layerNormalizer = LayerNorm(normalized_shape=192)
        self.classifier = Classifier(192, 1024, activityCount)

    def forward(self, x):
        x = self.sensorPatches(x)
        x = self.patchEncoder(x)
        x = self.layerNormalizer(x)
        input = x
        for index, _ in enumerate(self.convKernels):
            output = self.AttentionCombinedList[index](input)
            input = input + output
        representation = self.layerNormalizer(input)
        global_average = torch.mean(representation, dim=1)
        logits = self.classifier(global_average)
        return logits

processedDataX = np.load('x.npy')
processedDataY = np.load('y.npy')

processedDataY = processedDataY.reshape(-1)

x_tensor = torch.tensor(processedDataX, dtype=torch.float32)
y_tensor = torch.tensor(processedDataY, dtype=torch.long)

dataset = TensorDataset(x_tensor, y_tensor)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

# Dividi il dataset
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Crea DataLoader per training e test set
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = HART(activityCount=6, projection_dim=192, patchSize=16, timeStep=16, patchCount=8)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("-------------------: " + str(total_params))
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
sbs_seq = StepByStep(model, loss, optimizer)
sbs_seq.set_loaders(train_loader, test_loader)
sbs_seq.load_checkpoint('HART_NewPreprocessing_full_dataset_lr=0.0001_no_overlapping')
sbs_seq.train(100)
#sbs_seq.save_checkpoint('HART_300epochs')
sbs_seq.save_checkpoint('HART_NewPreprocessing_full_dataset_lr=0.0001_no_overlapping')
sbs_seq.plot_losses()
plt.show()
model = sbs_seq.model

class Submodel(nn.Module):
    def __init__(self, original_model, num_layers):
        super(Submodel, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:num_layers])

    def forward(self, x):
        x = self.features(x)
        return x

# Esempio di utilizzo
# model è il tuo modello PyTorch originale
# num_layers è il numero di layer che vuoi mantenere nel sottomodello
submodel = Submodel(model, -3)

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


features, labels = extract_features(test_loader, submodel)
from sklearn.manifold import TSNE
import numpy as np
for perplexity in range(1, 50, 10):
    tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity, n_iter=5000)
    tsne_results = tsne.fit_transform(features)

    plt.figure(figsize=(10, 5))
    for label in np.unique(labels):
        indices = labels == label
        plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], label=label, alpha=0.5)
    plt.legend(markerscale=2)
    plt.title('t-SNE of Submodel Features')
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    plt.show()


#sbs_seq.save_checkpoint('HART_1000epochs')
#fig = sbs_seq.plot_losses()
#plt.show()
'''
print('ok')
iterator = iter(dataloader)
x_batch, y_batch = next(iterator)
out = model(x_batch)
'''
print('ok')


#(32, 128, 6) -> (32, 128, 3),(32, 128, 3) -> (32, 8, 96), (32, 8, 96) -> (32, 8, 192) -> (32, 8, 48), (32, 8, 96), (32, 8, 48) -> (32, 8, 196) -> (32, 196) -> (32, 1024) -> (32, 10)
#                                                                                             MHA x 3     LC x 6      MHA x 3