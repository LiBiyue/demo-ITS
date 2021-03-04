# by guotong 2021-02-15
import numpy as np
import matplotlib.pyplot as plt
class K_Means:
    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self,data):

        self.centroids = {}

        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_iter):
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []

            for featureset in data:
                distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification],axis=0)

            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid-original_centroid)/original_centroid*100.0) > self.tol:
                    print(np.sum((current_centroid-original_centroid)/original_centroid*100.0))
                    optimized = False

            if optimized:
                break

    def predict(self,data):
        distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification

def relu(x):
  x[x<=0] = 0
  x[x>0] = 1
  return x

def SMOTE(input): # input is dict
    Cluster1_data1 = (input[0])[0]
    Cluster1_data2 = (input[0])[1]
    interpolation = Cluster1_data1 + 0.3 * (Cluster1_data2 - Cluster1_data1)
    X = np.array([(input[0])[0], (input[0])[1], (input[1])[0], (input[1])[1], interpolation])
    return X

def get_P(x): # x is a two-dimentional array (num, dim)
    P = np.zeros((x.shape[0], x.shape[0])) + 0.0001
    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            P[i,j] = np.exp(-1 * np.sum(np.square(x[i,:] - x[j,:])) / 2 / (np.var(x[i, :]) + 0.01))
        P[i,:] = P[i,:] / (np.sum(P[i,:]) - P[i,i])
    return P

def get_Q(x): # x is a two-dimentional array (num, dim)
    Q = np.zeros((x.shape[0], x.shape[0])) + 0.0001
    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            Q[i,j] = np.exp(-1 * np.sum(np.square(x[i,:] - x[j,:])) / 2)
        Q[i,:] = Q[i,:] / (np.sum(Q[i,:]) - Q[i,i])
    return Q

def pca(data, target_dim):
    normolized = data - np.sum(data, 0) / data.shape[0]
    covX = np.cov(normolized.T)
    val, vector = np.linalg.eig(covX)
    indice = np.argsort(-val)
    took_vector = np.matrix(vector[indice[:target_dim], :])  # dim~[target_dim,original_dim]
    W = took_vector.T  # dim~[original_dim,target_dim]
    final_data = normolized * W
    return final_data

def AutoEncoder(x):
    latent_dim = 1

    Encoder_weight = np.random.rand(x.shape[1], latent_dim)
    Encoder_bias = np.random.rand(x.shape[0], latent_dim)

    Decoder_weight = np.random.rand(latent_dim, x.shape[1])
    Decoder_bias = np.random.rand(x.shape[0], x.shape[1])

    print('weight of encoder is : ', np.around(Encoder_weight, decimals=2))
    print('bias of encoder is : ', np.around(Encoder_bias, decimals=2))
    print('weight of dencoder is : ', np.around(Decoder_weight, decimals=2))
    print('bias of dencoder is : ', np.around(Decoder_bias, decimals=2))

    # Encoder
    latent = np.dot(x , Encoder_weight) + Encoder_bias

    # Decoder
    rec = np.dot(latent , Decoder_weight) + Decoder_bias

    return rec, latent

x = np.random.rand(4,2)
print('the input data is : ', x)

model = K_Means()
model.fit(x)
X = SMOTE(model.classifications)
print('the data after SMOTE is : ', X)

Y, latent = AutoEncoder(X)
print('the output of AE is : ', Y)

# # data is fixed
# x = np.array([[1,0.92], [0.26,0.94], [0.84,0.08], [0.08,0.32]])
# X = np.array([[1,0.92], [0.26,0.94], [0.84,0.08], [0.08,0.32], [0.07,0.32]])
# Y = np.array([[0.22,0.76], [0.69,0.5], [0.58,0.72], [0.09,0.38], [0.4,0.57]])

P = get_P(x)
P[P>1] = 1
print('P matrix is : ', np.around(P, decimals=2))

Q = get_Q(Y)
print('Q matrix is : ', np.around(Q, decimals=2))

U = pca(P, 3)
U = pca(U.T, 3)
print('U matrix is : ', np.around(U, decimals=2))

V = pca(Q, 3)
V = pca(V.T, 3)
print('V matrix is : ', np.around(V, decimals=2))

model = K_Means()
model.fit(latent)
centroids = model.centroids # is a dict
print('centroids are : ', centroids)
centroids = np.squeeze(np.array([centroids[0], centroids[1]]))
latent_plot = latent
print('latents are : ', latent)

latent = np.squeeze(latent)

W_matrix = np.zeros((len(latent), len(centroids)))
for i in range(len(latent)):
    for j in range(len(centroids)):
        W_matrix[i,j] = np.linalg.norm(latent[i] - centroids[j])

print('W matrix is : ', np.around(W_matrix, decimals=2))
alpha = 0.5
G_matrix = np.exp(-1 * alpha * W_matrix)
G_matrix = G_matrix.T / np.sum(G_matrix, axis=1)
print('G matrix is : ', np.around(G_matrix, decimals=2))

y_plot = np.zeros((len(latent_plot)))
plt.scatter(X[:,0], X[:,1])
plt.scatter(latent_plot, y_plot)
plt.plot()
plt.show()
