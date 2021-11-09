from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_olivetti_faces
from sklearn.svm import LinearSVC

import matplotlib.pyplot as plt

digits = load_digits()
olivetti_faces = fetch_olivetti_faces()
elements = 5

X_train, X_test, y_train, y_test = train_test_split(olivetti_faces.data, olivetti_faces.target, test_size=0.3, random_state=42, shuffle=True)

# fig, axs = plt.subplots(int(len(X_test) / 10), 10)
# for i in range(int(len(X_test) / 10)):
#     for j in range(10):
#         axs[i][j].imshow(X_test[i * 10 + j], cmap='gray')
#         axs[i][j].axis("off")
# plt.show()

svc = LinearSVC(verbose=True)
svc.fit(X_train, y_train)
scr = svc.score(X_test, y_test)
print(scr)