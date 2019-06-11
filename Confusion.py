import matplotlib.pyplot as plt 
labels = ['Sincere', 'Insincere']
cm =    [[27249,923],
 [809,1019]]
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm, cmap='Blues')
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
for i in range(2):
    for j in range(2):
        text = ax.text(j, i, cm[i][j],
                       ha="center", va="center", color="black")
plt.show()