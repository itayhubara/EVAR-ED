from matplotlib import pyplot
import torch

auc_val=torch.load('auc_val')

pyplot.plot(auc_val[2],auc_val[1])
pyplot.title('AUROC Deep Learning')
pyplot.xlabel('True Positive')
pyplot.ylabel('False Positive')
pyplot.show()
