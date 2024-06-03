from model.nb_model import GaussianNaiveBayes
from model.knn_model import KNNClassifier
from service import app
import sys
sys.path.insert(0, 'D:/diabetes predictior using ML')
app.run(host='0.0.0.0', port=5000, debug=True)
