import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv('wustl-ehms-2020_with_attacks_categories.csv')

# 1. VERİ ÖNİŞLEME
# 1.1 NULL değer olup olmadığını kontrol et
#print(df.isnull().sum()) 
#df = df.dropna()  # NULL değer bulunana satırları çıkar
# df = df.fillna(df.median()) # boş değerleri sütunun medyanı ile doldur

# 1.2 eğitimde kullanılmayacak sütünları çıkar
df = df.drop(columns=['Dir'])  
# #, 'SrcAddr', 'DstAddr', 'Dport', 'SrcBytes', 'DstBytes', 'SrcGap',
#                       'DstGap', 'SIntPkt', 'sMaxPktSz', 'dMaxPktSz', 'sMinPktSz', 'dMinPktSz',
#                       'Trans', 'TotPkts', 'TotBytes', 'Loss', 'pDstLoss', 'Rate', 'SrcMac', 'DstMac',
#                        'Attack Category' ])  

# 1.3 ketegorik verileri etiketle (label encoding)
label_encoder = LabelEncoder()
categorical_columns = df.select_dtypes(include=['object', 'category']).columns
for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])

# 1.4 normalizasyon  ---> bu kısmı çıkardım çünkü normalizasyon yapınca çalışmadı
numerical_columns = df.drop(columns=['Label']).select_dtypes(include=['int64', 'float64']).columns
scaler = StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# 1.5 veriyi özellik ve hedef (feature-target) diye ayır
X = df.drop(columns=['Label'])
y = df['Label']
# 1.6 eğitim ve test verisini ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 2. MODEL EĞİTİMİ

mlp_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=52) #modeli oluştur
mlp_model.fit(X_train, y_train) #modeli eğitim verisiyle eğit
accuracy = mlp_model.score(X_test, y_test) #test verisiyle sınıflandırma yap

print(f'Model Accuracy: {accuracy * 100:.2f}%')

#confusion matrisi oluştur
y_pred = mlp_model.predict(X_test) 
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)


# class_report = classification_report(y_test, y_pred, target_names=target_classes)
# print("Classification Report:")
# print(class_report)

