from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
# Step 1: Collecting data
train_df = pd.read_csv(r"train.csv")
test_df = pd.read_csv(r"test.csv")

# Step 2: Preprocessing data by features engineering
features = ["Sex", "Pclass", "Age", "SibSp", "Parch", "Fare", "Embarked"]


def preprocessing(df):
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})
    return df


# step 3: Start training models
train_df = preprocessing(train_df)
test_df = preprocessing(test_df)
X = train_df[features]
Y = train_df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
# Step 4: Evaluate accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))
test_feature = test_df[features]
test_pred = model.predict(test_feature)
Submission = pd.DataFrame({
    "PassengerId": test_df["PassengerId"],
    "Survived": test_pred
})

importances = model.feature_importances_
feat_imp = pd.DataFrame({"Feature": features, "Importance": importances})
feat_imp = feat_imp.sort_values(by="Importance", ascending=False)

plt.figure(figsize=(7, 6))
plt.bar(feat_imp["Feature"], feat_imp["Importance"], color="skyblue")
plt.xticks(rotation=45)
plt.title("Feature Importance - Titanic Survival", fontsize=14)
plt.xlabel("Đặc trưng")
plt.ylabel("Độ quan trọng")
plt.show()