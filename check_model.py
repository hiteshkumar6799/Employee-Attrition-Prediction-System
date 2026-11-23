import pickle

with open("model.pkl", "rb") as file:
    model_info = pickle.load(file)

model = model_info["model"]
columns = model_info["columns"]

print("Model loaded successfully!")
print("Model:", type(model).__name__)
print("Number of features:", len(columns))
print("Features:", columns)