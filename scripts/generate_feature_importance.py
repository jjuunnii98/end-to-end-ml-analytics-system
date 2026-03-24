from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd

from src.data.load_data import load_telco_dataset
from src.data.preprocess import split_features_and_target, split_train_valid
from src.features.build_features import build_preprocessor, fit_transform_features, get_transformed_feature_names

# 경로
model_path = Path("artifacts/model/random_forest_model.joblib")
output_dir = Path("artifacts/figures")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "feature_importance.png"

# 데이터와 feature names 재생성
df = load_telco_dataset()
X, y = split_features_and_target(df)
X_train, X_valid, y_train, y_valid = split_train_valid(X, y)

preprocessor = build_preprocessor(X)
X_train_processed, X_valid_processed = fit_transform_features(preprocessor, X_train, X_valid)
feature_names = get_transformed_feature_names(preprocessor)

# 저장된 모델 로드
model = joblib.load(model_path)

# feature importance 추출
importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False).head(15)

# 시각화
plt.figure(figsize=(10, 7))
plt.barh(importance_df["feature"][::-1], importance_df["importance"][::-1])
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Top 15 Feature Importances - Random Forest")
plt.tight_layout()
plt.savefig(output_path, dpi=200, bbox_inches="tight")
plt.close()

print(f"Saved: {output_path}")