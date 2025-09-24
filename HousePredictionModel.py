# Advanced House Price Prediction with Comprehensive Data Preprocessing, Multiple Models, and Enhanced Visualizations


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (GridSearchCV, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# Set random seed for reproducibility
np.random.seed(42)

# Step 1: Create synthetic housing dataset
def create_synthetic_housing_data(n_samples=1000):
    """Create a synthetic dataset mimicking real housing data"""
    data = {
        'sqft_living': np.random.normal(2000, 800, n_samples),
        'bedrooms': np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.2, 0.4, 0.25, 0.05]),
        'bathrooms': np.random.normal(2.5, 1, n_samples),
        'floors': np.random.choice([1, 2, 3], n_samples, p=[0.6, 0.35, 0.05]),
        'waterfront': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
        'view': np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.6, 0.2, 0.1, 0.07, 0.03]),
        'condition': np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.05, 0.1, 0.6, 0.2, 0.05]),
        'grade': np.random.choice([4, 5, 6, 7, 8, 9, 10, 11, 12], n_samples,
                                p=[0.02, 0.05, 0.1, 0.2, 0.25, 0.2, 0.1, 0.06, 0.02]),
        'yr_built': np.random.choice(range(1900, 2021), n_samples),
        'zipcode': np.random.choice([98001, 98002, 98003, 98004, 98005], n_samples)
    }

    df = pd.DataFrame(data)

    # Create price based on features with some noise
    df['price'] = (
        df['sqft_living'] * 150 +
        df['bedrooms'] * 10000 +
        df['bathrooms'] * 15000 +
        df['floors'] * 20000 +
        df['waterfront'] * 200000 +
        df['view'] * 30000 +
        df['condition'] * 25000 +
        df['grade'] * 40000 +
        (2021 - df['yr_built']) * -500 +
        np.where(df['zipcode'] == 98004, 100000, 0) +  # Premium zipcode
        np.random.normal(0, 50000, n_samples)  # Random noise
    )

    # Introduce missing values randomly
    missing_cols = ['bathrooms', 'view', 'yr_built']
    for col in missing_cols:
        missing_idx = np.random.choice(df.index, size=int(0.1 * len(df)), replace=False)
        df.loc[missing_idx, col] = np.nan

    return df

# Step 2: Advanced Data Preprocessing
def advanced_preprocessing(df):
    """Perform comprehensive data preprocessing"""
    print("=== ADVANCED DATA PREPROCESSING ===")

    # Initial data exploration
    print(f"Dataset shape: {df.shape}")
    print(f"\nMissing values per column:")
    print(df.isnull().sum())

    # Separate numerical and categorical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # Remove target from numerical columns
    if 'price' in numerical_cols:
        numerical_cols.remove('price')

    # Handle missing values
    # KNN imputation for numerical features
    knn_imputer = KNNImputer(n_neighbors=5)
    df[numerical_cols] = knn_imputer.fit_transform(df[numerical_cols])

    # Feature Engineering
    print("\n=== FEATURE ENGINEERING ===")

    # Create new features
    df['house_age'] = 2021 - df['yr_built']
    df['price_per_sqft'] = df['price'] / df['sqft_living']
    df['total_rooms'] = df['bedrooms'] + df['bathrooms']
    df['is_luxury'] = ((df['grade'] >= 10) | (df['waterfront'] == 1)).astype(int)

    # Create interaction features
    df['sqft_grade_interaction'] = df['sqft_living'] * df['grade']
    df['bedrooms_bathrooms_ratio'] = df['bedrooms'] / (df['bathrooms'] + 1)  # +1 to avoid division by zero

    # Outlier detection using IQR method
    print("\n=== OUTLIER DETECTION ===")
    outlier_cols = ['sqft_living', 'price', 'bathrooms']
    outlier_indices = set()

    for col in outlier_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        col_outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
        outlier_indices.update(col_outliers)
        print(f"Column {col}: {len(col_outliers)} outliers detected")

    print(f"Total unique outliers: {len(outlier_indices)}")

    # Remove outliers
    df_clean = df.drop(outlier_indices).reset_index(drop=True)
    print(f"Dataset shape after outlier removal: {df_clean.shape}")

    return df_clean

# Step 3: Advanced Machine Learning Pipeline
def build_ml_pipeline(df):
    """Build and evaluate multiple ML models with proper validation"""
    print("\n=== MACHINE LEARNING PIPELINE ===")

    # Prepare features and target
    feature_cols = [col for col in df.columns if col not in ['price', 'price_per_sqft']]
    X = df[feature_cols]
    y = df['price']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define models with hyperparameter grids
    models = {
        'Linear Regression': {
            'model': Pipeline([
                ('scaler', StandardScaler()),
                ('poly', PolynomialFeatures(degree=2, include_bias=False)),
                ('selector', SelectKBest(f_regression, k=10)),
                ('regressor', LinearRegression())
            ]),
            'params': {
                'poly__degree': [1, 2],
                'selector__k': [8, 10, 12]
            }
        },
        'Random Forest': {
            'model': RandomForestRegressor(random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [10, 15, None],
                'min_samples_split': [2, 5]
            }
        },
        'Gradient Boosting': {
            'model': GradientBoostingRegressor(random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1],
                'max_depth': [3, 5]
            }
        }
    }

    # Train and evaluate models
    results = {}

    for name, config in models.items():
        print(f"\n--- Training {name} ---")

        # Grid search with cross-validation
        grid_search = GridSearchCV(
            config['model'],
            config['params'],
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )

        # Fit the model
        grid_search.fit(X_train, y_train)

        # Best model predictions
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results[name] = {
            'model': best_model,
            'y_pred': y_pred,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'best_params': grid_search.best_params_
        }

        print(f"Best parameters: {grid_search.best_params_}")
        print(f"RMSE: ${rmse:,.2f}")
        print(f"MAE: ${mae:,.2f}")
        print(f"R²: {r2:.4f}")

    return results, X_test, y_test, feature_cols

# Step 4: Advanced Visualization
def create_advanced_visualizations(results, X_test, y_test, feature_cols, df):
    """Create comprehensive visualizations"""
    print("\n=== CREATING ADVANCED VISUALIZATIONS ===")

    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")

    # Create a large figure with subplots
    fig = plt.figure(figsize=(20, 16))

    # 1. Model Performance Comparison
    ax1 = plt.subplot(3, 3, 1)
    models = list(results.keys())
    rmse_values = [results[model]['rmse'] for model in models]
    r2_values = [results[model]['r2'] for model in models]

    x_pos = np.arange(len(models))
    bars = ax1.bar(x_pos, rmse_values, alpha=0.7, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax1.set_xlabel('Models', fontsize=12, fontweight='bold')
    ax1.set_ylabel('RMSE ($)', fontsize=12, fontweight='bold')
    ax1.set_title('Model Performance Comparison\n(Lower RMSE is Better)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(models, rotation=45)

    # Add value labels on bars
    for bar, rmse in zip(bars, rmse_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'${rmse:,.0f}', ha='center', va='bottom', fontweight='bold')

    # 2. R² Score Comparison
    ax2 = plt.subplot(3, 3, 2)
    bars2 = ax2.bar(x_pos, r2_values, alpha=0.7, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax2.set_xlabel('Models', fontsize=12, fontweight='bold')
    ax2.set_ylabel('R² Score', fontsize=12, fontweight='bold')
    ax2.set_title('Model R² Score Comparison\n(Higher is Better)', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(models, rotation=45)
    ax2.set_ylim(0, 1)

    for bar, r2 in zip(bars2, r2_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{r2:.3f}', ha='center', va='bottom', fontweight='bold')

    # 3. Actual vs Predicted for Best Model
    best_model_name = min(results.keys(), key=lambda x: results[x]['rmse'])
    best_predictions = results[best_model_name]['y_pred']

    ax3 = plt.subplot(3, 3, 3)
    ax3.scatter(y_test, best_predictions, alpha=0.6, color='#4ECDC4', s=50)
    ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, alpha=0.8)
    ax3.set_xlabel('Actual Price ($)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Predicted Price ($)', fontsize=12, fontweight='bold')
    ax3.set_title(f'Actual vs Predicted\n{best_model_name}', fontsize=14, fontweight='bold')

    # Add R² to the plot
    r2_best = results[best_model_name]['r2']
    ax3.text(0.05, 0.95, f'R² = {r2_best:.3f}', transform=ax3.transAxes,
             fontsize=12, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # 4. Residual Plot
    ax4 = plt.subplot(3, 3, 4)
    residuals = y_test - best_predictions
    ax4.scatter(best_predictions, residuals, alpha=0.6, color='#FF6B6B', s=50)
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.8)
    ax4.set_xlabel('Predicted Price ($)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Residuals ($)', fontsize=12, fontweight='bold')
    ax4.set_title(f'Residual Plot\n{best_model_name}', fontsize=14, fontweight='bold')

    # 5. Feature Correlation Heatmap
    ax5 = plt.subplot(3, 3, 5)
    correlation_matrix = df[feature_cols + ['price']].corr()
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm',
                center=0, ax=ax5, fmt='.2f', square=True)
    ax5.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')

    # 6. Price Distribution
    ax6 = plt.subplot(3, 3, 6)
    ax6.hist(df['price'], bins=50, alpha=0.7, color='#45B7D1', edgecolor='black')
    ax6.axvline(df['price'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: ${df["price"].mean():,.0f}')
    ax6.axvline(df['price'].median(), color='orange', linestyle='--', linewidth=2, label=f'Median: ${df["price"].median():,.0f}')
    ax6.set_xlabel('House Price ($)', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax6.set_title('House Price Distribution', fontsize=14, fontweight='bold')
    ax6.legend()

    # 7. Feature Importance (for Random Forest)
    if 'Random Forest' in results:
        ax7 = plt.subplot(3, 3, 7)
        rf_model = results['Random Forest']['model']
        importances = rf_model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': importances
        }).sort_values('importance', ascending=True).tail(10)

        ax7.barh(range(len(feature_importance_df)), feature_importance_df['importance'],
                alpha=0.7, color='#96CEB4')
        ax7.set_yticks(range(len(feature_importance_df)))
        ax7.set_yticklabels(feature_importance_df['feature'])
        ax7.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
        ax7.set_title('Top 10 Feature Importance\n(Random Forest)', fontsize=14, fontweight='bold')

    # 8. Price vs Square Footage
    ax8 = plt.subplot(3, 3, 8)
    scatter = ax8.scatter(df['sqft_living'], df['price'], c=df['grade'],
                         cmap='viridis', alpha=0.6, s=50)
    ax8.set_xlabel('Square Footage', fontsize=12, fontweight='bold')
    ax8.set_ylabel('Price ($)', fontsize=12, fontweight='bold')
    ax8.set_title('Price vs Square Footage\n(Colored by Grade)', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, ax=ax8, label='Grade')

    # 9. Model Comparison Box Plot
    ax9 = plt.subplot(3, 3, 9)
    error_data = []
    model_names = []
    for name, result in results.items():
        errors = np.abs(y_test - result['y_pred']) / y_test * 100  # Percentage error
        error_data.append(errors)
        model_names.append(name)

    bp = ax9.boxplot(error_data, labels=model_names, patch_artist=True)
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax9.set_ylabel('Absolute Percentage Error (%)', fontsize=12, fontweight='bold')
    ax9.set_title('Model Error Distribution', fontsize=14, fontweight='bold')
    ax9.set_xticklabels(model_names, rotation=45)

    plt.tight_layout(pad=3.0)
    plt.show()

# Main execution function
def main():
    """Execute the complete machine learning pipeline"""
    print("🏠 ADVANCED HOUSE PRICE PREDICTION ANALYSIS 🏠")
    print("=" * 60)

    # Step 1: Create and load data
    print("Creating synthetic housing dataset...")
    df = create_synthetic_housing_data(1000)

    # Step 2: Advanced preprocessing
    df_processed = advanced_preprocessing(df)

    # Step 3: Machine learning pipeline
    results, X_test, y_test, feature_cols = build_ml_pipeline(df_processed)

    # Step 4: Create visualizations
    create_advanced_visualizations(results, X_test, y_test, feature_cols, df_processed)

    # Step 5: Summary
    print("\n" + "=" * 60)
    print("📊 FINAL RESULTS SUMMARY 📊")
    print("=" * 60)

    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  • RMSE: ${result['rmse']:,.2f}")
        print(f"  • MAE: ${result['mae']:,.2f}")
        print(f"  • R²: {result['r2']:.4f}")
        print(f"  • Best Parameters: {result['best_params']}")

    # Find best model
    best_model_name = min(results.keys(), key=lambda x: results[x]['rmse'])
    print(f"\n🏆 BEST MODEL: {best_model_name}")
    print(f"   └─ RMSE: ${results[best_model_name]['rmse']:,.2f}")

    return df_processed, results

# Execute the complete pipeline
if __name__ == "__main__":
    df_final, model_results = main()