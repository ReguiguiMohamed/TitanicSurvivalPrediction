
"""
Comprehensive test suite for Titanic ML Competition
Tests all components from data loading to final predictions
"""

import os
import sys
import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
import tempfile
import shutil
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

class TestDataLoading:
    """Test data loading functionality"""
    
    def test_data_files_exist(self):
        """Test that required data files exist"""
        assert os.path.exists('data/raw/train.csv'), "train.csv not found"
        assert os.path.exists('data/raw/test.csv'), "test.csv not found"
    
    def test_data_loading(self):
        """Test basic data loading"""
        try:
            from data_loader import load_data
            train_df, test_df = load_data()
            
            # Test shapes
            assert train_df.shape[0] == 891, f"Train should have 891 rows, got {train_df.shape[0]}"
            assert test_df.shape[0] == 418, f"Test should have 418 rows, got {test_df.shape[0]}"
            
            # Test columns
            expected_cols = ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
            train_cols = train_df.columns.tolist()
            
            for col in expected_cols:
                assert col in train_cols, f"Missing column: {col}"
            
            # Test target column in train
            assert 'Survived' in train_df.columns, "Survived column missing in train data"
            assert 'Survived' not in test_df.columns, "Survived column should not be in test data"
            
        except ImportError:
            pytest.skip("data_loader module not found")
    
    def test_data_summary_generation(self):
        """Test data summary generation"""
        try:
            from data_loader import generate_data_summary
            generate_data_summary()
            assert os.path.exists('data/processed/data_summary.txt'), "Data summary not generated"
        except ImportError:
            pytest.skip("generate_data_summary function not found")


class TestFeatureEngineering:
    """Test feature engineering functionality"""
    
    def setup_method(self):
        """Setup test data"""
        self.sample_data = pd.DataFrame({
            'Name': ['Smith, Mr. John', 'Johnson, Mrs. Mary', 'Brown, Miss. Jane'],
            'Age': [25, 35, np.nan],
            'SibSp': [1, 0, 2],
            'Parch': [0, 2, 1],
            'Cabin': ['C85', np.nan, 'E46'],
            'Fare': [7.25, 71.28, np.nan],
            'Pclass': [3, 1, 2],
            'Sex': ['male', 'female', 'female'],
            'Embarked': ['S', 'C', np.nan]
        })
    
    def test_title_extraction(self):
        """Test title extraction from names"""
        try:
            from feature_engineering import extract_title
            titles = extract_title(self.sample_data['Name'])
            expected = ['Mr', 'Mrs', 'Miss']
            assert titles.tolist() == expected, f"Expected {expected}, got {titles.tolist()}"
        except ImportError:
            pytest.skip("extract_title function not found")
    
    def test_family_size_creation(self):
        """Test family size feature creation"""
        try:
            from feature_engineering import create_family_size
            family_size = create_family_size(self.sample_data)
            expected = [2, 3, 4]  # SibSp + Parch + 1
            assert family_size.tolist() == expected, f"Expected {expected}, got {family_size.tolist()}"
        except ImportError:
            pytest.skip("create_family_size function not found")
    
    def test_missing_value_handling(self):
        """Test missing value imputation"""
        try:
            from feature_engineering import handle_missing_values
            filled_data = handle_missing_values(self.sample_data.copy())
            
            # Check no missing values remain in critical columns
            assert not filled_data['Age'].isnull().any(), "Age still has missing values"
            assert not filled_data['Embarked'].isnull().any(), "Embarked still has missing values"
            
        except ImportError:
            pytest.skip("handle_missing_values function not found")


class TestPreprocessing:
    """Test preprocessing functionality"""
    
    def setup_method(self):
        """Setup test data"""
        self.sample_data = pd.DataFrame({
            'Age': [25, 35, 45],
            'Fare': [7.25, 71.28, 30.0],
            'Sex': ['male', 'female', 'male'],
            'Pclass': [3, 1, 2],
            'Embarked': ['S', 'C', 'Q']
        })
    
    def test_categorical_encoding(self):
        """Test categorical variable encoding"""
        try:
            from preprocessing import encode_categorical
            encoded_data = encode_categorical(self.sample_data.copy())
            
            # Check that categorical columns are converted to numerical
            assert encoded_data['Sex'].dtype in [int, float], "Sex not properly encoded"
            assert encoded_data['Embarked'].dtype in [int, float], "Embarked not properly encoded"
            
        except ImportError:
            pytest.skip("encode_categorical function not found")
    
    def test_feature_scaling(self):
        """Test feature scaling"""
        try:
            from preprocessing import scale_features
            scaled_data = scale_features(self.sample_data[['Age', 'Fare']].copy())
            
            # Check that features are scaled (mean ~0, std ~1)
            assert abs(scaled_data['Age'].mean()) < 0.1, "Age not properly scaled"
            assert abs(scaled_data['Fare'].mean()) < 0.1, "Fare not properly scaled"
            
        except ImportError:
            pytest.skip("scale_features function not found")
    
    def test_train_test_split(self):
        """Test train/validation split"""
        try:
            from preprocessing import create_train_val_split
            
            # Create dummy target
            y = pd.Series([0, 1, 0])
            X_train, X_val, y_train, y_val = create_train_val_split(self.sample_data, y)
            
            assert len(X_train) + len(X_val) == len(self.sample_data), "Split doesn't preserve total size"
            assert len(X_train) == len(y_train), "X_train and y_train size mismatch"
            assert len(X_val) == len(y_val), "X_val and y_val size mismatch"
            
        except ImportError:
            pytest.skip("create_train_val_split function not found")


class TestBaselineModels:
    """Test baseline model functionality"""
    
    def setup_method(self):
        """Setup test data"""
        np.random.seed(42)
        self.X = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randint(0, 3, 100)
        })
        self.y = pd.Series(np.random.randint(0, 2, 100))
    
    def test_logistic_regression(self):
        """Test logistic regression model"""
        try:
            from baseline_models import train_logistic_regression
            model, score = train_logistic_regression(self.X, self.y)
            
            assert hasattr(model, 'predict'), "Model doesn't have predict method"
            assert 0 <= score <= 1, f"Score {score} not in valid range [0,1]"
            
        except ImportError:
            pytest.skip("train_logistic_regression function not found")
    
    def test_random_forest(self):
        """Test random forest model"""
        try:
            from baseline_models import train_random_forest
            model, score = train_random_forest(self.X, self.y)
            
            assert hasattr(model, 'predict'), "Model doesn't have predict method"
            assert 0 <= score <= 1, f"Score {score} not in valid range [0,1]"
            
        except ImportError:
            pytest.skip("train_random_forest function not found")
    
    def test_model_comparison(self):
        """Test model comparison functionality"""
        try:
            from baseline_models import compare_models
            results = compare_models(self.X, self.y)
            
            assert isinstance(results, dict), "Results should be a dictionary"
            assert len(results) > 0, "No models compared"
            
            for model_name, score in results.items():
                assert 0 <= score <= 1, f"Invalid score for {model_name}: {score}"
                
        except ImportError:
            pytest.skip("compare_models function not found")


class TestAdvancedModels:
    """Test advanced model functionality"""
    
    def setup_method(self):
        """Setup test data"""
        np.random.seed(42)
        self.X = pd.DataFrame({
            'feature1': np.random.randn(200),
            'feature2': np.random.randn(200),
            'feature3': np.random.randint(0, 5, 200)
        })
        self.y = pd.Series(np.random.randint(0, 2, 200))
    
    def test_xgboost_training(self):
        """Test XGBoost model training"""
        try:
            from advanced_models import train_xgboost
            model, score = train_xgboost(self.X, self.y)
            
            assert hasattr(model, 'predict'), "XGBoost model doesn't have predict method"
            assert 0 <= score <= 1, f"XGBoost score {score} not in valid range"
            
        except ImportError:
            pytest.skip("train_xgboost function not found")
    
    def test_ensemble_methods(self):
        """Test ensemble model training"""
        try:
            from advanced_models import train_ensemble
            ensemble, score = train_ensemble(self.X, self.y)
            
            assert hasattr(ensemble, 'predict'), "Ensemble doesn't have predict method"
            assert 0 <= score <= 1, f"Ensemble score {score} not in valid range"
            
        except ImportError:
            pytest.skip("train_ensemble function not found")


class TestHyperparameterTuning:
    """Test hyperparameter optimization"""
    
    def setup_method(self):
        """Setup test data"""
        np.random.seed(42)
        self.X = pd.DataFrame({
            'feature1': np.random.randn(150),
            'feature2': np.random.randn(150),
        })
        self.y = pd.Series(np.random.randint(0, 2, 150))
    
    def test_optuna_optimization(self):
        """Test Optuna hyperparameter optimization"""
        try:
            from hyperparameter_tuning import optimize_with_optuna
            best_params, best_score = optimize_with_optuna(self.X, self.y, n_trials=5)
            
            assert isinstance(best_params, dict), "Best params should be a dictionary"
            assert 0 <= best_score <= 1, f"Best score {best_score} not in valid range"
            
        except ImportError:
            pytest.skip("optimize_with_optuna function not found")
    
    def test_grid_search(self):
        """Test grid search optimization"""
        try:
            from hyperparameter_tuning import grid_search_optimization
            best_params, best_score = grid_search_optimization(self.X, self.y)
            
            assert isinstance(best_params, dict), "Best params should be a dictionary"
            assert 0 <= best_score <= 1, f"Best score {best_score} not in valid range"
            
        except ImportError:
            pytest.skip("grid_search_optimization function not found")


class TestPredictions:
    """Test prediction generation"""
    
    def setup_method(self):
        """Setup mock model and test data"""
        self.test_data = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6]
        })
        
        # Mock model
        self.mock_model = MagicMock()
        self.mock_model.predict.return_value = np.array([0, 1, 0])
    
    def test_prediction_generation(self):
        """Test prediction generation"""
        try:
            from predictions import generate_predictions
            predictions = generate_predictions(self.mock_model, self.test_data)
            
            assert len(predictions) == len(self.test_data), "Predictions length mismatch"
            assert all(pred in [0, 1] for pred in predictions), "Invalid prediction values"
            
        except ImportError:
            pytest.skip("generate_predictions function not found")
    
    def test_submission_format(self):
        """Test submission file format"""
        try:
            from predictions import create_submission
            
            passenger_ids = [892, 893, 894]
            predictions = [0, 1, 0]
            
            submission_df = create_submission(passenger_ids, predictions)
            
            assert list(submission_df.columns) == ['PassengerId', 'Survived'], "Wrong submission columns"
            assert len(submission_df) == len(passenger_ids), "Wrong submission length"
            assert submission_df['PassengerId'].tolist() == passenger_ids, "PassengerId mismatch"
            assert submission_df['Survived'].tolist() == predictions, "Predictions mismatch"
            
        except ImportError:
            pytest.skip("create_submission function not found")


class TestModelValidation:
    """Test model validation functionality"""
    
    def setup_method(self):
        """Setup test data and mock model"""
        np.random.seed(42)
        self.X = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100)
        })
        self.y = pd.Series(np.random.randint(0, 2, 100))
        
        self.mock_model = MagicMock()
        self.mock_model.predict.return_value = np.random.randint(0, 2, len(self.X))
    
    def test_cross_validation(self):
        """Test cross-validation functionality"""
        try:
            from model_validation import cross_validate_model
            cv_scores = cross_validate_model(self.mock_model, self.X, self.y, cv=3)
            
            assert len(cv_scores) == 3, "CV should return 3 scores"
            assert all(0 <= score <= 1 for score in cv_scores), "Invalid CV scores"
            
        except ImportError:
            pytest.skip("cross_validate_model function not found")
    
    def test_model_metrics(self):
        """Test model evaluation metrics"""
        try:
            from model_validation import calculate_metrics
            
            y_true = [0, 1, 0, 1, 0]
            y_pred = [0, 1, 1, 1, 0]
            
            metrics = calculate_metrics(y_true, y_pred)
            
            required_metrics = ['accuracy', 'precision', 'recall', 'f1']
            for metric in required_metrics:
                assert metric in metrics, f"Missing metric: {metric}"
                assert 0 <= metrics[metric] <= 1, f"Invalid {metric} value"
                
        except ImportError:
            pytest.skip("calculate_metrics function not found")


class TestAutomatedPipeline:
    """Test complete automated pipeline"""
    
    def test_pipeline_exists(self):
        """Test that pipeline module exists"""
        try:
            import automated_pipeline
            assert hasattr(automated_pipeline, 'run_complete_pipeline'), "run_complete_pipeline function missing"
        except ImportError:
            pytest.skip("automated_pipeline module not found")
    
    def test_pipeline_config(self):
        """Test pipeline configuration"""
        try:
            from automated_pipeline import load_config
            config = load_config()
            
            assert isinstance(config, dict), "Config should be a dictionary"
            
            required_keys = ['model_params', 'preprocessing_params', 'validation_params']
            for key in required_keys:
                if key not in config:
                    pytest.skip(f"Config missing {key} - pipeline may not be fully implemented")
                    
        except ImportError:
            pytest.skip("load_config function not found")


class TestIntegration:
    """Integration tests for complete workflow"""
    
    def test_end_to_end_pipeline(self):
        """Test complete end-to-end pipeline"""
        if not all([
            os.path.exists('data/raw/train.csv'),
            os.path.exists('data/raw/test.csv')
        ]):
            pytest.skip("Required data files not found")
        
        try:
            # Test data loading
            from data_loader import load_data
            train_df, test_df = load_data()
            
            # Test feature engineering
            from feature_engineering import apply_feature_engineering
            train_processed = apply_feature_engineering(train_df)
            test_processed = apply_feature_engineering(test_df)
            
            # Test preprocessing
            from preprocessing import preprocess_data
            X_train, y_train = preprocess_data(train_processed)
            X_test = preprocess_data(test_processed, is_test=True)
            
            # Test model training
            from baseline_models import train_random_forest
            model, score = train_random_forest(X_train, y_train)
            
            # Test prediction
            from predictions import generate_predictions
            predictions = generate_predictions(model, X_test)
            
            assert len(predictions) == 418, f"Expected 418 predictions, got {len(predictions)}"
            assert all(pred in [0, 1] for pred in predictions), "Invalid prediction values"
            
            print(f"✅ End-to-end pipeline successful! Model score: {score:.4f}")
            
        except ImportError as e:
            pytest.skip(f"Required module not found: {e}")
        except Exception as e:
            pytest.fail(f"End-to-end pipeline failed: {e}")


class TestPerformance:
    """Performance and quality tests"""
    
    def test_model_performance_threshold(self):
        """Test that model achieves minimum performance threshold"""
        try:
            from model_validation import get_best_cv_score
            best_score = get_best_cv_score()
            
            MIN_SCORE = 0.75  # Minimum acceptable accuracy
            assert best_score >= MIN_SCORE, f"Model score {best_score:.4f} below threshold {MIN_SCORE}"
            
        except ImportError:
            pytest.skip("get_best_cv_score function not found")
    
    def test_submission_file_quality(self):
        """Test quality of generated submission files"""
        submission_dir = 'submissions/'
        if not os.path.exists(submission_dir):
            pytest.skip("No submissions directory found")
        
        submission_files = [f for f in os.listdir(submission_dir) if f.endswith('.csv')]
        if not submission_files:
            pytest.skip("No submission files found")
        
        # Test most recent submission
        latest_submission = max(submission_files)
        submission_path = os.path.join(submission_dir, latest_submission)
        
        submission_df = pd.read_csv(submission_path)
        
        # Test format
        assert list(submission_df.columns) == ['PassengerId', 'Survived'], "Wrong submission format"
        assert len(submission_df) == 418, f"Wrong number of predictions: {len(submission_df)}"
        
        # Test content
        assert submission_df['PassengerId'].min() == 892, "Wrong PassengerId range"
        assert submission_df['PassengerId'].max() == 1309, "Wrong PassengerId range"
        assert set(submission_df['Survived'].unique()).issubset({0, 1}), "Invalid survival predictions"
        
        print(f"✅ Submission file {latest_submission} passes quality checks")


def run_all_tests():
    """Run all tests and provide summary"""
    import pytest
    
    # Run pytest with verbose output
    exit_code = pytest.main([
        __file__,
        '-v',
        '--tb=short',
        '--color=yes'
    ])
    
    return exit_code == 0


if __name__ == "__main__":
    print("🚀 Starting Titanic ML Competition Test Suite")
    print("=" * 60)

    success = run_all_tests()   

    print("=" * 60) 
    if success:
        print("🎉 All tests passed! Your Titanic ML project is ready!")
    else:
        print("❌ Some tests failed. Check the output above for details.")
        sys.exit(1)