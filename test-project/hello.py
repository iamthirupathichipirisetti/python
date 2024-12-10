import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from account import get_config, SaveData, run_model

def test_get_config():
    """Test the get_config function for different input scenarios."""
    with patch("os.path.exists") as mock_exists:
        mock_exists.return_value = True

        # Mock reading a YAML file
        with patch("builtins.open", MagicMock()) as mock_open:
            with patch("yaml.unsafe_load", return_value={"key": "value"}):
                config = get_config("/dummy/path")

                # Validate the configuration
                mock_open.assert_called_once_with("/dummy/path/static.yml")
                assert config == {"key": "value"}

    # Test fallback scenario
    mock_exists.return_value = False
    with patch("builtins.open", MagicMock()) as mock_open:
        with patch("yaml.unsafe_load", return_value={"fallback": "config"}):
            config = get_config()

            # Validate fallback path
            mock_open.assert_called_once_with("/static.yml")
            assert config == {"fallback": "config"}

@patch("account.DB")
def test_save_method(mock_db):
    """Test SaveData.save method to ensure it interacts correctly with DB."""
    mock_instance = mock_db.get_instance.return_value

    # Mock DataFrame
    df = pd.DataFrame({"col1": [1, 2, 3]})

    # Call the save method
    SaveData.save(df, "test_table", "append")

    # Validate DB interaction
    mock_instance.save_dataframe.assert_called_once_with(df, "test_table", "append")

@patch("account.DB")
def test_load_method(mock_db):
    """Test SaveData.load method to ensure data is fetched correctly."""
    mock_instance = mock_db.get_instance.return_value

    # Mock query result
    mock_instance.query.return_value = pd.DataFrame({"col1": [1, 2, 3]})

    # Call the load method
    result = SaveData.load("SELECT * FROM test_table")

    # Validate query execution and results
    mock_instance.query.assert_called_once_with("SELECT * FROM test_table")
    pd.testing.assert_frame_equal(result, pd.DataFrame({"col1": [1, 2, 3]}))

def test_save_data_initialization():
    """Test initialization of SaveData class."""
    with patch("account.get_config", return_value={"sample": "config"}):
        save_data = SaveData("Canada", "2024-12-10")

        assert save_data.country == "Canada"
        assert save_data.processing_date == pd.Timestamp("2024-12-10")
        assert save_data.cfg == {"sample": "config"}

def test_get_config_missing_file():
    """Test get_config when the configuration file is missing."""
    with patch("os.path.exists", return_value=False):
        with pytest.raises(FileNotFoundError):
            get_config("/nonexistent/path")

def test_save_empty_dataframe():
    """Test SaveData.save with an empty DataFrame."""
    with patch("account.DB.get_instance") as mock_db_instance:
        empty_df = pd.DataFrame()

        SaveData.save(empty_df, "empty_table", "append")

        # Ensure save_dataframe is not called with an empty DataFrame
        mock_db_instance.return_value.save_dataframe.assert_not_called()

def test_load_invalid_query():
    """Test SaveData.load with an invalid SQL query."""
    with patch("account.DB.get_instance") as mock_db_instance:
        mock_db_instance.return_value.query.side_effect = Exception("Invalid SQL")

        with pytest.raises(Exception, match="Invalid SQL"):
            SaveData.load("INVALID QUERY")

def test_db_instance_failure():
    """Test behavior when DB.get_instance() fails."""
    with patch("account.DB.get_instance", side_effect=Exception("DB instance error")):
        with pytest.raises(Exception, match="DB instance error"):
            SaveData.save(pd.DataFrame({"col1": [1]}), "test_table", "append")

def test_process_reference_data():
    """Test _process_reference_data for proper processing and interpolation."""
    save_data = SaveData("Canada", "2024-12-10")
    mock_csv_data = """name,date,code1,term,code2,rate
CADsCAD1dMid,2024-12-10,1,1,USD,0.5
CADsCAD1dMid,2024-12-10,1,3,USD,0.7
"""
    with patch("pandas.read_csv", return_value=pd.read_csv(pd.compat.StringIO(mock_csv_data))) as mock_read_csv:
        result = save_data._process_reference_data("dummy.csv")
        mock_read_csv.assert_called_once()
        assert not result.empty

def test_process_reference_data_empty_file():
    """Test _process_reference_data with an empty file."""
    save_data = SaveData("Canada", "2024-12-10")
    empty_csv = """name,date,code1,term,code2,rate
"""
    with patch("pandas.read_csv", return_value=pd.read_csv(pd.compat.StringIO(empty_csv))) as mock_read_csv:
        result = save_data._process_reference_data("empty.csv")
        mock_read_csv.assert_called_once()
        assert result is None or result.empty

def test_process_reference_data_missing_columns():
    """Test _process_reference_data with missing columns."""
    save_data = SaveData("Canada", "2024-12-10")
    mock_csv_data = """date,code1,term,rate
2024-12-10,1,1,0.5
"""
    with patch("pandas.read_csv", return_value=pd.read_csv(pd.compat.StringIO(mock_csv_data))) as mock_read_csv:
        with pytest.raises(Exception, match="Missing required columns"):
            save_data._process_reference_data("dummy.csv")

def test_process_reference_data_invalid_rate():
    """Test _process_reference_data with invalid rate data."""
    save_data = SaveData("Canada", "2024-12-10")
    mock_csv_data = """name,date,code1,term,code2,rate
CADsCAD1dMid,2024-12-10,1,1,USD,invalid
"""
    with patch("pandas.read_csv", return_value=pd.read_csv(pd.compat.StringIO(mock_csv_data))) as mock_read_csv:
        with pytest.raises(ValueError):
            save_data._process_reference_data("dummy.csv")

@patch("account.os.listdir")
@patch("account.DB")
def test_load_n_save_reference_rate_valid_files(mock_db, mock_listdir):
    """Test _load_n_save_reference_rate with valid files."""
    save_data = SaveData("Canada", "2024-12-10")
    mock_listdir.return_value = ["20241210.csv"]
    mock_db.get_instance().find_last_process_date.return_value = pd.Timestamp("2024-12-09")
    with patch.object(save_data, "_process_reference_data", return_value=pd.DataFrame({"col": [1, 2]})) as mock_process:
        save_data._load_n_save_reference_rate()
        mock_process.assert_called_once()
        mock_db.get_instance().save_dataframe.assert_called_once()

def test_run_model_invalid_country():
    """Test run_model with an invalid country name."""
    with pytest.raises(Exception, match="Invalid country name"):
        run_model("InvalidCountry", "2024-12-10")

def test_run_model_invalid_date():
    """Test run_model with an improperly formatted evaluation date."""
    with pytest.raises(Exception, match="Invalid date format"):
        run_model("Canada", "invalid-date")
