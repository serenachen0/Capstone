# test_unirep64.py
import os
from config.model_config import BaseConfig, UniRepModelConfig

def test_base_config():
    print("\n=== Testing BaseConfig ===")
    # Create a BaseConfig instance
    config = BaseConfig(param1=42, param2="test")

    # Test to_dict
    config_dict = config.to_dict()
    print(f"Config dictionary: {config_dict}")
    assert config_dict["param1"] == 42, "param1 value mismatch"
    assert config_dict["param2"] == "test", "param2 value mismatch"

    # Test save and load
    temp_path = "temp_config.json"
    config.save(temp_path)
    print(f"Config saved to {temp_path}")

    loaded_config = BaseConfig.load(temp_path)
    print(f"Loaded config: {loaded_config.to_dict()}")
    assert loaded_config.param1 == 42, "Loaded param1 value mismatch"
    assert loaded_config.param2 == "test", "Loaded param2 value mismatch"

    # Clean up
    os.remove(temp_path)
    print("BaseConfig tests passed!")

def test_unirep_model_config():
    print("\n=== Testing UniRepModelConfig ===")
    # Create a UniRepModelConfig instance
    config = UniRepModelConfig(
        rnn_size=64,
        embed_dim=10,
        num_layers=4,
        batch_size=256,
        dropout_rate=0.5,
        model_path="./weights"
    )

    # Test validate
    print("Validating configuration...")
    assert config.validate(), "Validation failed for valid configuration"

    # Test invalid configurations
    try:
        config.rnn_size = -1
        config.validate()
    except ValueError as e:
        print(f"Expected error for invalid rnn_size: {e}")
    else:
        assert False, "Validation did not fail for invalid rnn_size"

    try:
        config.embed_dim = 0
        config.validate()
    except ValueError as e:
        print(f"Expected error for invalid embed_dim: {e}")
    else:
        assert False, "Validation did not fail for invalid embed_dim"

    try:
        config.num_layers = -1
        config.validate()
    except ValueError as e:
        print(f"Expected error for invalid num_layers: {e}")
    else:
        assert False, "Validation did not fail for invalid num_layers"

    try:
        config.batch_size = 0
        config.validate()
    except ValueError as e:
        print(f"Expected error for invalid batch_size: {e}")
    else:
        assert False, "Validation did not fail for invalid batch_size"

    try:
        config.dropout_rate = 1.5
        config.validate()
    except ValueError as e:
        print(f"Expected error for invalid dropout_rate: {e}")
    else:
        assert False, "Validation did not fail for invalid dropout_rate"

    # Reset config to valid values
    config = UniRepModelConfig(
        rnn_size=64,
        embed_dim=10,
        num_layers=4,
        batch_size=256,
        dropout_rate=0.5,
        model_path="./weights"
    )

    # Test to_dict and from_dict
    config_dict = config.to_dict()
    print(f"Config dictionary: {config_dict}")
    new_config = UniRepModelConfig.from_dict(config_dict)
    print(f"New config from dictionary: {new_config.to_dict()}")
    assert new_config.rnn_size == 64, "rnn_size mismatch"
    assert new_config.embed_dim == 10, "embed_dim mismatch"
    assert new_config.num_layers == 4, "num_layers mismatch"
    assert new_config.batch_size == 256, "batch_size mismatch"
    assert new_config.dropout_rate == 0.5, "dropout_rate mismatch"
    assert new_config.model_path == "./weights", "model_path mismatch"

    # Test save and load
    temp_path = "temp_unirep_config.json"
    config.save(temp_path)
    print(f"Config saved to {temp_path}")

    loaded_config = UniRepModelConfig.load(temp_path)
    print(f"Loaded config: {loaded_config.to_dict()}")
    assert loaded_config.rnn_size == 64, "Loaded rnn_size mismatch"
    assert loaded_config.embed_dim == 10, "Loaded embed_dim mismatch"
    assert loaded_config.num_layers == 4, "Loaded num_layers mismatch"
    assert loaded_config.batch_size == 256, "Loaded batch_size mismatch"
    assert loaded_config.dropout_rate == 0.5, "Loaded dropout_rate mismatch"
    assert loaded_config.model_path == "./weights", "Loaded model_path mismatch"

    # Clean up
    os.remove(temp_path)
    print("UniRepModelConfig tests passed!")

def run_all_tests():
    print("\n=== Running all tests ===")
    test_base_config()
    test_unirep_model_config()
    print("\n✅ All tests passed!")

if __name__ == "__main__":
    run_all_tests()