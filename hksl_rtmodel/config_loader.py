import yaml
import os
import warnings

# Determine project root relative to this file's location
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, 'configs', 'config.yaml')

def load_config(config_path=DEFAULT_CONFIG_PATH):
    """Loads configuration from a YAML file and resolves relative paths."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        config['project_root'] = PROJECT_ROOT

        def resolve_path(base_config, key_path):
            value = base_config
            keys = key_path.split('.')
            valid_path = True
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    valid_path = False
                    break

            if not valid_path:
                 warnings.warn(f"Configuration key '{key_path}' not found. Skipping path resolution.")
                 return None

            # Resolve if it's a relative string path
            if isinstance(value, str) and not os.path.isabs(value):
                resolved = os.path.join(PROJECT_ROOT, value)
                # Update the config dict in place
                d = base_config
                for key in keys[:-1]:
                    d = d[key]
                d[keys[-1]] = resolved
                return resolved
            return value # Return original if absolute or not a string path

        # List of paths to resolve relative to project root
        paths_to_resolve = [
            'data.dataset_dir',
            'model.save_dir',
            'training.log_dir'
        ]
        for p in paths_to_resolve:
            resolve_path(config, p)

        # Construct derived paths (model filenames)
        model_cfg = config.get('model', {})
        model_name = model_cfg.get('name', 'default_model')
        model_dir = model_cfg.get('save_dir', os.path.join(PROJECT_ROOT, 'models'))

        # Ensure model_dir exists
        if not os.path.exists(model_dir):
             os.makedirs(model_dir, exist_ok=True)
             print(f"Created model save directory: {model_dir}")

        config['model']['model_save_path'] = os.path.join(model_dir, f'sign_language_model_{model_name}.keras')
        config['model']['label_encoder_path'] = os.path.join(model_dir, f'label_encoder_{model_name}.pkl')

        return config
    except Exception as e:
        print(f"Error loading or processing config file {config_path}: {e}")
        raise

# Example usage
if __name__ == '__main__':
    try:
        config = load_config()
        print("Config loaded successfully:")
        import json
        print(json.dumps(config, indent=2))
    except Exception as e:
        print(f"Failed to load config: {e}")