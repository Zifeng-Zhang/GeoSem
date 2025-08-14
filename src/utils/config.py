import yaml
from typing import Dict, List, Any
import os


class ConfigLoader:
    """统一的配置加载器，处理includes和路径"""

    @staticmethod
    def load_config(config_path: str, base_path: str = None) -> Dict[str, Any]:
        """
        加载配置文件，递归处理includes

        Args:
            config_path: 配置文件路径
            base_path: 基础路径，用于解析相对路径
        """
        if base_path is None:
            base_path = os.path.dirname(config_path)

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # 处理includes
        if 'includes' in config:
            includes = config.pop('includes')
            base_config = {}

            for include_path in includes:
                # 处理相对路径
                if not os.path.isabs(include_path):
                    include_path = os.path.join(base_path, include_path)

                included = ConfigLoader.load_config(include_path, base_path)
                base_config = ConfigLoader.deep_merge(base_config, included)

            # 合并当前配置
            config = ConfigLoader.deep_merge(base_config, config)

        return config

    @staticmethod
    def deep_merge(dict1: Dict, dict2: Dict) -> Dict:
        """深度合并两个字典"""
        result = dict1.copy()
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigLoader.deep_merge(result[key], value)
            else:
                result[key] = value
        return result


class ConfigValidator:
    """配置验证器，确保所需的键都存在"""

    @staticmethod
    def validate_trainer_config(cfg: Dict[str, Any]) -> None:
        """验证trainer所需的配置"""
        required_paths = [
            "model.vggt",
            "model.lifter",
            "model.clip_teacher",
            "model.clip.embed_dim",
            "model.projector",
            "optim.lr",
            "optim.weight_decay",
            "loss.temperature",
            "general.amp_dtype",
            "logging"
        ]

        for path in required_paths:
            keys = path.split('.')
            current = cfg
            for key in keys:
                if key not in current:
                    raise ValueError(f"Missing required config key: {path}")
                current = current[key]


if __name__ == '__main__':
    cfg_path = "../../configs/exp/poc_scannetpp.yaml"
    base_path = "../../"
    cfg = ConfigLoader.load_config(cfg_path, base_path)
    ConfigValidator.validate_trainer_config(cfg)

