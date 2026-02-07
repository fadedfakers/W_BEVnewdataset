"""
check_missing_keys.py - 26 个 Missing Keys 深度检查脚本

用于排查基座模型（如 model_e80.pth）性能塌陷的原因。
缺失的 key 会导致对应层使用随机初始化，严重影响 mIoU、AP 等指标。
"""
import os
import sys
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from configs.config import BEVConfig as cfg
from models.detector import WBEVFusionNet


def extract_state_dict(checkpoint):
    """兼容多种 checkpoint 格式"""
    if 'model_state_dict' in checkpoint:
        return checkpoint['model_state_dict']
    if 'model' in checkpoint:
        return checkpoint['model']
    if 'state_dict' in checkpoint:
        return checkpoint['state_dict']
    return checkpoint


def check_missing_keys(checkpoint_path, model):
    print(f"🔍 正在对比模型与权重文件: {checkpoint_path}\n")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = extract_state_dict(checkpoint)

    # 移除 module. 前缀（DataParallel 保存格式）
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())

    missing_keys = model_keys - ckpt_keys
    unexpected_keys = ckpt_keys - model_keys

    print("=" * 50)
    print(f"❌ 缺失的 Key ({len(missing_keys)} 个):")
    if missing_keys:
        prefixes = sorted(list(set(k.split('.')[0] for k in missing_keys)))
        for p in prefixes:
            group = [k for k in missing_keys if k.startswith(p + '.') or k == p]
            print(f"\n  [{p}] 模块共缺失 {len(group)} 个参数层:")
            for k in sorted(group)[:5]:
                print(f"    └─ {k}")
            if len(group) > 5:
                print(f"    └─ ... 及 {len(group) - 5} 个")
    else:
        print("  无")

    print("\n" + "=" * 50)
    print(f"❓ 多余的 Key ({len(unexpected_keys)} 个):")
    if unexpected_keys:
        for k in sorted(list(unexpected_keys))[:8]:
            print(f"  - {k}")
        if len(unexpected_keys) > 8:
            print(f"  ... 及 {len(unexpected_keys) - 8} 个")
    else:
        print("  无")
    print("=" * 50)

    # 排查建议
    print("\n📋 排查建议:")
    missing_str = ' '.join(missing_keys)
    if 'seg_head' in missing_str or 'segmentation' in missing_str or 'mask' in missing_str:
        print("  ⚠️ 缺失包含 seg_head / mask 的层 → 分割头可能是随机初始化，会导致 mIoU 极低")
    if 'head.' in missing_str:
        print("  ⚠️ 缺失 head 相关层 → 检测/分割/多项式头未正确加载")
    if 'fusion' in missing_str or 'neck' in missing_str:
        print("  ⚠️ 缺失 fusion/neck 相关层 → 跨模态融合未加载")
    if 'img_backbone' in missing_str or 'lidar_backbone' in missing_str:
        print("  ⚠️ 缺失 backbone 层 → 特征提取器未加载")
    if not missing_keys:
        print("  ✅ 无缺失 key，可进一步检查 shape 不匹配导致的 skipped 层")


def _get_ckpt_dir():
    if os.path.exists(cfg.CKPT_DIR):
        return cfg.CKPT_DIR
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "checkpoints")


if __name__ == "__main__":
    import glob

    ckpt_dir = _get_ckpt_dir()
    CHECKPOINT = None
    if len(sys.argv) > 1:
        CHECKPOINT = sys.argv[1]
    else:
        for name in ["model_e80.pth", "model_e60.pth"]:
            cand = os.path.join(ckpt_dir, name)
            if os.path.exists(cand):
                CHECKPOINT = cand
                break
        if CHECKPOINT is None:
            list_of_files = glob.glob(os.path.join(ckpt_dir, "*.pth"))
            if list_of_files:
                CHECKPOINT = max(list_of_files, key=os.path.getctime)

    if not CHECKPOINT or not os.path.exists(CHECKPOINT):
        print(f"❌ Checkpoint 未找到。用法: python check_missing_keys.py [checkpoint_path]")
        print(f"   已搜索目录: {ckpt_dir}")
        sys.exit(1)

    model = WBEVFusionNet(cfg)
    check_missing_keys(CHECKPOINT, model)
