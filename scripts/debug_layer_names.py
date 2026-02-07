"""
debug_layer_names.py - 模型与权重层对比，排查映射错位

并排打印 model 与 checkpoint 的 key，一眼看出 head 部分哪里映射歪了。
支持 --mapping 模式：对原始 checkpoint 模拟 fix 逻辑，打印 head 的映射关系。
"""
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from models.detector import WBEVFusionNet
from configs.config import BEVConfig as cfg


def extract_state_dict(checkpoint):
    if 'model_state_dict' in checkpoint:
        return checkpoint['model_state_dict']
    if 'model' in checkpoint:
        return checkpoint['model']
    if 'state_dict' in checkpoint:
        return checkpoint['state_dict']
    return checkpoint


def debug_layer_names(ckpt_path, show_mapping=False):
    model = WBEVFusionNet(cfg)
    model_keys = sorted(model.state_dict().keys())
    model_dict = model.state_dict()

    checkpoint = torch.load(ckpt_path, map_location='cpu')
    ckpt_sd = extract_state_dict(checkpoint)
    ckpt_sd = {k.replace('module.', ''): v for k, v in ckpt_sd.items()}
    ckpt_keys = sorted(ckpt_sd.keys())

    if show_mapping:
        # 模拟 fix 逻辑（与 fix_checkpoint_keys 一致）
        scripts_dir = os.path.dirname(os.path.abspath(__file__))
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        from fix_checkpoint_keys import get_fix_map
        fix_map = get_fix_map(use_sequential_head=True, use_mask_head=True)
        print("\n" + "=" * 80)
        print("HEAD 映射关系 (ckpt_key -> model_key, 基于 FIX_MAP)")
        print("=" * 80)
        for ck in sorted(ckpt_keys):
            if "head" not in ck:
                continue
            mk = fix_map.get(ck, ck)
            shape_ok = ""
            if mk in model_dict:
                shape_ok = " ✓" if ckpt_sd[ck].shape == model_dict[mk].shape else " ✗ shape mismatch"
            else:
                shape_ok = " (model 无此 key)"
            print(f"  {ck}")
            print(f"    -> {mk}{shape_ok}")
        print("=" * 80 + "\n")
        return

    w = 52
    print(f"\n{'=' * (w * 2 + 5)}")
    print(f"{'CURRENT MODEL KEYS':<{w}} | {'CHECKPOINT KEYS':<{w}}")
    print(f"{'-' * (w * 2 + 5)}")

    for i in range(max(len(model_keys), len(ckpt_keys))):
        m_k = model_keys[i] if i < len(model_keys) else "---"
        c_k = ckpt_keys[i] if i < len(ckpt_keys) else "---"

        tag = " <-- HEAD" if "head" in str(m_k) or "head" in str(c_k) else ""
        print(f"{m_k[:w]:<{w}} | {c_k[:w]:<{w}}{tag}")

    print(f"{'=' * (w * 2 + 5)}\n")
    print(f"Model 共 {len(model_keys)} 层, Checkpoint 共 {len(ckpt_keys)} 层")


def _get_ckpt_dir():
    if os.path.exists(cfg.CKPT_DIR):
        return cfg.CKPT_DIR
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "checkpoints")


if __name__ == "__main__":
    ckpt_dir = _get_ckpt_dir()
    args = sys.argv[1:]
    show_mapping = "--mapping" in args
    args = [a for a in args if a != "--mapping"]
    ckpt_path = args[0] if args else os.path.join(ckpt_dir, "model_e80.pth")

    if not os.path.exists(ckpt_path):
        print(f"❌ 文件不存在: {ckpt_path}")
        print("用法: python debug_layer_names.py [--mapping] [checkpoint_path]")
        print("  --mapping  对原始 ckpt 模拟映射，打印 head 的 ckpt_key -> model_key")
        sys.exit(1)

    debug_layer_names(ckpt_path, show_mapping=show_mapping)
