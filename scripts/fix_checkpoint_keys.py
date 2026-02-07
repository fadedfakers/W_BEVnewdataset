"""
fix_checkpoint_keys.py - 修复层导入问题，生成可正确加载的 checkpoint

使用基于层级偏移的硬映射，避免形状匹配导致 cls/reg/mask 张冠李戴。
Phase 1 checkpoint (直接 Conv2d) -> Phase 2 模型 (Sequential 内 .2 为输出层)
"""
import os
import sys
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from configs.config import BEVConfig as cfg
from models.detector import WBEVFusionNet

# 硬编码映射：ckpt_key -> model_key，只做名称/层级转换，不依赖形状匹配
# Phase 1: 直接 cls_head/reg_head, seg_head
# Phase 2 模型: cls_head.2/reg_head.2 为输出层, mask_head 与 seg_head 等价
FIX_MAP = {
    # 分割头：ckpt 用 seg_head，部分 Phase 2 模型用 mask_head
    "head.seg_head.0.weight": "head.seg_head.0.weight",
    "head.seg_head.0.bias": "head.seg_head.0.bias",
    "head.seg_head.1.weight": "head.seg_head.1.weight",
    "head.seg_head.1.bias": "head.seg_head.1.bias",
    "head.seg_head.1.running_mean": "head.seg_head.1.running_mean",
    "head.seg_head.1.running_var": "head.seg_head.1.running_var",
    "head.seg_head.1.num_batches_tracked": "head.seg_head.1.num_batches_tracked",
    "head.seg_head.3.weight": "head.seg_head.3.weight",
    "head.seg_head.3.bias": "head.seg_head.3.bias",
    # 检测头：Phase 1 直接输出 -> Phase 2 Sequential 的 .2 输出层
    # 若模型为直接 Conv2d，用下方 USE_DIRECT_HEAD 切换
    "head.cls_head.weight": "head.cls_head.weight",
    "head.cls_head.bias": "head.cls_head.bias",
    "head.reg_head.weight": "head.reg_head.weight",
    "head.reg_head.bias": "head.reg_head.bias",
}

# 若目标模型使用 Sequential cls_head/reg_head（.0, .2 结构），启用此映射
FIX_MAP_SEQUENTIAL_HEAD = {
    "head.cls_head.weight": "head.cls_head.2.weight",
    "head.cls_head.bias": "head.cls_head.2.bias",
    "head.reg_head.weight": "head.reg_head.2.weight",
    "head.reg_head.bias": "head.reg_head.2.bias",
}

# 若目标模型用 mask_head 而非 seg_head
FIX_MAP_MASK_HEAD = {
    "head.seg_head.0.weight": "head.mask_head.0.weight",
    "head.seg_head.0.bias": "head.mask_head.0.bias",
    "head.seg_head.1.weight": "head.mask_head.1.weight",
    "head.seg_head.1.bias": "head.mask_head.1.bias",
    "head.seg_head.1.running_mean": "head.mask_head.1.running_mean",
    "head.seg_head.1.running_var": "head.mask_head.1.running_var",
    "head.seg_head.1.num_batches_tracked": "head.mask_head.1.num_batches_tracked",
    "head.seg_head.3.weight": "head.mask_head.3.weight",
    "head.seg_head.3.bias": "head.mask_head.3.bias",
}


def get_fix_map(use_sequential_head=True, use_mask_head=True):
    """获取合并后的映射表，供 debug 等脚本使用"""
    m = dict(FIX_MAP)
    if use_sequential_head:
        m.update(FIX_MAP_SEQUENTIAL_HEAD)
    if use_mask_head:
        m.update(FIX_MAP_MASK_HEAD)
    return m


def fix_checkpoint(ckpt_path: str, output_path: str = None, use_sequential_head: bool = True, use_mask_head: bool = True):
    """
    use_sequential_head: Phase 2 模型 cls/reg_head 为 Sequential，输出在 .2 层
    use_mask_head: Phase 2 模型用 mask_head 而非 seg_head
    """
    print(f"🛠️ 正在修复权重文件: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint.get('state_dict', checkpoint)))
    if not isinstance(state_dict, dict):
        print("❌ checkpoint 中未找到有效的 state_dict")
        return False

    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    # 合并映射表
    fix_map = dict(FIX_MAP)
    if use_sequential_head:
        fix_map.update(FIX_MAP_SEQUENTIAL_HEAD)
    if use_mask_head:
        fix_map.update(FIX_MAP_MASK_HEAD)

    new_state_dict = {}
    mapping_log = []

    for ckpt_k, v in state_dict.items():
        model_k = fix_map.get(ckpt_k, ckpt_k)

        # 只做 key 变换，不做形状校验（避免 cls/reg 互换）
        new_state_dict[model_k] = v
        if model_k != ckpt_k:
            mapping_log.append((ckpt_k, model_k))

    # 校验：当输出兼容当前模型时尝试加载（use_sequential_head/use_mask_head 会改变 key，可能不兼容）
    model = WBEVFusionNet(cfg)
    try:
        msg = model.load_state_dict(new_state_dict, strict=False)
        if msg.missing_keys:
            print(f"\n⚠️ 当前模型缺失 {len(msg.missing_keys)} 层（输出可能面向 use_sequential_head/use_mask_head 模型）")
    except Exception:
        pass

    fix_count = len(mapping_log)
    print(f"✅ 修复完成！")
    print(f"📊 总计重命名: {fix_count} 层 | 共 {len(new_state_dict)} 层")
    if mapping_log:
        print("\n📋 Key 映射记录:")
        for a, b in mapping_log[:20]:
            print(f"   {a} -> {b}")
        if len(mapping_log) > 20:
            print(f"   ... 及 {len(mapping_log) - 20} 条")

    out = output_path or ckpt_path.replace('.pth', '_fixed.pth')
    save_ckpt = {
        'model_state_dict': new_state_dict,
        **{k: v for k, v in checkpoint.items() if k not in ('model_state_dict', 'model', 'state_dict')},
    }
    torch.save(save_ckpt, out)
    print(f"\n💾 已保存: {out}")
    return True


def _get_ckpt_dir():
    if os.path.exists(cfg.CKPT_DIR):
        return cfg.CKPT_DIR
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "checkpoints")


if __name__ == "__main__":
    import glob

    pos_args = [a for a in sys.argv[1:] if not a.startswith("--")]

    ckpt_dir = _get_ckpt_dir()
    ckpt_path = pos_args[0] if len(pos_args) >= 1 else None
    out_path = pos_args[1] if len(pos_args) >= 2 else None

    if not ckpt_path:
        for name in ["model_e80.pth", "model_e60.pth"]:
            cand = os.path.join(ckpt_dir, name)
            if os.path.exists(cand):
                ckpt_path = cand
                break
        if not ckpt_path:
            files = glob.glob(os.path.join(ckpt_dir, "*.pth"))
            if files:
                ckpt_path = max(files, key=os.path.getctime)

    if not ckpt_path or not os.path.exists(ckpt_path):
        print("用法: python fix_checkpoint_keys.py <checkpoint_path> [output_path] [--sequential-head] [--mask-head]")
        print("  --no-sequential-head  禁用 cls/reg -> .2 映射（目标为直接 Conv2d）")
        print("  --no-mask-head        禁用 seg_head -> mask_head 映射")
        sys.exit(1)

    use_seq = "--no-sequential-head" not in sys.argv  # 默认 True
    use_mask = "--no-mask-head" not in sys.argv  # 默认 True（目标模型用 mask_head）
    fix_checkpoint(ckpt_path, out_path, use_sequential_head=use_seq, use_mask_head=use_mask)
