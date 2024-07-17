import os
import importlib

NODE_CLASS_MAPPINGS = {}

# 自动导入 nodes 文件夹中的所有符合条件的节点
nodes_folder = os.path.dirname(__file__) + os.sep + 'nodes'
for node in os.listdir(nodes_folder):
    if node.startswith('ZZX_') and node.endswith('.py'):
        node = node.split('.')[0]
        node_import = importlib.import_module('custom_nodes.ComfyUI-ZZXYWQ.nodes.' + node)
        print('Imported node from nodes: ' + node)
        # 获取节点类映射并更新全局 NODE_CLASS_MAPPINGS
        NODE_CLASS_MAPPINGS.update(node_import.NODE_CLASS_MAPPINGS)

# 自动导入 Paints-UNDO 文件夹中的所有符合条件的节点
paints_undo_folder = os.path.dirname(__file__) + os.sep + 'Paints-UNDO'
for node in os.listdir(paints_undo_folder):
    if node.startswith('ZZX_') and node.endswith('.py'):
        node = node.split('.')[0]
        node_import = importlib.import_module('custom_nodes.ComfyUI-ZZXYWQ.Paints-UNDO.' + node)
        print('Imported node from Paints-UNDO: ' + node)
        # 获取节点类映射并更新全局 NODE_CLASS_MAPPINGS
        NODE_CLASS_MAPPINGS.update(node_import.NODE_CLASS_MAPPINGS)
