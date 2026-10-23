"""Load nodes relative to this package, independent of the installation name."""
import importlib
import logging

NODE_CLASS_MAPPINGS = {}
for module in ("Paints-UNDO.ZZX_PaintsUndo", "nodes.ZZX_VFC", "nodes.ZZX_Stream"):
    try:
        loaded = importlib.import_module("." + module, __name__)
        NODE_CLASS_MAPPINGS.update(loaded.NODE_CLASS_MAPPINGS)
    except ImportError:
        logging.exception("Unable to load optional ZZX node module %s", module)
