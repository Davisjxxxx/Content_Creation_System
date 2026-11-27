"""Load application modules from the verified desktop archive mounted read-only at /rc."""
import importlib.abc
import importlib.util
from pathlib import Path
import sys

from PyInstaller.archive.readers import CArchiveReader

archive = CArchiveReader("/rc/AvatarV2").open_embedded_archive("PYZ.pyz")


class ArchiveModules(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path=None, target=None):
        if name in archive.toc and (name == "avatar_v2" or name.startswith("avatar_v2.")):
            return importlib.util.spec_from_loader(name, self, is_package=archive.toc[name][0] == 1)
        return None

    def exec_module(self, module):
        code = archive.extract(module.__name__)
        module.__file__ = str(Path("/rc/_internal") / code.co_filename)
        exec(code, module.__dict__)


sys.meta_path.insert(0, ArchiveModules())
sys._MEIPASS = "/rc/_internal"
sys.frozen = True
