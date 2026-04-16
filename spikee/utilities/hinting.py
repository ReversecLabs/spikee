from typing import Union, List, Tuple

from spikee.utilities.content import Content
from spikee.utilities.enums import ModuleTag

ModuleDescriptionHint = Tuple[List[ModuleTag], str]
ModuleOptionsHint = Tuple[List[str], bool]

ContentHint = Union[str, Content]
