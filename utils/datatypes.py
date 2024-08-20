from typing import TypeVar, Generic, List, Type

T = TypeVar('T')

class TypedList(Generic[T], list):
    def __init__(self, *args: T):
        if args:
            super().__init__(args)
        else:
            super().__init__()

    def append(self, item: T) -> None:
        if not isinstance(item, self._item_type):
            raise TypeError(f"Expected type {self._item_type}, got {type(item)} instead")
        super().append(item)

    def extend(self, items: List[T]) -> None:
        if not all(isinstance(item, self._item_type) for item in items):
            raise TypeError(f"All items must be of type {self._item_type}")
        super().extend(items)

    @property
    def _item_type(self) -> Type[T]:
        return T