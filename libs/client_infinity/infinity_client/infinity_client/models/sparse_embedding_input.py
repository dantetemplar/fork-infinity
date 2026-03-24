from typing import Any, Dict, List, Type, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="SparseEmbeddingInput")


@_attrs_define
class SparseEmbeddingInput:
    """
    Attributes:
        input_ (Union[List[str], str]):
        model (str):  Default: "default/not-specified".
        prune_ratio (float):  Default: 0.0.
        task (str):  Default: "document".
    """

    input_: Union[List[str], str]
    model: str = "default/not-specified"
    prune_ratio: float = 0.0
    task: str = "document"
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        input_: Union[List[str], str]
        if isinstance(self.input_, list):
            input_ = self.input_
        else:
            input_ = self.input_

        model = self.model

        prune_ratio = self.prune_ratio

        task = self.task

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "input": input_,
                "model": model,
                "prune_ratio": prune_ratio,
                "task": task,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()

        def _parse_input_(data: object) -> Union[List[str], str]:
            try:
                if not isinstance(data, list):
                    raise TypeError()
                input_type_0 = cast(List[str], data)

                return input_type_0
            except:  # noqa: E722
                pass
            return cast(Union[List[str], str], data)

        input_ = _parse_input_(d.pop("input"))

        model = d.pop("model", "default/not-specified")

        prune_ratio = d.pop("prune_ratio", 0.0)

        task = d.pop("task", "document")

        sparse_embedding_input = cls(
            input_=input_,
            model=model,
            prune_ratio=prune_ratio,
            task=task,
        )

        sparse_embedding_input.additional_properties = d
        return sparse_embedding_input

    @property
    def additional_keys(self) -> List[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> Any:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
