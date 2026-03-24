from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.sparse_embedding_object import SparseEmbeddingObject
    from ..models.usage import Usage


T = TypeVar("T", bound="SparseEmbeddingResult")


@_attrs_define
class SparseEmbeddingResult:
    """
    Attributes:
        data (List['SparseEmbeddingObject']):
        model (str):
        usage (Usage):
        object_ (Union[Unset, str]):  Default: "list".
        id (Union[Unset, str]):
        created (Union[Unset, int]):
    """

    data: List["SparseEmbeddingObject"]
    model: str
    usage: "Usage"
    object_: Union[Unset, str] = "list"
    id: Union[Unset, str] = UNSET
    created: Union[Unset, int] = UNSET
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = []
        for data_item_data in self.data:
            data_item = data_item_data.to_dict()
            data.append(data_item)

        model = self.model

        usage = self.usage.to_dict()

        object_ = self.object_

        id = self.id

        created = self.created

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "data": data,
                "model": model,
                "usage": usage,
            }
        )
        if object_ is not UNSET:
            field_dict["object"] = object_
        if id is not UNSET:
            field_dict["id"] = id
        if created is not UNSET:
            field_dict["created"] = created

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        from ..models.sparse_embedding_object import SparseEmbeddingObject
        from ..models.usage import Usage

        d = src_dict.copy()
        data = []
        _data = d.pop("data")
        for data_item_data in _data:
            data_item = SparseEmbeddingObject.from_dict(data_item_data)

            data.append(data_item)

        model = d.pop("model")

        usage = Usage.from_dict(d.pop("usage"))

        object_ = d.pop("object", UNSET)

        id = d.pop("id", UNSET)

        created = d.pop("created", UNSET)

        sparse_embedding_result = cls(
            data=data,
            model=model,
            usage=usage,
            object_=object_,
            id=id,
            created=created,
        )

        sparse_embedding_result.additional_properties = d
        return sparse_embedding_result

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
