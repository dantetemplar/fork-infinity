from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.sparse_vector import SparseVector


T = TypeVar("T", bound="SparseEmbeddingObject")


@_attrs_define
class SparseEmbeddingObject:
    """
    Attributes:
        index (int):
        embedding (SparseVector):
        object_ (Union[Unset, str]):  Default: "sparse_embedding".
    """

    index: int
    embedding: "SparseVector"
    object_: Union[Unset, str] = "sparse_embedding"
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        index = self.index

        embedding = self.embedding.to_dict()

        object_ = self.object_

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "index": index,
                "embedding": embedding,
            }
        )
        if object_ is not UNSET:
            field_dict["object"] = object_

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        from ..models.sparse_vector import SparseVector

        d = src_dict.copy()
        index = d.pop("index")

        embedding = SparseVector.from_dict(d.pop("embedding"))

        object_ = d.pop("object", UNSET)

        sparse_embedding_object = cls(
            index=index,
            embedding=embedding,
            object_=object_,
        )

        sparse_embedding_object.additional_properties = d
        return sparse_embedding_object

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
