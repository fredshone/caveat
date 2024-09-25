from caveat.attribute_encoding.base import BaseLabelEncoder, tokenize
from caveat.attribute_encoding.onehot import OneHotAttributeEncoder
from caveat.attribute_encoding.tokenise import TokenAttributeEncoder

library = {"onehot": OneHotAttributeEncoder, "tokens": TokenAttributeEncoder}
