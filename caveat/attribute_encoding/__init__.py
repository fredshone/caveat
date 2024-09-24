from caveat.attribute_encoding.base import BaseAttributeEncoder, tokenize
from caveat.attribute_encoding.onehot import OneHotAttributeEncoder
from caveat.attribute_encoding.tokenise import TokenAttributeEncoder

library = {"onehot": OneHotAttributeEncoder, "tokens": TokenAttributeEncoder}
