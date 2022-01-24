# for type hint
from typing import Union, Tuple, Dict
from torch import Tensor


from .utils import (bha_coeff, bha_coeff_distance, hel_dist, l2_distance)
from .loss import softmax_cross_entropy_loss, bha_coeff_loss, l2_dist_loss

LogDictType = Dict[str, Tensor]



SimilarityType = Union[bha_coeff]
DistanceType = Union[hel_dist, bha_coeff_distance, l2_distance]
DistanceLossType = Union[softmax_cross_entropy_loss, l2_dist_loss, bha_coeff_loss]

