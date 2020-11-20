from .. import _ffi_api
from ..feature import DEFAULT_MAX_N_BUFS, unpack_feature

def get_per_stmt_features_from_cluster_states(cluster, states):
    byte_array = _ffi_api.GetPerStmtFeaturesFromClusterStates(
            cluster, states, DEFAULT_MAX_N_BUFS)
    return unpack_feature(byte_array)[0]
