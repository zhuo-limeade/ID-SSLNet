from .anchor_head_multi import AnchorHeadMulti
from .anchor_head_single import AnchorHeadSingle, AnchorHeadSingle_MD21,\
    AnchorHeadSingle_MD21_pillar


from .anchor_head_template import AnchorHeadTemplate
from .point_head_box import PointHeadBox
from .point_head_simple import PointHeadSimple
from .point_intra_part_head import PointIntraPartOffsetHead
from .center_head import CenterHead
from .voxelnext_head import VoxelNeXtHead
from .transfusion_head import TransFusionHead
from .IASSD_head import IASSD_Head
from .anchor_head_multi import AnchorHeadMulti

from .voxelnext_head_cih import CenterHead_CIH
from .voxelnext_head_voxel_atten import VoxelNeXtHead_MappingSplit

__all__ = {
    'AnchorHeadTemplate': AnchorHeadTemplate,
    'AnchorHeadSingle': AnchorHeadSingle,
    'PointIntraPartOffsetHead': PointIntraPartOffsetHead,
    'PointHeadSimple': PointHeadSimple,
    'PointHeadBox': PointHeadBox,
    'AnchorHeadMulti': AnchorHeadMulti,
    'CenterHead': CenterHead,
    'VoxelNeXtHead': VoxelNeXtHead,
    'TransFusionHead': TransFusionHead,

    'IASSD_Head': IASSD_Head,
    'AnchorHeadMulti' : AnchorHeadMulti,

    'AnchorHeadSingle_MD21': AnchorHeadSingle_MD21,
    'AnchorHeadSingle_MD21_pillar': AnchorHeadSingle_MD21_pillar,
    'CenterHead_CIH': CenterHead_CIH,
    'VoxelNeXtHead_MappingSplit': VoxelNeXtHead_MappingSplit,
}
