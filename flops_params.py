import torch
from torch import nn
from mmcv.cnn import build_norm_layer, build_conv_layer, build_upsample_layer
from voxel_layer import hard_voxelize
from torch.nn import functional as F
from mmdet3d.models.dense_heads.train_mixins import AnchorTrainMixin
from mmdet.core import multi_apply
from mmdet.core.bbox import BaseBBoxCoder
from mmcv import Config


class Voxelization(nn.Module):

    def __init__(self,
                 voxel_size,
                 point_cloud_range,
                 max_num_points,
                 max_voxels=20000):
        super(Voxelization, self).__init__()
        """
        Args:
            voxel_size (list): list [x, y, z] size of three dimension
            point_cloud_range (list):
                [x_min, y_min, z_min, x_max, y_max, z_max]
            max_num_points (int): max number of points per voxel
            max_voxels (tuple or int): max number of voxels in
                (training, testing) time
        """
        self.voxel_size = voxel_size
        self.coors_range = point_cloud_range
        self.max_points = max_num_points
        self.max_voxels = max_voxels[1]

    def forward(self, points):
        voxels = points.new_zeros(
            size=(self.max_voxels, self.max_points, points.size(1)))
        coors = points.new_zeros(size=(self.max_voxels, 3), dtype=torch.int)
        num_points_per_voxel = points.new_zeros(
            size=(self.max_voxels,), dtype=torch.int)
        voxel_num = hard_voxelize(points, voxels, coors,
                                  num_points_per_voxel, self.voxel_size,
                                  self.coors_range, self.max_points, self.max_voxels, 3)
        # select the valid voxels
        voxels_out = voxels[:voxel_num]
        coors_out = coors[:voxel_num]
        num_points_per_voxel_out = num_points_per_voxel[:voxel_num]
        return voxels_out, coors_out, num_points_per_voxel_out


def get_paddings_indicator(actual_num, max_num, axis=0):
    """Create boolean mask by actually number of a padded tensor.

    Args:
        actual_num (torch.Tensor): Actual number of points in each voxel.
        max_num (int): Max number of points in each voxel

    Returns:
        torch.Tensor: Mask indicates which points are valid inside a voxel.
    """
    actual_num = torch.unsqueeze(actual_num, axis + 1)
    # tiled_actual_num: [N, M, 1]
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num = torch.arange(
        max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
    # tiled_actual_num: [[3,3,3,3,3], [4,4,4,4,4], [2,2,2,2,2]]
    # tiled_max_num: [[0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4]]
    paddings_indicator = actual_num.int() > max_num
    # paddings_indicator shape: [batch_size, max_num]
    return paddings_indicator


class PFNLayer(nn.Module):
    """Pillar Feature Net Layer.

    The Pillar Feature Net is composed of a series of these layers, but the
    PointPillars paper results only used a single PFNLayer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        norm_cfg (dict): Config dict of normalization layers
        last_layer (bool): If last_layer, there is no concatenation of
            features.
        mode (str): Pooling model to gather features inside voxels.
            Default to 'max'.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 last_layer=False,
                 mode='max'):

        super().__init__()
        self.name = 'PFNLayer'
        self.last_vfe = last_layer
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units = out_channels

        self.norm = build_norm_layer(norm_cfg, self.units)[1]
        self.linear = nn.Linear(in_channels, self.units, bias=False)

        assert mode in ['max', 'avg']
        self.mode = mode

    def forward(self, inputs, num_voxels=None, aligned_distance=None):
        """Forward function.

        Args:
            inputs (torch.Tensor): Pillar/Voxel inputs with shape (N, M, C).
                N is the number of voxels, M is the number of points in
                voxels, C is the number of channels of point features.
            num_voxels (torch.Tensor, optional): Number of points in each
                voxel. Defaults to None.
            aligned_distance (torch.Tensor, optional): The distance of
                each points to the voxel center. Defaults to None.

        Returns:
            torch.Tensor: Features of Pillars.
        """
        x = self.linear(inputs)
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2,
                                                               1).contiguous()
        x = F.relu(x)

        if self.mode == 'max':
            if aligned_distance is not None:
                x = x.mul(aligned_distance.unsqueeze(-1))
            x_max = torch.max(x, dim=1, keepdim=True)[0]
        elif self.mode == 'avg':
            if aligned_distance is not None:
                x = x.mul(aligned_distance.unsqueeze(-1))
            x_max = x.sum(
                dim=1, keepdim=True) / num_voxels.type_as(inputs).view(
                -1, 1, 1)

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


class PillarFeatureNet(nn.Module):
    """Pillar Feature Net.

    The network prepares the pillar features and performs forward pass
    through PFNLayers.

    Args:
        in_channels (int, optional): Number of input features,
            either x, y, z or x, y, z, r. Defaults to 4.
        feat_channels (tuple, optional): Number of features in each of the
            N PFNLayers. Defaults to (64, ).
        with_distance (bool, optional): Whether to include Euclidean distance
            to points. Defaults to False.
        with_cluster_center (bool, optional): [description]. Defaults to True.
        with_voxel_center (bool, optional): [description]. Defaults to True.
        voxel_size (tuple[float], optional): Size of voxels, only utilize x
            and y size. Defaults to (0.2, 0.2, 4).
        point_cloud_range (tuple[float], optional): Point cloud range, only
            utilizes x and y min. Defaults to (0, -40, -3, 70.4, 40, 1).
        norm_cfg ([type], optional): [description].
            Defaults to dict(type='BN1d', eps=1e-3, momentum=0.01).
        mode (str, optional): The mode to gather point features. Options are
            'max' or 'avg'. Defaults to 'max'.
    """

    def __init__(self,
                 type,
                 in_channels=4,
                 feat_channels=(64,),
                 with_distance=False,
                 with_cluster_center=True,
                 with_voxel_center=True,
                 voxel_size=(0.2, 0.2, 4),
                 point_cloud_range=(0, -40, -3, 70.4, 40, 1),
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 mode='max'):
        super(PillarFeatureNet, self).__init__()
        self.type = type
        assert len(feat_channels) > 0
        if with_cluster_center:
            in_channels += 3
        if with_voxel_center:
            in_channels += 2
        if with_distance:
            in_channels += 1
        self._with_distance = with_distance
        self._with_cluster_center = with_cluster_center
        self._with_voxel_center = with_voxel_center

        # Create PillarFeatureNet layers
        self.in_channels = in_channels
        feat_channels = [in_channels] + list(feat_channels)
        pfn_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            if i < len(feat_channels) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(
                PFNLayer(
                    in_filters,
                    out_filters,
                    norm_cfg=norm_cfg,
                    last_layer=last_layer,
                    mode=mode))
        self.pfn_layers = nn.ModuleList(pfn_layers)

        # Need pillar (voxel) size and x/y offset in order to calculate offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.point_cloud_range = point_cloud_range

    def forward(self, features, num_points, coors):
        """Forward function.

        Args:
            features (torch.Tensor): Point features or raw points in shape
                (N, M, C).
            num_points (torch.Tensor): Number of points in each pillar.
            coors (torch.Tensor): Coordinates of each voxel

        Returns:
            torch.Tensor: Features of pillars.
        """
        features_ls = [features]
        # Find distance of x, y, and z from cluster center
        if self._with_cluster_center:
            points_mean = features[:, :, :3].sum(
                dim=1, keepdim=True) / num_points.type_as(features).view(
                -1, 1, 1)
            f_cluster = features[:, :, :3] - points_mean
            features_ls.append(f_cluster)

        # Find distance of x, y, and z from pillar center
        if self._with_voxel_center:
            f_center = features[:, :, :2]
            f_center[:, :, 0] = f_center[:, :, 0] - (
                    coors[:, 3].type_as(features).unsqueeze(1) * self.vx +
                    self.x_offset)
            f_center[:, :, 1] = f_center[:, :, 1] - (
                    coors[:, 2].type_as(features).unsqueeze(1) * self.vy +
                    self.y_offset)
            features_ls.append(f_center)

        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)

        # Combine together feature decorations
        features = torch.cat(features_ls, dim=-1)
        # The feature decorations were calculated without regard to whether
        # pillar was empty. Need to ensure that
        # empty pillars remain set to zeros.
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features *= mask

        for pfn in self.pfn_layers:
            features = pfn(features, num_points)

        return features.squeeze()


class PointPillarsScatter(nn.Module):
    """Point Pillar's Scatter.

    Converts learned features from dense tensor to sparse pseudo image.

    Args:
        in_channels (int): Channels of input features.
        output_shape (list[int]): Required output shape of features.
    """

    def __init__(self, type, in_channels, output_shape):
        super(PointPillarsScatter, self).__init__()
        self.type = type
        self.output_shape = output_shape
        self.ny = output_shape[0]
        self.nx = output_shape[1]
        self.in_channels = in_channels

    def forward(self, voxel_features, coors, batch_size=None):
        """Foraward function to scatter features."""
        # TODO: rewrite the function in a batch manner
        # no need to deal with different batch cases
        return self.forward_batch(voxel_features, coors, batch_size)

    def forward_batch(self, voxel_features, coors, batch_size):
        """Scatter features of single sample.

        Args:
            voxel_features (torch.Tensor): Voxel features in shape (N, M, C).
            coors (torch.Tensor): Coordinates of each voxel in shape (N, 4).
                The first column indicates the sample ID.
            batch_size (int): Number of samples in the current batch.
        """
        # batch_canvas will be the final output.
        batch_canvas = []
        for batch_itt in range(batch_size):
            # Create the canvas for this sample
            canvas = torch.zeros(
                self.in_channels,
                self.nx * self.ny,
                dtype=voxel_features.dtype,
                device=voxel_features.device)

            # Only include non-empty pillars
            batch_mask = coors[:, 0] == batch_itt
            this_coors = coors[batch_mask, :]
            indices = this_coors[:, 2] * self.nx + this_coors[:, 3]
            indices = indices.type(torch.long)
            voxels = voxel_features[batch_mask, :]
            voxels = voxels.t()

            # Now scatter the blob back to the canvas.
            canvas[:, indices] = voxels

            # Append to a list for later stacking.
            batch_canvas.append(canvas)

        # Stack to 3-dim tensor (batch-size, in_channels, nrows*ncols)
        batch_canvas = torch.stack(batch_canvas, 0)

        # Undo the column stacking to final 4-dim tensor
        batch_canvas = batch_canvas.view(batch_size, self.in_channels, self.ny,
                                         self.nx)

        return batch_canvas


class SECOND(nn.Module):
    """Backbone network for SECOND/PointPillars/PartA2/MVXNet.

    Args:
        in_channels (int): Input channels.
        out_channels (list[int]): Output channels for multi-scale feature maps.
        layer_nums (list[int]): Number of layers in each stage.
        layer_strides (list[int]): Strides of each stage.
        norm_cfg (dict): Config dict of normalization layers.
        conv_cfg (dict): Config dict of convolutional layers.
    """

    def __init__(self,
                 type,
                 in_channels=128,
                 out_channels=[128, 128, 256],
                 layer_nums=[3, 5, 5],
                 layer_strides=[2, 2, 2],
                 norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                 conv_cfg=dict(type='Conv2d', bias=False)):
        super(SECOND, self).__init__()
        self.type = type
        assert len(layer_strides) == len(layer_nums)
        assert len(out_channels) == len(layer_nums)

        in_filters = [in_channels, *out_channels[:-1]]
        # note that when stride > 1, conv2d with same padding isn't
        # equal to pad-conv2d. we should use pad-conv2d.
        blocks = []
        for i, layer_num in enumerate(layer_nums):
            block = [
                build_conv_layer(
                    conv_cfg,
                    in_filters[i],
                    out_channels[i],
                    3,
                    stride=layer_strides[i],
                    padding=1),
                build_norm_layer(norm_cfg, out_channels[i])[1],
                nn.ReLU(inplace=True),
            ]
            for j in range(layer_num):
                block.append(
                    build_conv_layer(
                        conv_cfg,
                        out_channels[i],
                        out_channels[i],
                        3,
                        padding=1))
                block.append(build_norm_layer(norm_cfg, out_channels[i])[1])
                block.append(nn.ReLU(inplace=True))

            block = nn.Sequential(*block)
            blocks.append(block)

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): Input with shape (N, C, H, W).

        Returns:
            tuple[torch.Tensor]: Multi-scale features.
        """
        outs = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            outs.append(x)
        return tuple(outs)


class SECONDFPN(nn.Module):
    """FPN used in SECOND/PointPillars/PartA2/MVXNet.

    Args:
        in_channels (list[int]): Input channels of multi-scale feature maps
        out_channels (list[int]): Output channels of feature maps
        upsample_strides (list[int]): Strides used to upsample the feature maps
        norm_cfg (dict): Config dict of normalization layers
        upsample_cfg (dict): Config dict of upsample layers
    """

    def __init__(self,
                 type,
                 in_channels=[128, 128, 256],
                 out_channels=[256, 256, 256],
                 upsample_strides=[1, 2, 4],
                 norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                 upsample_cfg=dict(type='deconv', bias=False)):
        # if for GroupNorm,
        # cfg is dict(type='GN', num_groups=num_groups, eps=1e-3, affine=True)
        super(SECONDFPN, self).__init__()
        self.type = type
        assert len(out_channels) == len(upsample_strides) == len(in_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels

        deblocks = []
        for i, out_channel in enumerate(out_channels):
            upsample_layer = build_upsample_layer(
                upsample_cfg,
                in_channels=in_channels[i],
                out_channels=out_channel,
                kernel_size=upsample_strides[i],
                stride=upsample_strides[i])
            deblock = nn.Sequential(upsample_layer,
                                    build_norm_layer(norm_cfg, out_channel)[1],
                                    nn.ReLU(inplace=True))
            deblocks.append(deblock)
        self.deblocks = nn.ModuleList(deblocks)

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): 4D Tensor in (N, C, H, W) shape.

        Returns:
            list[torch.Tensor]: Multi-level feature maps.
        """
        assert len(x) == len(self.in_channels)
        ups = [deblock(x[i]) for i, deblock in enumerate(self.deblocks)]

        if len(ups) > 1:
            out = torch.cat(ups, dim=1)
        else:
            out = ups[0]
        return [out]


class Anchor3DRangeGenerator(object):
    """3D Anchor Generator by range.

    This anchor generator generates anchors by the given range in different
    feature levels.
    Due the convention in 3D detection, different anchor sizes are related to
    different ranges for different categories. However we find this setting
    does not effect the performance much in some datasets, e.g., nuScenes.

    Args:
        ranges (list[list[float]]): Ranges of different anchors.
            The ranges are the same across different feature levels. But may
            vary for different anchor sizes if size_per_range is True.
        sizes (list[list[float]]): 3D sizes of anchors.
        scales (list[int]): Scales of anchors in different feature levels.
        rotations (list[float]): Rotations of anchors in a feature grid.
        custom_values (tuple[float]): Customized values of that anchor. For
            example, in nuScenes the anchors have velocities.
        reshape_out (bool): Whether to reshape the output into (N x 4).
        size_per_range: Whether to use separate ranges for different sizes.
            If size_per_range is True, the ranges should have the same length
            as the sizes, if not, it will be duplicated.
    """

    def __init__(self,
                 type,
                 ranges,
                 sizes=[[1.6, 3.9, 1.56]],
                 scales=[1],
                 rotations=[0, 1.5707963],
                 custom_values=(),
                 reshape_out=True,
                 size_per_range=True):
        self.type = type
        if size_per_range:
            if len(sizes) != len(ranges):
                assert len(ranges) == 1
                ranges = ranges * len(sizes)
            assert len(ranges) == len(sizes)
        else:
            assert len(ranges) == 1
        self.sizes = sizes
        self.scales = scales
        self.ranges = ranges
        self.rotations = rotations
        self.custom_values = custom_values
        self.cached_anchors = None
        self.reshape_out = reshape_out
        self.size_per_range = size_per_range

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += f'anchor_range={self.ranges},\n'
        s += f'scales={self.scales},\n'
        s += f'sizes={self.sizes},\n'
        s += f'rotations={self.rotations},\n'
        s += f'reshape_out={self.reshape_out},\n'
        s += f'size_per_range={self.size_per_range})'
        return s

    @property
    def num_base_anchors(self):
        """list[int]: Total number of base anchors in a feature grid."""
        num_rot = len(self.rotations)
        num_size = torch.tensor(self.sizes).reshape(-1, 3).size(0)
        return num_rot * num_size

    @property
    def num_levels(self):
        """int: Number of feature levels that the generator is applied to."""
        return len(self.scales)

    def grid_anchors(self, featmap_sizes, device='cuda'):
        """Generate grid anchors in multiple feature levels.

        Args:
            featmap_sizes (list[tuple]): List of feature map sizes in
                multiple feature levels.
            device (str): Device where the anchors will be put on.

        Returns:
            list[torch.Tensor]: Anchors in multiple feature levels. \
                The sizes of each tensor should be [N, 4], where \
                N = width * height * num_base_anchors, width and height \
                are the sizes of the corresponding feature lavel, \
                num_base_anchors is the number of anchors for that level.
        """
        assert self.num_levels == len(featmap_sizes)
        multi_level_anchors = []
        for i in range(self.num_levels):
            anchors = self.single_level_grid_anchors(
                featmap_sizes[i], self.scales[i], device=device)
            if self.reshape_out:
                anchors = anchors.reshape(-1, anchors.size(-1))
            multi_level_anchors.append(anchors)
        return multi_level_anchors

    def single_level_grid_anchors(self, featmap_size, scale, device='cuda'):
        """Generate grid anchors of a single level feature map.

        This function is usually called by method ``self.grid_anchors``.

        Args:
            featmap_size (tuple[int]): Size of the feature map.
            scale (float): Scale factor of the anchors in the current level.
            device (str, optional): Device the tensor will be put on.
                Defaults to 'cuda'.

        Returns:
            torch.Tensor: Anchors in the overall feature map.
        """
        # We reimplement the anchor generator using torch in cuda
        # torch: 0.6975 s for 1000 times
        # numpy: 4.3345 s for 1000 times
        # which is ~5 times faster than the numpy implementation
        if not self.size_per_range:
            return self.anchors_single_range(
                featmap_size,
                self.ranges[0],
                scale,
                self.sizes,
                self.rotations,
                device=device)

        mr_anchors = []
        for anchor_range, anchor_size in zip(self.ranges, self.sizes):
            mr_anchors.append(
                self.anchors_single_range(
                    featmap_size,
                    anchor_range,
                    scale,
                    anchor_size,
                    self.rotations,
                    device=device))
        mr_anchors = torch.cat(mr_anchors, dim=-3)
        return mr_anchors

    def anchors_single_range(self,
                             feature_size,
                             anchor_range,
                             scale=1,
                             sizes=[[1.6, 3.9, 1.56]],
                             rotations=[0, 1.5707963],
                             device='cuda'):
        """Generate anchors in a single range.

        Args:
            feature_size (list[float] | tuple[float]): Feature map size. It is
                either a list of a tuple of [D, H, W](in order of z, y, and x).
            anchor_range (torch.Tensor | list[float]): Range of anchors with
                shape [6]. The order is consistent with that of anchors, i.e.,
                (x_min, y_min, z_min, x_max, y_max, z_max).
            scale (float | int, optional): The scale factor of anchors.
            sizes (list[list] | np.ndarray | torch.Tensor): Anchor size with
                shape [N, 3], in order of x, y, z.
            rotations (list[float] | np.ndarray | torch.Tensor): Rotations of
                anchors in a single feature grid.
            device (str): Devices that the anchors will be put on.

        Returns:
            torch.Tensor: Anchors with shape \
                [*feature_size, num_sizes, num_rots, 7].
        """
        if len(feature_size) == 2:
            feature_size = [1, feature_size[0], feature_size[1]]
        anchor_range = torch.tensor(anchor_range, device=device)
        z_centers = torch.linspace(
            anchor_range[2], anchor_range[5], feature_size[0], device=device)
        y_centers = torch.linspace(
            anchor_range[1], anchor_range[4], feature_size[1], device=device)
        x_centers = torch.linspace(
            anchor_range[0], anchor_range[3], feature_size[2], device=device)
        sizes = torch.tensor(sizes, device=device).reshape(-1, 3) * scale
        rotations = torch.tensor(rotations, device=device)

        # torch.meshgrid default behavior is 'id', np's default is 'xy'
        rets = torch.meshgrid(x_centers, y_centers, z_centers, rotations)
        # torch.meshgrid returns a tuple rather than list
        rets = list(rets)
        tile_shape = [1] * 5
        tile_shape[-2] = int(sizes.shape[0])
        for i in range(len(rets)):
            rets[i] = rets[i].unsqueeze(-2).repeat(tile_shape).unsqueeze(-1)

        sizes = sizes.reshape([1, 1, 1, -1, 1, 3])
        tile_size_shape = list(rets[0].shape)
        tile_size_shape[3] = 1
        sizes = sizes.repeat(tile_size_shape)
        rets.insert(3, sizes)

        ret = torch.cat(rets, dim=-1).permute([2, 1, 0, 3, 4, 5])
        # [1, 200, 176, N, 2, 7] for kitti after permute

        if len(self.custom_values) > 0:
            custom_ndim = len(self.custom_values)
            custom = ret.new_zeros([*ret.shape[:-1], custom_ndim])
            # custom[:] = self.custom_values
            ret = torch.cat([ret, custom], dim=-1)
            # [1, 200, 176, N, 2, 9] for nus dataset after permute
        return ret


class DeltaXYZWLHRBBoxCoder(BaseBBoxCoder):
    """Bbox Coder for 3D boxes.

    Args:
        code_size (int): The dimension of boxes to be encoded.
    """

    def __init__(self, code_size=7):
        super(DeltaXYZWLHRBBoxCoder, self).__init__()
        self.code_size = code_size

    @staticmethod
    def encode(src_boxes, dst_boxes):
        """Get box regression transformation deltas (dx, dy, dz, dw, dh, dl,
        dr, dv*) that can be used to transform the `src_boxes` into the
        `target_boxes`.

        Args:
            src_boxes (torch.Tensor): source boxes, e.g., object proposals.
            dst_boxes (torch.Tensor): target of the transformation, e.g.,
                ground-truth boxes.

        Returns:
            torch.Tensor: Box transformation deltas.
        """
        box_ndim = src_boxes.shape[-1]
        cas, cgs, cts = [], [], []
        if box_ndim > 7:
            xa, ya, za, wa, la, ha, ra, *cas = torch.split(
                src_boxes, 1, dim=-1)
            xg, yg, zg, wg, lg, hg, rg, *cgs = torch.split(
                dst_boxes, 1, dim=-1)
            cts = [g - a for g, a in zip(cgs, cas)]
        else:
            xa, ya, za, wa, la, ha, ra = torch.split(src_boxes, 1, dim=-1)
            xg, yg, zg, wg, lg, hg, rg = torch.split(dst_boxes, 1, dim=-1)
        za = za + ha / 2
        zg = zg + hg / 2
        diagonal = torch.sqrt(la ** 2 + wa ** 2)
        xt = (xg - xa) / diagonal
        yt = (yg - ya) / diagonal
        zt = (zg - za) / ha
        lt = torch.log(lg / la)
        wt = torch.log(wg / wa)
        ht = torch.log(hg / ha)
        rt = rg - ra
        return torch.cat([xt, yt, zt, wt, lt, ht, rt, *cts], dim=-1)

    @staticmethod
    def decode(anchors, deltas):
        """Apply transformation `deltas` (dx, dy, dz, dw, dh, dl, dr, dv*) to
        `boxes`.

        Args:
            anchors (torch.Tensor): Parameters of anchors with shape (N, 7).
            deltas (torch.Tensor): Encoded boxes with shape
                (N, 7+n) [x, y, z, w, l, h, r, velo*].

        Returns:
            torch.Tensor: Decoded boxes.
        """
        cas, cts = [], []
        box_ndim = anchors.shape[-1]
        if box_ndim > 7:
            xa, ya, za, wa, la, ha, ra, *cas = torch.split(anchors, 1, dim=-1)
            xt, yt, zt, wt, lt, ht, rt, *cts = torch.split(deltas, 1, dim=-1)
        else:
            xa, ya, za, wa, la, ha, ra = torch.split(anchors, 1, dim=-1)
            xt, yt, zt, wt, lt, ht, rt = torch.split(deltas, 1, dim=-1)

        za = za + ha / 2
        diagonal = torch.sqrt(la ** 2 + wa ** 2)
        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * ha + za

        lg = torch.exp(lt) * la
        wg = torch.exp(wt) * wa
        hg = torch.exp(ht) * ha
        rg = rt + ra
        zg = zg - hg / 2
        cgs = [t + a for t, a in zip(cts, cas)]
        return torch.cat([xg, yg, zg, wg, lg, hg, rg, *cgs], dim=-1)


class Anchor3DHead(nn.Module, AnchorTrainMixin):
    """Anchor head for SECOND/PointPillars/MVXNet/PartA2.

    Args:
        num_classes (int): Number of classes.
        in_channels (int): Number of channels in the input feature map.
        train_cfg (dict): Train configs.
        test_cfg (dict): Test configs.
        feat_channels (int): Number of channels of the feature map.
        use_direction_classifier (bool): Whether to add a direction classifier.
        anchor_generator(dict): Config dict of anchor generator.
        assigner_per_size (bool): Whether to do assignment for each separate
            anchor size.
        assign_per_class (bool): Whether to do assignment for each class.
        diff_rad_by_sin (bool): Whether to change the difference into sin
            difference for box regression loss.
        dir_offset (float | int): The offset of BEV rotation angles.
            (TODO: may be moved into box coder)
        dir_limit_offset (float | int): The limited range of BEV
            rotation angles. (TODO: may be moved into box coder)
        bbox_coder (dict): Config dict of box coders.
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        loss_dir (dict): Config of direction classifier loss.
    """

    def __init__(self,
                 type,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 use_direction_classifier=True,
                 anchor_generator=dict(
                     type='Anchor3DRangeGenerator',
                     range=[0, -39.68, -1.78, 69.12, 39.68, -1.78],
                     strides=[2],
                     sizes=[[1.6, 3.9, 1.56]],
                     rotations=[0, 1.57],
                     custom_values=[],
                     reshape_out=False),
                 bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
                 diff_rad_by_sin=True,
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
                 loss_dir=dict(
                     type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2)
                 ):
        super(Anchor3DHead, self).__init__()
        self.type = type
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.use_direction_classifier = use_direction_classifier

        # build anchor generator
        self.anchor_generator = Anchor3DRangeGenerator(**anchor_generator)
        # In 3D detection, the anchor stride is connected with anchor size
        self.num_anchors = self.anchor_generator.num_base_anchors
        # build box coder
        if bbox_coder:
            self.bbox_coder = DeltaXYZWLHRBBoxCoder()
        self.box_code_size = self.bbox_coder.code_size
        self._init_layers()

    def _init_layers(self):
        """Initialize neural network layers of the head."""
        self.cls_out_channels = self.num_anchors * self.num_classes
        self.conv_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels, 1)
        self.conv_reg = nn.Conv2d(self.feat_channels,
                                  self.num_anchors * self.box_code_size, 1)
        if self.use_direction_classifier:
            self.conv_dir_cls = nn.Conv2d(self.feat_channels,
                                          self.num_anchors * 2, 1)

    def forward_single(self, x):
        """Forward function on a single-scale feature map.

        Args:
            x (torch.Tensor): Input features.

        Returns:
            tuple[torch.Tensor]: Contain score of each class, bbox \
                regression and direction classification predictions.
        """
        cls_score = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)
        dir_cls_preds = None
        if self.use_direction_classifier:
            dir_cls_preds = self.conv_dir_cls(x)
        return [cls_score, bbox_pred, dir_cls_preds]

    def forward(self, feats):
        """Forward pass.

        Args:
            feats (list[torch.Tensor]): Multi-level features, e.g.,
                features produced by FPN.

        Returns:
            tuple[list[torch.Tensor]]: Multi-level class score, bbox \
                and direction predictions.
        """
        return multi_apply(self.forward_single, feats)


class VoxelNet(nn.Module):
    r"""`VoxelNet <https://arxiv.org/abs/1711.06396>`_ for 3D detection."""

    def __init__(self, type, voxel_layer, voxel_encoder, middle_encoder, backbone, neck, bbox_head):
        super(VoxelNet, self).__init__()
        self.type = type
        self.voxel_layer = Voxelization(**voxel_layer)
        self.voxel_encoder = PillarFeatureNet(**voxel_encoder)
        self.middle_encoder = PointPillarsScatter(**middle_encoder)
        self.backbone = SECOND(**backbone)
        self.neck = SECONDFPN(**neck)
        self.bbox_head = Anchor3DHead(**bbox_head)

    def extract_feat(self, points):
        """Extract features from points."""
        voxels, num_points, coors = self.voxelize(points)
        voxel_features = self.voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0].item() + 1
        x = self.middle_encoder(voxel_features, coors, batch_size)
        x = self.backbone(x)
        x = self.neck(x)
        return x

    def voxelize(self, points):
        """Apply hard voxelization to points."""
        voxels, coors, num_points = [], [], []
        res = points
        res_voxels, res_coors, res_num_points = self.voxel_layer(res)
        voxels.append(res_voxels)
        coors.append(res_coors)
        num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    def forward(self, points):
        """Training forward function.

        Args:
            points (list[torch.Tensor]): Point cloud of each sample.
        Returns:
            dict: Losses of each branch.
        """
        # x = self.extract_feat(points)
        voxels, num_points, coors = self.voxelize(points)
        voxel_features = self.voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0].item() + 1
        x = self.middle_encoder(voxel_features, coors, batch_size)
        x = self.backbone(x)
        x = self.neck(x)
        outs = self.bbox_head(x)
        return outs


if __name__ == "__main__":
    import thop
    cfg = Config.fromfile(
        "/home/mk/mmdetection3d-master/configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py")
    network_arch = cfg._cfg_dict.model
    model_mmdet = torch.load("/home/mk/mmdetection3d-master/benchmark_model/pp_originSec/epoch_149.pth")

    model = VoxelNet(**network_arch)
    model.load_state_dict(model_mmdet["state_dict"])
    data_input = torch.randn(19364, 4).cuda()
    dynamic_axes = {"points": [0]}

    model = model.cuda()
    flops, params = thop.profile(model, inputs=(data_input,))
    flops, params = thop.clever_format([flops, params], "%.3f")
    print("flops: {} params: {}".format(flops, params))
    # torch.onnx.export(model, data_input, "pointpillars.onnx", input_names=["points"], dynamic_axes=dynamic_axes, verbose=True, opset_version=11)
    # out = model(data_input)

    print("hahahahhah")



