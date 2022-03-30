# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
import math
import warnings
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

Device = Union[str, torch.device]

# Default values for rotation and translation matrices.
_R = torch.eye(3)[None]  # (1, 3, 3)
_T = torch.zeros(1, 3)  # (1, 3)


# Provide get_origin and get_args even in Python 3.7.

if sys.version_info >= (3, 8, 0):
    from typing import get_args, get_origin
elif sys.version_info >= (3, 7, 0):

    def get_origin(cls):  # pragma: no cover
        return getattr(cls, "__origin__", None)

    def get_args(cls):  # pragma: no cover
        return getattr(cls, "__args__", None)


else:
    raise ImportError("This module requires Python 3.7+")

################################################################
##   ██████╗██╗      █████╗ ███████╗███████╗███████╗███████╗  ##
##  ██╔════╝██║     ██╔══██╗██╔════╝██╔════╝██╔════╝██╔════╝  ##
##  ██║     ██║     ███████║███████╗███████╗█████╗  ███████╗  ##
##  ██║     ██║     ██╔══██║╚════██║╚════██║██╔══╝  ╚════██║  ##
##  ╚██████╗███████╗██║  ██║███████║███████║███████╗███████║  ##
##   ╚═════╝╚══════╝╚═╝  ╚═╝╚══════╝╚══════╝╚══════╝╚══════╝  ##
################################################################

class Translate(Transform3d):
    def __init__(
        self,
        x,
        y=None,
        z=None,
        dtype: torch.dtype = torch.float32,
        device: Optional[Device] = None,
    ) -> None:
        """
        Create a new Transform3d representing 3D translations.

        Option I: Translate(xyz, dtype=torch.float32, device='cpu')
            xyz should be a tensor of shape (N, 3)

        Option II: Translate(x, y, z, dtype=torch.float32, device='cpu')
            Here x, y, and z will be broadcast against each other and
            concatenated to form the translation. Each can be:
                - A python scalar
                - A torch scalar
                - A 1D torch tensor
        """
        xyz = _handle_input(x, y, z, dtype, device, "Translate")
        super().__init__(device=xyz.device, dtype=dtype)
        N = xyz.shape[0]

        mat = torch.eye(4, dtype=dtype, device=self.device)
        mat = mat.view(1, 4, 4).repeat(N, 1, 1)
        mat[:, 3, :3] = xyz
        self._matrix = mat

    def _get_matrix_inverse(self) -> torch.Tensor:
        """
        Return the inverse of self._matrix.
        """
        inv_mask = self._matrix.new_ones([1, 4, 4])
        inv_mask[0, 3, :3] = -1.0
        i_matrix = self._matrix * inv_mask
        return i_matrix

class Rotate(Transform3d):
    def __init__(
        self,
        R: torch.Tensor,
        dtype: torch.dtype = torch.float32,
        device: Optional[Device] = None,
        orthogonal_tol: float = 1e-5,
    ) -> None:
        """
        Create a new Transform3d representing 3D rotation using a rotation
        matrix as the input.

        Args:
            R: a tensor of shape (3, 3) or (N, 3, 3)
            orthogonal_tol: tolerance for the test of the orthogonality of R

        """
        device_ = get_device(R, device)
        super().__init__(device=device_, dtype=dtype)
        if R.dim() == 2:
            R = R[None]
        if R.shape[-2:] != (3, 3):
            msg = "R must have shape (3, 3) or (N, 3, 3); got %s"
            raise ValueError(msg % repr(R.shape))
        R = R.to(device=device_, dtype=dtype)
        _check_valid_rotation_matrix(R, tol=orthogonal_tol)
        N = R.shape[0]
        mat = torch.eye(4, dtype=dtype, device=device_)
        mat = mat.view(1, 4, 4).repeat(N, 1, 1)
        mat[:, :3, :3] = R
        self._matrix = mat

    def _get_matrix_inverse(self) -> torch.Tensor:
        """
        Return the inverse of self._matrix.
        """
        return self._matrix.permute(0, 2, 1).contiguous()

class Transform3d:
    """
    A Transform3d object encapsulates a batch of N 3D transformations, and knows
    how to transform points and normal vectors. Suppose that t is a Transform3d;
    then we can do the following:

    .. code-block:: python

        N = len(t)
        points = torch.randn(N, P, 3)
        normals = torch.randn(N, P, 3)
        points_transformed = t.transform_points(points)    # => (N, P, 3)
        normals_transformed = t.transform_normals(normals)  # => (N, P, 3)


    BROADCASTING
    Transform3d objects supports broadcasting. Suppose that t1 and tN are
    Transform3d objects with len(t1) == 1 and len(tN) == N respectively. Then we
    can broadcast transforms like this:

    .. code-block:: python

        t1.transform_points(torch.randn(P, 3))     # => (P, 3)
        t1.transform_points(torch.randn(1, P, 3))  # => (1, P, 3)
        t1.transform_points(torch.randn(M, P, 3))  # => (M, P, 3)
        tN.transform_points(torch.randn(P, 3))     # => (N, P, 3)
        tN.transform_points(torch.randn(1, P, 3))  # => (N, P, 3)


    COMBINING TRANSFORMS
    Transform3d objects can be combined in two ways: composing and stacking.
    Composing is function composition. Given Transform3d objects t1, t2, t3,
    the following all compute the same thing:

    .. code-block:: python

        y1 = t3.transform_points(t2.transform_points(t1.transform_points(x)))
        y2 = t1.compose(t2).compose(t3).transform_points(x)
        y3 = t1.compose(t2, t3).transform_points(x)


    Composing transforms should broadcast.

    .. code-block:: python

        if len(t1) == 1 and len(t2) == N, then len(t1.compose(t2)) == N.

    We can also stack a sequence of Transform3d objects, which represents
    composition along the batch dimension; then the following should compute the
    same thing.

    .. code-block:: python

        N, M = len(tN), len(tM)
        xN = torch.randn(N, P, 3)
        xM = torch.randn(M, P, 3)
        y1 = torch.cat([tN.transform_points(xN), tM.transform_points(xM)], dim=0)
        y2 = tN.stack(tM).transform_points(torch.cat([xN, xM], dim=0))

    BUILDING TRANSFORMS
    We provide convenience methods for easily building Transform3d objects
    as compositions of basic transforms.

    .. code-block:: python

        # Scale by 0.5, then translate by (1, 2, 3)
        t1 = Transform3d().scale(0.5).translate(1, 2, 3)

        # Scale each axis by a different amount, then translate, then scale
        t2 = Transform3d().scale(1, 3, 3).translate(2, 3, 1).scale(2.0)

        t3 = t1.compose(t2)
        tN = t1.stack(t3, t3)


    BACKPROP THROUGH TRANSFORMS
    When building transforms, we can also parameterize them by Torch tensors;
    in this case we can backprop through the construction and application of
    Transform objects, so they could be learned via gradient descent or
    predicted by a neural network.

    .. code-block:: python

        s1_params = torch.randn(N, requires_grad=True)
        t_params = torch.randn(N, 3, requires_grad=True)
        s2_params = torch.randn(N, 3, requires_grad=True)

        t = Transform3d().scale(s1_params).translate(t_params).scale(s2_params)
        x = torch.randn(N, 3)
        y = t.transform_points(x)
        loss = compute_loss(y)
        loss.backward()

        with torch.no_grad():
            s1_params -= lr * s1_params.grad
            t_params -= lr * t_params.grad
            s2_params -= lr * s2_params.grad

    CONVENTIONS
    We adopt a right-hand coordinate system, meaning that rotation about an axis
    with a positive angle results in a counter clockwise rotation.

    This class assumes that transformations are applied on inputs which
    are row vectors. The internal representation of the Nx4x4 transformation
    matrix is of the form:

    .. code-block:: python

        M = [
                [Rxx, Ryx, Rzx, 0],
                [Rxy, Ryy, Rzy, 0],
                [Rxz, Ryz, Rzz, 0],
                [Tx,  Ty,  Tz,  1],
            ]

    To apply the transformation to points which are row vectors, the M matrix
    can be pre multiplied by the points:

    .. code-block:: python

        points = [[0, 1, 2]]  # (1 x 3) xyz coordinates of a point
        transformed_points = points * M

    """

    def __init__(
        self,
        dtype: torch.dtype = torch.float32,
        device: Device = "cpu",
        matrix: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Args:
            dtype: The data type of the transformation matrix.
                to be used if `matrix = None`.
            device: The device for storing the implemented transformation.
                If `matrix != None`, uses the device of input `matrix`.
            matrix: A tensor of shape (4, 4) or of shape (minibatch, 4, 4)
                representing the 4x4 3D transformation matrix.
                If `None`, initializes with identity using
                the specified `device` and `dtype`.
        """

        if matrix is None:
            self._matrix = torch.eye(4, dtype=dtype, device=device).view(1, 4, 4)
        else:
            if matrix.ndim not in (2, 3):
                raise ValueError('"matrix" has to be a 2- or a 3-dimensional tensor.')
            if matrix.shape[-2] != 4 or matrix.shape[-1] != 4:
                raise ValueError(
                    '"matrix" has to be a tensor of shape (minibatch, 4, 4)'
                )
            # set dtype and device from matrix
            dtype = matrix.dtype
            device = matrix.device
            self._matrix = matrix.view(-1, 4, 4)

        self._transforms = []  # store transforms to compose
        self._lu = None
        self.device = make_device(device)
        self.dtype = dtype

    def __len__(self) -> int:
        return self.get_matrix().shape[0]

    def __getitem__(
        self, index: Union[int, List[int], slice, torch.Tensor]
    ) -> "Transform3d":
        """
        Args:
            index: Specifying the index of the transform to retrieve.
                Can be an int, slice, list of ints, boolean, long tensor.
                Supports negative indices.

        Returns:
            Transform3d object with selected transforms. The tensors are not cloned.
        """
        if isinstance(index, int):
            index = [index]
        return self.__class__(matrix=self.get_matrix()[index])

    def compose(self, *others: "Transform3d") -> "Transform3d":
        """
        Return a new Transform3d representing the composition of self with the
        given other transforms, which will be stored as an internal list.

        Args:
            *others: Any number of Transform3d objects

        Returns:
            A new Transform3d with the stored transforms
        """
        out = Transform3d(dtype=self.dtype, device=self.device)
        out._matrix = self._matrix.clone()
        for other in others:
            if not isinstance(other, Transform3d):
                msg = "Only possible to compose Transform3d objects; got %s"
                raise ValueError(msg % type(other))
        out._transforms = self._transforms + list(others)
        return out

    def get_matrix(self) -> torch.Tensor:
        """
        Return a matrix which is the result of composing this transform
        with others stored in self.transforms. Where necessary transforms
        are broadcast against each other.
        For example, if self.transforms contains transforms t1, t2, and t3, and
        given a set of points x, the following should be true:

        .. code-block:: python

            y1 = t1.compose(t2, t3).transform(x)
            y2 = t3.transform(t2.transform(t1.transform(x)))
            y1.get_matrix() == y2.get_matrix()

        Returns:
            A transformation matrix representing the composed inputs.
        """
        composed_matrix = self._matrix.clone()
        if len(self._transforms) > 0:
            for other in self._transforms:
                other_matrix = other.get_matrix()
                composed_matrix = _broadcast_bmm(composed_matrix, other_matrix)
        return composed_matrix

    def _get_matrix_inverse(self) -> torch.Tensor:
        """
        Return the inverse of self._matrix.
        """
        return torch.inverse(self._matrix)

    def inverse(self, invert_composed: bool = False) -> "Transform3d":
        """
        Returns a new Transform3d object that represents an inverse of the
        current transformation.

        Args:
            invert_composed:
                - True: First compose the list of stored transformations
                  and then apply inverse to the result. This is
                  potentially slower for classes of transformations
                  with inverses that can be computed efficiently
                  (e.g. rotations and translations).
                - False: Invert the individual stored transformations
                  independently without composing them.

        Returns:
            A new Transform3d object containing the inverse of the original
            transformation.
        """

        tinv = Transform3d(dtype=self.dtype, device=self.device)

        if invert_composed:
            # first compose then invert
            tinv._matrix = torch.inverse(self.get_matrix())
        else:
            # self._get_matrix_inverse() implements efficient inverse
            # of self._matrix
            i_matrix = self._get_matrix_inverse()

            # 2 cases:
            if len(self._transforms) > 0:
                # a) Either we have a non-empty list of transforms:
                # Here we take self._matrix and append its inverse at the
                # end of the reverted _transforms list. After composing
                # the transformations with get_matrix(), this correctly
                # right-multiplies by the inverse of self._matrix
                # at the end of the composition.
                tinv._transforms = [t.inverse() for t in reversed(self._transforms)]
                last = Transform3d(dtype=self.dtype, device=self.device)
                last._matrix = i_matrix
                tinv._transforms.append(last)
            else:
                # b) Or there are no stored transformations
                # we just set inverted matrix
                tinv._matrix = i_matrix

        return tinv

    def stack(self, *others: "Transform3d") -> "Transform3d":
        """
        Return a new batched Transform3d representing the batch elements from
        self and all the given other transforms all batched together.

        Args:
            *others: Any number of Transform3d objects

        Returns:
            A new Transform3d.
        """
        transforms = [self] + list(others)
        matrix = torch.cat([t.get_matrix() for t in transforms], dim=0)
        out = Transform3d(dtype=self.dtype, device=self.device)
        out._matrix = matrix
        return out

    def transform_points(self, points, eps: Optional[float] = None) -> torch.Tensor:
        """
        Use this transform to transform a set of 3D points. Assumes row major
        ordering of the input points.

        Args:
            points: Tensor of shape (P, 3) or (N, P, 3)
            eps: If eps!=None, the argument is used to clamp the
                last coordinate before performing the final division.
                The clamping corresponds to:
                last_coord := (last_coord.sign() + (last_coord==0)) *
                torch.clamp(last_coord.abs(), eps),
                i.e. the last coordinates that are exactly 0 will
                be clamped to +eps.

        Returns:
            points_out: points of shape (N, P, 3) or (P, 3) depending
            on the dimensions of the transform
        """
        points_batch = points.clone()
        if points_batch.dim() == 2:
            points_batch = points_batch[None]  # (P, 3) -> (1, P, 3)
        if points_batch.dim() != 3:
            msg = "Expected points to have dim = 2 or dim = 3: got shape %r"
            raise ValueError(msg % repr(points.shape))

        N, P, _3 = points_batch.shape
        ones = torch.ones(N, P, 1, dtype=points.dtype, device=points.device)
        points_batch = torch.cat([points_batch, ones], dim=2)

        composed_matrix = self.get_matrix()
        points_out = _broadcast_bmm(points_batch, composed_matrix)
        denom = points_out[..., 3:]  # denominator
        if eps is not None:
            denom_sign = denom.sign() + (denom == 0.0).type_as(denom)
            denom = denom_sign * torch.clamp(denom.abs(), eps)
        points_out = points_out[..., :3] / denom

        # When transform is (1, 4, 4) and points is (P, 3) return
        # points_out of shape (P, 3)
        if points_out.shape[0] == 1 and points.dim() == 2:
            points_out = points_out.reshape(points.shape)

        return points_out

    def transform_normals(self, normals) -> torch.Tensor:
        """
        Use this transform to transform a set of normal vectors.

        Args:
            normals: Tensor of shape (P, 3) or (N, P, 3)

        Returns:
            normals_out: Tensor of shape (P, 3) or (N, P, 3) depending
            on the dimensions of the transform
        """
        if normals.dim() not in [2, 3]:
            msg = "Expected normals to have dim = 2 or dim = 3: got shape %r"
            raise ValueError(msg % (normals.shape,))
        composed_matrix = self.get_matrix()

        # TODO: inverse is bad! Solve a linear system instead
        mat = composed_matrix[:, :3, :3]
        normals_out = _broadcast_bmm(normals, mat.transpose(1, 2).inverse())

        # This doesn't pass unit tests. TODO investigate further
        # if self._lu is None:
        #     self._lu = self._matrix[:, :3, :3].transpose(1, 2).lu()
        # normals_out = normals.lu_solve(*self._lu)

        # When transform is (1, 4, 4) and normals is (P, 3) return
        # normals_out of shape (P, 3)
        if normals_out.shape[0] == 1 and normals.dim() == 2:
            normals_out = normals_out.reshape(normals.shape)

        return normals_out

    def translate(self, *args, **kwargs) -> "Transform3d":
        return self.compose(
            Translate(device=self.device, dtype=self.dtype, *args, **kwargs)
        )

    def scale(self, *args, **kwargs) -> "Transform3d":
        return self.compose(
            Scale(device=self.device, dtype=self.dtype, *args, **kwargs)
        )

    def rotate(self, *args, **kwargs) -> "Transform3d":
        return self.compose(
            Rotate(device=self.device, dtype=self.dtype, *args, **kwargs)
        )

    def rotate_axis_angle(self, *args, **kwargs) -> "Transform3d":
        return self.compose(
            RotateAxisAngle(device=self.device, dtype=self.dtype, *args, **kwargs)
        )

    def clone(self) -> "Transform3d":
        """
        Deep copy of Transforms object. All internal tensors are cloned
        individually.

        Returns:
            new Transforms object.
        """
        other = Transform3d(dtype=self.dtype, device=self.device)
        if self._lu is not None:
            other._lu = [elem.clone() for elem in self._lu]
        other._matrix = self._matrix.clone()
        other._transforms = [t.clone() for t in self._transforms]
        return other

    def to(
        self,
        device: Device,
        copy: bool = False,
        dtype: Optional[torch.dtype] = None,
    ) -> "Transform3d":
        """
        Match functionality of torch.Tensor.to()
        If copy = True or the self Tensor is on a different device, the
        returned tensor is a copy of self with the desired torch.device.
        If copy = False and the self Tensor already has the correct torch.device,
        then self is returned.

        Args:
          device: Device (as str or torch.device) for the new tensor.
          copy: Boolean indicator whether or not to clone self. Default False.
          dtype: If not None, casts the internal tensor variables
              to a given torch.dtype.

        Returns:
          Transform3d object.
        """
        device_ = make_device(device)
        dtype_ = self.dtype if dtype is None else dtype
        skip_to = self.device == device_ and self.dtype == dtype_

        if not copy and skip_to:
            return self

        other = self.clone()

        if skip_to:
            return other

        other.device = device_
        other.dtype = dtype_
        other._matrix = other._matrix.to(device=device_, dtype=dtype_)
        other._transforms = [
            t.to(device_, copy=copy, dtype=dtype_) for t in other._transforms
        ]
        return other

    def cpu(self) -> "Transform3d":
        return self.to("cpu")

    def cuda(self) -> "Transform3d":
        return self.to("cuda")

class FoVPerspectiveCameras(CamerasBase):
    """
    A class which stores a batch of parameters to generate a batch of
    projection matrices by specifying the field of view.
    The definition of the parameters follow the OpenGL perspective camera.

    The extrinsics of the camera (R and T matrices) can also be set in the
    initializer or passed in to `get_full_projection_transform` to get
    the full transformation from world -> ndc.

    The `transform_points` method calculates the full world -> ndc transform
    and then applies it to the input points.

    The transforms can also be returned separately as Transform3d objects.

    * Setting the Aspect Ratio for Non Square Images *

    If the desired output image size is non square (i.e. a tuple of (H, W) where H != W)
    the aspect ratio needs special consideration: There are two aspect ratios
    to be aware of:
        - the aspect ratio of each pixel
        - the aspect ratio of the output image
    The `aspect_ratio` setting in the FoVPerspectiveCameras sets the
    pixel aspect ratio. When using this camera with the differentiable rasterizer
    be aware that in the rasterizer we assume square pixels, but allow
    variable image aspect ratio (i.e rectangle images).

    In most cases you will want to set the camera `aspect_ratio=1.0`
    (i.e. square pixels) and only vary the output image dimensions in pixels
    for rasterization.
    """

    # For __getitem__
    _FIELDS = (
        "K",
        "znear",
        "zfar",
        "aspect_ratio",
        "fov",
        "R",
        "T",
        "degrees",
    )

    _SHARED_FIELDS = ("degrees",)

    def __init__(
        self,
        znear=1.0,
        zfar=100.0,
        aspect_ratio=1.0,
        fov=60.0,
        degrees: bool = True,
        R: torch.Tensor = _R,
        T: torch.Tensor = _T,
        K: Optional[torch.Tensor] = None,
        device: Device = "cpu",
    ) -> None:
        """

        Args:
            znear: near clipping plane of the view frustrum.
            zfar: far clipping plane of the view frustrum.
            aspect_ratio: aspect ratio of the image pixels.
                1.0 indicates square pixels.
            fov: field of view angle of the camera.
            degrees: bool, set to True if fov is specified in degrees.
            R: Rotation matrix of shape (N, 3, 3)
            T: Translation matrix of shape (N, 3)
            K: (optional) A calibration matrix of shape (N, 4, 4)
                If provided, don't need znear, zfar, fov, aspect_ratio, degrees
            device: Device (as str or torch.device)
        """
        # The initializer formats all inputs to torch tensors and broadcasts
        # all the inputs to have the same batch dimension where necessary.
        super().__init__(
            device=device,
            znear=znear,
            zfar=zfar,
            aspect_ratio=aspect_ratio,
            fov=fov,
            R=R,
            T=T,
            K=K,
        )

        # No need to convert to tensor or broadcast.
        self.degrees = degrees

    def compute_projection_matrix(
        self, znear, zfar, fov, aspect_ratio, degrees: bool
    ) -> torch.Tensor:
        """
        Compute the calibration matrix K of shape (N, 4, 4)

        Args:
            znear: near clipping plane of the view frustrum.
            zfar: far clipping plane of the view frustrum.
            fov: field of view angle of the camera.
            aspect_ratio: aspect ratio of the image pixels.
                1.0 indicates square pixels.
            degrees: bool, set to True if fov is specified in degrees.

        Returns:
            torch.FloatTensor of the calibration matrix with shape (N, 4, 4)
        """
        K = torch.zeros((self._N, 4, 4), device=self.device, dtype=torch.float32)
        ones = torch.ones((self._N), dtype=torch.float32, device=self.device)
        if degrees:
            fov = (np.pi / 180) * fov

        if not torch.is_tensor(fov):
            fov = torch.tensor(fov, device=self.device)
        tanHalfFov = torch.tan((fov / 2))
        max_y = tanHalfFov * znear
        min_y = -max_y
        max_x = max_y * aspect_ratio
        min_x = -max_x

        # NOTE: In OpenGL the projection matrix changes the handedness of the
        # coordinate frame. i.e the NDC space positive z direction is the
        # camera space negative z direction. This is because the sign of the z
        # in the projection matrix is set to -1.0.
        # In pytorch3d we maintain a right handed coordinate system throughout
        # so the so the z sign is 1.0.
        z_sign = 1.0

        K[:, 0, 0] = 2.0 * znear / (max_x - min_x)
        K[:, 1, 1] = 2.0 * znear / (max_y - min_y)
        K[:, 0, 2] = (max_x + min_x) / (max_x - min_x)
        K[:, 1, 2] = (max_y + min_y) / (max_y - min_y)
        K[:, 3, 2] = z_sign * ones

        # NOTE: This maps the z coordinate from [0, 1] where z = 0 if the point
        # is at the near clipping plane and z = 1 when the point is at the far
        # clipping plane.
        K[:, 2, 2] = z_sign * zfar / (zfar - znear)
        K[:, 2, 3] = -(zfar * znear) / (zfar - znear)

        return K

    def get_projection_transform(self, **kwargs) -> Transform3d:
        """
        Calculate the perspective projection matrix with a symmetric
        viewing frustrum. Use column major order.
        The viewing frustrum will be projected into ndc, s.t.
        (max_x, max_y) -> (+1, +1)
        (min_x, min_y) -> (-1, -1)

        Args:
            **kwargs: parameters for the projection can be passed in as keyword
                arguments to override the default values set in `__init__`.

        Return:
            a Transform3d object which represents a batch of projection
            matrices of shape (N, 4, 4)

        .. code-block:: python

            h1 = (max_y + min_y)/(max_y - min_y)
            w1 = (max_x + min_x)/(max_x - min_x)
            tanhalffov = tan((fov/2))
            s1 = 1/tanhalffov
            s2 = 1/(tanhalffov * (aspect_ratio))

            # To map z to the range [0, 1] use:
            f1 =  far / (far - near)
            f2 = -(far * near) / (far - near)

            # Projection matrix
            K = [
                    [s1,   0,   w1,   0],
                    [0,   s2,   h1,   0],
                    [0,    0,   f1,  f2],
                    [0,    0,    1,   0],
            ]
        """
        K = kwargs.get("K", self.K)
        if K is not None:
            if K.shape != (self._N, 4, 4):
                msg = "Expected K to have shape of (%r, 4, 4)"
                raise ValueError(msg % (self._N))
        else:
            K = self.compute_projection_matrix(
                kwargs.get("znear", self.znear),
                kwargs.get("zfar", self.zfar),
                kwargs.get("fov", self.fov),
                kwargs.get("aspect_ratio", self.aspect_ratio),
                kwargs.get("degrees", self.degrees),
            )

        # Transpose the projection matrix as PyTorch3D transforms use row vectors.
        transform = Transform3d(
            matrix=K.transpose(1, 2).contiguous(), device=self.device
        )
        return transform

    def unproject_points(
        self,
        xy_depth: torch.Tensor,
        world_coordinates: bool = True,
        scaled_depth_input: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """>!
        FoV cameras further allow for passing depth in world units
        (`scaled_depth_input=False`) or in the [0, 1]-normalized units
        (`scaled_depth_input=True`)

        Args:
            scaled_depth_input: If `True`, assumes the input depth is in
                the [0, 1]-normalized units. If `False` the input depth is in
                the world units.
        """

        # obtain the relevant transformation to ndc
        if world_coordinates:
            to_ndc_transform = self.get_full_projection_transform()
        else:
            to_ndc_transform = self.get_projection_transform()

        if scaled_depth_input:
            # the input is scaled depth, so we don't have to do anything
            xy_sdepth = xy_depth
        else:
            # parse out important values from the projection matrix
            K_matrix = self.get_projection_transform(**kwargs.copy()).get_matrix()
            # parse out f1, f2 from K_matrix
            unsqueeze_shape = [1] * xy_depth.dim()
            unsqueeze_shape[0] = K_matrix.shape[0]
            f1 = K_matrix[:, 2, 2].reshape(unsqueeze_shape)
            f2 = K_matrix[:, 3, 2].reshape(unsqueeze_shape)
            # get the scaled depth
            sdepth = (f1 * xy_depth[..., 2:3] + f2) / xy_depth[..., 2:3]
            # concatenate xy + scaled depth
            xy_sdepth = torch.cat((xy_depth[..., 0:2], sdepth), dim=-1)

        # unproject with inverse of the projection
        unprojection_transform = to_ndc_transform.inverse()
        return unprojection_transform.transform_points(xy_sdepth)

    def is_perspective(self):
        return True

    def in_ndc(self):
        return True

#######################################################################################
##  ██████╗ ███████╗███████╗██╗███╗   ██╗██╗████████╗██╗ ██████╗ ███╗   ██╗███████╗  ##
##  ██╔══██╗██╔════╝██╔════╝██║████╗  ██║██║╚══██╔══╝██║██╔═══██╗████╗  ██║██╔════╝  ##
##  ██║  ██║█████╗  █████╗  ██║██╔██╗ ██║██║   ██║   ██║██║   ██║██╔██╗ ██║███████╗  ##
##  ██║  ██║██╔══╝  ██╔══╝  ██║██║╚██╗██║██║   ██║   ██║██║   ██║██║╚██╗██║╚════██║  ##
##  ██████╔╝███████╗██║     ██║██║ ╚████║██║   ██║   ██║╚██████╔╝██║ ╚████║███████║  ##
##  ╚═════╝ ╚══════╝╚═╝     ╚═╝╚═╝  ╚═══╝╚═╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚══════╝  ##
#######################################################################################                                                                               

def make_device(device: Device) -> torch.device:
    """
    Makes an actual torch.device object from the device specified as
    either a string or torch.device object. If the device is `cuda` without
    a specific index, the index of the current device is assigned.
    Args:
        device: Device (as str or torch.device)
    Returns:
        A matching torch.device object
    """
    device = torch.device(device) if isinstance(device, str) else device
    if device.type == "cuda" and device.index is None:  # pyre-ignore[16]
        # If cuda but with no index, then the current cuda device is indicated.
        # In that case, we fix to that device
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
    return device

def get_world_to_view_transform(
    R: torch.Tensor = _R, T: torch.Tensor = _T
) -> Transform3d:
    """
    This function returns a Transform3d representing the transformation
    matrix to go from world space to view space by applying a rotation and
    a translation.

    PyTorch3D uses the same convention as Hartley & Zisserman.
    I.e., for camera extrinsic parameters R (rotation) and T (translation),
    we map a 3D point `X_world` in world coordinates to
    a point `X_cam` in camera coordinates with:
    `X_cam = X_world R + T`

    Args:
        R: (N, 3, 3) matrix representing the rotation.
        T: (N, 3) matrix representing the translation.

    Returns:
        a Transform3d object which represents the composed RT transformation.

    """
    # TODO: also support the case where RT is specified as one matrix
    # of shape (N, 4, 4).

    if T.shape[0] != R.shape[0]:
        msg = "Expected R, T to have the same batch dimension; got %r, %r"
        raise ValueError(msg % (R.shape[0], T.shape[0]))
    if T.dim() != 2 or T.shape[1:] != (3,):
        msg = "Expected T to have shape (N, 3); got %r"
        raise ValueError(msg % repr(T.shape))
    if R.dim() != 3 or R.shape[1:] != (3, 3):
        msg = "Expected R to have shape (N, 3, 3); got %r"
        raise ValueError(msg % repr(R.shape))

    # Create a Transform3d object
    T_ = Translate(T, device=T.device)
    R_ = Rotate(R, device=R.device)
    return R_.compose(T_)

def get_full_projection_transform(self, **kwargs) -> Transform3d:
    """
    Return the full world-to-camera transform composing the
    world-to-view and view-to-camera transforms.
    If camera is defined in NDC space, the projected points are in NDC space.
    If camera is defined in screen space, the projected points are in screen space.

    Args:
        **kwargs: parameters for the projection transforms can be passed in
            as keyword arguments to override the default values
            set in __init__.
    Setting R and T here will update the values set in init as these
    values may be needed later on in the rendering pipeline e.g. for
    lighting calculations.

    Returns:
        a Transform3d object which represents a batch of transforms
        of shape (N, 3, 3)
    """
    self.R: torch.Tensor = kwargs.get("R", self.R)  # pyre-ignore[16]
    self.T: torch.Tensor = kwargs.get("T", self.T)  # pyre-ignore[16]
    world_to_view_transform = self.get_world_to_view_transform(R=self.R, T=self.T)
    view_to_proj_transform = self.get_projection_transform(**kwargs)
    return world_to_view_transform.compose(view_to_proj_transform)
    
def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))

def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, torch.unbind(euler_angles, -1))
    ]
    # return functools.reduce(torch.matmul, matrices)
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])
