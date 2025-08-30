import xrt_splines
import mitsuba as mi
mi.set_variant("llvm_ad_rgb")
import collections.abc as cabc
import pdb
import time 
import math
import drjit as dr
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from typing import Any
#from scipy import fftpack
matplotlib.use('WebAgg')
import timeit



# If Mitsuba AD-variants used, then DrJIT AD-types are required.
gpu = False
if gpu:
    mi.set_variant("cuda_ad_rgb")
    import cupy as xp
    from drjit.cuda.ad import Array3f, Array3u, Float, Loop, UInt32
else:
    mi.set_variant("llvm_ad_rgb")
    import numpy as xp
    from drjit.llvm.ad import Array3f, Array3u, Float, Loop, UInt32


# Internal Functions ==========================================================
def as_canonical_shape(x, _type) -> tuple:
    # Transform a lone number into tuple-based format.
    if isinstance(x, cabc.Iterable):
        x = tuple(x)
    else:
        x = (x,)
    sh = tuple(map(_type, x))
    return sh


def ray_step(r: mi.Ray3f) -> mi.Ray3f:
    # Advance ray until next unit-step lattice intersection.
    #
    # Parameters
    #   r(o, d): ray to move. (`d` assumed normalized.)
    # Returns
    #   r_next(o_next, d): next ray position on unit-step lattice intersection.
    eps = 1e-4  # threshold for proximity tests with 0
    # Go to local coordinate system.
    o_ref = dr.floor(r.o)
    r_local = mi.Ray3f(o=r.o - o_ref, d=r.d)

    # Find bounding box for ray-intersection tests.
    on_boundary = r_local.o <= eps
    any_boundary = dr.any(on_boundary)
    bbox_border = dr.select(
        any_boundary,
        dr.select(on_boundary, dr.sign(r.d), 1),
        1,
    )
    bbox = mi.BoundingBox3f(
        dr.minimum(0, bbox_border),
        dr.maximum(0, bbox_border),
    )

    # Compute step size to closest bounding box wall.
    # (a1, a2) may contain negative values or Infs.
    # In any case, we must always choose min(a1, a2) > 0.
    _, a1, a2 = bbox.ray_intersect(r_local)
    a_min = dr.minimum(a1, a2)
    a_max = dr.maximum(a1, a2)
    a = dr.select(a_min >= eps, a_min, a_max)

    # Move ray to new position in global coordinates.
    # r_next located on lattice intersection (up to FP error).
    r_next = mi.Ray3f(o=o_ref + r_local(a), d=r.d)
    return r_next


def xray_apply(
    o: Array3f,
    pitch: Array3f,
    N: Array3u,
    I: Float,
    r: mi.Ray3f,
) -> Float:
    # X-Ray Forward-Projection.
    #
    # Parameters
    #   o:     bottom-left coordinate of I[0,0,0]
    #   pitch: cell dimensions \in \bR_{+}
    #   N:     (Nx,Ny,Nz) lattice size
    #   I:     (Nx*Ny*Nz,) cell weights \in \bR [C-order]
    #   r:     (L,) ray descriptors
    # Returns
    #   P:     (L,) forward-projected samples \in \bR
    # Go to normalized coordinates
    ipitch = dr.rcp(pitch)
    rs = r
    r = mi.Ray3f(
        o=(r.o - o) * ipitch,
        d=dr.normalize(r.d * ipitch),
    )

    stride = Array3u(N[1] * N[2], N[2], 1)
    flat_index = lambda i: dr.dot(stride, mi.Point3u(i))  # Point3f (int-valued) -> UInt32

    L = max(dr.shape(r.o)[1], dr.shape(r.d)[1])
    P = dr.zeros(Float, shape=L)  # Forward-Projection samples
    idx_P = dr.arange(UInt32, L)

    # Move (intersecting) rays to volume surface
    bbox_vol = mi.BoundingBox3f(0, Array3f(N))
    active, a1, a2 = bbox_vol.ray_intersect(r)
    a_min = dr.minimum(a1, a2)
    r.o = dr.select(active, r(a_min), r.o)

    r_next = ray_step(r)
    active &= bbox_vol.contains(r_next.o)
    loop = Loop("X-Ray FW Project", lambda: (r, r_next, active))

  
    while loop(active):
        #length = dr.norm((r_next.o - r.o) * pitch)

        #theta = dr.atan(((r_next.o - r.o)[1] + dr.pi/3602)/(dr.pi/3602+(r_next.o - r.o)[0]) )
        
        theta = dr.atan((r_next.o - r.o)[1] /(r_next.o - r.o)[0])
        #theta = dr.select(dr.abs(r_next.d[1]) < 1e-10, dr.pi/2, theta)
        evaluate_spline = radon_transform_zwart_powell_adjusted(theta)
        idx_I = dr.floor(0.5 * (r_next.o + r.o))
        #y is distance between center of the pixel and the ray
        center_pixel = idx_I + 0.5
        center_pixel = center_pixel - r.o
        
        dot = center_pixel[0]*dr.cos(theta) + center_pixel[1]*dr.sin(theta)
        y = dr.sqrt(dr.norm(center_pixel)**2 - dot**2)
        
        length = evaluate_spline(y)






        #move perp direction and get neighbor_left and neighbor_right
        # [...]
        center_line = 0.5 * (r_next.o + r.o)
        '''neighbor_left = center_line

        border_val = (1.)/(dr.cos(theta))  #+ dr.abs(dr.sin(theta)) #+ 0.289 for pi/smth
        neighbor_left[0] = neighbor_left[0] + dr.abs(border_val) * (-dr.sin(theta))
        neighbor_left[1] = neighbor_left[1] + dr.abs(border_val) * dr.cos(theta)'''

        idx_I_left = idx_I
        idx_I_left[0] = idx_I[0] + 1
        idx_I_left[1] = idx_I[1] + 1
        #idx_I_left[2] = 6
        idx_I_left = dr.floor(idx_I_left)
        #y is distance between center of the pixel and the ray
        center_pixel = idx_I_left + 0.5
        center_pixel = center_pixel - r.o
        
        dot = center_pixel[0]*dr.cos(theta) + center_pixel[1]*dr.sin(theta)
        y = dr.sqrt(dr.norm(center_pixel)**2 - dot**2)
        
        length_left = evaluate_spline(y)
        #sum their contribution
        #length = dr.norm((r_next.o - r.o) * pitch)
        

        '''#move perp direction and get neighbor_left and neighbor_right
        # [...]
        center_line = 0.5 * (r_next.o + r.o)

        neighbor_right = center_line

        neighbor_right[0] = neighbor_right[0] + dr.sqrt(2)/2* (dr.sin(theta))
        neighbor_right[1] = neighbor_right[1] + dr.sqrt(2)/2* -dr.cos(theta)

        idx_I_right = dr.floor(neighbor_right)
        #y is distance between center of the pixel and the ray
        center_pixel = idx_I_right + 0.5
        center_pixel = center_pixel - r.o
        
        dot = center_pixel[0]*dr.cos(theta) + center_pixel[1]*dr.sin(theta)
        y = dr.sqrt(dr.norm(center_pixel)**2 - dot**2)
        
        length_right = evaluate_spline(y)'''

        weight = dr.gather(
            type(I),
            I,
            flat_index(idx_I),
            active & dr.all(idx_I >= 0),
            # Careful to disable out-of-bound queries.
            # [This may occur if FP-error caused r_next(above) to not enter the lattice; auto-rectified at next iteration.]
        )

        weight_left = dr.gather(
            type(I),
            I,
            flat_index(idx_I_left),
            active & dr.all(idx_I_left >= 0),
            # Careful to disable out-of-bound queries.
            # [This may occur if FP-error caused r_next(above) to not enter the lattice; auto-rectified at next iteration.]
        )

        '''weight_right = dr.gather(
            type(I),
            I,
            flat_index(idx_I_right),
            active & dr.all(idx_I_right >= 0),
            # Careful to disable out-of-bound queries.
            # [This may occur if FP-error caused r_next(above) to not enter the lattice; auto-rectified at next iteration.]
        )'''


        # Update line integral estimates
        #dr.scatter_reduce(dr.ReduceOp.Add, P, ipitch[2] , idx_P, active)
        dr.scatter_reduce(dr.ReduceOp.Add, P, weight * length + weight_left * length_left  , idx_P, active)
        
        # Walk to next lattice intersection.
        r = r_next
        r_next = ray_step(r)
        active &= bbox_vol.contains(r_next.o)

    return P

def finite_difference(func, h):
    """
    Compute the finite difference operator with step h : (f(y+h/2) - f(y - h/2)) / h
    """
    h = dr.select(dr.abs(h) < 1e-3, 1e-3, h)
    function_ =  lambda x: (func(x+h/2) - func(x - h/2)) / h
    return function_

def radon_transform_zwart_powell_adjusted(theta):
    """
    Compute the Radon transform of the Zwart-Powell box spline for a given angle theta and y value.
    """
    
    # Directions based on theta
    zetas = [
        #(dr.cos(theta) - dr.sin(theta)), #stop here for pice-wise quadratic
        (dr.cos(theta) + dr.sin(theta)), #stop here for pice-wise linear
        dr.sin(theta), #stop here for pice-wise constant
        dr.cos(theta),
    ]
    
    # Using the explicit formula for Radon transform of Zwart-Powell box spline
    #proj = lambda x: np.where(x >= 0, (x** (len(zetas) - 1)) , 0)/math.factorial(len(zetas) - 1)
    #drjit syntax
    proj = lambda x: dr.select(x >= 0, (x** (len(zetas) - 1)) , 0)/math.factorial(len(zetas) - 1)

    for zeta in zetas:

        proj = finite_difference(proj, zeta)
        
    return proj





def xray_adjoint(
    o: Array3f,
    pitch: Array3f,
    N: Array3u,
    P: Float,
    r: mi.Ray3f,
) -> Float:
    # X-Ray Back-Projection.
    #
    # Parameters
    #   o:     bottom-left coordinate of I[0,0,0]
    #   pitch: cell dimensions \in \bR_{+}
    #   N:     (Nx,Ny,Nz) lattice size
    #   P:     (L,) X-Ray samples \in \bR
    #   r:     (L,) ray descriptors
    # Returns
    #   I: (Nx*Ny*Nz,) back-projected samples \in \bR [C-order]

    # Go to normalized coordinates
    ipitch = dr.rcp(pitch)
    r = mi.Ray3f(
        o=(r.o - o) * ipitch,
        d=dr.normalize(r.d * ipitch),
    )

    stride = Array3u(N[1] * N[2], N[2], 1)
    flat_index = lambda i: dr.dot(stride, mi.Point3u(i))  # Point3f (int-valued) -> UInt32

    I = dr.zeros(Float, dr.prod(N)[0])  # Back-Projection samples

    # Move (intersecting) rays to volume surface
    bbox_vol = mi.BoundingBox3f(0, Array3f(N))
    active, a1, a2 = bbox_vol.ray_intersect(r)
    a_min = dr.minimum(a1, a2)
    r.o = dr.select(active, r(a_min), r.o)

    r_next = ray_step(r)
    active &= bbox_vol.contains(r_next.o)
    active &= dr.neq(P, 0)
    loop = Loop("X-Ray BW Project", lambda: (r, r_next, active))
    while loop(active):
        length = dr.norm((r_next.o - r.o) * pitch)

        idx_I = dr.floor(0.5 * (r_next.o + r.o))
        dr.scatter_reduce(
            dr.ReduceOp.Add,
            I,
            P * length,
            flat_index(idx_I),
            active & dr.all(idx_I >= 0),
            # Careful to disable out-of-bound queries.
            # [This may occur if FP-error caused r_next(above) to not enter the lattice; auto-rectified at next iteration.]
        )

        # Walk to next lattice intersection.
        r = r_next
        r_next = ray_step(r)
        active &= bbox_vol.contains(r_next.o)
    return I


# =============================================================================


# Public Interface ============================================================
class XRayTransform:
    def __init__(
        self,
        arg_shape,
        ray_spec,
        origin=0,
        pitch=1,
    ):
        # arg_shape: (D,)
        # ray_spec: dict(n=n_spec, t=t_spec)
        #   n_spec = (N_ray, D)
        #   t_spec = (N_ray, D)
        # origin: float | (D,)
        # pitch: float | (D,)
        self._arg_shape = as_canonical_shape(arg_shape, int)
        D = len(self._arg_shape)
        assert D == 3, "[D=3 supported only for now; embed 2D problems into 3D.]"

        origin = as_canonical_shape(origin, float)
        if len(origin) == 1:
            origin = origin * D
        self._origin = origin

        pitch = as_canonical_shape(pitch, float)
        if len(pitch) == 1:
            pitch = pitch * D
        self._pitch = pitch

        self._ray_n = ray_spec["n"]
        self._ray_t = ray_spec["t"]
        self._N_ray = len(self._ray_n)

    def apply(self, arr):
        # arr: (*arg_shape,) NP/CP
        # out: (N_ray,)

        P = xray_apply(
            o=Array3f(*self._origin),
            pitch=Array3f(*self._pitch),
            N=Array3u(*self._arg_shape),
            I=Float(arr.reshape(-1)),
            r=mi.Ray3f(
                o=Array3f(*self._ray_t.T),
                d=Array3f(*self._ray_n.T),
            ),
        )
        out = xp.asarray(P)
        return out

    def adjoint(self, arr):
        # arr: (N_ray,) NP/CP
        # out: (*arg_shape,)

        I = xray_adjoint(
            o=Array3f(*self._origin),
            pitch=Array3f(*self._pitch),
            N=Array3u(*self._arg_shape),
            P=Float(arr),
            r=mi.Ray3f(
                o=Array3f(*self._ray_t.T),
                d=Array3f(*self._ray_n.T),
            ),
        )

        out = xp.asarray(I).reshape(self._arg_shape)
        return out


def set_detector_param(N_phantom, source, sdd, sod, detector_nb, detector_length, left_bottom_position = np.array([0.0])):
        # angles of source-detector line with x axis
        center_image_coords = (N_phantom/2, N_phantom/2) #+0.5 to be in the middle of the pixel
        alpha = np.arctan((source[1] - center_image_coords[1]) / (source[0] - center_image_coords[0]))
        # detector positions
        detector_screen_abs = np.zeros(detector_nb)
        detector_ord = np.linspace(-detector_length/2, detector_length/2, num=detector_nb)

        # rotation of the detector

        alpha = alpha
        detector_screen_abs_rot = detector_screen_abs * np.cos(alpha) - detector_ord * np.sin(alpha)
        detector_ord_rot = detector_screen_abs * np.sin(alpha) + detector_ord * np.cos(alpha)


        detector_abs = source[0] + detector_screen_abs_rot
        detector_ord = source[1] + detector_ord_rot
        #move the detector to the right position with sdd
        detector_abs = detector_abs + np.abs((sdd) * np.cos(alpha)) * np.sign(center_image_coords[0] - source[0])
        detector_ord = detector_ord + np.abs((sdd) * np.sin(alpha)) * np.sign(center_image_coords[1] - source[1])
        #flip vectors if source is on the right of the image
        detector_abs = np.where(center_image_coords[0] - source[0] <= 0, np.flip(detector_abs), detector_abs)
        detector_ord = np.where(center_image_coords[0] - source[0] <= 0, np.flip(detector_ord), detector_ord)
        
        return detector_abs, detector_ord
    




#++++++++++++++++++MAIN SCRIPT++++++++++++++++++++++++++++++++




if __name__ == "__main__":
    
    N_side = 5  # N_px = N_side**2
    pitch = 1.  # m/px [can differ per axis]
    #phantom = np.zeros((N_side, N_side, 1), dtype=np.float32) #x,y,z(depth)
    phantom = np.zeros((N_side, N_side), dtype=np.float32)
    #indices = np.random.randint(0, N_side, size=(1, 2))
    #phantom[indices[:,0], indices[:,1], 0] = 1.
    phantom[N_side//2,N_side//2] = 1
    arg_shape = phantom.shape
    plt.figure('Phantom')
    plt.imshow(phantom)

    N_angle = 1
    N_offset = 300
  
    # Let's build the necessary components to instantiate the operator . ========================
    angles = np.linspace(0, np.pi, N_angle, endpoint=False)

    #angles = np.where(np.abs(angles - np.pi/2) < 0.5e-1, angles + 1e-3, angles)
    #angles = np.where(np.abs(angles - 3*np.pi/2) < 0.5e-1, angles + 1e-3, angles)
    #angles = np.where(np.abs(angles - np.pi) < 0.5e-1, angles + 1e-3, angles)
    #angles = np.where(np.abs(angles - 0) < 0.5e-1, angles + 1e-3, angles)

    n = np.stack([np.cos(angles), np.sin(angles)], axis=1)

    t = n[:, [1, 0]] * np.r_[-1, 1]  # <n, t> = 0
    t_max = pitch * N_side / 2 * 1.1  # 10% over ball radius
    t_offset = np.linspace(-t_max, t_max, N_offset, endpoint=True)

    n_spec = np.broadcast_to(n.reshape(N_angle, 1, 2), (N_angle, N_offset, 2))  # (N_angle, N_offset, 2)
    t_spec = t.reshape(N_angle, 1, 2) * t_offset.reshape(N_offset, 1)  # (N_angle, N_offset, 2)
    t_spec += pitch * N_side / 2  # Move anchor point to the center of volume.
    extra = np.ones((1,N_angle*N_offset,1)) 
    #t_spec = np.concatenate((t_spec, extra* 0.5), axis=2)
    #n_spec = np.concatenate((n_spec, extra * 0.), axis=2)
    origin = (0., 0.)
    pitch = (1.,1.)


    '''op = XRayTransform(
        arg_shape=arg_shape,
        ray_spec=dict(t=t_spec[0], n=n_spec[0]),
        origin=origin,
        pitch=pitch,
    )'''
    op = xrt_splines.RayXRT(
        arg_shape=arg_shape,
        t_spec=t_spec[0], 
        n_spec=n_spec[0],
        origin=origin,
        pitch=pitch,
    )
    
    fwd = op.apply(phantom)
    print(fwd)
    #fwd = fwd.reshape()
    plt.figure('X-ray spline Projection')
    
    plt.plot(np.linspace(-2.5,2.5, num= len(fwd)), fwd)
    plt.show()








    '''
    NN = 200
    data = np.zeros((NN, NN, 1))
    #5 random coefficients
    data[NN//2, NN//2, 0] = 1
    indices = np.random.randint(0, NN, size=(5, 2))
    data[indices[:,0], indices[:,1], 0] = 1
    plt.figure('data')
    plt.imshow(data[:,:,0], cmap='gray')
    plt.show()

    pitch = (1, 1, 1)
    sinogram_3D_real = data
    N_angles = 1 #data.shape[0]
    N = data.shape[2]
    Nz = data.shape[1]
    arg_shape = data.shape
    N_detectors = 1 # image sized detector
    angles = np.linspace(np.pi/4, 2*np.pi, N_angles, endpoint=False)
    ray_t = np.ones((1, 3))*0.5
    ray_t[0,0] = 12
    ray_t[0,1] = 40
    ray_n = np.ones((1, 3))
    ray_n[0,1] = 0.4
    ray_n[0,2] = 0
    print("ray_t", ray_t)
    print("ray_n", ray_n)

    #parallel beam setup, 100 detectors, 100 angles
    N_detectors = 4000
    N_angles = 1000
    def rotate_points(pos_detectors, theta, center):
        """
        Rotate 2D points by theta radians around a specified center.
        
        Args:
        - pos_detectors (numpy array): Array of shape (N, 2) representing N points.
        - theta (float): Rotation angle in radians.
        - center (list or tuple): The center of rotation as [x, y].
        
        Returns:
        - numpy array: Rotated points of shape (N, 2).
        """
        # Translate points so that the center of rotation becomes the origin
        translated_points = pos_detectors - center
        
        # Define the rotation matrix
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])
        
        # Use matrix multiplication to rotate the points
        rotated_points = translated_points @ rotation_matrix.T
        
        # Translate points back to their original positions
        final_points = rotated_points + center
        
        return final_points

    angles = np.linspace(0, 2*np.pi, N_angles, endpoint=False)
    print(angles)
    angles = np.where(np.abs(angles - np.pi/2) < 0.5e-1, angles + 1e-3, angles)
    angles = np.where(np.abs(angles - 3*np.pi/2) < 0.5e-1, angles + 1e-3, angles)
    angles = np.where(np.abs(angles - np.pi) < 0.5e-1, angles + 1e-3, angles)
    angles = np.where(np.abs(angles - 0) < 0.5e-1, angles + 1e-3, angles)

    ray_t = np.ones((N_detectors*N_angles, 3))*0.5
    ray_n = np.zeros((N_detectors*N_angles, 3))
    abs_init_detectors = np.zeros(N_detectors)
    ord_init_detectors = np.linspace(-NN/2, NN/2, N_detectors)

    for i, angle in enumerate(angles):
        ray_t[i*N_detectors:(i+1)*N_detectors,0] = rotate_points(np.array([abs_init_detectors, ord_init_detectors]).T, angle, np.array([0, 0]))[:,0]
        ray_t[i*N_detectors:(i+1)*N_detectors,1] = rotate_points(np.array([abs_init_detectors, ord_init_detectors]).T, angle, np.array([0, 0]))[:,1]
        #offset to center the image
        ray_t[i*N_detectors:(i+1)*N_detectors,0] += NN/2
        ray_t[i*N_detectors:(i+1)*N_detectors,1] += NN//2
        ray_n[i*N_detectors:(i+1)*N_detectors,0] = np.cos(angle)
        ray_n[i*N_detectors:(i+1)*N_detectors,1] = np.sin(angle)
        ray_n[i*N_detectors:(i+1)*N_detectors,2] = 0

    #replace zero coefficients by 1e-10 to avoid division by zero

    origin = (0, 0, 0)

    op = XRayTransform(
        arg_shape=arg_shape,
        ray_spec=dict(t=ray_t, n=ray_n),
        origin=origin,
        pitch=pitch,
    )

    fwd = op.apply(sinogram_3D_real)
    fwd = fwd.reshape(N_angles, N_detectors)
    print(fwd.shape)
    plt.imshow(fwd)
    plt.show()
    
    '''


    '''
    y = sinogram_3D_real.reshape(1,-1).flatten()
    J = 0.5*pynorm.SquaredL2Norm(dim=y.shape[0]).asloss(y)*Myclone 
    #breakpoint()

    R = pynorm.SquaredL1Norm(dim = N*N*Nz) 
    reg_param = 0.3
    f = J
    g = reg_param*R
    R = lamb * TVFunc(arg_shape=image.shape)
    mySolver = PGD(f = f, g=g, show_progress=True, verbosity=1)
    # definition stopping critertion
    #stop_crit = pystop.RelError(eps=1e-2, satisfy_all=True)
    stop_crit = pystop.MaxIter(12) 
    # run solver
    x0 = np.zeros(N*N*Nz)

    with pycrt.Precision(pycrt.Width.SINGLE):
        mySolver.fit(mode=Mode.BLOCK, x0=x0, acceleration=True, stop_crit=stop_crit)

    opt_jax = mySolver.solution()

    np.save('opt_3D_l1.npy', opt_jax)
    plt.figure('opt3DL1')
    plt.imshow(opt_jax.reshape(N, N, Nz)[:,:,Nz//2], cmap='gray')
    plt.show()

    


    
    starttime = timeit.default_timer()
    A = FanBeam_JAX(N, N, sdd, det_length, N, num_angles)

    y = fwd.reshape(1,-1).flatten()
    J = 0.5*pynorm.SquaredL2Norm(dim=num_angles*N).asloss(y)*A 
    R = lamb * TVFunc(arg_shape=image.shape)
    reg_param = 1
    f = J
    g = reg_param*R

    # definition solver
    mySolver = PGD(f = f, show_progress=True, verbosity=1)
    # definition stopping critertion
    stop_crit = pystop.RelError(eps=1e-2, satisfy_all=True)
    #stop_crit = pystop.MaxIter(1) |pystop.RelError(eps=1e-6) 
    # run solver
    x0 = np.zeros(N*N)
    stop = timeit.default_timer()
    print('time : ', stop - starttime)

    starttime = timeit.default_timer()
    with pycrt.Precision(pycrt.Width.SINGLE):
        mySolver.fit(mode=Mode.BLOCK, x0=x0, acceleration=True, stop_crit=stop_crit)

    opt_jax = mySolver.solution()
    stop = timeit.default_timer()
    print('time : ', stop - starttime)
    plt.figure('opt_jax_l1')
    plt.imshow(opt_jax.reshape((N,N)), cmap='gray')
    plt.colorbar()
    '''

    '''
    op = XRayTransform(
        arg_shape=arg_shape,
        ray_spec=dict(t=ray_t, n=ray_n),
        origin=origin,
        pitch=pitch,
    )


    t_start = time.time()
    I_BW = op.adjoint(sinogram_3D_real.reshape(-1))
    t_end = time.time()
    #save
    np.save('I_BW2.npy', I_BW)
    print(f"BW: {t_end-t_start} [s]")
    print(I_BW.shape)
   
    plt.figure("BW")
    plt.imshow(I_BW.reshape(N, N, Nz)[:,:,Nz//2], cmap="gray")
    plt.show()
    '''
    

    


    