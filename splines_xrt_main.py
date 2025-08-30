import numpy as np 
import pyxu.experimental.xray as pxr
import pyxu.opt.stop as pxst
import matplotlib.pyplot as plt
import xrt_splines
import pat_xrt
import matplotlib
import cupy as cp
from cupyx.profiler import benchmark
from skimage.transform import iradon

matplotlib.use('WebAgg')

# Launcher for GPU/CPU CPWL spline X-ray projection

# Use GPU number 0.
cp.cuda.Device(0).use()

#plt.rcParams['text.usetex'] = True
if __name__ == "__main__":
    
    N_side = 200 # N_px = N_side**2
    pitch = 1.  # m/px [can differ per axis]
    phantom = np.zeros((N_side, N_side))
    
    #phantom[N_side//2-30: N_side//2+30, N_side//2-30: N_side//2+30 ] = 1
    xx = np.linspace(-N_side//2,N_side//2, num=N_side)
    yy = xx
    X, Y = np.meshgrid(xx,yy)
    r = 700
    mask = np.sqrt(X**2 + Y**2) < r
    phantom = np.exp(-((X-500)**2 + (Y+400)**2)/400000) + 0.5*np.exp(-((X+700)**2 + (Y-700)**2)/400000)

    arg_shape = phantom.shape

    plt.figure('Phantom')
    plt.imshow(phantom)

    N_angle  = 1
    N_offset = 1
   
    # Let's build the necessary components to instantiate the operator . ========================
    angles = np.linspace(np.pi/2, 2*np.pi, N_angle, endpoint=False)

    n = np.stack([np.cos(angles), np.sin(angles)], axis=1)

    t        = n[:, [1, 0]] * np.r_[-1, 1]  # <n, t> = 0
    t_max    = pitch * (N_side-1) / 2 * 1.  # 10% over ball radius
    t_offset = np.linspace(-t_max, t_max, N_offset, endpoint=True)

    n_spec  = np.broadcast_to(n.reshape(N_angle, 1, 2), (N_angle, N_offset, 2))  # (N_angle, N_offset, 2)
    t_spec  = t.reshape(N_angle, 1, 2) * t_offset.reshape(N_offset, 1)  # (N_angle, N_offset, 2)
    t_spec += pitch * N_side / 2  # Move anchor point to the center of volume.
    extra   = np.ones((1,N_angle*N_offset,1))
    origin  = (0., 0.)
    pitch   = (1.,1.)

    # Convert to cupy arrays
    phantom = cp.array(phantom)
    t_spec  = cp.array(t_spec)
    n_spec  = cp.array(n_spec)
    

    op_splines_1 = xrt_splines.RayXRT(
        deg=1,
        arg_shape=arg_shape,
        t_spec=t_spec.reshape(-1,2), 
        n_spec=n_spec.reshape(-1,2),
        origin=origin,
        pitch=pitch,
    )
    op_splines_2 = xrt_splines.RayXRT(
        deg=2,
        arg_shape=arg_shape,
        t_spec=t_spec.reshape(-1,2), 
        n_spec=n_spec.reshape(-1,2),
        origin=origin,
        pitch=pitch,
    )

    op_pixels = pxr.XRayTransform.init(
        arg_shape=arg_shape,
        t_spec=t_spec.reshape(-1,2), 
        n_spec=n_spec.reshape(-1,2),
        origin=origin,
        pitch=pitch,
    )


    bp = op_splines_1.adjoint(cp.ones(N_offset).reshape(-1))
    plt.figure('trace back bp')
    plt.imshow(bp.get().reshape(phantom.shape), cmap='gray')
    plt.show()
    breakpoint()

    #N_angle  = 50
    N_offset_true = 1000
    pitch = 1.
   
    # Let's build the necessary components to instantiate the operator . ========================
    angles = np.linspace(0, 2*np.pi, N_angle, endpoint=False)

    n = np.stack([np.cos(angles), np.sin(angles)], axis=1)

    t        = n[:, [1, 0]] * np.r_[-1, 1]  # <n, t> = 0
    t_max    = pitch * N_side / 2 * 1.1  # 10% over ball radius
    t_offset = np.linspace(-t_max, t_max, N_offset_true, endpoint=True)

    n_spec_true  = np.broadcast_to(n.reshape(N_angle, 1, 2), (N_angle, N_offset_true, 2))  # (N_angle, N_offset_true, 2)
    t_spec_true  = t.reshape(N_angle, 1, 2) * t_offset.reshape(N_offset_true, 1)  # (N_angle, N_offset_true, 2)
    t_spec_true += pitch * N_side / 2  # Move anchor point to the center of volume.
    extra   = np.ones((1,N_angle*N_offset_true,1))
    origin  = (0., 0.)
    pitch   = (1.,1.)

    # Convert to cupy arrays
    phantom = cp.array(phantom)
    t_spec_true  = cp.array(t_spec_true)
    n_spec_true  = cp.array(n_spec_true)

    op_true = pxr.XRayTransform.init( #for reconstruction
        arg_shape=arg_shape,
        t_spec=t_spec_true.reshape(-1,2), 
        n_spec=n_spec_true.reshape(-1,2),
        origin=origin,
        pitch=pitch,
    )

    fwd_splines_1 = op_splines_1.apply(phantom.reshape(-1))
    fwd_splines_2 = op_splines_2.apply(phantom.reshape(-1))
    fwd_pixels = op_pixels.apply(phantom.reshape(-1))

    y_data = fwd_pixels

    # benchmarking 
    '''
    bench_splines_1 = benchmark(op_splines_1.apply, (phantom.reshape(-1),), n_repeat=10)
    print("splines : ", bench_splines_1)
    bench_pixels    = benchmark(op_pixels.apply, (phantom.reshape(-1),), n_repeat=10)
    print("pixels : ", bench_pixels)
    breakpoint()
    '''

    fwd_splines_1 = fwd_splines_1.reshape((N_angle, N_offset)).get() #remove .get() for CPU
    fwd_splines_2 = fwd_splines_2.reshape((N_angle, N_offset)).get() #remove .get() for CPU
    fwd_pixels = fwd_pixels.reshape((N_angle, N_offset)).get() #remove .get() for CPU

    stop_crit = pxst.MaxIter(30) #30 before

    oversampling = 50
    op_splines_1 = xrt_splines.RayXRT( #for reconstruction
        deg=1,
        arg_shape=(oversampling, oversampling),
        t_spec=t_spec.reshape(-1,2), 
        n_spec=n_spec.reshape(-1,2),
        origin=origin,
        pitch=(phantom.shape[0]/oversampling, phantom.shape[0]/oversampling)
    )
    op_splines_2 = xrt_splines.RayXRT( #for reconstruction
        deg=2,
        arg_shape=(oversampling, oversampling),
        t_spec=t_spec.reshape(-1,2), 
        n_spec=n_spec.reshape(-1,2),
        origin=origin,
        pitch=(phantom.shape[0]/oversampling, phantom.shape[0]/oversampling)
    )
    op_pixels = pxr.XRayTransform.init( #for reconstruction
        arg_shape=(oversampling, oversampling),
        t_spec=t_spec.reshape(-1,2), 
        n_spec=n_spec.reshape(-1,2),
        origin=origin,
        pitch=(phantom.shape[0]/oversampling, phantom.shape[0]/oversampling)
    )

    recon_pixels = op_pixels.pinv(y_data, damp=1., kwargs_fit=dict(stop_crit=stop_crit)).reshape((oversampling, oversampling))
    recon_box_1 = op_splines_1.pinv(y_data, damp=1., kwargs_fit=dict(stop_crit=stop_crit)).reshape((oversampling, oversampling))
    recon_box_2 = op_splines_2.pinv(y_data, damp=1., kwargs_fit=dict(stop_crit=stop_crit)).reshape((oversampling, oversampling))


    plt.figure('sino')
    plt.imshow(fwd_splines_1.reshape((N_angle, N_offset)).T)

    plt.figure('recon pix')
    plt.imshow(recon_pixels.get(), cmap='gray')

    plt.figure('recon splines 1')
    plt.imshow(recon_box_1.get(), cmap='gray')

    plt.figure('recon splines 2')
    plt.imshow(recon_box_2.get(), cmap='gray')

    plt.figure('fwd - recon')
    fwd = op_splines_1.apply(recon_box_1.reshape(-1)).get()
    fwd_splines = fwd.reshape((N_angle, N_offset))
    plt.plot(fwd_splines[0], c='green', label='splines 2')
    fwd = op_pixels.apply(recon_pixels.reshape(-1)).get()
    fwd_pixels = fwd.reshape((N_angle, N_offset))
    plt.plot(fwd_pixels[0], c='red', label='pixels')
    true_proj = y_data.get().reshape((N_angle, N_offset))
    plt.plot(true_proj[0], c='blue', label='true')
    print('sample normal')
    norm1 = np.linalg.norm(true_proj - fwd_splines)
    print('norm diff splines : ', norm1)
    norm2 = np.linalg.norm(true_proj - fwd_pixels)
    print('norm diff pixels  : ', norm2)
    plt.legend()
    






    #N_angle  = 50
    N_offset = 1000 
    pitch = 1.
   
    # Let's build the necessary components to instantiate the operator . ========================
    angles = np.linspace(0, 2*np.pi, N_angle, endpoint=False)

    n = np.stack([np.cos(angles), np.sin(angles)], axis=1)

    t        = n[:, [1, 0]] * np.r_[-1, 1]  # <n, t> = 0
    t_max    = pitch * N_side / 2 * 1.1  # 10% over ball radius
    t_offset = np.linspace(-t_max, t_max, N_offset, endpoint=True)

    n_spec  = np.broadcast_to(n.reshape(N_angle, 1, 2), (N_angle, N_offset, 2))  # (N_angle, N_offset, 2)
    t_spec  = t.reshape(N_angle, 1, 2) * t_offset.reshape(N_offset, 1)  # (N_angle, N_offset, 2)
    t_spec += pitch * N_side / 2  # Move anchor point to the center of volume.
    extra   = np.ones((1,N_angle*N_offset,1))
    origin  = (0., 0.)
    pitch   = (1.,1.)

    # Convert to cupy arrays
    phantom = cp.array(phantom)
    t_spec  = cp.array(t_spec)
    n_spec  = cp.array(n_spec)

    op_splines_1 = xrt_splines.RayXRT( #for reconstruction
        deg=1,
        arg_shape=(oversampling, oversampling),
        t_spec=t_spec.reshape(-1,2), 
        n_spec=n_spec.reshape(-1,2),
        origin=origin,
        pitch=(phantom.shape[0]/oversampling,phantom.shape[0]/oversampling)
    )
    op_splines_2 = xrt_splines.RayXRT( #for reconstruction
        deg=2,
        arg_shape=(oversampling, oversampling),
        t_spec=t_spec.reshape(-1,2), 
        n_spec=n_spec.reshape(-1,2),
        origin=origin,
        pitch=(phantom.shape[0]/oversampling,phantom.shape[0]/oversampling)
    )
    op_pixels = pxr.XRayTransform.init( #for reconstruction
        arg_shape=(oversampling, oversampling),
        t_spec=t_spec.reshape(-1,2), 
        n_spec=n_spec.reshape(-1,2),
        origin=origin,
        pitch=(phantom.shape[0]/oversampling,phantom.shape[0]/oversampling)
    )



    plt.figure('fwd - recon (many offsets)')
    fwd = op_splines_2.apply(recon_box_2.reshape(-1)).get()
    fwd_splines = fwd.reshape((N_angle, N_offset))
    plt.plot(fwd_splines[10], c='green', label='splines 2')
    fwd = op_pixels.apply(recon_pixels.reshape(-1)).get()
    fwd_pixels = fwd.reshape((N_angle, N_offset))
    plt.plot(fwd_pixels[10], c='red', label='pixels')
    true_proj = op_true.apply(phantom.reshape(-1)).get()
    true_proj = true_proj.reshape((N_angle, N_offset_true))
    plt.plot(true_proj[10], c='blue', label='true')
    plt.legend()

    print('Oversampled')
    norm1 = np.linalg.norm(true_proj - fwd_splines)
    print('norm diff splines : ', norm1)
    norm2 = np.linalg.norm(true_proj - fwd_pixels)
    print('norm diff pixels  : ', norm2)
    plt.show()
    breakpoint()
    















    #visualisation of spline projections
    plt.figure('X-ray spline Projection')
    for k in range(len(angles)):
        num = k
        plt.plot(np.linspace(-N_side/2, N_side/2, num= len(fwd_pixels[num])), fwd_pixels[num], label=r'Pixel, $\theta=0$')
        plt.plot(np.linspace(-N_side/2, N_side/2, num= len(fwd_splines_1[num])), fwd_splines_1[num], label=r'Linear Spline, $\theta=0$')
        plt.plot(np.linspace(-N_side/2, N_side/2, num= len(fwd_splines_2[num])), fwd_splines_2[num], label=r'Quadratic Spline, $\theta=0$')

        plt.legend()

    

    pix = fwd_pixels[0]/np.sum(fwd_pixels[0])
    conv_test = np.convolve(fwd_pixels[0], pix, mode='same')
    plt.figure('separable splines convolution : linear')
    plt.plot(conv_test)

    conv_test2 = np.convolve(conv_test, conv_test, mode='same')
    plt.figure('separable splines convolution : quad')
    plt.plot(conv_test2)


    plt.figure('Spline construction : CPWL')
    for k in range(len(angles)):
        num = k
        
        #plt.plot(np.linspace(-N_side/2, N_side/2, num= len(fwd_splines_1[num])), fwd_splines_1[num])
        plt.fill(np.linspace(-N_side/2, N_side/2, num= len(fwd_splines_1[num])), fwd_splines_1[num], alpha=0.35)
        
        phantom[N_side//2,N_side//2] = 0
        phantom[N_side//2,N_side//2+1] = 4
        fwd_splines_1 = op_splines_1.apply(phantom.reshape(-1))
        fwd_splines_1 = fwd_splines_1.reshape((N_angle, N_offset)).get() #remove .get() for CPU
        #plt.plot(np.linspace(-N_side/2, N_side/2, num= len(fwd_splines_1[num])), fwd_splines_1[num])
        plt.fill(np.linspace(-N_side/2, N_side/2, num= len(fwd_splines_1[num])), fwd_splines_1[num], alpha=0.35)
        
        
        phantom[N_side//2,N_side//2+1] = 0
        phantom[N_side//2,N_side//2-1] = 2
        fwd_splines_1 = op_splines_1.apply(phantom.reshape(-1))
        fwd_splines_1 = fwd_splines_1.reshape((N_angle, N_offset)).get() #remove .get() for CPU
        #plt.plot(np.linspace(-N_side/2, N_side/2, num= len(fwd_splines_1[num])), fwd_splines_1[num])
        plt.fill(np.linspace(-N_side/2, N_side/2, num= len(fwd_splines_1[num])), fwd_splines_1[num], alpha=0.35)
        
        phantom[N_side//2,N_side//2] = 1
        phantom[N_side//2,N_side//2+1] = 4
        phantom[N_side//2,N_side//2-1] = 2
        fwd_splines_1 = op_splines_1.apply(phantom.reshape(-1))
        fwd_splines_1 = fwd_splines_1.reshape((N_angle, N_offset)).get() #remove .get() for CPU
        plt.plot(np.linspace(-N_side/2, N_side/2, num= len(fwd_splines_1[num])), fwd_splines_1[num], color='k')
        
        



    plt.figure('Spline construction : CPWQ')
    for k in range(len(angles)):
        num = k
        
        #plt.plot(np.linspace(-N_side/2, N_side/2, num= len(fwd_splines_1[num])), fwd_splines_1[num])
        plt.fill(np.linspace(-N_side/2, N_side/2, num= len(fwd_splines_2[num])), fwd_splines_2[num], alpha=0.35)
        
        phantom[N_side//2,N_side//2] = 0
        phantom[N_side//2,N_side//2-1] = 0
        phantom[N_side//2,N_side//2+1] = 4
        fwd_splines_2 = op_splines_2.apply(phantom.reshape(-1))
        fwd_splines_2 = fwd_splines_2.reshape((N_angle, N_offset)).get() #remove .get() for CPU
        #plt.plot(np.linspace(-N_side/2, N_side/2, num= len(fwd_splines_1[num])), fwd_splines_1[num])
        plt.fill(np.linspace(-N_side/2, N_side/2, num= len(fwd_splines_2[num])), fwd_splines_2[num], alpha=0.35)
        
        
        phantom[N_side//2,N_side//2+1] = 0
        phantom[N_side//2,N_side//2-1] = 2
        fwd_splines_2 = op_splines_2.apply(phantom.reshape(-1))
        fwd_splines_2 = fwd_splines_2.reshape((N_angle, N_offset)).get() #remove .get() for CPU
        #plt.plot(np.linspace(-N_side/2, N_side/2, num= len(fwd_splines_1[num])), fwd_splines_1[num])
        plt.fill(np.linspace(-N_side/2, N_side/2, num= len(fwd_splines_2[num])), fwd_splines_2[num], alpha=0.35)
        
        phantom[N_side//2,N_side//2] = 1
        phantom[N_side//2,N_side//2+1] = 4
        phantom[N_side//2,N_side//2-1] = 2
        fwd_splines_2 = op_splines_2.apply(phantom.reshape(-1))
        fwd_splines_2 = fwd_splines_2.reshape((N_angle, N_offset)).get() #remove .get() for CPU
        plt.plot(np.linspace(-N_side/2, N_side/2, num= len(fwd_splines_2[num])), fwd_splines_2[num], color='k')
        
    #phantom[N_side//2,N_side//2] = 0
    phantom[N_side//2,N_side//2+1] = 0
    phantom[N_side//2,N_side//2-1] = 0
    fwd_splines_1 = op_splines_1.apply(phantom.reshape(-1))
    fwd_splines_2 = op_splines_2.apply(phantom.reshape(-1))
    fwd_pixels = op_pixels.apply(phantom.reshape(-1))

    fwd_splines_1 = fwd_splines_1.reshape((N_angle, N_offset)).get() #remove .get() for CPU
    fwd_splines_2 = fwd_splines_2.reshape((N_angle, N_offset)).get() #remove .get() for CPU
    fwd_pixels = fwd_pixels.reshape((N_angle, N_offset)).get() #remove .get() for CPU

    plt.figure('X-ray Box-splines')

    for k in range(len(angles)):
        num = k
        #plt.plot(np.linspace(-N_side/2, N_side/2, num= len(fwd_pixels[num])), fwd_pixels[num], label=r'Pixel, $\theta=0$')
        #plt.plot(np.linspace(-N_side/2, N_side/2, num= len(fwd_splines_1[num])), fwd_splines_1[num], label=r'Linear Spline, $\theta=0$', c='blue')
        plt.plot(np.linspace(-N_side/2, N_side/2, num= len(fwd_splines_2[num])), fwd_splines_2[num], label=r'Quadratic Spline, $\theta=0$', c='red')

    plt.show()
    breakpoint()

    plt.figure('X-ray separable splines via convolution')

    for k in range(len(angles)):
        num = k
        plt.plot(np.linspace(-N_side/2, N_side/2, num= len(fwd_pixels[num])), fwd_pixels[num], label=r'Pixel, $\theta=0$')
        conv = np.convolve(fwd_pixels[num],pix, mode='same')
        plt.plot(np.linspace(-N_side/2, N_side/2, num= len(fwd_pixels[num])), conv, label=r'conv deg 1, $\theta=0$')
        plt.plot(np.linspace(-N_side/2, N_side/2, num= len(fwd_splines_1[num])), fwd_splines_1[num], label=r'Linear Spline, $\theta=0$')
        #plt.plot(np.linspace(-N_side/2, N_side/2, num= len(fwd_splines_2[num])), fwd_splines_2[num], label=r'Quadratic Spline, $\theta=0$')

        plt.legend()
        


    N_side = 512 # N_px = N_side**2
    pitch = 1.  # m/px [can differ per axis]

    phantom = np.zeros((N_side, N_side))

    phantom[N_side//2-60:N_side//2+60, N_side//2-60:N_side//2+60] = 1

    arg_shape = phantom.shape
    plt.figure('Phantom')
    plt.imshow(phantom)

    N_angle = 400
    N_offset = 400
  
    # Let's build the necessary components to instantiate the operator . ========================
    angles = np.linspace(0, 2*np.pi, N_angle, endpoint=False)
    n = np.stack([np.cos(angles), np.sin(angles)], axis=1)

    t = n[:, [1, 0]] * np.r_[-1, 1]  # <n, t> = 0
    t_max = pitch * N_side / 2 * 1.1  # 10% over ball radius
    t_offset = np.linspace(-t_max, t_max, N_offset, endpoint=True)

    n_spec = np.broadcast_to(n.reshape(N_angle, 1, 2), (N_angle, N_offset, 2))  # (N_angle, N_offset, 2)
    t_spec = t.reshape(N_angle, 1, 2) * t_offset.reshape(N_offset, 1)  # (N_angle, N_offset, 2)
    t_spec += pitch * N_side / 2  # Move anchor point to the center of volume.
    extra = np.ones((1,N_angle*N_offset,1))
    origin = (0., 0.)
    pitch = (1.,1.)

    print(arg_shape, t_spec.shape, n_spec.shape, origin, pitch)

    # Convert to cupy arrays
    phantom = cp.array(phantom)
    t_spec = cp.array(t_spec)
    n_spec = cp.array(n_spec)

    op = pxr.XRayTransform.init(
        arg_shape=arg_shape,
        t_spec=t_spec.reshape(-1,2), 
        n_spec=n_spec.reshape(-1,2),
        origin=origin,
        pitch=pitch,
    )
    op_splines_1 = xrt_splines.RayXRT(
        deg=1,
        arg_shape=arg_shape,
        t_spec=t_spec.reshape(-1,2), 
        n_spec=n_spec.reshape(-1,2),
        origin=origin,
        pitch=pitch,
    )
    op_splines_2 = xrt_splines.RayXRT(
        deg=2,
        arg_shape=arg_shape,
        t_spec=t_spec.reshape(-1,2), 
        n_spec=n_spec.reshape(-1,2),
        origin=origin,
        pitch=pitch,
    )
    
    fwd = op.apply(phantom.reshape(-1)).reshape((N_angle, N_offset)).get() #remove .get() for CPU
    fwd_splines_1 = op_splines_1.apply(phantom.reshape(-1)).reshape((N_angle, N_offset)).get() #remove .get() for CPU
    fwd_splines_2 = op_splines_2.apply(phantom.reshape(-1)).reshape((N_angle, N_offset)).get() #remove .get() for CPU

    plt.figure('Sino (Spline quadratic - pixels)')
    plt.imshow(fwd_splines_2 - fwd)


    bp_splines_2 = op_splines_2.adjoint(cp.array(fwd_splines_2).reshape(-1))
    bp_pixels = op.adjoint(cp.array(fwd).reshape(-1))

    plt.figure('Bp splines')
    plt.imshow(bp_splines_2.get().reshape(phantom.shape)[N_side//2-100:N_side//2+100, N_side//2-100:N_side//2+100], cmap='gray')
    plt.figure('Bp pixels')
    plt.imshow(bp_pixels.get().reshape(phantom.shape)[N_side//2-100:N_side//2+100, N_side//2-100:N_side//2+100], cmap='gray')




    plt.show()

    