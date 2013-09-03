"""
CUDA Tool for SCons
"""

def CUDANVCCStaticObjectEmitter(target, source, env):
    import os
    import SCons.Defaults
    tgt, src = SCons.Defaults.StaticObjectEmitter(target, source, env)
    for file in src:
        linkfn = os.path.basename(
            os.path.splitext(src[0].rstr())[0]) + '.linkinfo'
        env.Clean(src, linkfn)
    return tgt, src
def CUDANVCCSharedObjectEmitter(target, source, env):
    import os
    import SCons.Defaults
    tgt, src = SCons.Defaults.SharedObjectEmitter(target, source, env)
    for file in src:
        linkfn = os.path.basename(
            os.path.splitext(src[0].rstr())[0]) + '.linkinfo'
        env.Clean(src, linkfn)
    return tgt, src

def find_paths(paths):
    """
    @param paths: paths to geuss.
    @type paths: list
    @return: the path found or None (not found).
    @rtype: str
    """
    import os
    for path in paths:
        if os.path.isdir(path):
            return path
    return ''

def determine_paths(env):
    """
    Fill the 'CUDA_TOOLKIT_PATH' into environment if it is not there.

    @return: the paths.
    @rtype: tuple
    """
    import sys
    import os
    from warnings import warn
    home = os.environ.get('HOME', '')
    programfiles = os.environ.get('PROGRAMFILES', '')
    homedrive = os.environ.get('HOMEDRIVE', '')

    # find CUDA Toolkit path and set CUDA_TOOLKIT_PATH.
    cudaToolkitPath = os.environ.get('CUDA_TOOLKIT_PATH', '')
    if not cudaToolkitPath:
        paths = [
            '/'.join([home, 'NVIDIA_CUDA_TOOLKIT']),
            '/'.join([home, 'Apps', 'NVIDIA_CUDA_TOOLKIT']),
            '/'.join([home, 'Apps', 'CudaToolkit']),
            '/'.join([home, 'Apps', 'CudaTK']),
            '/'.join(['/usr', 'local', 'NVIDIA_CUDA_TOOLKIT']),
            '/'.join(['/usr', 'local', 'CUDA_TOOLKIT']),
            '/'.join(['/usr', 'local', 'cuda_toolkit']),
            '/'.join(['/usr', 'local', 'CUDA']),
            '/'.join(['/usr', 'local', 'cuda']),
            '/'.join(['/Developer', 'NVIDIA CUDA TOOLKIT']),
            '/'.join(['/Developer', 'CUDA TOOLKIT']),
            '/'.join(['/Developer', 'CUDA']),
            '/'.join([programfiles, 'NVIDIA Corporation',
                'NVIDIA CUDA TOOLKIT']),
            '/'.join([programfiles, 'NVIDIA Corporation', 'NVIDIA CUDA']),
            '/'.join([programfiles, 'NVIDIA Corporation', 'CUDA TOOLKIT']),
            '/'.join([programfiles, 'NVIDIA Corporation', 'CUDA']),
            '/'.join([programfiles, 'NVIDIA', 'NVIDIA CUDA TOOLKIT']),
            '/'.join([programfiles, 'NVIDIA', 'NVIDIA CUDA']),
            '/'.join([programfiles, 'NVIDIA', 'CUDA TOOLKIT']),
            '/'.join([programfiles, 'NVIDIA', 'CUDA']),
            '/'.join([programfiles, 'CUDA TOOLKIT']),
            '/'.join([programfiles, 'CUDA']),
            '/'.join([homedrive, 'CUDA TOOLKIT']),
            '/'.join([homedrive, 'CUDA']),
        ]
        cudaToolkitPath = find_paths(paths)
        if cudaToolkitPath:
            sys.stdout.write(
                'scons: CUDA Toolkit found in %s\n' % cudaToolkitPath)
        else:
            warn('Cannot find the CUDA Toolkit path. '
                'Please set it to CUDA_TOOLKIT_PATH environment variable.')
    env['CUDA_TOOLKIT_PATH'] = cudaToolkitPath

    return cudaToolkitPath

def generate(env):
    """
    In order to use this tool, user must have 
    CUDA_TOOLKIT_PATH environmental variable defined.
    """
    import os
    import SCons.Tool
    import SCons.Scanner.C
    cudaToolkitPath = determine_paths(env)

    # scanners and builders.
    CUDAScanner = SCons.Scanner.C.CScanner()
    staticObjBuilder, sharedObjBuilder = SCons.Tool.createObjBuilders(env);
    staticObjBuilder.add_action('.cu', '$STATICNVCCCMD')
    staticObjBuilder.add_emitter('.cu', CUDANVCCStaticObjectEmitter)
    sharedObjBuilder.add_action('.cu', '$SHAREDNVCCCMD')
    sharedObjBuilder.add_emitter('.cu', CUDANVCCSharedObjectEmitter)
    SCons.Tool.SourceFileScanner.add_scanner('.cu', CUDAScanner)

    # build commands.
    env['STATICNVCCCMD'] = ' '.join([
        '$NVCC',
	'$NVCCDEFINES',
        '$NVCCPATH',
        '$NVCCFLAGS',
        '$STATICNVCCFLAGS',
        '-o $TARGET',
        '-c $SOURCES',
    ])
    env['SHAREDNVCCCMD'] = ' '.join([
        '$NVCC',
	'$NVCCDEFINES',
        '$NVCCPATH',
        '$NVCCFLAGS',
        '$SHAREDNVCCFLAGS',
        '$ENABLESHAREDNVCCFLAG',
        '-o $TARGET',
        '-c $SOURCES',
    ])

    # compiler.
    env['NVCC'] = 'nvcc'
    env.PrependENVPath('PATH', '/'.join([cudaToolkitPath, 'bin']))

    # defines

    for i, p in enumerate(env['NVCCDEFINES']):
	env['NVCCDEFINES'][i] = '-D' + p

    # includes

    for i, p in enumerate(env['NVCCPATH']):
	env['NVCCPATH'][i] = '-I' + p

    env['STATICNVCCFLAGS'] = ''
    env['SHAREDNVCCFLAGS'] = ''
    env['ENABLESHAREDNVCCFLAG'] = '-shared'

    # libraries.

    env.Append(LIBPATH=[
        '/'.join([cudaToolkitPath, 'lib64']),
        '/'.join([cudaToolkitPath, 'lib']),
    ])

def exists(env):
    return env.Detect('nvcc')

