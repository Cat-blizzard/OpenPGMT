"""pytest 全局配置。

KMP_DUPLICATE_LIB_OK：Anaconda numpy(MKL) 与 torch 同时导入后，
MKL SVD 会因重复 OpenMP 运行时直接 Abort（服务器实测：torch 导入后
retarget 的 np.linalg.svd 崩溃）。必须在任何测试模块导入前设置。
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
