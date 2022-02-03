## Benchmarks

* Yocto BSP 5.10.70_2.2.0 (modified with TF Lite 2.5.0)
* Input file:
FPS: 30
Frame Count: 151
Frame width: 520
Frame height: 520
Repetitions count: 10

Entries marked with `-` will not be measured.
Entries marked with `?` are to be measured.

| Model       | qm_cpu      | qm_gpu      | mp_cpu      | mp_npu      | num_det     | score_thold | mAP         |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| lite0       | 653.85      | 171.24      | (370.37)  ? | (869.56) ?  | 100         | 0           | 0.266       |
| lite0       | -           | -           | -           | -           | 25          | 0           | 0.262       |
| lite0       | -           | -           | -           | -           | 100         | 0.4         | 0.216       |
| lite0-quant | 253.35      | 178.48      | (335.57) ?  | (217.39) ?  | 100         | 0           | 0.262       |
| lite0-quant | -           | -           | -           | -           | 25          | 0           | 0.258       |
| lite1       | 1542.89     | 318.32      | ?           | ?           | 100         | 0           | 0.313       |
| lite1-quant | 392.69      | 341.73      | ?           | ?           | 100         | 0           | 0.309       |
| lite2       | ?           | ?           | ?           | ?           | 100         | 0           | 0.346       |
| lite2-quant | ?           | ?           | ?           | ?           | 100         | 0           | 0.342       |
| lite3       | -           | -           | -           | -           | 100         | 0           | 0.380       |
| lite3-quant | -           | -           | -           | -           | 100         | 0           | 0.376       |
| d0          | 2325.82     | 568.12      | (2,114.16) ?| (2444.98) ? | 100         | 0           | 0.331       |
| d0-quant    | 1226.87     | 1114.17     | (1,315.78) ?| (1052.63) ? | 100         | 0           | 0.187       |
| d1          | 6477.22     | 1147.42     | ?           | ?           | 100         | 0           | 0.383       |
| d1-quant    | 2257.88     | 2184.99     | ?           | ?           | 100         | 0           | 0.268       |

*Inference time measured in ms.*

Models are available at: https://nl-nxrm.sw.nxp.com:8443/#browse/browse:ml-nn-models:efficientdet-imx
