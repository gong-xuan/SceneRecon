import argparse



def args():
    parser = argparse.ArgumentParser(description='A PyTorch Implementation of NeuralRecon')

    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        # required=True,
                        default='neuralrecon',
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # distributed training
    # parser.add_argument('--gpu', default='0,1', type=str, help='GPU device ID (default: -1)')
    parser.add_argument('--world-size',
                        default=1,
                        type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--dist-url',
                        default='tcp://127.0.0.1:23456',
                        type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--local_rank',
                        default=0,
                        type=int,
                        help='node rank for distributed training') 
    parser.add_argument('--proj_wdepth', default=0, type=float)
    parser.add_argument('--concat_depth', action='store_true')
    parser.add_argument('--val', action='store_true')
    # parser.add_argument('--evalp3d', action='store_true')
    parser.add_argument('--agg3dv', default='', type=str)
    parser.add_argument('--dataset', default='', type=str)

    parser.add_argument('--log', default='debug', type=str)
    parser.add_argument('--batchsize', default='', type=str)
    parser.add_argument('--catpdepth', action='store_true')
    # parser.add_argument('--vol_org_z', default=-0.2, type=str)
    parser.add_argument('--mode', default='', type=str) #neuv1
    parser.add_argument('--tsdf', default='1,1,1', type=str)
    parser.add_argument('--tsch', default=24, type=int)
    parser.add_argument('--tenso', default='', type=str) #smpl, ms, or default
    parser.add_argument('--fuse', default='concat', type=str)
    parser.add_argument('--up', default='nearest', type=str)
    parser.add_argument('--newgridmask', action='store_true')
    parser.add_argument('--tvloss', default=0, type=float)
    parser.add_argument('--tvcloss', default=0, type=float)
    parser.add_argument('--tvzloss', default=0, type=float)
    parser.add_argument('--concatimz', default=1, type=int)
    parser.add_argument('--coarse2d', action='store_true')#cannot be 1 due to limited memory
    parser.add_argument('--msloss', action='store_true')
    # for pretrain
    parser.add_argument('--pretrain', action='store_true')
    # for test
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--eps', default=47, type=int)
    parser.add_argument('--epe', default=48, type=int)

    # parse arguments and check
    args = parser.parse_args()

    return args