import os
import torch
from tensorboardX import SummaryWriter
from loguru import logger
import ray
import json

from utils.utils import tensor2float, DictAverageMeter, make_nograd_func, split_list
from utils.io_utils import save_scalars, SaveScene
from utils.eval_utils import EvalMetrics
from models import NeuralRecon
from config import cfg, update_config, args
from datasets import  dataloader
from utils.io_utils import print_metrics


class Infer():
    def __init__(self, model, cfg, tb_writer):
        # dataloader
        self.TestImgLoader = dataloader.build_dataloader(cfg, 'test')
        logger.info(f"Test Batchsize: {cfg.TEST.BATCH_SIZE}")
        # logger.info(f"TestData: {len(self.TestImgLoader)}") 
        self.model = model
        self.model.eval()
        self.cfg = cfg
        self.tb_writer = tb_writer

    def is_exist_preds(self, pred_dir, scene_list):
        for scene in scene_list:
            if not os.path.exists(f'{pred_dir}/{scene}.ply'):
                return False
        return True

    def __call__(self, ckpt, pred_dir, scene_list):
        loadckpt = os.path.join(self.cfg.LOGDIR, ckpt)
        if not os.path.exists(loadckpt):
            logger.info(f"Not exist {loadckpt}")
            return False
        elif self.is_exist_preds(pred_dir, scene_list):
            logger.info(f"Exist {len(scene_list)} predicted mesh")
            return True
        
        logger.info("Loading " + str(loadckpt))
        state_dict = torch.load(loadckpt)
        self.model.load_state_dict(state_dict['model'])
        self.infer_model(state_dict['epoch'], pred_dir)
        return True

    @make_nograd_func
    def infer_model(self, epoch_idx, save_dir):
        self.TestImgLoader.dataset.tsdf_cashe = {}
        avg_test_scalars = DictAverageMeter()
        savescene = SaveScene(self.cfg, save_dir)

        for batch_idx, sample in enumerate(self.TestImgLoader):
            save_scene = (batch_idx == len(self.TestImgLoader) - 1) #last
            outputs, loss_dict = model(sample, save_scene)
            scalar_outputs = tensor2float(loss_dict)
            avg_test_scalars.update(scalar_outputs)
            del scalar_outputs
            # import ipdb; ipdb.set_trace()
            savescene(outputs) #each scene: overid by the last one
        
        logger.info(avg_test_scalars.mean())
        
        if self.tb_writer is not None:
            save_scalars(tb_writer, 'fulltest', avg_test_scalars.mean(), epoch_idx)#test loss in tensorboard





def eval_multiprocess(pred_dir, info_files):
    all_proc = n_proc * n_gpu
    info_files = split_list(info_files, all_proc)
    ray.init(num_cpus=all_proc * n_cpu_proc, num_gpus=n_gpu, ignore_reinit_error=True)

    ray_worker_ids = []
    for w_idx in range(all_proc):
        ray_worker_ids.append(process_with_single_worker.remote(pred_dir, info_files[w_idx]))

    results = ray.get(ray_worker_ids)
    metrics = {}
    for r in results:
        metrics.update(r)
    
    rslt_file = f'{pred_dir}/metrics_new.json'
    json.dump(metrics, open(rslt_file, 'w'))
    metric = print_metrics(rslt_file)
    json.dump(metric, open(rslt_file, 'a'))
    logger.info(f"Completed: {rslt_file}")
    ray.shutdown()


if __name__ == '__main__':
    n_proc = 2
    n_cpu_proc = 8
    args_config = args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args_config.gpu
    update_config(cfg, args_config)
    n_gpu = len(args_config.gpu.split(','))

    # create logger
    assert os.path.isdir(cfg.LOGDIR)
    # current_time_str = str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    logfile_path = os.path.join(cfg.LOGDIR, f'test.log')#TODO
    print('creating log file', logfile_path)
    logger.add(logfile_path, format="{time} {level} {message}", level="INFO")

    tb_writer = None #SummaryWriter(f"{cfg.LOGDIR}/test")
    
    # model
    model = NeuralRecon(cfg, mode=args_config.mode)
    model = torch.nn.DataParallel(model, device_ids=[0])

    # multi-process
    @ray.remote(num_cpus=n_cpu_proc, num_gpus=(1 / n_proc))
    def process_with_single_worker(pred_dir, scene_list):
        return eval(pred_dir, scene_list)

    

    # test
    infer = Infer(model, cfg, tb_writer)
    eval = EvalMetrics(cfg, 'cuda', eval3D=True, loader_num_workers=n_proc*n_gpu)
    scene_list = sorted(os.listdir(cfg.TEST.PATH))

    for epoch in range(args_config.eps, args_config.epe):
        ckpt = f'model_0000{epoch}.ckpt'
        pred_dir = f'{cfg.TEST.SAVE_SCENE_MESH}_ep{epoch}'

        pred_finished = infer(ckpt, pred_dir, scene_list)

        if pred_finished:
            if n_proc>1:
                eval_multiprocess(pred_dir, scene_list)
            else:
                metrics = eval(pred_dir, scene_list)

                rslt_file = f'{pred_dir}/metrics_new.json'
                json.dump(metrics, open(rslt_file, 'w'))
                metric = print_metrics(rslt_file)
                json.dump(metric, open(rslt_file, 'a'))
                logger.info(f"Completed: {rslt_file}")