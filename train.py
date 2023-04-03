import os
import time
import datetime
import torch
from torch.nn.parallel import DistributedDataParallel
from tensorboardX import SummaryWriter
from loguru import logger


from utils.utils import tensor2float
from utils.io_utils import save_scalars
from models import NeuralRecon
from config import cfg, update_config, args
from ops.comm import is_main_process, synchronize

from datasets import  dataloader
from utils.utils import tocuda, todevice


class Trainer():
    def __init__(self, model, cfg, distributed):
        self.cfg = cfg
        self.model = model
        self.distributed = distributed

        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                        lr=cfg.TRAIN.LR, 
                                        betas=(0.9, 0.999), 
                                        weight_decay=cfg.TRAIN.WD)
        self.load_params()
        
        milestones = [int(epoch_idx) for epoch_idx in cfg.TRAIN.LREPOCHS.split(':')[0].split(',')]
        lr_gamma = 1 / float(cfg.TRAIN.LREPOCHS.split(':')[1])
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones, gamma=lr_gamma,
                                                            last_epoch=self.start_epoch - 1)

        self.TrainImgLoader = dataloader.build_dataloader(cfg, 'train')

    def load_params(self):
        cfg = self.cfg
        # load parameters
        if cfg.RESUME:
            saved_models = [fn for fn in os.listdir(cfg.LOGDIR) if fn.endswith(".ckpt")]
            saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
            if len(saved_models) != 0:
                # use the latest checkpoint file
                loadckpt = os.path.join(cfg.LOGDIR, saved_models[-1])
                logger.info("resuming " + str(loadckpt))
                map_location = {'cuda:%d' % 0: 'cuda:%d' % cfg.LOCAL_RANK}
                state_dict = torch.load(loadckpt, map_location=map_location)
                self.model.load_state_dict(state_dict['model'], strict=False)
                self.optimizer.param_groups[0]['initial_lr'] = state_dict['optimizer']['param_groups'][0]['lr']
                self.optimizer.param_groups[0]['lr'] = state_dict['optimizer']['param_groups'][0]['lr']
                self.start_epoch = state_dict['epoch'] + 1
            else:
                self.start_epoch = 0
        elif cfg.LOADCKPT != '':
            # load checkpoint file specified by args.loadckpt
            logger.info("loading model {}".format(cfg.LOADCKPT))
            map_location = {'cuda:%d' % 0: 'cuda:%d' % cfg.LOCAL_RANK}
            state_dict = torch.load(cfg.LOADCKPT, map_location=map_location)
            self.model.load_state_dict(state_dict['model'])
            self.optimizer.param_groups[0]['initial_lr'] = state_dict['optimizer']['param_groups'][0]['lr']
            self.optimizer.param_groups[0]['lr'] = state_dict['optimizer']['param_groups'][0]['lr']
            self.start_epoch = state_dict['epoch'] + 1
        else:
            self.start_epoch = 0
    
    def train_batch(self, sample, global_step):
        model.train()
        self.optimizer.zero_grad()

        if self.distributed:
            sample = tocuda(sample)
        
        outputs, loss_dict = model(sample)

        if global_step%100==0:
            print_loss = 'Loss: '
            for k, v in loss_dict.items():
                print_loss += f'{k}: {v} '
            logger.info(print_loss)

        loss = loss_dict['total_loss']
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        self.optimizer.step()
        return tensor2float(loss), tensor2float(loss_dict)

    def train(self, tb_writer):
        cfg = self.cfg
        TrainImgLoader = self.TrainImgLoader

        logger.info("start at epoch {}".format(self.start_epoch))
        logger.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in self.model.parameters()])))
        
        for epoch_idx in range(self.start_epoch, cfg.TRAIN.EPOCHS):
            logger.info('Epoch {}:'.format(epoch_idx))
            TrainImgLoader.dataset.epoch = epoch_idx
            TrainImgLoader.dataset.tsdf_cashe = {}
            # training
            for batch_idx, sample in enumerate(TrainImgLoader):
                global_step = len(TrainImgLoader) * epoch_idx + batch_idx
                do_summary = global_step % cfg.SUMMARY_FREQ == 0
                start_time = time.time()
                loss, scalar_outputs = self.train_batch(sample, global_step)
                if is_main_process():
                    logger.info(
                        'Epoch {}/{}, Iter {}/{}, train loss = {:.3f}, time = {:.3f}'.format(epoch_idx, cfg.TRAIN.EPOCHS,
                                                                                            batch_idx,
                                                                                            len(TrainImgLoader), loss,
                                                                                            time.time() - start_time))
                if do_summary and is_main_process() and tb_writer is not None:
                    save_scalars(tb_writer, 'train', scalar_outputs, global_step)
                del scalar_outputs
            
            self.lr_scheduler.step()
            # checkpoint
            if (epoch_idx + 1) % cfg.TRAIN.SAVE_FREQ == 0 and is_main_process():
                torch.save({
                    'epoch': epoch_idx,
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict()},
                    "{}/model_{:0>6}.ckpt".format(cfg.LOGDIR, epoch_idx))


    
if __name__ == '__main__':
    args_config = args()
    update_config(cfg, args_config)
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    print('number of gpus: {}'.format(num_gpus))
    distributed = num_gpus>1
    if distributed:
        if False:
            torch.distributed.init_process_group(
                backend="nccl", init_method="env://"
            )
            torch.cuda.set_device(cfg.LOCAL_RANK)
            # torch.distributed.init_process_group(backend='nccl')
            # torch.cuda.set_device(int(local_rank))
            synchronize()
            # torch.cuda.synchronize(device=cfg.LOCAL_RANK)
        else:
            from ops.comm import setup_DDP
            rank, local_rank, world_size, device = setup_DDP(verbose=True)
            device = torch.device("cuda:{}".format(local_rank))
            # logger.info(f'Device:{device}...GPU:{local_rank}..')
    else:
        local_rank = 0
        device = torch.device("cuda:0")
    cfg.defrost()
    cfg.LOCAL_RANK = local_rank
    cfg.DISTRIBUTED = num_gpus > 1
    cfg.freeze()
    # local_rank = os.environ["LOCAL_RANK"]
    # print('LocalRank: {}..{}..{}'.format(local_rank, cfg.LOCAL_RANK, args_config.local_rank)) 

    # print(f"Synchronized: rank{cfg.LOCAL_RANK}")
    # torch.manual_seed(cfg.SEED)
    # torch.cuda.manual_seed(cfg.SEED)

    # create logger
    if not os.path.isdir(cfg.LOGDIR):
        os.makedirs(cfg.LOGDIR)

    current_time_str = str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    logfile_path = os.path.join(cfg.LOGDIR, f'{current_time_str}_{cfg.MODE}.log')
    logger.info('creating log file', logfile_path)
    logger.add(logfile_path, format="{time} {level} {message}", level="INFO")


    tb_writer = SummaryWriter(cfg.LOGDIR)
    

    # model
    model = NeuralRecon(cfg, mode=args_config.mode)
    print(f"Build Model: rank{cfg.LOCAL_RANK}, {local_rank}, distributed {cfg.DISTRIBUTED}")
    if cfg.DISTRIBUTED:
        model.to(device)
        model = DistributedDataParallel(
            model, 
            device_ids= [local_rank], #[cfg.LOCAL_RANK], 
            output_device=local_rank, #cfg.LOCAL_RANK,
            # this should be removed if we update BatchNorm stats
            # broadcast_buffers=qFalse,
            # find_unused_parameters=True
        )
    else:
        model = torch.nn.DataParallel(model, device_ids=[0])
        model.cuda()

    logger.info(f"Start traning: rank{local_rank}")
    trainer = Trainer(model, cfg, cfg.DISTRIBUTED)
    trainer.train(tb_writer)
    