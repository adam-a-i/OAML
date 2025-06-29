import torch
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning import seed_everything
import config
import os
from utils import dotdict
import train_val
import data
import inspect
import time
import traceback


def main(args):
    print("\n" + "="*80)
    print("STARTING ADAFACE + TRANSMATCHER TRAINING")
    print("="*80)
    
    start_time = time.time()
    
    print(f"[MAIN] Starting training with args:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")

    hparams = dotdict(vars(args))

    # Set up TransMatcher parameters (adjusted for divisibility)
    hparams.transmatcher_params = {
        'seq_len': 49,  # 7x7 for 112x112 input
        'd_model': 512,  # Match the actual feature map channels from backbone
        'num_decoder_layers': 3,  # Original value
        'dim_feedforward': 2048,  # Original value
    }
    print(f"[MAIN] TransMatcher params: {hparams.transmatcher_params}")
    
    # Note: TransMatcher is now integrated into the backbone, no wrapper needed
    print("[MAIN] TransMatcher is integrated into the backbone (like QAConv in root implementation)")

    print("[MAIN] Initializing trainer module...")
    trainer_start = time.time()
    trainer_mod = train_val.Trainer(**hparams)
    print(f"[MAIN] Trainer module initialized in {time.time() - trainer_start:.2f}s")

    print("[MAIN] Initializing data module...")
    data_start = time.time()
    data_mod = data.DataModule(**hparams)
    print(f"[MAIN] Data module initialized in {time.time() - data_start:.2f}s")

    if hparams.seed is not None:
        print(f"[MAIN] Setting random seed: {hparams.seed}")
        seed_everything(hparams.seed)

    # create model checkpoint callback
    print("[MAIN] Setting up checkpoint callback...")
    monitor = 'val_combined_acc_epoch'
    mode = 'max'
    save_top_k = hparams.epochs+1 if hparams.save_all_models else 1
    checkpoint_callback = ModelCheckpoint(dirpath=hparams.output_dir, save_last=True,
                                          save_top_k=save_top_k, monitor=monitor, mode=mode)
    print(f"[MAIN] Checkpoint callback created with monitor={monitor}, mode={mode}, save_top_k={save_top_k}")

    # create logger
    print("[MAIN] Setting up loggers...")
    csv_logger = CSVLogger(save_dir=hparams.output_dir, name='result')
    my_loggers = [csv_logger]
    if args.use_wandb:
        print("[MAIN] Adding WandB logger...")
        wandb_logger = WandbLogger(save_dir=hparams.output_dir,
                                   name=os.path.basename(args.output_dir), project='adaface_face_recognition')
        my_loggers.append(wandb_logger)

    resume_from_checkpoint = hparams.resume_from_checkpoint if hparams.resume_from_checkpoint else None
    if resume_from_checkpoint:
        print(f"[MAIN] Resuming from checkpoint: {resume_from_checkpoint}")

    print("[MAIN] Creating PyTorch Lightning trainer...")
    trainer_start = time.time()
    
    params = inspect.signature(pl.Trainer).parameters.values()
    if 'strategy' in [param.name for param in params]:
        # recent pytorch lightning
        print("[MAIN] Using recent PyTorch Lightning version...")
        trainer = pl.Trainer(resume_from_checkpoint=resume_from_checkpoint,
                             default_root_dir=hparams.output_dir,
                             logger=my_loggers,
                             gpus=hparams.gpus,
                             max_epochs=hparams.epochs,
                             accelerator='cpu' if hparams.gpus == 0 else 'gpu',
                             strategy=hparams.distributed_backend,
                             precision=16 if hparams.use_16bit else 32,
                             fast_dev_run=hparams.fast_dev_run,
                             callbacks=[checkpoint_callback],
                             num_sanity_val_steps=0,
                             val_check_interval=1.0,
                             limit_train_batches=50 if hparams.test_run else 1.0,
                             # gradient_clip_val=1.0  # Not supported with manual optimization
                             )
    else:
        # pytorch lightning before 1.4.4
        print("[MAIN] Using older PyTorch Lightning version...")
        trainer = pl.Trainer(resume_from_checkpoint=resume_from_checkpoint,
                             default_root_dir=hparams.output_dir,
                             logger=my_loggers,
                             gpus=hparams.gpus,
                             max_epochs=hparams.epochs,
                             accelerator=hparams.distributed_backend,
                             precision=16 if hparams.use_16bit else 32,
                             fast_dev_run=hparams.fast_dev_run,
                             callbacks=[checkpoint_callback],
                             num_sanity_val_steps=0,
                             val_check_interval=1.0,
                             limit_train_batches=50 if hparams.test_run else 1.0,
                             # gradient_clip_val=1.0  # Not supported with manual optimization
                             # accumulate_grad_batches=hparams.accumulate_grad_batches,  # Not supported with manual optimization
                             )
    
    print(f"[MAIN] PyTorch Lightning trainer created in {time.time() - trainer_start:.2f}s")
    print(f"[MAIN] Trainer configuration:")
    print(f"  - Max epochs: {hparams.epochs}")
    print(f"  - GPUs: {hparams.gpus}")
    print(f"  - Precision: {16 if hparams.use_16bit else 32}")
    print(f"  - Distributed backend: {hparams.distributed_backend}")
    print(f"  - Test run: {hparams.test_run}")

    if not hparams.evaluate:
        # train / val
        print('\n[MAIN] Starting training phase...')
        training_start = time.time()
        
        try:
            trainer.fit(trainer_mod, data_mod)
            print(f'[MAIN] Training completed in {time.time() - training_start:.2f}s')
        except Exception as e:
            print(f'[MAIN] ERROR during training: {e}')
            print(traceback.format_exc())
            raise
        
        print('[MAIN] Starting evaluation phase...')
        evaluation_start = time.time()
        print('evaluating from ', checkpoint_callback.best_model_path)
        
        try:
            trainer.test(ckpt_path='best', datamodule=data_mod)
            print(f'[MAIN] Evaluation completed in {time.time() - evaluation_start:.2f}s')
        except Exception as e:
            print(f'[MAIN] ERROR during evaluation: {e}')
            print(traceback.format_exc())
            raise
    else:
        # eval only
        print('[MAIN] Starting evaluation-only mode...')
        evaluation_start = time.time()
        
        try:
            trainer.test(trainer_mod, datamodule=data_mod)
            print(f'[MAIN] Evaluation completed in {time.time() - evaluation_start:.2f}s')
        except Exception as e:
            print(f'[MAIN] ERROR during evaluation: {e}')
            print(traceback.format_exc())
            raise

    total_time = time.time() - start_time
    print(f"\n[MAIN] Total execution time: {total_time:.2f}s")
    print("="*80)
    print("TRAINING COMPLETED")
    print("="*80)


if __name__ == '__main__':
    print("\n" + "="*80)
    print("ADAFACE + TRANSMATCHER TRAINING SCRIPT")
    print("="*80)

    args = config.get_args()

    if args.distributed_backend == 'ddp' and args.gpus > 0:
        print(f"[MAIN] Using DDP with {args.gpus} GPUs, adjusting batch size...")
        # When using a single GPU per process and per
        # DistributedDataParallel, we need to divide the batch size
        # ourselves based on the total number of GPUs we have
        torch.set_num_threads(1)
        args.total_batch_size = args.batch_size
        args.batch_size = int(args.batch_size / max(1, args.gpus))
        args.num_workers = min(args.num_workers, 2)  # Further reduced max workers
        print(f"[MAIN] Adjusted batch size: {args.total_batch_size} -> {args.batch_size}")
        
        # Force smaller batch size for memory constraints
        if args.batch_size > 4:
            args.batch_size = 4
            print(f"[MAIN] Further reduced batch size to {args.batch_size} for memory constraints")

    if args.resume_from_checkpoint:
        assert args.resume_from_checkpoint.endswith('.ckpt')
        args.output_dir = os.path.dirname(args.resume_from_checkpoint)
        print(f'[MAIN] Resuming from checkpoint: {args.output_dir}')

    # Print TransMatcher status
    print('[MAIN] TransMatcher branch ENABLED with hardcoded params:')
    print('  seq_len=49, d_model=512, num_layers=3, dim_feedforward=2048, loss_weight=0.5')

    print(f"[MAIN] Output directory: {args.output_dir}")
    print(f"[MAIN] Data root: {args.data_root}")
    print(f"[MAIN] Architecture: {args.arch}")
    print(f"[MAIN] Head type: {args.head}")

    main(args)
