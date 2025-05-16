import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from data_utils import create_dataloader
from model import TrajectoryDiffusionModel

def main():
    # 创建数据加载器
    train_loader = create_dataloader(
        csv_path="trajectories.csv",
        batch_size=32,
        max_length=30,
        num_ids=3953,
        miss_ratio=0.3
    )
    
    # 创建模型
    model = TrajectoryDiffusionModel(
        num_ids=3953,
        embedding_dim=64,
        hidden_dim=128,
        num_layers=4,
        learning_rate=1e-4
    )
    
    # 设置检查点回调
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="trajectory-diffusion-{epoch:02d}-{train_loss:.2f}",
        save_top_k=3,
        monitor="train_loss",
        mode="min",
        every_n_epochs=1
    )
    
    # 创建训练器
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="gpu" if pl.utilities.device.is_cuda_available() else "cpu",
        callbacks=[checkpoint_callback],
        gradient_clip_val=1.0,
        log_every_n_steps=50
    )
    
    # 开始训练
    trainer.fit(model, train_loader)

if __name__ == "__main__":
    main()
