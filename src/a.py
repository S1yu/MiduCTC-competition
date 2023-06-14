from src.baseline.trainer import TrainerCtc

if __name__ == '__main__':

    config = {
        "out_model_dir":"./",
        'in_model_dir':"hfl/chinese-macbert-base",
        "learning_rate":4e-5,
        "max_seq_len":256,
        "train_fp":'D:\project\working\XF\data\example_input.json',
        "dev_fp":'D:\project\working\XF\data\example_input.json',
        "test_fp":'D:\project\working\XF\data\example_input.json',
        "batch_size":4,
        "random_seed_num":42,
        "early_stop_times":1,
        "freeze_embedding":1,
        "max_grad_norm":1,
        "dev_data_ratio":1,
        "n_fold":4,
        "with_train_epoch_metric":1,
    }


    for i in range(4):
        trainer = TrainerCtc(epochs=i,**config, fold = i)
        trainer.train()