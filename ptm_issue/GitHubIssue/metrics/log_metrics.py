

def log_metrics(logger, metrics_dict, prefix='', global_step=0):
    for key, value in metrics_dict.items():
        if isinstance(value, dict):
            # 如果值是字典，递归调用
            log_metrics(logger, value, prefix=f'{prefix}{key}_', global_step=global_step)
        else:
            # 记录指标
            if "precision" or "recall" in f'{prefix}{key}':
                logger.experiment.add_scalar(f'{prefix}{key}', value, global_step=global_step)
