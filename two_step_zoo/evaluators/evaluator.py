from inspect import getmembers, isfunction
import pdb 

from . import metrics


metric_fn_dict = dict(getmembers(metrics, predicate=isfunction))


class Evaluator:
    def __init__(self, module, *,
                 valid_loader, test_loader,
                 valid_metrics=None,
                 test_metrics=None,
                 metric_kwargs=None):
        
        self.module = module
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.valid_metrics = valid_metrics or {}
        self.test_metrics = test_metrics or valid_metrics
        self.valid_cache = {}
        self.test_cache = {}
        self.metric_kwargs = metric_kwargs or {}

    def evaluate(self, dataloader, metric, cache=None):
        assert metric in metric_fn_dict, f"Metric name {metric} not present in `metrics.py`"
      
        metric_fn = metric_fn_dict[metric]
        metric_fn_kwargs = self.metric_kwargs.get(metric) or {}
        self.module.eval()
        return metric_fn(self.module, dataloader, cache=cache, **metric_fn_kwargs)

    def validate(self):
        print(f"Validating {self.valid_metrics} for {self.module.model_type}")
        return self._create_metrics_dict(self.valid_metrics, self.valid_loader, self.valid_cache)

    def test(self):
        print(f"Testing {self.test_metrics} for {self.module.model_type}")
        return self._create_metrics_dict(self.test_metrics, self.test_loader, self.test_cache)

    def _create_metrics_dict(self, metrics, loader, cache):
        metrics_dict = {}
        for metric in metrics:
            metric_result = self.evaluate(loader, metric, cache=cache)
            if isinstance(metric_result, dict):
                metrics_dict.update(metric_result)
            else:
                metrics_dict[metric] = metric_result
        return metrics_dict



class NullEvaluator(Evaluator):
    def __init__(self, module, *, valid_loader, test_loader):
        super().__init__(
            module, valid_loader=valid_loader, test_loader=test_loader,
            valid_metrics={"null_metric"},
        )


class AutoencodingEvaluator(Evaluator):
    def __init__(self, module, *, valid_loader, test_loader):
        super().__init__(
            module, valid_loader=valid_loader, test_loader=test_loader,
            valid_metrics={"l2_reconstruction_error"},
        )


class ImageEvaluator(Evaluator):
    def __init__(self, module, *, train_loader, valid_loader, test_loader):
        super().__init__(
            module, valid_loader=valid_loader, test_loader=test_loader,
            valid_metrics={"l2_reconstruction_error"},
            test_metrics={"l2_reconstruction_error", "fid"},
            metric_kwargs={
                "fid": {
                    "train_loader": train_loader,
                }
            }
        )


class ImageFIDEvaluator(Evaluator):
    def __init__(self, module, *, train_loader, valid_loader, test_loader):
        super().__init__(
            module, valid_loader=valid_loader, test_loader=test_loader,
            valid_metrics={"l2_reconstruction_error", "fid"},
            test_metrics={"l2_reconstruction_error", "fid"},
            metric_kwargs={
                "fid": {
                    "train_loader": train_loader,
                }
            }
        )


class OODEvaluator(Evaluator):
    def __init__(
            self,
            module,
            *,
            is_test_loader,
            oos_test_loader,
            is_train_loader,
            oos_train_loader,
            savedir,
            low_dim
        ):
        super().__init__(
            module, valid_loader=None, test_loader=is_test_loader,
            valid_metrics=None,
            test_metrics={"likelihood_ood_acc"},
            metric_kwargs={
                "likelihood_ood_acc": {
                    "oos_test_loader": oos_test_loader,
                    "is_train_loader": is_train_loader,
                    "oos_train_loader": oos_train_loader,
                    "savedir": savedir,
                    "low_dim": low_dim
                }
            }
        )